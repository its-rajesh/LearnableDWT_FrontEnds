"""
Effective Bleed Matrix — TensorFlow GPU implementation
=======================================================
Handles batched data of shape (N, S_or_M, T):
    Xtest : (N, S, T)  — clean source signals
    Ytest : (N, M, T)  — microphone recordings

Key change vs v1: per-(mic, source) delay estimation.
A single global delay fails on real re-recordings where each source-mic
pair has a different propagation lag. This version estimates and corrects
delays independently for every (m, s) pair before solving the FIR model.

Example
-------
    from bleed_matrix_tf import BleedMatrixTF, batch_summary
    bm      = BleedMatrixTF(K=512, lam=1e-3, sr=22050)
    results = bm.compute_batch(Xtest, Ytest)

    print(results[0].B)           # (M, S) bleed matrix
    print(results[0].delays_ms)   # (M, S) per-pair delays in ms
    print(batch_summary(results))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    raise ImportError("TensorFlow not found.  pip install tensorflow")


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class BleedResult:
    """All metrics for a single (sources, mics) example."""

    B:               np.ndarray   # (M, S)  normalised bleed matrix
    E:               np.ndarray   # (M, S)  unnormalised energy matrix
    unmodeled_ratio: np.ndarray   # (M,)    residual-based U_m = SSE/SST
    legacy_unmodeled_ratio: np.ndarray  # (M,) legacy U from fitted component energies
    unmodeled_ratio_uncentered: np.ndarray  # (M,) SSE / signal energy
    bleed_off:       float        # scalar  mean off-diagonal bleed
    sir_db:          np.ndarray   # (M,)    per-mic SIR in dB
    ltr_db:          np.ndarray   # (M,)    Leakage-to-Target Ratio in dB
    delays:          np.ndarray   # (M, S)  per-pair delays in samples
    sr:              int          # sample rate (for unit conversion)

    mean_unmodeled: float      = field(init=False)
    delays_ms:      np.ndarray = field(init=False)

    def __post_init__(self):
        self.mean_unmodeled = float(np.mean(self.unmodeled_ratio))
        self.delays_ms      = (self.delays / self.sr * 1000).astype(np.float32)

    def summary(self) -> str:
        d = self.delays
        return "\n".join([
            "── Bleed Matrix Summary ──────────────────────────",
            f"  Shape              : {self.B.shape}  (M x S)",
            f"  Per-pair delays    : min={d.min():+d}  max={d.max():+d}"
            f"  mean={d.mean():+.0f} samples",
            f"  Mean unmodeled U   : {self.mean_unmodeled:.4f}",
            f"  Mean legacy U      : {float(np.mean(self.legacy_unmodeled_ratio)):.4f}",
            f"  BleedOff           : {self.bleed_off:.4f}",
            f"  SIR (mean)   [dB]  : {float(np.mean(self.sir_db)):.2f}",
            f"  LTR (mean)   [dB]  : {float(np.mean(self.ltr_db)):.2f}",
            "──────────────────────────────────────────────────",
        ])


# ─── Main class ───────────────────────────────────────────────────────────────

class BleedMatrixTF:
    """
    TensorFlow GPU-accelerated Effective Bleed Matrix with per-pair delay
    correction.

    Parameters
    ----------
    K : int
        FIR filter length in samples (default 512 ~23 ms @ 22.05 kHz).
        Use 2048-8192 for real room recordings.
    lam : float
        Tikhonov regularisation lambda > 0.
    sr : int
        Sample rate Hz (used for delay reporting in ms).
    max_delay_samples : int
        Search range +/- for per-pair delay estimation.
        Default 44100 (2 s @ 22.05 kHz) covers any realistic studio I/O
        latency. Increase if your device latency is known to be larger.
    use_envelope : bool
        If True, cross-correlation is computed on the signal envelope |x|
        rather than the raw waveform. More robust for tonal/harmonic sources
        where raw xcorr can be periodic and pick spurious peaks. Default True.
    verbose : bool
        Print per-example progress, delay matrices and U values.
    """

    def __init__(
        self,
        K:                 int   = 512,
        lam:               float = 1e-3,
        sr:                int   = 22050,
        max_delay_samples: int   = 44100,
        use_envelope:      bool  = True,
        verbose:           bool  = False,
    ):
        self.K                 = K
        self.lam               = lam
        self.sr                = sr
        self.max_delay_samples = max_delay_samples
        self.use_envelope      = use_envelope
        self.verbose           = verbose

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self._device = "/GPU:0"
        else:
            warnings.warn("No GPU found — running on CPU.")
            self._device = "/CPU:0"

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_batch(
        self,
        sources: np.ndarray,   # (N, S, T)
        mics:    np.ndarray,   # (N, M, T)
    ) -> List[BleedResult]:
        """Process a batch, return list of N BleedResult objects."""
        sources = np.asarray(sources, dtype=np.float32)
        mics    = np.asarray(mics,    dtype=np.float32)

        if sources.ndim == 2:
            sources = sources[np.newaxis]
            mics    = mics[np.newaxis]

        N = sources.shape[0]
        results = []
        for n in range(N):
            if self.verbose:
                print(f"  Example {n+1}/{N} ...", flush=True)
            results.append(self._compute_single(sources[n], mics[n]))
        return results

    def compute(
        self,
        sources: np.ndarray,   # (S, T)
        mics:    np.ndarray,   # (M, T)
    ) -> BleedResult:
        """Process a single example."""
        return self._compute_single(
            np.asarray(sources, dtype=np.float32),
            np.asarray(mics,    dtype=np.float32),
        )

    # ── Single-example pipeline ───────────────────────────────────────────────

    def _compute_single(self, sources: np.ndarray, mics: np.ndarray) -> BleedResult:

        # Step 1: estimate per-(m,s) delays  (M, S)
        delays = self._estimate_per_pair_delays(sources, mics)

        if self.verbose:
            print(f"    delays (samples):\n{delays}")
            print(f"    delays (ms):\n{np.round(delays / self.sr * 1000, 1)}")

        # Step 2: build delay-corrected source array  (M, S, T)
        sources_aligned = _apply_per_pair_delays(sources, delays)

        # Step 3: FIR energy estimation and fitted waveform on GPU.
        E, fit = self._estimate_energy_tf(sources_aligned, mics)

        # Step 4: unmodeled ratios (M,).
        # Corrected U is residual energy divided by centered mic energy, so
        # EXP = 100 * (1 - U) is a standard linear-fit R^2-style score.
        mics64 = mics.astype(np.float64)
        residual = mics64 - fit.astype(np.float64)
        centered = mics64 - np.mean(mics64, axis=1, keepdims=True)
        sse = np.sum(residual ** 2, axis=1)
        sst = np.sum(centered ** 2, axis=1)
        mic_power = np.sum(mics64 ** 2, axis=1)
        U = (sse / np.maximum(sst, 1e-12)).astype(np.float32)
        U0 = (sse / np.maximum(mic_power, 1e-12)).astype(np.float32)

        # Legacy U is retained for backward comparison. It mixed fitted
        # component energies with mic signal energy, which is not a residual.
        explained = E.sum(axis=1).astype(np.float64)
        U_legacy = (1.0 - np.clip(
            explained / np.where(mic_power == 0, 1.0, mic_power), 0.0, 1.0
        )).astype(np.float32)

        if self.verbose:
            print(f"    U per mic: {np.round(U, 4)}")
            print(f"    legacy U per mic: {np.round(U_legacy, 4)}")

        # Step 5: row-normalise -> bleed matrix  (M, S)
        row_sum = E.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        B = (E / row_sum).astype(np.float32)

        return BleedResult(
            B=B, E=E,
            unmodeled_ratio=U,
            legacy_unmodeled_ratio=U_legacy,
            unmodeled_ratio_uncentered=U0,
            bleed_off=_mean_off_diagonal(B),
            sir_db=_sir(B),
            ltr_db=_ltr(E),
            delays=delays,
            sr=self.sr,
        )

    # ── Per-pair delay estimation ─────────────────────────────────────────────

    def _estimate_per_pair_delays(
        self, sources: np.ndarray, mics: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the delay between every (mic m, source s) pair via batched
        FFT cross-correlation.

        Returns
        -------
        delays : (M, S) int array
            delays[m, s] = lag in samples such that source s shifted by this
            amount best aligns with mic m.
            Positive -> source leads mic (mic arrives later in time).
        """
        S, T  = sources.shape
        M     = mics.shape[0]
        D     = min(self.max_delay_samples, T - 1)
        n_fft = 1 << (2 * T - 1).bit_length()

        if self.use_envelope:
            src_proc = np.abs(sources).astype(np.float64)
            mic_proc = np.abs(mics).astype(np.float64)
        else:
            src_proc = sources.astype(np.float64)
            mic_proc = mics.astype(np.float64)

        # Batch FFT of all sources and mics
        Xf = np.fft.rfft(src_proc, n=n_fft, axis=1)   # (S, F)
        Yf = np.fft.rfft(mic_proc, n=n_fft, axis=1)   # (M, F)

        # Signed lag array covering +-D
        lags = np.concatenate([np.arange(0, D + 1), np.arange(-D, 0)])
        idx  = (lags % n_fft).astype(int)

        # All-pairs cross-correlation in one broadcast
        # cross_all[m, s, f] = conj(Xf[s, f]) * Yf[m, f]
        cross_all   = Xf[np.newaxis, :, :].conj() * Yf[:, np.newaxis, :]  # (M, S, F)
        xcorr_all   = np.fft.irfft(cross_all, n=n_fft, axis=2)            # (M, S, n_fft)

        # Find best lag within +-D for every (m, s)
        xcorr_search = xcorr_all[:, :, idx]            # (M, S, 2D+1)
        best_pos     = np.argmax(xcorr_search, axis=2) # (M, S)
        delays       = lags[best_pos]                   # (M, S)

        return delays.astype(int)

    # ── TF GPU energy estimation ──────────────────────────────────────────────

    def _estimate_energy_tf(
        self,
        sources_aligned: np.ndarray,   # (M, S, T)  delay-corrected per mic
        mics:            np.ndarray,   # (M, T)
    ) -> np.ndarray:
        """
        Solves M independent regularised least-squares systems on GPU.

        Because each mic has its own pre-aligned sources, Rxx is now a
        batch (M, SK, SK) rather than a single shared matrix.

        Steps
        -----
        1. rfft of sources_aligned (M, S, T) and mics (M, T).
        2. Build Rxx_batch (M, SK, SK) via batched all-pairs xcorr.
        3. Build Rxy_batch (M, SK) via batched xcorr.
        4. Solve all M systems at once: tf.linalg.solve(Rxx_batch, Rxy_batch).
        5. Compute filter energies and the fitted mic waveform via batched irfft.
        """
        with tf.device(self._device):
            M, S, T = sources_aligned.shape
            K       = self.K
            SK      = S * K
            n_fft   = 1 << (T + K - 2).bit_length()
            pad_T   = n_fft - T

            # ── FFTs ──────────────────────────────────────────────────────
            src_tf  = tf.constant(sources_aligned, dtype=tf.float32)  # (M, S, T)
            mic_tf  = tf.constant(mics,            dtype=tf.float32)  # (M, T)

            src_pad = tf.pad(src_tf, [[0, 0], [0, 0], [0, pad_T]])    # (M, S, n_fft)
            mic_pad = tf.pad(mic_tf, [[0, 0], [0, pad_T]])             # (M, n_fft)

            Xs = tf.signal.rfft(src_pad)    # (M, S, F)
            Ys = tf.signal.rfft(mic_pad)    # (M, F)

            # ── Rxx_batch (M, SK, SK) ─────────────────────────────────────
            # Rxx[m, s1*K+l1, s2*K+l2] = xcorr( xs1_aligned[m], xs2_aligned[m] )[l1-l2]
            Xs_a     = tf.expand_dims(Xs, 2)                 # (M, S, 1, F)
            Xs_b     = tf.expand_dims(Xs, 1)                 # (M, 1, S, F)
            cross_ss = tf.math.conj(Xs_a) * Xs_b            # (M, S, S, F)
            xcorr_ss = tf.signal.irfft(cross_ss)             # (M, S, S, n_fft)

            # Lag index table (K, K)
            l1       = tf.cast(tf.range(K)[:, tf.newaxis], tf.int32)
            l2       = tf.cast(tf.range(K)[tf.newaxis, :], tf.int32)
            lag_idx  = tf.math.floormod(l1 - l2, n_fft)     # (K, K)
            lag_flat = tf.reshape(lag_idx, [-1])              # (K*K,)

            # Gather lags: xcorr_ss[:, :, :, lag_flat] -> (M, S, S, K*K)
            xcorr_flat = tf.gather(xcorr_ss, lag_flat, axis=3)         # (M, S, S, K*K)
            Rxx_blocks = tf.reshape(xcorr_flat, [M, S, S, K, K])       # (M, S, S, K, K)

            # (M, S, S, K, K) -> (M, S, K, S, K) -> (M, SK, SK)
            Rxx = tf.reshape(
                tf.transpose(Rxx_blocks, [0, 1, 3, 2, 4]),
                [M, SK, SK]
            )

            # Regularise with batch identity
            Rxx_reg = Rxx + self.lam * tf.eye(SK, batch_shape=[M])     # (M, SK, SK)

            # ── Rxy_batch (M, SK) ─────────────────────────────────────────
            # Rxy[m, s*K+l] = xcorr( xs_aligned[m], ym )[l]
            Ys_exp   = tf.expand_dims(Ys, 1)                           # (M, 1, F)
            cross_my = tf.math.conj(Xs) * Ys_exp                      # (M, S, F)
            xcorr_my = tf.signal.irfft(cross_my)                       # (M, S, n_fft)
            Rxy      = tf.reshape(xcorr_my[:, :, :K], [M, SK])         # (M, SK)

            # ── Solve M systems in float64 for numerical stability ─────────
            Rxx_f64  = tf.cast(Rxx_reg, tf.float64)
            Rxy_f64  = tf.cast(tf.expand_dims(Rxy, -1), tf.float64)   # (M, SK, 1)

            A = tf.linalg.solve(Rxx_f64, Rxy_f64)                     # (M, SK, 1)
            A = tf.cast(tf.squeeze(A, axis=-1), tf.float32)            # (M, SK)

            # ── Filter energies via batched irfft ─────────────────────────
            A_3d  = tf.reshape(A, [M, S, K])                           # (M, S, K)
            pad_K = n_fft - K
            A_pad = tf.pad(A_3d, [[0, 0], [0, 0], [0, pad_K]])        # (M, S, n_fft)
            A_fft = tf.signal.rfft(A_pad)                              # (M, S, F)

            # filtered[m, s] = irfft( A_fft[m,s] * Xs[m,s] )
            filtered_fft = A_fft * Xs                                  # (M, S, F)
            filtered     = tf.signal.irfft(filtered_fft)               # (M, S, n_fft)
            filtered     = filtered[:, :, :T + K - 1]                  # (M, S, T+K-1)

            E = tf.reduce_sum(filtered ** 2, axis=2)                   # (M, S)
            fit = tf.reduce_sum(filtered[:, :, :T], axis=1)             # (M, T)

            return E.numpy().astype(np.float32), fit.numpy().astype(np.float32)


# ─── Static helpers ───────────────────────────────────────────────────────────

def _apply_per_pair_delays(sources: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """
    Build sources_aligned (M, S, T).
    sources_aligned[m, s] = source s shifted by delays[m, s] samples.
    Positive delay -> shift right (source moved later to match mic).
    """
    S, T = sources.shape
    M    = delays.shape[0]
    out  = np.zeros((M, S, T), dtype=sources.dtype)

    for m in range(M):
        for s in range(S):
            d = int(delays[m, s])
            if d == 0:
                out[m, s] = sources[s]
            elif d > 0:
                out[m, s, d:] = sources[s, :T - d]
            else:
                out[m, s, :T + d] = sources[s, -d:]

    return out


def _mean_off_diagonal(B: np.ndarray) -> float:
    M, S = B.shape
    mask = ~np.eye(min(M, S), S, dtype=bool)
    if M > S:
        mask = np.vstack([mask, np.ones((M - S, S), dtype=bool)])
    return float(B[mask].mean()) if mask.any() else 0.0


def _sir(B: np.ndarray) -> np.ndarray:
    M, S  = B.shape
    diag  = np.array([B[m, m] if m < S else 0.0 for m in range(M)])
    off   = B.sum(axis=1) - diag
    ratio = np.where(off > 0, diag / off, 1e12)
    return (10.0 * np.log10(np.clip(ratio, 1e-12, 1e12))).astype(np.float32)


def _ltr(E: np.ndarray) -> np.ndarray:
    M, S  = E.shape
    diag  = np.array([E[m, m] if m < S else 0.0 for m in range(M)])
    off   = E.sum(axis=1) - diag
    ratio = np.where(diag > 0, off / diag, 1e12)
    return (10.0 * np.log10(np.clip(ratio, 1e-12, 1e12))).astype(np.float32)


# ─── Batch summary ────────────────────────────────────────────────────────────

def batch_summary(results: List[BleedResult]) -> dict:
    """Aggregate statistics across all N BleedResult objects."""
    bleed_offs = np.array([r.bleed_off      for r in results])
    sirs       = np.stack([r.sir_db         for r in results])   # (N, M)
    ltrs       = np.stack([r.ltr_db         for r in results])
    unmodeled  = np.array([r.mean_unmodeled  for r in results])
    delays     = np.stack([r.delays          for r in results])  # (N, M, S)

    return {
        "N":                    len(results),
        "bleed_off_mean":       float(bleed_offs.mean()),
        "bleed_off_std":        float(bleed_offs.std()),
        "sir_db_mean":          float(sirs.mean()),
        "sir_db_std":           float(sirs.std()),
        "ltr_db_mean":          float(ltrs.mean()),
        "ltr_db_std":           float(ltrs.std()),
        "mean_unmodeled_mean":  float(unmodeled.mean()),
        "mean_unmodeled_std":   float(unmodeled.std()),
        "delay_mean_samples":   float(delays.mean()),
        "delay_std_samples":    float(delays.std()),
    }


# ─── Diagnostic helper ────────────────────────────────────────────────────────

def diagnose_pairing(
    sources:           np.ndarray,
    mics:              np.ndarray,
    sr:                int = 22050,
    max_delay_samples: int = 44100,
) -> None:
    """
    Print per-(m,s) normalised peak cross-correlation and best lag.

    Use this to verify that sources and mics are from the same take before
    running the full bleed matrix computation.

    Interpretation
    --------------
    ncorr > 0.30  : same take, delay correction should work well      (tick)
    ncorr 0.05-0.30 : same take but heavily processed / reverberant   (~)
    ncorr < 0.05  : likely different takes -- bleed matrix not valid  (cross)
    """
    S, T  = sources.shape
    M     = mics.shape[0]
    D     = min(max_delay_samples, T - 1)
    n_fft = 1 << (2 * T - 1).bit_length()

    Xf = np.fft.rfft(np.abs(sources).astype(np.float64), n=n_fft, axis=1)
    Yf = np.fft.rfft(np.abs(mics).astype(np.float64),    n=n_fft, axis=1)

    lags = np.concatenate([np.arange(0, D + 1), np.arange(-D, 0)])
    idx  = (lags % n_fft).astype(int)

    header = "           " + "".join(f"  source {s:<12}" for s in range(S))
    print(header)

    for m in range(M):
        row = f"  mic {m}    "
        for s in range(S):
            xcorr = np.fft.irfft(Xf[s].conj() * Yf[m], n=n_fft)
            peak  = np.max(xcorr[idx])
            norm  = np.sqrt(np.sum(sources[s] ** 2) * np.sum(mics[m] ** 2))
            lag   = int(lags[np.argmax(xcorr[idx])])
            ncorr = peak / norm if norm > 0 else 0.0
            flag  = "OK" if ncorr > 0.30 else ("~~ " if ncorr > 0.05 else "BAD")
            row  += f"  {flag} r={ncorr:.3f} lag={lag:+6d} ({lag/sr*1000:+7.1f}ms)"
        print(row)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    import time

    path = "/home/rrame12/Desktop/Datasets/Re-recorded/preprocessed_v2/"
    Xtest = np.load(path + "mixture_rec_pp.npy")
    Ytest = np.load(path + "gt_diag_rec_pp.npy")

    Xtest = Xtest[:10, :, :]
    Ytest = Ytest[:10, :, :]
    N, S, T = 10, 3, 220500
    sr      = 22050

    #bm = BleedMatrixTF(K=8192, lam=1e-3, sr=sr, max_delay_samples=44100, verbose=False)
    bm = BleedMatrixTF(K=256, lam=1e-3, sr=sr, max_delay_samples=44100)
    print(f"Device: {bm._device}\n")

    print("── Pairing diagnostic  example 0 ───────────────────────────────")
    diagnose_pairing(Xtest[0], Ytest[0], sr=sr)
    print()

    t0      = time.time()
    results = bm.compute_batch(Xtest, Ytest)
    elapsed = time.time() - t0
    print(f"Processed {N} examples in {elapsed:.2f}s  ({elapsed/N:.2f}s/example)\n")

    print("── Identity examples (0-4) ──────────────────")
    for i in range(5):
        r = results[i]
        print(f"  [{i}] BleedOff={r.bleed_off:.4f}  "
              f"SIR={np.mean(r.sir_db):+.1f} dB  U={r.mean_unmodeled:.4f}")

    print("\n── Bleed examples (5-9) ─────────────────────")
    for i in range(5, 10):
        r = results[i]
        print(f"  [{i}] BleedOff={r.bleed_off:.4f}  "
              f"SIR={np.mean(r.sir_db):+.1f} dB  U={r.mean_unmodeled:.4f}")

    print("\n── Bleed matrix  example 5 ──────────────────")
    print(np.round(results[5].B, 3))

    print("\n── Per-pair delays  example 5 (ms) ──────────")
    print(np.round(results[5].delays_ms, 1))

    print("\n── Batch aggregate ──────────────────────────")
    summary = batch_summary(results)
    for k, v in summary.items():
        fmt = f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}"
        print(fmt)


if __name__ == "__main__":
    _demo()
