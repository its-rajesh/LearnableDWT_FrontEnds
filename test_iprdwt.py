#!/usr/bin/env python3
"""
test_iprdwt.py

Evaluation script for the NEW iprdwt model (PR-DWT-U-Net w/ UpsampleTo HF fix).
Robustly loads:
  (A) a normal saved base model (.keras), OR
  (B) a TwoStageTrainer-saved .keras (weights stored under "base/...") by:
      - building base model
      - extracting model.weights.h5 from the .keras container
      - loading weights into a wrapper with attribute `.base`

Outputs:
  - metrics_per_sample.csv
  - metrics_summary.json + metrics_summary.csv
  - examples/{wav, png}
  - npy taps
  - pr_residuals plots

Requirements:
  - iprdwt.py available in PYTHONPATH (same folder OK)
"""

import os
import csv
import json
import argparse
import tempfile
import zipfile
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Optional WAV saving
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False

# ---- Import the new model/layers from iprdwt.py ----
from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
)

# ============================================================
# Utilities
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def normalize_audio(x, peak=0.99):
    x = np.asarray(x)
    m = np.max(np.abs(x)) + 1e-12
    return (x / m) * peak

def save_wav(path, audio, sr):
    if not HAVE_SF:
        return
    sf.write(path, audio, sr)

def plot_wave(path, signals, labels, sr=22050, max_sec=3.0):
    T = min([len(s) for s in signals])
    T = min(T, int(max_sec * sr))
    t = np.arange(T) / sr
    plt.figure(figsize=(12, 4))
    for s, lab in zip(signals, labels):
        plt.plot(t, s[:T], label=lab)
    plt.xlabel("Time (s)")
    plt.ylabel("Amp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_feature_map(path, X, title="", max_frames=800, max_bins=256):
    A = X
    if A.ndim == 3:
        A = A[:, :, 0]
    A = A[:max_frames, :max_bins]
    plt.figure(figsize=(10, 4))
    plt.imshow(A.T, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ============================================================
# Metrics
# ============================================================

def sisdr_np(y, yhat, eps=1e-8):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    y0 = y - y.mean()
    x0 = yhat - yhat.mean()
    dot = np.sum(y0 * x0)
    den = np.sum(y0 * y0) + eps
    a = dot / den
    s = a * y0
    e = x0 - s
    return 10.0 * np.log10((np.sum(s*s) + eps) / (np.sum(e*e) + eps))

def snr_np(y, yhat, eps=1e-8):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    e = yhat - y
    return 10.0 * np.log10((np.sum(y*y) + eps) / (np.sum(e*e) + eps))

def sir_sar_np(y, interf, yhat, eps=1e-8):
    """
    2D subspace projection onto span{target=y, interference=interf}
    Returns (SIR, SAR)
    """
    y0 = y - y.mean()
    v0 = interf - interf.mean()
    x0 = yhat - yhat.mean()

    if np.sum(v0*v0) < eps:
        sdr = sisdr_np(y0, x0, eps)
        return sdr, sdr

    n1 = np.sqrt(np.sum(y0*y0) + eps)
    u1 = y0 / n1
    vproj = v0 - np.sum(v0*u1) * u1
    n2 = np.sqrt(np.sum(vproj*vproj) + eps)
    u2 = vproj / n2

    c1 = np.sum(x0 * u1)
    c2 = np.sum(x0 * u2)
    s_target = c1 * u1
    s_interf = c2 * u2
    s_tot = s_target + s_interf
    art = x0 - s_tot

    SIR = 10.0 * np.log10((np.sum(s_target*s_target) + eps) / (np.sum(s_interf*s_interf) + eps))
    SAR = 10.0 * np.log10((np.sum(s_tot*s_tot) + eps) / (np.sum(art*art) + eps))
    return SIR, SAR

# ============================================================
# PR residual plots (from learned h0)
# ============================================================

def pr_even_shift_residuals(h0, max_m=20):
    h0 = np.asarray(h0).astype(np.float64)
    L = len(h0)
    ms = np.arange(0, max_m + 1)
    r2m = []
    for m in ms:
        k = 2*m
        if k == 0:
            r = np.sum(h0*h0)
        elif k < L:
            r = np.sum(h0[k:] * h0[:-k])
        else:
            r = 0.0
        r2m.append(r)
    return ms, np.array(r2m, dtype=np.float64)

def save_pr_residual_plots(model, out_root, levels=2, pr_shifts=12):
    pr_dir = ensure_dir(os.path.join(out_root, "pr_residuals"))
    overlay_path = os.path.join(pr_dir, "pr_residuals_overlay.png")

    plt.figure(figsize=(10, 4))
    for i in range(levels):
        layer = model.get_layer(f"prdwt_{i}")
        # PRDWT1D has weight h0_raw
        w = layer.get_weights()
        if not w:
            print(f"[PR residual] WARNING: no weights found for prdwt_{i}")
            continue
        h0_raw = w[0]
        h0 = h0_raw / (np.linalg.norm(h0_raw) + 1e-12)

        ms, r2m = pr_even_shift_residuals(h0, max_m=pr_shifts)

        csv_path = os.path.join(pr_dir, f"pr_residuals_level{i}.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["m", "shift", "r_2m"])
            for m, val in zip(ms, r2m):
                wr.writerow([int(m), int(2*m), float(val)])

        # per-level stem plot
        plt.figure(figsize=(10, 4))
        markerline, stemlines, baseline = plt.stem(2*ms, r2m)
        plt.setp(stemlines, linewidth=1.5)
        plt.setp(markerline, markersize=5)
        plt.axhline(0.0, linewidth=1)
        plt.title(f"PR residuals (even-shift autocorr) - level {i}")
        plt.xlabel("even shift (samples)")
        plt.ylabel("r[2m]")
        plt.tight_layout()
        plt.savefig(os.path.join(pr_dir, f"pr_residuals_level{i}.png"), dpi=150)
        plt.close()

        plt.plot(2*ms[1:], np.abs(r2m[1:]), marker="o", label=f"level {i}")

        print(f"[PR residual] level {i}: r0={r2m[0]:.6f}, max|r2m|(m>=1)={np.max(np.abs(r2m[1:])):.6e}")
        print("[Saved]", csv_path)

    plt.yscale("log")
    plt.title("PR residuals overlay (|r[2m]| for m>=1)")
    plt.xlabel("even shift (samples)")
    plt.ylabel("|r[2m]| (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print("[Saved]", overlay_path)

# ============================================================
# Loading logic (robust for TwoStageTrainer checkpoints)
# ============================================================

def relink_pridwt_to_prdwt(model, levels):
    """
    After loading weights or building a new model, make sure each PRIDWT1D has its ._dwt set.
    This is necessary because object references don't survive serialization.
    """
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)

def build_base_model(T, C, levels, filt_len, pr_shifts, pr_lambda, pr_dc_lambda, pr_nyq_lambda, unet_depth, base_filters):
    base = build_pr_dwt_unet(
        time_length=T,
        channels=C,
        levels=levels,
        filter_length=filt_len,
        pr_shifts=pr_shifts,
        pr_lambda=pr_lambda,
        pr_dc_lambda=pr_dc_lambda,
        pr_nyq_lambda=pr_nyq_lambda,
        unet_depth=unet_depth,
        base_filters=base_filters,
        return_taps=False
    )
    # Build variables
    _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(base, levels)
    return base

def load_base_model_from_any(best_model_path, custom_objs, base_builder, C, T, levels):
    """
    Try normal tf.keras.models.load_model().
    If it fails (e.g., TwoStageTrainer not deserializable), restore weights from inside .keras:
      - model.weights.h5 (most common)
      - variables/variables.* (checkpoint fallback)
    Handles weights saved under "base/..." by using a wrapper with attribute `.base`.
    Returns: base_model
    """
    # 1) Try normal load_model (works if saved object is base model)
    try:
        loaded = tf.keras.models.load_model(best_model_path, custom_objects=custom_objs, compile=False)
        # If it loaded a wrapper with `.base`, use it; else assume it's already base model
        base = getattr(loaded, "base", None)
        if base is None:
            base = loaded
        # Ensure built and relinked
        _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
        relink_pridwt_to_prdwt(base, levels)
        print("[Load] load_model() success.")
        return base
    except Exception as e:
        print("[WARN] load_model() failed; using restore-from-archive fallback.")
        print("       Reason:", repr(e))

    # 2) Build fresh base model
    base_model = base_builder()

    # 3) Extract .keras archive (zip) if needed
    tmpdir = tempfile.mkdtemp(prefix="keras_restore_")
    extracted_dir = None
    try:
        if os.path.isdir(best_model_path):
            extracted_dir = best_model_path
        else:
            with zipfile.ZipFile(best_model_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted_dir = tmpdir

        # 3a) Preferred: model.weights.h5
        h5_path = os.path.join(extracted_dir, "model.weights.h5")
        if os.path.exists(h5_path):
            print("[Restore] model.weights.h5 detected.")

            # Build base vars first (already built inside base_builder, but safe)
            _ = base_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)
            relink_pridwt_to_prdwt(base_model, levels)

            # ---- Try DIRECT load first (most common when best.keras stores base-model weights)
            print("[Restore] Trying direct base_model.load_weights(...) ...")
            try:
                base_model.load_weights(h5_path)
                relink_pridwt_to_prdwt(base_model, levels)
                print("[Restore] OK (H5 direct).")
                return base_model
            except Exception as e1:
                print("[Restore] Direct load failed:", repr(e1))

            # ---- Fallback: weights saved under "base/..." (TwoStageTrainer-style)
            print("[Restore] Trying wrapper(.base=base_model) load (base/ prefix) ...")

            class _BaseWrapper(tf.keras.Model):
                def __init__(self, base):
                    super().__init__(name="Trainer_StageB")
                    self.base = base
                def call(self, x, training=False):
                    return self.base(x, training=training)

            wrapper = _BaseWrapper(base_model)
            _ = wrapper(tf.zeros((1, C, T), dtype=tf.float32), training=False)

            wrapper.load_weights(h5_path)
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] OK (H5 wrapper).")
            return base_model

        # 3b) Fallback: TF checkpoint variables
        ckpt_prefix = os.path.join(extracted_dir, "variables", "variables")
        if os.path.exists(ckpt_prefix + ".index"):
            print("[Restore] TF checkpoint variables detected. Restoring via tf.train.Checkpoint ...")
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] OK (TF checkpoint).")
            return base_model

        # 3c) Nothing found -> print listing for debugging
        print("[ERROR] Could not find model.weights.h5 or variables/variables.* inside:")
        for root, _, files in os.walk(extracted_dir):
            rel = os.path.relpath(root, extracted_dir)
            for f in files[:200]:
                print("  -", os.path.join(rel, f))
        raise FileNotFoundError(
            f"Could not restore weights from {best_model_path}. Expected model.weights.h5 or variables/variables.index"
        )

    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR/")
    p.add_argument("--X", type=str, default="Xtest.npy")
    p.add_argument("--Y", type=str, default="Ytest.npy")
    p.add_argument("--best_model", type=str, required=True)

    p.add_argument("--T", type=int, default=220448)
    p.add_argument("--sr", type=int, default=22050)

    # Model hyperparams (must match training)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--filter_length", type=int, default=101)
    p.add_argument("--pr_shifts", type=int, default=24)
    p.add_argument("--pr_lambda", type=float, default=1e-2)
    p.add_argument("--pr_dc_lambda", type=float, default=1e-2)
    p.add_argument("--pr_nyq_lambda", type=float, default=1e-2)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=64)

    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--n_examples", type=int, default=6)
    p.add_argument("--example_idx", type=int, nargs="*", default=None)

    return p.parse_args()

def main():
    args = parse_args()

    X_path = os.path.join(args.dpath, args.X)
    Y_path = os.path.join(args.dpath, args.Y)

    print("[Load] Xtest:", X_path)
    print("[Load] Ytest:", Y_path)

    X = np.load(X_path).astype(np.float32)
    Y = np.load(Y_path).astype(np.float32)

    # enforce T
    T = args.T
    X = X[:, :, :T]
    Y = Y[:, :, :T]

    N, C, _ = X.shape
    print("Shapes:", X.shape, Y.shape)

    # Output dirs
    out_root = ensure_dir(os.path.join(os.path.dirname(args.best_model), "test_outputs_iprdwt"))
    ex_dir = ensure_dir(os.path.join(out_root, "examples"))
    npy_dir = ensure_dir(os.path.join(out_root, "npy"))
    fig_dir = ensure_dir(os.path.join(out_root, "figs"))

    # custom objects for deserialization attempt
    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    # Build base builder closure
    def _base_builder():
        return build_base_model(
            T=T, C=C,
            levels=args.levels,
            filt_len=args.filter_length,
            pr_shifts=args.pr_shifts,
            pr_lambda=args.pr_lambda,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,
            unet_depth=args.unet_depth,
            base_filters=args.base_filters,
        )

    print("[Load model]", args.best_model)
    base_model = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=_base_builder,
        C=C, T=T, levels=args.levels,
    )

    # Build tap model and copy weights from base
    tap_model, tap_keys = build_pr_dwt_unet(
        time_length=T,
        channels=C,
        levels=args.levels,
        filter_length=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
        unet_depth=args.unet_depth,
        base_filters=args.base_filters,
        return_taps=True
    )
    _ = tap_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(tap_model, args.levels)

    tap_model.set_weights(base_model.get_weights())

    # PR residual plots
    save_pr_residual_plots(tap_model, out_root, levels=args.levels, pr_shifts=args.pr_shifts)

    # Predict all
    Yhat = np.zeros_like(Y, dtype=np.float32)
    BATCH = args.batch

    for s in range(0, N, BATCH):
        e = min(N, s + BATCH)
        xb = X[s:e]
        outs = tap_model.predict(xb, verbose=0)
        yb = outs[0]
        Yhat[s:e] = yb

    # Metrics per sample/channel
    rows = []
    for n in range(N):
        for ch in range(C):
            y = Y[n, ch]
            x_in = X[n, ch]
            yhat = Yhat[n, ch]

            sisdr_in = sisdr_np(y, x_in)
            snr_in = snr_np(y, x_in)

            interf = np.sum(Y[n, np.arange(C) != ch], axis=0)
            sir_in, sar_in = sir_sar_np(y, interf, x_in)

            sisdr = sisdr_np(y, yhat)
            snr = snr_np(y, yhat)
            sir, sar = sir_sar_np(y, interf, yhat)

            rows.append({
                "idx": n, "ch": ch,
                "sisdr_in": sisdr_in, "snr_in": snr_in, "sir_in": sir_in, "sar_in": sar_in,
                "sisdr": sisdr, "snr": snr, "sir": sir, "sar": sar,
                "sisdr_impr": sisdr - sisdr_in,
                "snr_impr": snr - snr_in,
                "sir_impr": sir - sir_in,
                "sar_impr": sar - sar_in,
            })

    # Save per-sample metrics
    per_csv = os.path.join(out_root, "metrics_per_sample.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[Saved]", per_csv)

    # Summaries
    def summarize(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(vals.mean()), float(np.median(vals)), float(vals.std())

    summary_keys = [
        "sisdr_in","snr_in","sir_in","sar_in",
        "sisdr","snr","sir","sar",
        "sisdr_impr","snr_impr","sir_impr","sar_impr"
    ]
    summ = {k: {"mean": summarize(k)[0], "median": summarize(k)[1], "std": summarize(k)[2]}
            for k in summary_keys}

    summ_path = os.path.join(out_root, "metrics_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2)
    print("[Saved]", summ_path)

    summ_csv = os.path.join(out_root, "metrics_summary.csv")
    with open(summ_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "std"])
        for k in summary_keys:
            w.writerow([k, summ[k]["mean"], summ[k]["median"], summ[k]["std"]])
    print("[Saved]", summ_csv)

    # Examples
    rng = np.random.RandomState(0)
    idx_all = np.arange(N)
    if args.example_idx is None:
        ex_idx = rng.choice(idx_all, size=min(args.n_examples, N), replace=False)
    else:
        ex_idx = np.array(args.example_idx, dtype=int)

    ex_meta = []
    for n in ex_idx:
        xb = X[n:n+1]
        outs = tap_model.predict(xb, verbose=0)
        yhat = outs[0][0]  # [C,T]

        # save audio & wave plots
        for ch in range(C):
            x_in = normalize_audio(X[n, ch])
            y_gt = normalize_audio(Y[n, ch])
            y_pd = normalize_audio(yhat[ch])

            if HAVE_SF:
                save_wav(os.path.join(ex_dir, f"ex{n:04d}_ch{ch}_input.wav"), x_in, args.sr)
                save_wav(os.path.join(ex_dir, f"ex{n:04d}_ch{ch}_target.wav"), y_gt, args.sr)
                save_wav(os.path.join(ex_dir, f"ex{n:04d}_ch{ch}_pred.wav"),   y_pd, args.sr)

            plot_wave(
                os.path.join(fig_dir, f"ex{n:04d}_ch{ch}_wave.png"),
                [X[n, ch], Y[n, ch], yhat[ch]],
                ["input", "target", "pred"],
                sr=args.sr,
                max_sec=3.0
            )

        # save taps
        tap_pack = {"y": outs[0][0]}
        for k, arr in zip(tap_keys, outs[1:]):
            tap_pack[k] = arr[0]
        np.save(os.path.join(npy_dir, f"ex{n:04d}_taps.npy"), tap_pack, allow_pickle=True)

        # a couple internal visuals
        if "feat_before_unet" in tap_pack:
            plot_feature_map(os.path.join(fig_dir, f"ex{n:04d}_feat_before_unet.png"),
                             tap_pack["feat_before_unet"], title="feat_before_unet")
        if "recon_pre_match" in tap_pack:
            A = tap_pack["recon_pre_match"]
            if A.ndim == 2:
                plot_wave(os.path.join(fig_dir, f"ex{n:04d}_recon_pre_match_ch0.png"),
                          [A[:, 0]], ["recon_pre_match_ch0"], sr=args.sr, max_sec=3.0)

        ex_meta.append({"idx": int(n), "saved_wav": bool(HAVE_SF)})

    with open(os.path.join(out_root, "examples.json"), "w") as f:
        json.dump(ex_meta, f, indent=2)

    print("\n[DONE]")
    print("Outputs saved to:", out_root)
    if not HAVE_SF:
        print("NOTE: soundfile not installed; wav saving skipped. `pip install soundfile` to enable.")

if __name__ == "__main__":
    main()
