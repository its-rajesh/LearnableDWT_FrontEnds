#!/usr/bin/env python3
"""preprocess_rerecorded_dataset.py

Create aligned training pairs (mix -> gt) from re-recorded dataset.

Inputs (.npy, shape (N,C,T)):
  - digital: clean played signal (anchor)
  - gt     : diagonal re-recorded (each speaker individually)
  - mix    : bleed mixture

Outputs (saved to --out_dir):
  - mixture_rec_pp.npy   (aligned mix)
  - gt_diag_rec_pp.npy   (gt aligned to digital)
  - keep_mask_NC.npy     (uint8 mask over (N,C))
  - preprocess_report.json

Main idea:
  1) Align GT -> Digital (fix latency; optional gain match)
  2) Align Mix -> GT (so training pairs are time-consistent)
  3) Keep only (n,c) pairs whose mix-vs-gt xcorr >= threshold

This is intentionally conservative: it tries to *remove bad pairs* that will
poison fine-tuning on real re-recorded data.
"""

import os
import json
import argparse
import numpy as np


def _ensure_3d(x, name="array"):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape (N,C,T), got {x.shape}")
    return x


def dc_remove(x):
    return x - np.mean(x, axis=-1, keepdims=True)


def safe_norm(x, eps=1e-12):
    return np.sqrt(np.sum(x * x) + eps)


def sisdr(s_hat, s, eps=1e-8):
    """Scale-invariant SDR (dB)."""
    s_hat = np.asarray(s_hat, dtype=np.float64).reshape(-1)
    s     = np.asarray(s,     dtype=np.float64).reshape(-1)
    T = min(len(s_hat), len(s))
    if T < 2:
        return float("nan")
    s_hat = s_hat[:T] - np.mean(s_hat[:T])
    s     = s[:T]     - np.mean(s[:T])
    denom = np.dot(s, s) + eps
    alpha = np.dot(s_hat, s) / denom
    s_target = alpha * s
    e_noise  = s_hat - s_target
    num = np.dot(s_target, s_target) + eps
    den = np.dot(e_noise,  e_noise)  + eps
    return 10.0 * np.log10(num / den)


def best_lag_xcorr(x, y, max_lag=8192, eps=1e-12):
    """Robust brute-force normalized dot-product xcorr over integer lags.

    We test lags in [-L,+L], where lag means we compare x[t] with y[t+lag].

    Returns (best_lag, best_score).
    Guards against empty overlap slices.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    T = min(len(x), len(y))
    if T < 2:
        return 0, 0.0
    x = x[:T]
    y = y[:T]

    L = int(max_lag)
    L = max(0, min(L, T - 1))

    best_lag = 0
    best_score = -1e9

    for lag in range(-L, L + 1):
        if lag >= 0:
            a = x[:T - lag]
            b = y[lag:]
        else:
            a = x[-lag:]
            b = y[:T + lag]
        if a.size < 2 or b.size < 2:
            continue
        sc = float(np.dot(a, b) / (safe_norm(a, eps) * safe_norm(b, eps) + eps))
        if sc > best_score:
            best_score = sc
            best_lag = lag

    if best_score < -1e8:
        return 0, 0.0
    return int(best_lag), float(best_score)


def apply_integer_lag(x, lag):
    """Shift x by -lag (consistent with earlier alignment code)."""
    return np.roll(x, -int(lag))


def gain_match_ls(x, y, eps=1e-8):
    """Scalar g minimizing ||g*x - y||^2."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.dot(x, x) + eps))


def peak_normalize(x, peak=0.99, eps=1e-8):
    m = float(np.max(np.abs(x)))
    if m < eps:
        return x
    return (peak / m) * x


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--digital", type=str, required=True)
    ap.add_argument("--gt", type=str, required=True)
    ap.add_argument("--mix", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--max_lag_dig", type=int, default=65536)
    ap.add_argument("--max_lag_mix", type=int, default=8192)

    ap.add_argument("--dc_remove", action="store_true")
    ap.add_argument("--no_gain_match", action="store_true")
    ap.add_argument("--peak", type=float, default=0.99)
    ap.add_argument("--keep_xcorr_thr", type=float, default=0.25)
    ap.add_argument("--keep_sisdr_thr", type=float, default=None)
    ap.add_argument("--zero_out_dropped", action="store_true")

    args = ap.parse_args()

    path_d = os.path.join(args.root, args.digital)
    path_g = os.path.join(args.root, args.gt)
    path_m = os.path.join(args.root, args.mix)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(args.root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[Load]")
    Xd = _ensure_3d(np.load(path_d), "digital")
    Yg = _ensure_3d(np.load(path_g), "gt")
    Xm = _ensure_3d(np.load(path_m), "mix")

    # Crop to common min shape if needed
    N = min(Xd.shape[0], Yg.shape[0], Xm.shape[0])
    C = min(Xd.shape[1], Yg.shape[1], Xm.shape[1])
    T = min(Xd.shape[2], Yg.shape[2], Xm.shape[2])
    Xd = Xd[:N, :C, :T].astype(np.float32)
    Yg = Yg[:N, :C, :T].astype(np.float32)
    Xm = Xm[:N, :C, :T].astype(np.float32)

    if args.dc_remove:
        Xd = dc_remove(Xd)
        Yg = dc_remove(Yg)
        Xm = dc_remove(Xm)

    do_gain = (not args.no_gain_match)

    Yg_pp = np.empty_like(Yg)
    Xm_pp = np.empty_like(Xm)
    keep_mask = np.zeros((N, C), dtype=np.uint8)

    lag_gd = np.zeros((N, C), dtype=np.int32)
    sc_gd  = np.zeros((N, C), dtype=np.float32)
    lag_mg = np.zeros((N, C), dtype=np.int32)
    sc_mg  = np.zeros((N, C), dtype=np.float32)

    # --- Align GT -> Digital
    for n in range(N):
        for c in range(C):
            d = Xd[n, c]
            g = Yg[n, c]
            lag, sc = best_lag_xcorr(g, d, max_lag=args.max_lag_dig)
            g_al = apply_integer_lag(g, lag)
            if do_gain:
                g_al = gain_match_ls(g_al, d) * g_al
            g_al = peak_normalize(g_al, peak=args.peak)
            Yg_pp[n, c] = g_al.astype(np.float32)
            lag_gd[n, c] = lag
            sc_gd[n, c] = sc

    # --- Align Mix -> (aligned) GT and filter
    for n in range(N):
        for c in range(C):
            g = Yg_pp[n, c]
            m = Xm[n, c]
            lag, sc = best_lag_xcorr(m, g, max_lag=args.max_lag_mix)
            m_al = apply_integer_lag(m, lag)
            m_al = peak_normalize(m_al, peak=args.peak)
            Xm_pp[n, c] = m_al.astype(np.float32)
            lag_mg[n, c] = lag
            sc_mg[n, c] = sc

            keep = (sc >= float(args.keep_xcorr_thr))
            if args.keep_sisdr_thr is not None:
                keep = keep and (sisdr(m_al, g) >= float(args.keep_sisdr_thr))
            keep_mask[n, c] = 1 if keep else 0

    if args.zero_out_dropped:
        bad = (keep_mask == 0)
        for n in range(N):
            for c in range(C):
                if bad[n, c]:
                    Xm_pp[n, c] = 0.0
                    Yg_pp[n, c] = 0.0

    # Probe SI-SDR stats
    rng = np.random.default_rng(0)
    probe_k = min(256, N * C)
    probe_idx = rng.choice(N * C, size=probe_k, replace=False)
    probe_raw = []
    probe_al  = []
    for idx in probe_idx:
        n = int(idx // C)
        c = int(idx % C)
        probe_raw.append(float(sisdr(Xm[n, c],  Yg_pp[n, c])))
        probe_al.append(float(sisdr(Xm_pp[n, c], Yg_pp[n, c])))

    report = {
        "paths": {"digital": path_d, "gt": path_g, "mix": path_m},
        "shape": {"N": int(N), "C": int(C), "T": int(T)},
        "fs": int(args.sr),
        "params": {
            "max_lag_dig": int(args.max_lag_dig),
            "max_lag_mix": int(args.max_lag_mix),
            "dc_remove": bool(args.dc_remove),
            "gain_match": bool(do_gain),
            "peak": float(args.peak),
            "keep_xcorr_thr": float(args.keep_xcorr_thr),
            "keep_sisdr_thr": float(args.keep_sisdr_thr) if args.keep_sisdr_thr is not None else None,
            "zero_out_dropped": bool(args.zero_out_dropped),
        },
        "stats": {
            "xcorr_gt_vs_digital": summarize(sc_gd.reshape(-1)),
            "lag_gt_vs_digital": summarize(lag_gd.reshape(-1)),
            "xcorr_mix_vs_gt": summarize(sc_mg.reshape(-1)),
            "lag_mix_vs_gt": summarize(lag_mg.reshape(-1)),
            "probe_sisdr_raw_mix_vs_gt": summarize(np.asarray(probe_raw)),
            "probe_sisdr_aligned_mix_vs_gt": summarize(np.asarray(probe_al)),
            "keep_fraction": float(np.mean(keep_mask)),
        },
    }

    mix_out = os.path.join(out_dir, "mixture_rec_pp.npy")
    gt_out  = os.path.join(out_dir, "gt_diag_rec_pp.npy")
    km_out  = os.path.join(out_dir, "keep_mask_NC.npy")
    rep_out = os.path.join(out_dir, "preprocess_report.json")

    np.save(mix_out, Xm_pp.astype(np.float32))
    np.save(gt_out,  Yg_pp.astype(np.float32))
    np.save(km_out,  keep_mask)
    with open(rep_out, "w") as f:
        json.dump(report, f, indent=2)

    print("[Saved]")
    print(" ", mix_out)
    print(" ", gt_out)
    print(" ", km_out)
    print(" ", rep_out)
    print("[Keep fraction]", float(np.mean(keep_mask)))


if __name__ == "__main__":
    main()
