#!/usr/bin/env python3
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

from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
)


EPS = 1e-12


# ============================================================
# Utilities
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)


def nmse_db(x, xhat, eps=1e-12):
    num = np.sum((x - xhat) ** 2)
    den = np.sum(x ** 2) + eps
    return 10.0 * np.log10((num + eps) / den)


def pr_snr_db(x, xhat, eps=1e-12):
    num = np.sum(x ** 2)
    den = np.sum((x - xhat) ** 2) + eps
    return 10.0 * np.log10((num + eps) / den)


def nrmse(x, xhat, eps=1e-12):
    num = np.sqrt(np.mean((x - xhat) ** 2))
    den = np.sqrt(np.mean(x ** 2)) + eps
    return num / den


# ============================================================
# Filter-level PR residuals
# ============================================================

def pr_even_shift_residuals(h0, max_m=20):
    h0 = np.asarray(h0).astype(np.float64)
    L = len(h0)
    ms = np.arange(0, max_m + 1)
    r2m = []
    for m in ms:
        k = 2 * m
        if k == 0:
            r = np.sum(h0 * h0)
        elif k < L:
            r = np.sum(h0[k:] * h0[:-k])
        else:
            r = 0.0
        r2m.append(r)
    return ms, np.array(r2m, dtype=np.float64)


def dc_nyq_stats(h0):
    h0 = np.asarray(h0, dtype=np.float64)
    dc = np.sum(h0)
    alt = np.ones_like(h0)
    alt[1::2] = -1.0
    nyq = np.sum(alt * h0)
    return dc, nyq


def save_filter_pr_metrics(model, out_root, levels=2, pr_shifts=12):
    pr_dir = ensure_dir(os.path.join(out_root, "pr_filter_metrics"))
    overlay_path = os.path.join(pr_dir, "pr_residuals_overlay.png")

    summary_rows = []

    plt.figure(figsize=(10, 4))
    for i in range(levels):
        layer = model.get_layer(f"prdwt_{i}")
        w = layer.get_weights()
        if not w:
            print(f"[PR filter] WARNING: no weights found for prdwt_{i}")
            continue

        h0_raw = w[0]
        h0 = h0_raw / (np.linalg.norm(h0_raw) + 1e-12)

        ms, r2m = pr_even_shift_residuals(h0, max_m=pr_shifts)
        dc, nyq = dc_nyq_stats(h0)

        max_off = float(np.max(np.abs(r2m[1:]))) if len(r2m) > 1 else 0.0
        mean_off = float(np.mean(np.abs(r2m[1:]))) if len(r2m) > 1 else 0.0

        csv_path = os.path.join(pr_dir, f"pr_residuals_level{i}.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["m", "shift", "r_2m"])
            for m, val in zip(ms, r2m):
                wr.writerow([int(m), int(2 * m), float(val)])

        plt.figure(figsize=(10, 4))
        markerline, stemlines, baseline = plt.stem(2 * ms, r2m)
        plt.setp(stemlines, linewidth=1.5)
        plt.setp(markerline, markersize=5)
        plt.axhline(0.0, linewidth=1)
        plt.title(f"PR residuals (even-shift autocorr) - level {i}")
        plt.xlabel("even shift (samples)")
        plt.ylabel("r[2m]")
        plt.tight_layout()
        plt.savefig(os.path.join(pr_dir, f"pr_residuals_level{i}.png"), dpi=200)
        plt.close()

        if len(ms) > 1:
            plt.plot(2 * ms[1:], np.abs(r2m[1:]), marker="o", linewidth=2, label=f"level {i}")

        summary_rows.append({
            "level": i,
            "r0": float(r2m[0]),
            "max_abs_r2m_m_ge_1": max_off,
            "mean_abs_r2m_m_ge_1": mean_off,
            "dc_sum": float(dc),
            "dc_target_sqrt2": float(np.sqrt(2.0)),
            "dc_error": float(dc - np.sqrt(2.0)),
            "nyq_sum": float(nyq),
        })

        print(
            f"[PR filter] level={i} "
            f"r0={r2m[0]:.6f} "
            f"max|r2m|(m>=1)={max_off:.6e} "
            f"dc={dc:.6f} "
            f"nyq={nyq:.6e}"
        )

    plt.yscale("log")
    plt.title("PR residuals overlay (|r[2m]| for m>=1)")
    plt.xlabel("even shift (samples)")
    plt.ylabel("|r[2m]| (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    summary_csv = os.path.join(pr_dir, "pr_filter_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    summary_json = os.path.join(pr_dir, "pr_filter_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("[Saved]", summary_csv)
    print("[Saved]", summary_json)
    print("[Saved]", overlay_path)


# ============================================================
# Signal-domain PR metrics
# ============================================================

def build_pr_analysis_synthesis_model(base_model, T, C, levels, filter_length,
                                      pr_shifts, pr_lambda, pr_dc_lambda, pr_nyq_lambda,
                                      unet_depth, base_filters):
    """
    Build a pure analysis-synthesis PR model:
      x -> DWT pyramid -> IDWT pyramid -> x_hat
    No U-Net path, no learned separator path.
    Uses the trained PRDWT/PRIDWT weights copied from base_model.
    """
    inp = tf.keras.Input(shape=(C, T), name="x_in")
    x = tf.keras.layers.Permute((2, 1), name="to_time_channels")(inp)  # [B,T,C]

    dwt_layers = []
    idwt_layers = []
    for i in range(levels):
        dwt = PRDWT1D(
            filter_length=filter_length,
            pr_shifts=pr_shifts,
            pr_lambda=pr_lambda,
            pr_dc_lambda=pr_dc_lambda,
            pr_nyq_lambda=pr_nyq_lambda,
            name=f"prdwt_{i}",
        )
        dwt_layers.append(dwt)

    approx = x
    details = []
    for i in range(levels):
        a, d = dwt_layers[i](approx)
        details.append(d)
        approx = a

    for i in range(levels):
        idwt = PRIDWT1D(name=f"pridwt_{i}").set_dwt(dwt_layers[i])
        idwt_layers.append(idwt)

    recon = approx
    for i in reversed(range(levels)):
        recon = idwt_layers[i]([recon, details[i]])

    recon = MatchTimeLen(name="match_out_len")([recon, x])
    y = tf.keras.layers.Permute((2, 1), name="to_channels_time")(recon)

    pr_model = tf.keras.Model(inp, y, name="PR_AnalysisSynthesis")
    _ = pr_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)

    # copy only DWT weights from base_model
    for i in range(levels):
        pr_model.get_layer(f"prdwt_{i}").set_weights(base_model.get_layer(f"prdwt_{i}").get_weights())
    relink_pridwt_to_prdwt(pr_model, levels)

    return pr_model


def compute_signal_pr_metrics(pr_model, X, out_root, batch_size=2):
    pr_dir = ensure_dir(os.path.join(out_root, "pr_signal_metrics"))

    N, C, T = X.shape
    Xhat = np.zeros_like(X, dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        Xhat[s:e] = pr_model.predict(X[s:e], verbose=0).astype(np.float32)

    rows = []
    for n in range(N):
        for ch in range(C):
            x = X[n, ch].astype(np.float64)
            xh = Xhat[n, ch].astype(np.float64)

            rows.append({
                "idx": n,
                "ch": ch,
                "pr_snr_db": float(pr_snr_db(x, xh)),
                "nmse_db": float(nmse_db(x, xh)),
                "nrmse": float(nrmse(x, xh)),
                "max_abs_err": float(np.max(np.abs(x - xh))),
            })

    per_csv = os.path.join(pr_dir, "pr_signal_per_sample.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def summarize(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    summary = {
        "pr_snr_db": summarize("pr_snr_db"),
        "nmse_db": summarize("nmse_db"),
        "nrmse": summarize("nrmse"),
        "max_abs_err": summarize("max_abs_err"),
    }

    summary_json = os.path.join(pr_dir, "pr_signal_summary.json")
    summary_csv = os.path.join(pr_dir, "pr_signal_summary.csv")

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "std", "min", "max"])
        for k, v in summary.items():
            w.writerow([k, v["mean"], v["median"], v["std"], v["min"], v["max"]])

    # Plot histogram of PR-SNR
    vals = np.array([r["pr_snr_db"] for r in rows], dtype=np.float64)
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=30)
    plt.xlabel("PR-SNR (dB)")
    plt.ylabel("Count")
    plt.title("Signal-domain PR reconstruction quality")
    plt.tight_layout()
    plt.savefig(os.path.join(pr_dir, "pr_snr_hist.png"), dpi=200)
    plt.close()

    print("[Saved]", per_csv)
    print("[Saved]", summary_json)
    print("[Saved]", summary_csv)
    print("[Saved]", os.path.join(pr_dir, "pr_snr_hist.png"))

    return summary


# ============================================================
# Loading logic
# ============================================================

def build_base_model(T, C, levels, filt_len, pr_shifts,
                     pr_lambda, pr_dc_lambda, pr_nyq_lambda,
                     unet_depth, base_filters):
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
        return_taps=False,
    )
    _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(base, levels)
    return base


def load_base_model_from_any(best_model_path, custom_objs, base_builder, C, T, levels):
    try:
        loaded = tf.keras.models.load_model(best_model_path, custom_objects=custom_objs, compile=False)
        base = getattr(loaded, "base", None)
        if base is None:
            base = loaded
        _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
        relink_pridwt_to_prdwt(base, levels)
        print("[Load] load_model() success.")
        return base
    except Exception as e:
        print("[WARN] load_model() failed; using restore fallback.")
        print("       Reason:", repr(e))

    base_model = base_builder()

    tmpdir = tempfile.mkdtemp(prefix="keras_restore_")
    extracted_dir = None
    try:
        if os.path.isdir(best_model_path):
            extracted_dir = best_model_path
        else:
            with zipfile.ZipFile(best_model_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted_dir = tmpdir

        h5_path = os.path.join(extracted_dir, "model.weights.h5")
        if os.path.exists(h5_path):
            _ = base_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)
            relink_pridwt_to_prdwt(base_model, levels)

            try:
                base_model.load_weights(h5_path)
                relink_pridwt_to_prdwt(base_model, levels)
                print("[Restore] OK (H5 direct).")
                return base_model
            except Exception as e1:
                print("[Restore] Direct load failed:", repr(e1))

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

        ckpt_prefix = os.path.join(extracted_dir, "variables", "variables")
        if os.path.exists(ckpt_prefix + ".index"):
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] OK (TF checkpoint).")
            return base_model

        raise FileNotFoundError(f"Could not restore weights from {best_model_path}")

    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpath", type=str, required=True)
    p.add_argument("--X", type=str, default="Xtest.npy")
    p.add_argument("--best_model", type=str, required=True)

    p.add_argument("--T", type=int, default=220448)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--filter_length", type=int, default=11)
    p.add_argument("--pr_shifts", type=int, default=32)
    p.add_argument("--pr_lambda", type=float, default=1e-1)
    p.add_argument("--pr_dc_lambda", type=float, default=1.0)
    p.add_argument("--pr_nyq_lambda", type=float, default=1.0)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    X_path = os.path.join(args.dpath, args.X)
    print("[Load] X:", X_path)
    X = np.load(X_path).astype(np.float32)
    X = X[:, :, :args.T]
    N, C, T = X.shape
    print("[Shapes]", X.shape)

    out_root = ensure_dir(os.path.join(os.path.dirname(args.best_model), "pr_metrics_paper"))

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    def _base_builder():
        return build_base_model(
            T=T,
            C=C,
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

    # 1) filter-level PR
    save_filter_pr_metrics(base_model, out_root, levels=args.levels, pr_shifts=args.pr_shifts)

    # 2) signal-level PR
    pr_model = build_pr_analysis_synthesis_model(
        base_model=base_model,
        T=T,
        C=C,
        levels=args.levels,
        filter_length=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
        unet_depth=args.unet_depth,
        base_filters=args.base_filters,
    )

    summary = compute_signal_pr_metrics(pr_model, X, out_root, batch_size=args.batch)

    print("\n===== PR SIGNAL SUMMARY =====")
    for k, v in summary.items():
        print(
            f"{k:12s} mean={v['mean']:.6f}  median={v['median']:.6f}  "
            f"std={v['std']:.6f}  min={v['min']:.6f}  max={v['max']:.6f}"
        )


if __name__ == "__main__":
    main()