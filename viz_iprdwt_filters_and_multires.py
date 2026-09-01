#!/usr/bin/env python3
"""
viz_iprdwt_filters_and_multires.py

Saves (into RUN_DIR):
  ./filters/
    - level{l}_time.png
    - level{l}_freq.png
  ./multiresolution/
    - ex{idx}_ch{ch}_multires_wave.png
    - ex{idx}_ch{ch}_multires_image.png

It robustly loads a Keras v3 .keras saved either as:
  (A) a base model, or
  (B) a TwoStageTrainer wrapper (weights stored under "base/..."),
by extracting model.weights.h5 and loading via a wrapper with attribute `.base`.

Usage example:
  python viz_iprdwt_filters_and_multires.py \
    --run_dir ./runs_pr/20251219_131806 \
    --best_model ./runs_pr/20251219_131806/best.keras \
    --dpath /home/rrame12/Projects/dwtIR/multichannel/march2025/ \
    --xfile Xtest.npy \
    --ex 14 --ch 0

Assumptions:
  - iprdwt.py is importable (same folder or PYTHONPATH)
  - levels=2, filter_length=101 (defaults set accordingly)
"""

import os
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

# -------------------------
# QMF helper (numpy)
# -------------------------
def qmf_highpass_from_lowpass_np(h0: np.ndarray) -> np.ndarray:
    """QMF: h1[n] = (-1)^n * h0[L-1-n]"""
    h0 = np.asarray(h0, dtype=np.float64)
    L = len(h0)
    h0_rev = h0[::-1]
    alt = np.ones(L, dtype=np.float64)
    alt[1::2] = -1.0
    return alt * h0_rev

# -------------------------
# Build + relink
# -------------------------
def relink_pridwt_to_prdwt(model: tf.keras.Model, levels: int):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)

def build_base_model(
    T: int,
    C: int,
    levels: int,
    filter_length: int,
    pr_shifts: int,
    pr_lambda: float,
    pr_dc_lambda: float,
    pr_nyq_lambda: float,
    unet_depth: int,
    base_filters: int,
) -> tf.keras.Model:
    base = build_pr_dwt_unet(
        time_length=T,
        channels=C,
        levels=levels,
        filter_length=filter_length,
        pr_shifts=pr_shifts,
        pr_lambda=pr_lambda,
        pr_dc_lambda=pr_dc_lambda,
        pr_nyq_lambda=pr_nyq_lambda,
        unet_depth=unet_depth,
        base_filters=base_filters,
        return_taps=False,
    )
    # Build variables
    _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(base, levels)
    return base

# -------------------------
# Robust loading
# -------------------------
def load_base_model_from_any(
    best_model_path: str,
    base_builder,
    custom_objects: dict,
    C: int,
    T: int,
    levels: int,
) -> tf.keras.Model:
    """
    Try load_model(). If it fails (e.g., TwoStageTrainer not deserializable),
    extract model.weights.h5 from .keras and load via wrapper with `.base`.
    """
    # 1) direct load (works if base model saved)
    try:
        loaded = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        base = getattr(loaded, "base", None) or loaded
        # ensure built + relink
        _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
        relink_pridwt_to_prdwt(base, levels)
        print("[Load] load_model() success.")
        return base
    except Exception as e:
        print("[WARN] load_model() failed; trying archive restore.")
        print("       ", repr(e))

    # 2) build base
    base = base_builder()

    # 3) extract model.weights.h5 from .keras container (zip) or directory
    tmpdir = tempfile.mkdtemp(prefix="keras_restore_")
    extracted = None
    try:
        if os.path.isdir(best_model_path):
            extracted = best_model_path
        else:
            with zipfile.ZipFile(best_model_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted = tmpdir

        h5_path = os.path.join(extracted, "model.weights.h5")
        if not os.path.exists(h5_path):
            # show listing for debugging
            print("[ERROR] model.weights.h5 not found. Contents:")
            for root, _, files in os.walk(extracted):
                rel = os.path.relpath(root, extracted)
                for f in files[:200]:
                    print("  -", os.path.join(rel, f))
            raise FileNotFoundError(f"model.weights.h5 not found inside: {best_model_path}")

        # wrapper for trainer-saved weights (stored under "base/...")
        class _Wrapper(tf.keras.Model):
            def __init__(self, base_model):
                super().__init__(name="Trainer_StageB")
                self.base = base_model
            def call(self, x, training=False):
                return self.base(x, training=training)

        wrap = _Wrapper(base)
        _ = wrap(tf.zeros((1, C, T), dtype=tf.float32), training=False)

        print("[Restore] Loading model.weights.h5 into wrapper(.base=base) ...")
        wrap.load_weights(h5_path)

        relink_pridwt_to_prdwt(base, levels)
        print("[Restore] OK.")
        return base
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# -------------------------
# Extract filters
# -------------------------
def extract_filters_from_model(model: tf.keras.Model, levels: int):
    """
    Returns list of dicts per level: h0,h1,g0,g1 as numpy arrays
    where:
      h0 = normalized learned lowpass
      h1 = QMF derived highpass
      g0 = reverse(h0)
      g1 = reverse(h1)
    """
    out = []
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        w = dwt.get_weights()
        if not w:
            raise RuntimeError(f"prdwt_{i} has no weights (did load fail?).")
        h0_raw = w[0].astype(np.float64)
        h0 = h0_raw / (np.linalg.norm(h0_raw) + 1e-12)
        h1 = qmf_highpass_from_lowpass_np(h0)
        g0 = h0[::-1]
        g1 = h1[::-1]
        out.append({"h0": h0, "h1": h1, "g0": g0, "g1": g1})
    return out

# -------------------------
# Plot + save filters
# -------------------------
def save_filter_plots(filters_per_level, sr: int, filter_dir: str):
    os.makedirs(filter_dir, exist_ok=True)

    n_fft = 16384
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    for lvl, d in enumerate(filters_per_level):
        h0, h1, g0, g1 = d["h0"], d["h1"], d["g0"], d["g1"]

        # time-domain taps
        plt.figure(figsize=(12, 4))
        plt.plot(h0, label="h0 (analysis LP)")
        plt.plot(h1, label="h1 (analysis HP)")
        plt.plot(g0, "--", label="g0 (synth LP)")
        plt.plot(g1, "--", label="g1 (synth HP)")
        plt.title(f"Learned/derived taps (Level {lvl})")
        plt.xlabel("Sample index n")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        out_time = os.path.join(filter_dir, f"level{lvl}_time.png")
        plt.savefig(out_time, dpi=200)
        plt.close()

        # frequency responses (normalized dB)
        plt.figure(figsize=(12, 4))
        for name, h in [("h0", h0), ("h1", h1), ("g0", g0), ("g1", g1)]:
            H = np.fft.rfft(h, n=n_fft)
            mag = np.abs(H)
            mag_db = 20.0 * np.log10(mag / (np.max(mag) + 1e-12) + 1e-12)
            plt.plot(freqs, mag_db, label=name)

        plt.ylim([-120, 5])
        plt.xlim([0, sr / 2])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB, normalized)")
        plt.title(f"Frequency responses (Level {lvl})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_freq = os.path.join(filter_dir, f"level{lvl}_freq.png")
        plt.savefig(out_freq, dpi=200)
        plt.close()

        print("[Saved]", out_time)
        print("[Saved]", out_freq)

# -------------------------
# Multiresolution decomposition
# -------------------------
def dwt_multiresolution(model: tf.keras.Model, x_bct: np.ndarray, levels: int):
    """
    x_bct: [1, C, T]
    returns approx_list, detail_list where each item is numpy [T_i, C] (channels-last)
    """
    x = tf.convert_to_tensor(x_bct, dtype=tf.float32)
    x_tc = tf.transpose(x, [0, 2, 1])  # [B,T,C]
    approx = x_tc
    approx_list, detail_list = [], []
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        a, d = dwt(approx)  # [B,T_i,C]
        approx_list.append(a.numpy()[0])  # [T_i,C]
        detail_list.append(d.numpy()[0])  # [T_i,C]
        approx = a
    return approx_list, detail_list

def save_multiresolution_wave(
    approx_list,
    detail_list,
    sr: int,
    channel: int,
    ex_id: int,
    out_dir: str,
    max_sec: float = 4.0,
):
    os.makedirs(out_dir, exist_ok=True)
    levels = len(approx_list)

    plt.figure(figsize=(14, 3 * levels))
    for i in range(levels):
        a = approx_list[i][:, channel]
        d = detail_list[i][:, channel]
        sr_i = sr / (2 ** (i + 1))
        Tshow = min(len(a), int(max_sec * sr_i))
        t = np.arange(Tshow) / sr_i

        plt.subplot(levels, 2, 2 * i + 1)
        plt.plot(t, a[:Tshow])
        plt.title(f"Level {i} Approximation a{i}  (sr≈{sr_i:.1f} Hz)")
        plt.xlabel("Time (s)")
        plt.grid(alpha=0.3)

        plt.subplot(levels, 2, 2 * i + 2)
        plt.plot(t, d[:Tshow])
        plt.title(f"Level {i} Detail d{i}  (sr≈{sr_i:.1f} Hz)")
        plt.xlabel("Time (s)")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f"ex{ex_id:04d}_ch{channel}_multires_wave.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("[Saved]", out)

def save_multiresolution_image(
    approx_list,
    detail_list,
    ex_id: int,
    channel: int,
    out_dir: str,
):
    """
    Paper-ready image: stack [a0,d0,a1,d1,...] as rows; pad to equal length.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    labels = []
    for i in range(len(approx_list)):
        rows.append(approx_list[i][:, channel])
        labels.append(f"a{i}")
        rows.append(detail_list[i][:, channel])
        labels.append(f"d{i}")

    max_len = max(len(r) for r in rows)
    rows_pad = [np.pad(r, (0, max_len - len(r))) for r in rows]
    M = np.vstack(rows_pad)  # [2*levels, max_len]

    # robust scaling for visualization
    vmax = np.percentile(np.abs(M), 99.5) + 1e-12

    plt.figure(figsize=(14, 4))
    plt.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=-vmax, vmax=vmax)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Time (samples, padded)")
    plt.title("Learned DWT multiresolution decomposition (approx/detail per level)")
    cbar = plt.colorbar()
    cbar.set_label("Amplitude")
    plt.tight_layout()

    out = os.path.join(out_dir, f"ex{ex_id:04d}_ch{channel}_multires_image.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("[Saved]", out)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Run folder to save plots into (e.g., ./runs_pr/20251219_131806)")
    p.add_argument("--best_model", type=str, required=True, help="Path to best.keras (usually inside run_dir)")
    p.add_argument("--dpath", type=str, default="/home/rrame12/Projects/dwtIR/multichannel/march2025/")
    p.add_argument("--xfile", type=str, default="Xtest.npy")

    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--T", type=int, default=220448)

    # fixed per your note
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--filter_length", type=int, default=101)

    # must match architecture used in training
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=64)

    # these don't affect architecture except internal losses, but keep consistent
    p.add_argument("--pr_shifts", type=int, default=24)
    p.add_argument("--pr_lambda", type=float, default=1e-2)
    p.add_argument("--pr_dc_lambda", type=float, default=1e-2)
    p.add_argument("--pr_nyq_lambda", type=float, default=1e-2)

    p.add_argument("--ex", type=int, default=14, help="Example index from Xtest to visualize")
    p.add_argument("--ch", type=int, default=0, help="Channel index (0..C-1) to visualize")
    p.add_argument("--C", type=int, default=3, help="Number of channels (default 3 for your dataset)")

    return p.parse_args()

def main():
    args = parse_args()

    run_dir = args.run_dir
    filter_dir = os.path.join(run_dir, "filters")
    multires_dir = os.path.join(run_dir, "multiresolution")
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(multires_dir, exist_ok=True)

    x_path = os.path.join(args.dpath, args.xfile)
    X = np.load(x_path).astype(np.float32)  # [N,C,T]
    X = X[:, :, : args.T]

    if args.ex < 0 or args.ex >= X.shape[0]:
        raise ValueError(f"--ex out of range. Got {args.ex}, dataset has {X.shape[0]} examples.")
    if args.ch < 0 or args.ch >= X.shape[1]:
        raise ValueError(f"--ch out of range. Got {args.ch}, input has {X.shape[1]} channels.")

    custom_objects = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    def base_builder():
        return build_base_model(
            T=args.T,
            C=args.C,
            levels=args.levels,
            filter_length=args.filter_length,
            pr_shifts=args.pr_shifts,
            pr_lambda=args.pr_lambda,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,
            unet_depth=args.unet_depth,
            base_filters=args.base_filters,
        )

    print("[Load model]", args.best_model)
    model = load_base_model_from_any(
        best_model_path=args.best_model,
        base_builder=base_builder,
        custom_objects=custom_objects,
        C=args.C,
        T=args.T,
        levels=args.levels,
    )

    # 1) Save learned/derived filters
    filters = extract_filters_from_model(model, levels=args.levels)
    save_filter_plots(filters, sr=args.sr, filter_dir=filter_dir)

    # 2) Multiresolution decomposition (one example)
    x = X[args.ex : args.ex + 1]  # [1,C,T]
    approx_list, detail_list = dwt_multiresolution(model, x, levels=args.levels)

    save_multiresolution_wave(
        approx_list, detail_list,
        sr=args.sr, channel=args.ch, ex_id=args.ex,
        out_dir=multires_dir, max_sec=4.0
    )
    save_multiresolution_image(
        approx_list, detail_list,
        ex_id=args.ex, channel=args.ch,
        out_dir=multires_dir
    )

    print("\n[DONE]")
    print("Saved to:")
    print(" -", filter_dir)
    print(" -", multires_dir)

if __name__ == "__main__":
    main()
