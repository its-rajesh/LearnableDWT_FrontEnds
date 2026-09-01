#!/usr/bin/env python3
import os
import io
import json
import zipfile
import shutil
import random
import tempfile
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False

from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
    qmf_highpass_from_lowpass,
)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)


def build_base_model(T, C, levels, filt_len, pr_shifts, pr_lambda,
                     pr_dc_lambda, pr_nyq_lambda, unet_depth, base_filters):
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
        print("[Load] load_model() success")
        return base
    except Exception as e:
        print("[Load] load_model() failed, fallback restore.")
        print("Reason:", repr(e))

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
            try:
                base_model.load_weights(h5_path)
                relink_pridwt_to_prdwt(base_model, levels)
                print("[Restore] direct H5 load success")
                return base_model
            except Exception as e1:
                print("[Restore] direct H5 load failed:", repr(e1))

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
            print("[Restore] wrapped H5 load success")
            return base_model

        ckpt_prefix = os.path.join(extracted_dir, "variables", "variables")
        if os.path.exists(ckpt_prefix + ".index"):
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] TF checkpoint restore success")
            return base_model

        raise FileNotFoundError("Could not find model.weights.h5 or variables/variables.*")
    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------- filter helpers ----------

def normalize_h0_from_layer(layer):
    w = layer.get_weights()
    if not w:
        raise RuntimeError(f"No weights in layer {layer.name}")
    h0_raw = w[0].astype(np.float64)
    return h0_raw / (np.linalg.norm(h0_raw) + 1e-12)


def qmf_highpass_np(h0):
    n = np.arange(len(h0))
    return ((-1.0) ** n) * h0[::-1]


def upsample_filter(h, factor):
    h = np.asarray(h, dtype=np.float64)
    if factor <= 1:
        return h.copy()
    out = np.zeros((len(h) - 1) * factor + 1, dtype=np.float64)
    out[::factor] = h
    return out


def equivalent_analysis_filters(model, levels):
    """
    Returns one list element per decomposition level.
    Each element has learned stage filters and equivalent filters viewed at input sampling grid.

    Convention (0-indexed level):
      eq_low[level]  = cascade lowpass to reach approx at this level
      eq_high[level] = cascade lowpass prefix + this level highpass
    """
    out = []
    prefix = np.array([1.0], dtype=np.float64)
    for lev in range(levels):
        h0 = normalize_h0_from_layer(model.get_layer(f"prdwt_{lev}"))
        h1 = qmf_highpass_np(h0)
        up = 2 ** lev
        h0_up = upsample_filter(h0, up)
        h1_up = upsample_filter(h1, up)
        eq_low = np.convolve(prefix, h0_up)
        eq_high = np.convolve(prefix, h1_up)
        out.append({
            "level": lev + 1,
            "h0": h0,
            "h1": h1,
            "h0_up": h0_up,
            "h1_up": h1_up,
            "eq_low": eq_low,
            "eq_high": eq_high,
        })
        prefix = eq_low
    return out


# ---------- band decomposition helpers ----------

def partial_reconstruct_band(model, x_bt_c, levels, band_type, band_level):
    """
    x_bt_c: [1, T, C]
    band_type: 'A' or 'D'
    band_level: 1..levels

    Returns full-rate reconstruction [1, Trec, C] of only that subband.
    """
    a_list, d_list = [], []
    cur = x_bt_c
    for lev in range(levels):
        dwt = model.get_layer(f"prdwt_{lev}")
        a, d = dwt(cur)
        a_list.append(a)
        d_list.append(d)
        cur = a

    active_a = [tf.zeros_like(a) for a in a_list]
    active_d = [tf.zeros_like(d) for d in d_list]

    idx = band_level - 1
    if band_type.upper() == 'A':
        if idx != levels - 1:
            raise ValueError("Approx band only exists at final level: use band_level == levels")
        active_a[-1] = a_list[-1]
    elif band_type.upper() == 'D':
        active_d[idx] = d_list[idx]
    else:
        raise ValueError("band_type must be 'A' or 'D'")

    recon = active_a[-1]
    for lev in reversed(range(levels)):
        idwt = model.get_layer(f"pridwt_{lev}")
        recon = idwt([recon, active_d[lev]])
    return recon


def plot_filter_panels(eq_filters, out_dir):
    # Learned stage filters
    fig, axes = plt.subplots(len(eq_filters), 2, figsize=(10, 3.2 * len(eq_filters)))
    if len(eq_filters) == 1:
        axes = np.array([axes])
    for row, info in enumerate(eq_filters):
        ax = axes[row, 0]
        ax.plot(info["h0"], label="h0 learned")
        ax.plot(info["h1"], label="h1 derived")
        ax.set_title(f"Level {info['level']} stage filters")
        ax.set_xlabel("n")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.plot(info["eq_low"], label=f"A{info['level']} equivalent")
        ax.plot(info["eq_high"], label=f"D{info['level']} equivalent")
        ax.set_title(f"Level {info['level']} equivalent filters")
        ax.set_xlabel("n")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "filters_impulse_responses.png"), dpi=180)
    plt.close(fig)

    # one figure with only equivalent filters overlayed
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for info in eq_filters:
        ax.plot(info["eq_low"], label=f"A{info['level']}")
        ax.plot(info["eq_high"], linestyle='--', label=f"D{info['level']}")
    ax.set_title("Equivalent analysis filters across levels")
    ax.set_xlabel("n")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "filters_equivalent_overlay.png"), dpi=180)
    plt.close(fig)

    np.save(os.path.join(out_dir, "equivalent_filters.npy"), eq_filters, allow_pickle=True)



def plot_multiresolution_views(example_idx, x_ct, coeffs_native, recon_bands, out_dir, sr, channel_names=None):
    C, T = x_ct.shape
    levels = len(coeffs_native["A"])
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]

    # Raw decimated coefficients at native resolutions
    for c in range(C):
        nrows = 1 + levels + 1  # input + D1..DL + AL
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.0 * nrows), sharex=False)
        axes[0].plot(np.arange(T) / sr, x_ct[c], linewidth=0.8)
        axes[0].set_title(f"Example {example_idx} | {channel_names[c]} | input waveform")
        axes[0].grid(True, alpha=0.3)

        for lev in range(levels):
            d = coeffs_native["D"][lev][0, :, c]
            td = np.arange(len(d)) / (sr / (2 ** (lev + 1)))
            axes[lev + 1].plot(td, d, linewidth=0.8)
            axes[lev + 1].set_title(f"D{lev+1} native coefficients (decimated)")
            axes[lev + 1].grid(True, alpha=0.3)

        aL = coeffs_native["A"][-1][0, :, c]
        ta = np.arange(len(aL)) / (sr / (2 ** levels))
        axes[-1].plot(ta, aL, linewidth=0.8)
        axes[-1].set_title(f"A{levels} native coefficients (decimated)")
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_native_coeffs.png"), dpi=180)
        plt.close(fig)

    # Full-rate band reconstructions
    for c in range(C):
        nrows = 1 + levels + 1
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.0 * nrows), sharex=True)
        tt = np.arange(T) / sr
        axes[0].plot(tt, x_ct[c], linewidth=0.8)
        axes[0].set_title(f"Example {example_idx} | {channel_names[c]} | input waveform")
        axes[0].grid(True, alpha=0.3)

        for lev in range(levels):
            band = recon_bands[f"D{lev+1}"][0, :, c]
            axes[lev + 1].plot(tt[:len(band)], band, linewidth=0.8)
            axes[lev + 1].set_title(f"D{lev+1} reconstructed to full rate")
            axes[lev + 1].grid(True, alpha=0.3)

        aband = recon_bands[f"A{levels}"][0, :, c]
        axes[-1].plot(tt[:len(aband)], aband, linewidth=0.8)
        axes[-1].set_title(f"A{levels} reconstructed to full rate")
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_band_recons.png"), dpi=180)
        plt.close(fig)


def maybe_save_audio_bands(example_idx, recon_bands, out_dir, sr, channel_names=None):
    if not HAVE_SF:
        return
    C = next(iter(recon_bands.values())).shape[-1]
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]
    for band_name, arr in recon_bands.items():
        arr = arr[0]  # [T,C]
        for c in range(C):
            x = arr[:, c]
            m = np.max(np.abs(x)) + 1e-12
            sf.write(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_{band_name}.wav"), 0.99 * x / m, sr)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PR-DWT filters and one random test example for paper figures")
    p.add_argument("--best_model", type=str, required=True)
    p.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")
    p.add_argument("--X", type=str, default="Xtest.npy")
    p.add_argument("--Y", type=str, default=None)
    p.add_argument("--T", type=int, default=220448)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--example_idx", type=int, default=None)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--filter_length", type=int, default=101)
    p.add_argument("--pr_shifts", type=int, default=32)
    p.add_argument("--pr_lambda", type=float, default=1e-1)
    p.add_argument("--pr_dc_lambda", type=float, default=1.0)
    p.add_argument("--pr_nyq_lambda", type=float, default=1.0)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=64)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X_path = os.path.join(args.dpath, args.X)
    X = np.load(X_path).astype(np.float32)
    X = X[:, :, :args.T]
    N, C, T = X.shape

    if args.example_idx is None:
        ex_idx = np.random.randint(0, N)
    else:
        ex_idx = int(args.example_idx)
    print(f"[Data] X shape = {X.shape}")
    print(f"[Data] chosen example index = {ex_idx}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(args.best_model), "paper_analysis_one_random_example")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "audio_bands"))

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

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

    model = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=_base_builder,
        C=C, T=T, levels=args.levels,
    )

    # 1) Extract and plot learned + equivalent filters
    eq_filters = equivalent_analysis_filters(model, args.levels)
    plot_filter_panels(eq_filters, out_dir)

    # Save raw filter arrays in json-friendly form
    filt_dump = []
    for info in eq_filters:
        filt_dump.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()})
    with open(os.path.join(out_dir, "filters_dump.json"), "w") as f:
        json.dump(filt_dump, f, indent=2)

    # 2) One random example decomposition
    x = X[ex_idx:ex_idx + 1]             # [1,C,T]
    x_bt_c = tf.convert_to_tensor(np.transpose(x, (0, 2, 1)), dtype=tf.float32)  # [1,T,C]

    coeffs_A, coeffs_D = [], []
    cur = x_bt_c
    for lev in range(args.levels):
        dwt = model.get_layer(f"prdwt_{lev}")
        a, d = dwt(cur)
        coeffs_A.append(a.numpy())
        coeffs_D.append(d.numpy())
        cur = a

    recon_bands = {}
    for lev in range(1, args.levels + 1):
        recon_bands[f"D{lev}"] = partial_reconstruct_band(model, x_bt_c, args.levels, band_type='D', band_level=lev).numpy()
    recon_bands[f"A{args.levels}"] = partial_reconstruct_band(model, x_bt_c, args.levels, band_type='A', band_level=args.levels).numpy()

    coeffs_native = {"A": coeffs_A, "D": coeffs_D}
    plot_multiresolution_views(
        example_idx=ex_idx,
        x_ct=x[0],
        coeffs_native=coeffs_native,
        recon_bands=recon_bands,
        out_dir=out_dir,
        sr=args.sr,
        channel_names=["vocals", "bass", "drums"] if C == 3 else None,
    )
    maybe_save_audio_bands(ex_idx, recon_bands, os.path.join(out_dir, "audio_bands"), args.sr,
                           channel_names=["vocals", "bass", "drums"] if C == 3 else None)

    np.save(os.path.join(out_dir, f"example_{ex_idx:04d}_native_coeffs.npy"), coeffs_native, allow_pickle=True)
    np.save(os.path.join(out_dir, f"example_{ex_idx:04d}_band_recons.npy"), recon_bands, allow_pickle=True)

    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({
            "best_model": args.best_model,
            "X_path": X_path,
            "chosen_example_idx": ex_idx,
            "shape": [int(N), int(C), int(T)],
            "levels": args.levels,
            "filter_length": args.filter_length,
            "pr_shifts": args.pr_shifts,
            "pr_lambda": args.pr_lambda,
            "pr_dc_lambda": args.pr_dc_lambda,
            "pr_nyq_lambda": args.pr_nyq_lambda,
            "sr": args.sr,
        }, f, indent=2)

    print("\nSaved outputs to:", out_dir)
    print("Main figures:")
    print("  - filters_impulse_responses.png")
    print("  - filters_equivalent_overlay.png")
    print("  - example_*_native_coeffs.png")
    print("  - example_*_band_recons.png")


if __name__ == "__main__":
    main()
