#!/usr/bin/env python3
import os
import json
import zipfile
import shutil
import random
import tempfile
import argparse
import numpy as np
import tensorflow as tf
import matplotlib as mpl
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
)

# -------------------- global paper style --------------------
PAPER_COLORS = {
    "blue":   "#2B6CB0",
    "teal":   "#2C7A7B",
    "orange": "#DD6B20",
    "red":    "#C53030",
    "gold":   "#B7791F",
    "purple": "#6B46C1",
    "gray":   "#4A5568",
    "light":  "#E2E8F0",
}

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def strip_axes(ax, keep_left=True, keep_bottom=True):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)
    ax.tick_params(length=3.5, width=0.8)


def center_by_energy(h):
    h = np.asarray(h, dtype=np.float64)
    n = np.arange(len(h), dtype=np.float64)
    e = h * h
    c = int(np.round((n * e).sum() / (e.sum() + 1e-12)))
    x = np.arange(len(h)) - c
    return x, h


def norm_to_peak(h):
    h = np.asarray(h, dtype=np.float64)
    return h / (np.max(np.abs(h)) + 1e-12)


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


def compute_frequency_response(h, n_fft=8192):
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.rfft(h, n=n_fft)
    w = np.linspace(0.0, 0.5, len(H))
    mag_db = 20.0 * np.log10(np.abs(H) / (np.max(np.abs(H)) + 1e-12) + 1e-12)
    return w, mag_db


# ---------- band decomposition helpers ----------

def partial_reconstruct_band(model, x_bt_c, levels, band_type, band_level):
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


# ---------- plotting ----------

def plot_filter_panels(eq_filters, out_dir):
    # Figure 1: stage filters and equivalent filters, centered and normalized
    nrows = len(eq_filters)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.5, 3.0 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    for row, info in enumerate(eq_filters):
        ax = axes[row, 0]
        x0, h0 = center_by_energy(norm_to_peak(info["h0"]))
        x1, h1 = center_by_energy(norm_to_peak(info["h1"]))
        ax.plot(x0, h0, lw=2.0, color=PAPER_COLORS["blue"], label=r"$h_0$")
        ax.plot(x1, h1, lw=2.0, color=PAPER_COLORS["orange"], label=r"$h_1$")
        ax.axhline(0.0, lw=0.7, color="#A0AEC0")
        ax.set_title(f"Level {info['level']} stage filters")
        ax.set_ylabel("Amplitude")
        if row == nrows - 1:
            ax.set_xlabel("Centered tap index")
        strip_axes(ax)
        ax.legend(frameon=False, loc="upper right")

        ax = axes[row, 1]
        xa, aeq = center_by_energy(norm_to_peak(info["eq_low"]))
        xd, deq = center_by_energy(norm_to_peak(info["eq_high"]))
        ax.plot(xa, aeq, lw=2.0, color=PAPER_COLORS["teal"], label=f"A{info['level']}")
        ax.plot(xd, deq, lw=2.0, color=PAPER_COLORS["red"], label=f"D{info['level']}")
        ax.axhline(0.0, lw=0.7, color="#A0AEC0")
        ax.set_title(f"Level {info['level']} equivalent filters")
        if row == nrows - 1:
            ax.set_xlabel("Centered tap index")
        strip_axes(ax)
        ax.legend(frameon=False, loc="upper right")

    fig.savefig(os.path.join(out_dir, "filters_impulse_responses_pretty.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: frequency responses of equivalent filters
    fig, axes = plt.subplots(nrows, 1, figsize=(8.8, 2.7 * nrows), sharex=True, constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    for ax, info in zip(axes, eq_filters):
        wA, HA = compute_frequency_response(info["eq_low"])
        wD, HD = compute_frequency_response(info["eq_high"])
        ax.plot(wA, HA, lw=2.0, color=PAPER_COLORS["teal"], label=f"A{info['level']}")
        ax.plot(wD, HD, lw=2.0, color=PAPER_COLORS["red"], label=f"D{info['level']}")
        ax.set_ylim([-80, 3])
        ax.set_ylabel("Mag. (dB)")
        ax.set_title(f"Level {info['level']} equivalent frequency responses")
        strip_axes(ax)
        ax.legend(frameon=False, loc="lower left")
    axes[-1].set_xlabel("Normalized frequency (cycles/sample)")
    fig.savefig(os.path.join(out_dir, "filters_frequency_responses_pretty.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: compact centered overlay only for equivalent filters
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    color_pairs = [
        (PAPER_COLORS["blue"], PAPER_COLORS["orange"]),
        (PAPER_COLORS["teal"], PAPER_COLORS["red"]),
        (PAPER_COLORS["purple"], PAPER_COLORS["gold"]),
    ]
    for i, info in enumerate(eq_filters):
        cA, cD = color_pairs[i % len(color_pairs)]
        xa, aeq = center_by_energy(norm_to_peak(info["eq_low"]))
        xd, deq = center_by_energy(norm_to_peak(info["eq_high"]))
        ax.plot(xa, aeq, lw=2.0, color=cA, label=f"A{info['level']}")
        ax.plot(xd, deq, lw=2.0, ls="--", color=cD, label=f"D{info['level']}")
    ax.axhline(0.0, lw=0.7, color="#A0AEC0")
    ax.set_title("Equivalent analysis filters across levels")
    ax.set_xlabel("Centered tap index")
    ax.set_ylabel("Normalized amplitude")
    strip_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.savefig(os.path.join(out_dir, "filters_equivalent_overlay_pretty.png"), bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(out_dir, "equivalent_filters.npy"), eq_filters, allow_pickle=True)


def plot_multiresolution_views(example_idx, x_ct, coeffs_native, recon_bands, out_dir, sr, channel_names=None):
    C, T = x_ct.shape
    levels = len(coeffs_native["A"])
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]

    wave_color = PAPER_COLORS["gray"]
    detail_colors = [PAPER_COLORS["orange"], PAPER_COLORS["red"], PAPER_COLORS["gold"], PAPER_COLORS["purple"]]
    approx_color = PAPER_COLORS["teal"]

    # Native decimated coefficients
    for c in range(C):
        nrows = 1 + levels + 1
        fig, axes = plt.subplots(nrows, 1, figsize=(11.2, 1.7 * nrows), constrained_layout=True)
        tt = np.arange(T) / sr
        axes[0].plot(tt, x_ct[c], lw=1.0, color=wave_color)
        axes[0].set_title(f"Example {example_idx} | {channel_names[c]} | Input waveform")
        axes[0].set_ylabel("Amp.")
        strip_axes(axes[0])

        for lev in range(levels):
            d = coeffs_native["D"][lev][0, :, c]
            td = np.arange(len(d)) / (sr / (2 ** (lev + 1)))
            axes[lev + 1].plot(td, d, lw=0.95, color=detail_colors[lev % len(detail_colors)])
            axes[lev + 1].set_title(f"D{lev+1} native coefficients")
            axes[lev + 1].set_ylabel("Amp.")
            strip_axes(axes[lev + 1])

        aL = coeffs_native["A"][-1][0, :, c]
        ta = np.arange(len(aL)) / (sr / (2 ** levels))
        axes[-1].plot(ta, aL, lw=0.95, color=approx_color)
        axes[-1].set_title(f"A{levels} native coefficients")
        axes[-1].set_ylabel("Amp.")
        axes[-1].set_xlabel("Time (s)")
        strip_axes(axes[-1])
        fig.savefig(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_native_coeffs_pretty.png"), bbox_inches="tight")
        plt.close(fig)

    # Full-rate band reconstructions
    for c in range(C):
        nrows = 1 + levels + 1
        fig, axes = plt.subplots(nrows, 1, figsize=(11.2, 1.7 * nrows), sharex=True, constrained_layout=True)
        tt = np.arange(T) / sr
        axes[0].plot(tt, x_ct[c], lw=1.0, color=wave_color)
        axes[0].set_title(f"Example {example_idx} | {channel_names[c]} | Input waveform")
        axes[0].set_ylabel("Amp.")
        strip_axes(axes[0])

        for lev in range(levels):
            band = recon_bands[f"D{lev+1}"][0, :, c]
            axes[lev + 1].plot(tt[:len(band)], band, lw=0.95, color=detail_colors[lev % len(detail_colors)])
            axes[lev + 1].set_title(f"D{lev+1} reconstructed to full rate")
            axes[lev + 1].set_ylabel("Amp.")
            strip_axes(axes[lev + 1])

        aband = recon_bands[f"A{levels}"][0, :, c]
        axes[-1].plot(tt[:len(aband)], aband, lw=0.95, color=approx_color)
        axes[-1].set_title(f"A{levels} reconstructed to full rate")
        axes[-1].set_ylabel("Amp.")
        axes[-1].set_xlabel("Time (s)")
        strip_axes(axes[-1])
        fig.savefig(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_band_recons_pretty.png"), bbox_inches="tight")
        plt.close(fig)


def maybe_save_audio_bands(example_idx, recon_bands, out_dir, sr, channel_names=None):
    if not HAVE_SF:
        return
    C = next(iter(recon_bands.values())).shape[-1]
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]
    for band_name, arr in recon_bands.items():
        arr = arr[0]
        for c in range(C):
            x = arr[:, c]
            m = np.max(np.abs(x)) + 1e-12
            sf.write(os.path.join(out_dir, f"example_{example_idx:04d}_{channel_names[c]}_{band_name}.wav"), 0.99 * x / m, sr)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PR-DWT filters and one random test example with paper-style figures")
    p.add_argument("--best_model", type=str, required=True)
    p.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")
    p.add_argument("--X", type=str, default="Xtest.npy")
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

    ex_idx = np.random.randint(0, N) if args.example_idx is None else int(args.example_idx)
    print(f"[Data] X shape = {X.shape}")
    print(f"[Data] chosen example index = {ex_idx}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.best_model), "paper_analysis_one_random_example_pretty")
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

    eq_filters = equivalent_analysis_filters(model, args.levels)
    plot_filter_panels(eq_filters, out_dir)

    filt_dump = []
    for info in eq_filters:
        filt_dump.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()})
    with open(os.path.join(out_dir, "filters_dump.json"), "w") as f:
        json.dump(filt_dump, f, indent=2)

    x = X[ex_idx:ex_idx + 1]
    x_bt_c = tf.convert_to_tensor(np.transpose(x, (0, 2, 1)), dtype=tf.float32)

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
    channel_names = ["vocals", "bass", "drums"] if C == 3 else None
    plot_multiresolution_views(ex_idx, x[0], coeffs_native, recon_bands, out_dir, args.sr, channel_names)
    maybe_save_audio_bands(ex_idx, recon_bands, os.path.join(out_dir, "audio_bands"), args.sr, channel_names)

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
    print("Main pretty figures:")
    print("  - filters_impulse_responses_pretty.png")
    print("  - filters_frequency_responses_pretty.png")
    print("  - filters_equivalent_overlay_pretty.png")
    print("  - example_*_native_coeffs_pretty.png")
    print("  - example_*_band_recons_pretty.png")


if __name__ == "__main__":
    main()
