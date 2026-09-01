#!/usr/bin/env python3
"""
Evaluation script for PR-DWT-U-Net on rerecorded / measured-RIR datasets.

What it does:
1) Metric evaluation on activity-aware crops (closer to training validation)
2) Full-song overlap-add inference for a few listening examples
3) Writes raw full-length audio examples without independent normalization

Supports both layouts:
  - X/Y
  - Xaligned/Yaligned

Typical usage:
python test_iprdwt_rerecorded_final.py \
  --rerec_root /path/to/dataset \
  --split test \
  --best_model /path/to/best.keras \
  --x_subdir X --y_subdir Y \
  --target_sr 22050 \
  --crop_T 32768 \
  --n_crops_per_song 8 \
  --min_active_stems 2 \
  --full_example_ids 0 10
"""

import os
import csv
import json
import math
import argparse
import tempfile
import zipfile
import shutil
import random
from typing import List, Tuple

import numpy as np
import soundfile as sf
import scipy.signal as sp
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

STEMS = ["vocals", "bass", "drums"]


# ============================================================
# General utilities
# ============================================================

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def sanitize_audio(x):
    x = np.asarray(x, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def audio_rms(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def audio_peak(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return float(np.max(np.abs(x)) + eps)


def maybe_resample(x, sr, target_sr=22050):
    x = sanitize_audio(x)
    sr = int(sr)
    target_sr = int(target_sr)
    if sr == target_sr:
        return x.astype(np.float32)
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return sp.resample_poly(x, up=up, down=down).astype(np.float32)


def read_mono(path, target_sr=22050):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = maybe_resample(x, sr, target_sr)
    return sanitize_audio(x), int(target_sr)


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
    return 10.0 * np.log10((np.sum(s * s) + eps) / (np.sum(e * e) + eps))


def snr_np(y, yhat, eps=1e-8):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    e = yhat - y
    return 10.0 * np.log10((np.sum(y * y) + eps) / (np.sum(e * e) + eps))


def sir_sar_np(y, interf, yhat, eps=1e-8):
    y0 = y - y.mean()
    v0 = interf - interf.mean()
    x0 = yhat - yhat.mean()

    if np.sum(v0 * v0) < eps:
        sdr = sisdr_np(y0, x0, eps)
        return sdr, sdr

    n1 = np.sqrt(np.sum(y0 * y0) + eps)
    u1 = y0 / n1
    vproj = v0 - np.sum(v0 * u1) * u1
    n2 = np.sqrt(np.sum(vproj * vproj) + eps)
    u2 = vproj / n2

    c1 = np.sum(x0 * u1)
    c2 = np.sum(x0 * u2)
    s_target = c1 * u1
    s_interf = c2 * u2
    s_tot = s_target + s_interf
    art = x0 - s_tot

    sir = 10.0 * np.log10((np.sum(s_target * s_target) + eps) / (np.sum(s_interf * s_interf) + eps))
    sar = 10.0 * np.log10((np.sum(s_tot * s_tot) + eps) / (np.sum(art * art) + eps))
    return sir, sar


# ============================================================
# Model loading (copied/adapted from original evaluator)
# ============================================================

def relink_pridwt_to_prdwt(model, levels):
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
        print("[WARN] load_model() failed; using restore-from-archive fallback.")
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
            print("[Restore] model.weights.h5 detected.")
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
            print("[Restore] TF checkpoint variables detected.")
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] OK (TF checkpoint).")
            return base_model

        raise FileNotFoundError(
            f"Could not restore weights from {best_model_path}. Expected model.weights.h5 or variables/variables.index"
        )
    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Dataset loading
# ============================================================

def load_song(song_dir: str, x_subdir="X", y_subdir="Y", target_sr=22050):
    xdir = os.path.join(song_dir, x_subdir)
    ydir = os.path.join(song_dir, y_subdir)

    X = []
    Y = []
    for stem in STEMS:
        xp = os.path.join(xdir, f"{stem}.wav")
        yp = os.path.join(ydir, f"{stem}.wav")
        x, _ = read_mono(xp, target_sr=target_sr)
        y, _ = read_mono(yp, target_sr=target_sr)
        X.append(x)
        Y.append(y)

    minlen = min(min(len(a) for a in X), min(len(b) for b in Y))
    X = np.stack([a[:minlen] for a in X], axis=0).astype(np.float32)
    Y = np.stack([b[:minlen] for b in Y], axis=0).astype(np.float32)
    return X, Y


def list_song_dirs(root: str, split: str, x_subdir="X", y_subdir="Y") -> List[Tuple[str, str]]:
    split_dir = os.path.join(root, split)
    out = []
    for name in sorted(os.listdir(split_dir)):
        song_dir = os.path.join(split_dir, name)
        if not os.path.isdir(song_dir):
            continue
        if not (os.path.isdir(os.path.join(song_dir, x_subdir)) and os.path.isdir(os.path.join(song_dir, y_subdir))):
            continue
        ok = True
        for stem in STEMS:
            if not os.path.isfile(os.path.join(song_dir, x_subdir, f"{stem}.wav")):
                ok = False
            if not os.path.isfile(os.path.join(song_dir, y_subdir, f"{stem}.wav")):
                ok = False
        if ok:
            out.append((name, song_dir))
    return out


# ============================================================
# Activity-aware crop evaluation
# ============================================================

def stem_activity_mask(Y, rms_thresh=1e-4, peak_thresh=1e-3):
    flags = []
    for c in range(Y.shape[0]):
        rr = audio_rms(Y[c])
        pk = audio_peak(Y[c])
        flags.append((rr > rms_thresh) and (pk > peak_thresh))
    return np.asarray(flags, dtype=bool)


def crop_is_valid(y, min_active_stems=2, rms_thresh=1e-4, peak_thresh=1e-3, require_vocal_active=False):
    act = stem_activity_mask(y, rms_thresh=rms_thresh, peak_thresh=peak_thresh)
    ok = int(np.sum(act)) >= int(min_active_stems)
    if require_vocal_active:
        ok = ok and bool(act[0])
    return bool(ok), act


def sample_active_crop(X, Y, T, min_active_stems=2, rms_thresh=1e-4, peak_thresh=1e-3,
                       require_vocal_active=False, aligned_to=1, rng=None, max_tries=80,
                       strict_activity=False):
    rng = np.random.RandomState() if rng is None else rng
    C, L = X.shape

    if L <= T:
        x = X[:, :T]
        y = Y[:, :T]
        if x.shape[1] < T:
            x = np.pad(x, ((0, 0), (0, T - x.shape[1])))
            y = np.pad(y, ((0, 0), (0, T - y.shape[1])))
        return x, y, 0, stem_activity_mask(y, rms_thresh, peak_thresh)

    max_start = L - T
    for _ in range(max_tries):
        s = rng.randint(0, max_start + 1)
        s = (s // aligned_to) * aligned_to
        x = X[:, s:s + T]
        y = Y[:, s:s + T]
        ok, act = crop_is_valid(
            y,
            min_active_stems=min_active_stems,
            rms_thresh=rms_thresh,
            peak_thresh=peak_thresh,
            require_vocal_active=require_vocal_active,
        )
        if ok:
            return x, y, s, act

    # fallback: first aligned crop
    s = 0
    x = X[:, s:s + T]
    y = Y[:, s:s + T]
    if x.shape[1] < T:
        x = np.pad(x, ((0, 0), (0, T - x.shape[1])))
        y = np.pad(y, ((0, 0), (0, T - y.shape[1])))
    ok, act = crop_is_valid(
        y,
        min_active_stems=min_active_stems,
        rms_thresh=rms_thresh,
        peak_thresh=peak_thresh,
        require_vocal_active=require_vocal_active,
    )
    if strict_activity and (not ok):
        raise RuntimeError(
            f"Could not find valid crop: min_active_stems={min_active_stems}, require_vocal_active={require_vocal_active}"
        )
    return x, y, s, act


# ============================================================
# Full-song overlap-add inference
# ============================================================

def build_fade_window(T):
    w = np.hanning(T).astype(np.float32)
    if np.max(w) <= 0:
        w = np.ones((T,), np.float32)
    return w


def run_full_inference(model, x_full, chunk_T=32768, hop_T=16384, batch=1):
    """
    x_full: [C, L]
    returns y_full: [C, L]
    """
    x_full = sanitize_audio(x_full)
    C, L = x_full.shape

    if L <= chunk_T:
        xin = np.pad(x_full, ((0, 0), (0, max(0, chunk_T - L))))
        y = model.predict(xin[None, ...], verbose=0)[0]
        return y[:, :L].astype(np.float32)

    starts = list(range(0, max(1, L - chunk_T + 1), hop_T))
    if starts[-1] != (L - chunk_T):
        starts.append(L - chunk_T)

    win = build_fade_window(chunk_T)
    acc = np.zeros((C, L), dtype=np.float32)
    wsum = np.zeros((L,), dtype=np.float32)

    for i in range(0, len(starts), batch):
        sb = starts[i:i + batch]
        xb = np.stack([x_full[:, s:s + chunk_T] for s in sb], axis=0).astype(np.float32)
        yb = model.predict(xb, verbose=0)
        for j, s in enumerate(sb):
            acc[:, s:s + chunk_T] += yb[j] * win[None, :]
            wsum[s:s + chunk_T] += win

    acc /= np.maximum(wsum[None, :], 1e-8)
    return acc.astype(np.float32)


# ============================================================
# Visualization helpers
# ============================================================

def save_wave_plot(path, signals, labels, sr=22050, max_sec=3.0):
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


def evaluate_full_song_metrics(X_full, Y_full, Yp_full):
    rows = []
    C = Y_full.shape[0]
    for ch in range(C):
        y = Y_full[ch]
        x_in = X_full[ch]
        yhat = Yp_full[ch]
        interf = np.sum(Y_full[np.arange(C) != ch], axis=0)
        sisdr_in = sisdr_np(y, x_in)
        snr_in = snr_np(y, x_in)
        sir_in, sar_in = sir_sar_np(y, interf, x_in)
        sisdr = sisdr_np(y, yhat)
        snr = snr_np(y, yhat)
        sir, sar = sir_sar_np(y, interf, yhat)
        rows.append({
            "ch": ch,
            "stem": STEMS[ch],
            "sisdr_in": sisdr_in,
            "snr_in": snr_in,
            "sir_in": sir_in,
            "sar_in": sar_in,
            "sisdr": sisdr,
            "snr": snr,
            "sir": sir,
            "sar": sar,
            "sisdr_impr": sisdr - sisdr_in,
            "snr_impr": snr - snr_in,
            "sir_impr": sir - sir_in,
            "sar_impr": sar - sar_in,
        })
    return rows


def simple_input_mask_postprocess(x_in, y_pred, eps=1e-8, power=1.0):
    x_in = sanitize_audio(x_in)
    y_pred = sanitize_audio(y_pred)
    X = np.fft.rfft(x_in)
    Yp = np.fft.rfft(y_pred)
    magX = np.abs(X)
    magY = np.abs(Yp)
    mask = np.clip(magY / (magX + eps), 0.0, 1.0) ** power
    Ypp = X * mask * np.exp(1j * np.angle(Yp))
    y = np.fft.irfft(Ypp, n=len(y_pred)).astype(np.float32)
    return sanitize_audio(y)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rerec_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--best_model", required=True)

    p.add_argument("--x_subdir", type=str, default="X")
    p.add_argument("--y_subdir", type=str, default="Y")
    p.add_argument("--target_sr", type=int, default=22050)

    # Metric-eval crop params
    p.add_argument("--crop_T", type=int, default=32768)
    p.add_argument("--n_crops_per_song", type=int, default=8)
    p.add_argument("--min_active_stems", type=int, default=2)
    p.add_argument("--require_vocal_every_n", type=int, default=0)
    p.add_argument("--strict_activity", type=int, default=0)
    p.add_argument("--stem_rms_thresh", type=float, default=1e-4)
    p.add_argument("--stem_peak_thresh", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)

    # Full-song export params
    p.add_argument("--full_example_ids", type=int, nargs="*", default=[0, 10])
    p.add_argument("--full_chunk_T", type=int, default=32768)
    p.add_argument("--full_hop_T", type=int, default=16384)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--save_postprocessed", type=int, default=0)

    # Model hyperparams (must match training)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--filter_length", type=int, default=101)
    p.add_argument("--pr_shifts", type=int, default=24)
    p.add_argument("--pr_lambda", type=float, default=1e-2)
    p.add_argument("--pr_dc_lambda", type=float, default=1e-2)
    p.add_argument("--pr_nyq_lambda", type=float, default=1e-2)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    songs = list_song_dirs(args.rerec_root, args.split, x_subdir=args.x_subdir, y_subdir=args.y_subdir)
    if not songs:
        raise RuntimeError("No valid songs found.")

    print("kept songs:", len(songs))
    print("first kept songs:", [s[0] for s in songs[:5]])

    # Build temporary crop arrays for metric evaluation
    crop_X = []
    crop_Y = []
    crop_meta = []

    align = 2 ** int(args.levels)
    crop_counter = 0

    for song_name, song_dir in songs:
        X_full, Y_full = load_song(song_dir, x_subdir=args.x_subdir, y_subdir=args.y_subdir, target_sr=args.target_sr)
        for k in range(int(args.n_crops_per_song)):
            need_vocal = (args.require_vocal_every_n > 0) and ((crop_counter % int(args.require_vocal_every_n)) == 0)
            x, y, s, act = sample_active_crop(
                X_full, Y_full,
                T=int(args.crop_T),
                min_active_stems=int(args.min_active_stems),
                rms_thresh=float(args.stem_rms_thresh),
                peak_thresh=float(args.stem_peak_thresh),
                require_vocal_active=need_vocal,
                aligned_to=align,
                rng=rng,
                strict_activity=bool(args.strict_activity),
            )
            crop_X.append(x)
            crop_Y.append(y)
            crop_meta.append({
                "song": song_name,
                "crop_idx": k,
                "start": int(s),
                "active_mask": act.astype(int).tolist(),
            })
            crop_counter += 1

    X = np.stack(crop_X, axis=0).astype(np.float32)
    Y = np.stack(crop_Y, axis=0).astype(np.float32)
    print("crop dataset shape", X.shape, Y.shape)

    out_root = ensure_dir(os.path.join(os.path.dirname(args.best_model), "test_outputs_iprdwt_rerecorded"))
    ex_dir = ensure_dir(os.path.join(out_root, "examples_full_audio"))
    fig_dir = ensure_dir(os.path.join(out_root, "figs"))

    # Save temp npy for reproducibility/debug
    np.save(os.path.join(out_root, "Xtest_crops.npy"), X)
    np.save(os.path.join(out_root, "Ytest_crops.npy"), Y)
    with open(os.path.join(out_root, "crop_meta.json"), "w") as f:
        json.dump(crop_meta, f, indent=2)

    C = X.shape[1]
    T = X.shape[2]

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

    print("[Load model]", args.best_model)
    base_model = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=_base_builder,
        C=C, T=T, levels=args.levels,
    )

    # Metric evaluation on crops
    Yhat = np.zeros_like(Y, dtype=np.float32)
    for s in range(0, len(X), args.batch):
        e = min(len(X), s + args.batch)
        Yhat[s:e] = base_model.predict(X[s:e], verbose=0)

    rows = []
    for n in range(len(X)):
        song = crop_meta[n]["song"]
        crop_idx = crop_meta[n]["crop_idx"]
        start = crop_meta[n]["start"]
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
                "song": song,
                "crop_idx": crop_idx,
                "start": start,
                "ch": ch,
                "active_mask": "".join(str(int(v)) for v in crop_meta[n]["active_mask"]),
                "n_active": int(np.sum(crop_meta[n]["active_mask"])),
                "sisdr_in": sisdr_in,
                "snr_in": snr_in,
                "sir_in": sir_in,
                "sar_in": sar_in,
                "sisdr": sisdr,
                "snr": snr,
                "sir": sir,
                "sar": sar,
                "sisdr_impr": sisdr - sisdr_in,
                "snr_impr": snr - snr_in,
                "sir_impr": sir - sir_in,
                "sar_impr": sar - sar_in,
            })

    per_csv = os.path.join(out_root, "metrics_per_crop.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[Saved]", per_csv)

    def summarize(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(vals.mean()), float(np.median(vals)), float(vals.std())

    summary_keys = [
        "sisdr_in", "snr_in", "sir_in", "sar_in",
        "sisdr", "snr", "sir", "sar",
        "sisdr_impr", "snr_impr", "sir_impr", "sar_impr",
    ]
    summ = {k: {"mean": summarize(k)[0], "median": summarize(k)[1], "std": summarize(k)[2]}
            for k in summary_keys}

    summ_json = os.path.join(out_root, "metrics_summary.json")
    with open(summ_json, "w") as f:
        json.dump(summ, f, indent=2)
    print("[Saved]", summ_json)

    summ_csv = os.path.join(out_root, "metrics_summary.csv")
    with open(summ_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "std"])
        for k in summary_keys:
            w.writerow([k, summ[k]["mean"], summ[k]["median"], summ[k]["std"]])
    print("[Saved]", summ_csv)

    # Song-level aggregation
    song_to_rows = {}
    for r in rows:
        song_to_rows.setdefault(r["song"], []).append(r)

    song_csv = os.path.join(out_root, "metrics_per_song.csv")
    with open(song_csv, "w", newline="") as f:
        fieldnames = ["song"] + summary_keys
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for song, rr in song_to_rows.items():
            row = {"song": song}
            for k in summary_keys:
                row[k] = float(np.mean([x[k] for x in rr]))
            w.writerow(row)
    print("[Saved]", song_csv)

    # Full-song exports for listening
    full_ids = [i for i in args.full_example_ids if 0 <= i < len(songs)]
    if not full_ids:
        full_ids = [0, min(1, len(songs) - 1)]

    full_meta = []
    full_song_rows = []
    for idx in full_ids[:2]:
        song_name, song_dir = songs[idx]
        X_full, Y_full = load_song(song_dir, x_subdir=args.x_subdir, y_subdir=args.y_subdir, target_sr=args.target_sr)
        Yp_full = run_full_inference(
            base_model,
            X_full,
            chunk_T=int(args.full_chunk_T),
            hop_T=int(args.full_hop_T),
            batch=int(args.batch),
        )

        song_out_dir = ensure_dir(os.path.join(ex_dir, f"{idx:03d}_{song_name.replace('/', '_')}"))

        full_rows = evaluate_full_song_metrics(X_full, Y_full, Yp_full)
        for rr in full_rows:
            rr["song_index"] = idx
            rr["song"] = song_name
            full_song_rows.append(rr)

        Ypp_full = None
        if bool(args.save_postprocessed):
            Ypp_full = np.stack([
                simple_input_mask_postprocess(X_full[ch], Yp_full[ch]) for ch in range(len(STEMS))
            ], axis=0)

        for ch, stem in enumerate(STEMS):
            in_path = os.path.join(song_out_dir, f"{stem}_input_full.wav")
            tg_path = os.path.join(song_out_dir, f"{stem}_target_full.wav")
            pr_path = os.path.join(song_out_dir, f"{stem}_pred_full.wav")

            # raw export: no independent normalization
            sf.write(in_path, sanitize_audio(X_full[ch]), args.target_sr)
            sf.write(tg_path, sanitize_audio(Y_full[ch]), args.target_sr)
            sf.write(pr_path, sanitize_audio(Yp_full[ch]), args.target_sr)
            if Ypp_full is not None:
                pp_path = os.path.join(song_out_dir, f"{stem}_pred_post_full.wav")
                sf.write(pp_path, sanitize_audio(Ypp_full[ch]), args.target_sr)

            save_wave_plot(
                os.path.join(song_out_dir, f"{stem}_wave_preview.png"),
                [X_full[ch], Y_full[ch], Yp_full[ch]],
                ["input", "target", "pred"],
                sr=args.target_sr,
                max_sec=3.0,
            )

        full_meta.append({
            "song_index": idx,
            "song": song_name,
            "out_dir": song_out_dir,
        })

    with open(os.path.join(out_root, "full_examples.json"), "w") as f:
        json.dump(full_meta, f, indent=2)

    if full_song_rows:
        full_song_csv = os.path.join(out_root, "metrics_full_song_examples.csv")
        with open(full_song_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(full_song_rows[0].keys()))
            w.writeheader()
            for r in full_song_rows:
                w.writerow(r)
        print("[Saved]", full_song_csv)

    print("\n[DONE]")
    print("Outputs saved to:", out_root)


if __name__ == "__main__":
    main()
