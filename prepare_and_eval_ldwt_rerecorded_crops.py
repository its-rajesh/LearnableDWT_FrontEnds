#!/usr/bin/env python3
"""
prepare_and_eval_ldwt_rerecorded_crops.py

Builds rerecorded evaluation arrays using activity-aware random crops
(to match training/validation style), then runs LDWT on those crops.

Outputs:
  - Xrerecorded.npy              [N, 3, crop_T]
  - Yrerecorded.npy              [N, 3, crop_T]
  - Ypred_rerecorded_ldwt.npy    [N, 3, crop_T]
  - rerecorded_crop_meta.json

This intentionally does NOT pad full songs to a common max length.
Instead it samples fixed-length crops with activity constraints.

Typical usage:
python prepare_and_eval_ldwt_rerecorded_crops.py \
  --rerec_root /path/to/rerecorded_dataset \
  --split test \
  --best_model /home/rrame12/Desktop/Research/DWT_IR/runs_pr/20260116_170304/best.keras \
  --out_dir /home/rrame12/Desktop/Research/DWT_IR/rerecorded_eval_ldwt \
  --x_subdir X \
  --y_subdir Y \
  --crop_T 32768 \
  --n_crops_per_song 8 \
  --min_active_stems 3 \
  --strict_activity 1
"""

import os
import json
import math
import argparse
import tempfile
import zipfile
import shutil
from typing import List, Tuple

import numpy as np
import soundfile as sf
import scipy.signal as sp
import tensorflow as tf

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
# Audio utils
# ============================================================

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def sanitize_audio(x):
    x = np.asarray(x, np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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
# Model loading (same robust style as test_iprdwt.py)
# ============================================================

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
# Activity-aware crop sampling
# ============================================================

def stem_activity_mask(Y, rms_thresh=1e-4, peak_thresh=1e-3):
    flags = []
    for c in range(Y.shape[0]):
        rr = audio_rms(Y[c])
        pk = audio_peak(Y[c])
        flags.append((rr > rms_thresh) and (pk > peak_thresh))
    return np.asarray(flags, dtype=bool)


def crop_is_valid(y, min_active_stems=3, rms_thresh=1e-4, peak_thresh=1e-3,
                  require_vocal_active=False):
    act = stem_activity_mask(y, rms_thresh=rms_thresh, peak_thresh=peak_thresh)
    ok = int(np.sum(act)) >= int(min_active_stems)
    if require_vocal_active:
        ok = ok and bool(act[0])
    return bool(ok), act


def sample_active_crop(X, Y, T, min_active_stems=3, rms_thresh=1e-4, peak_thresh=1e-3,
                       require_vocal_active=False, aligned_to=1, rng=None, max_tries=200,
                       strict_activity=True):
    rng = np.random.RandomState() if rng is None else rng
    C, L = X.shape

    if L <= T:
        x = X[:, :T]
        y = Y[:, :T]
        if x.shape[1] < T:
            pad = T - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            y = np.pad(y, ((0, 0), (0, pad)))
        ok, act = crop_is_valid(
            y,
            min_active_stems=min_active_stems,
            rms_thresh=rms_thresh,
            peak_thresh=peak_thresh,
            require_vocal_active=require_vocal_active,
        )
        if strict_activity and (not ok):
            raise RuntimeError("Song shorter than crop_T and does not satisfy activity constraint.")
        return x, y, 0, act

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

    if strict_activity:
        raise RuntimeError(
            f"Could not find valid crop after {max_tries} tries "
            f"(min_active_stems={min_active_stems}, require_vocal_active={require_vocal_active})"
        )

    s = 0
    x = X[:, s:s + T]
    y = Y[:, s:s + T]
    ok, act = crop_is_valid(
        y,
        min_active_stems=min_active_stems,
        rms_thresh=rms_thresh,
        peak_thresh=peak_thresh,
        require_vocal_active=require_vocal_active,
    )
    return x, y, s, act


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rerec_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--best_model", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--x_subdir", type=str, default="X")
    p.add_argument("--y_subdir", type=str, default="Y")
    p.add_argument("--target_sr", type=int, default=22050)

    p.add_argument("--crop_T", type=int, default=32768)
    p.add_argument("--n_crops_per_song", type=int, default=8)
    p.add_argument("--min_active_stems", type=int, default=3)
    p.add_argument("--require_vocal_every_n", type=int, default=0)
    p.add_argument("--strict_activity", type=int, default=1)
    p.add_argument("--stem_rms_thresh", type=float, default=1e-4)
    p.add_argument("--stem_peak_thresh", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch", type=int, default=2)

    # model hyperparams
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
    ensure_dir(args.out_dir)
    rng = np.random.RandomState(args.seed)

    songs = list_song_dirs(args.rerec_root, args.split, x_subdir=args.x_subdir, y_subdir=args.y_subdir)
    if not songs:
        raise RuntimeError("No valid songs found.")
    print("kept songs:", len(songs))
    print("first kept songs:", [s[0] for s in songs[:5]])

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
            crop_X.append(x.astype(np.float32))
            crop_Y.append(y.astype(np.float32))
            crop_meta.append({
                "song": song_name,
                "crop_idx": int(k),
                "start": int(s),
                "active_mask": act.astype(int).tolist(),
                "n_active": int(np.sum(act)),
            })
            crop_counter += 1

    X = np.stack(crop_X, axis=0).astype(np.float32)
    Y = np.stack(crop_Y, axis=0).astype(np.float32)
    print("crop dataset shape:", X.shape, Y.shape)

    x_out = os.path.join(args.out_dir, "Xrerecorded.npy")
    y_out = os.path.join(args.out_dir, "Yrerecorded.npy")
    meta_out = os.path.join(args.out_dir, "rerecorded_crop_meta.json")

    np.save(x_out, X)
    np.save(y_out, Y)
    with open(meta_out, "w") as f:
        json.dump(crop_meta, f, indent=2)

    print("[Saved]", x_out)
    print("[Saved]", y_out)
    print("[Saved]", meta_out)

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

    Yhat = np.zeros_like(Y, dtype=np.float32)
    for s in range(0, len(X), int(args.batch)):
        e = min(len(X), s + int(args.batch))
        Yhat[s:e] = base_model.predict(X[s:e], verbose=0)

    ypred_out = os.path.join(args.out_dir, "Ypred_rerecorded_ldwt.npy")
    np.save(ypred_out, Yhat.astype(np.float32))
    print("[Saved]", ypred_out)
    print("Final shapes:")
    print("  Xrerecorded:", X.shape)
    print("  Yrerecorded:", Y.shape)
    print("  Ypred_rerecorded_ldwt:", Yhat.shape)


if __name__ == "__main__":
    main()
