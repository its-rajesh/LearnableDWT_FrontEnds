#!/usr/bin/env python3
"""
prepare_and_eval_ldwt_rerecorded.py

Build common fixed-shape rerecorded arrays and run LDWT full-song inference.

Outputs:
  - Xrerecorded.npy              : [N, 3, Tmax]
  - Yrerecorded.npy              : [N, 3, Tmax]
  - Ypred_rerecorded_ldwt.npy    : [N, 3, Tmax]
  - rerecorded_lengths.npy       : [N] original per-song lengths before padding
  - rerecorded_song_order.json   : song names / folders in saved order

Why padded arrays?
  Rerecorded songs can have different durations. To make one common .npy usable by
  all baselines, we pad every song to the global maximum length Tmax and save the
  true lengths separately.

The LDWT model itself still runs on the true song length using overlap-add chunked
inference, then the prediction is padded back to Tmax before saving.
"""

import os
import json
import math
import argparse
import tempfile
import zipfile
import shutil

import numpy as np
import soundfile as sf
import scipy.signal as sp
import tensorflow as tf
from tqdm import tqdm

from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
)

STEMS = ["vocals", "bass", "drums"]


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def sanitize_audio(x):
    x = np.asarray(x, np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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


def list_song_dirs(root: str, split: str, x_subdir="X", y_subdir="Y"):
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


def build_fade_window(T):
    w = np.hanning(T).astype(np.float32)
    if np.max(w) <= 0:
        w = np.ones((T,), np.float32)
    return w


def run_full_inference(model, x_full, chunk_T=32768, hop_T=16384, batch=1):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rerec_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--best_model", required=True)

    p.add_argument("--x_subdir", type=str, default="X")
    p.add_argument("--y_subdir", type=str, default="Y")
    p.add_argument("--target_sr", type=int, default=22050)

    p.add_argument("--full_chunk_T", type=int, default=32768)
    p.add_argument("--full_hop_T", type=int, default=16384)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--out_dir", type=str, default=None)

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

    songs = list_song_dirs(args.rerec_root, args.split, x_subdir=args.x_subdir, y_subdir=args.y_subdir)
    if not songs:
        raise RuntimeError("No valid songs found.")

    out_dir = args.out_dir if args.out_dir is not None else os.path.dirname(args.best_model)
    ensure_dir(out_dir)

    print("kept songs:", len(songs))
    print("first kept songs:", [s[0] for s in songs[:5]])

    X_list, Y_list, lengths, song_meta = [], [], [], []
    for idx, (song_name, song_dir) in enumerate(tqdm(songs, desc="Loading rerecorded songs")):
        X_full, Y_full = load_song(song_dir, x_subdir=args.x_subdir, y_subdir=args.y_subdir, target_sr=args.target_sr)
        L = int(X_full.shape[1])
        X_list.append(X_full)
        Y_list.append(Y_full)
        lengths.append(L)
        song_meta.append({
            "index": idx,
            "song": song_name,
            "song_dir": song_dir,
            "length": L,
        })

    lengths = np.asarray(lengths, dtype=np.int64)
    Tmax = int(lengths.max())
    N = len(X_list)
    C = X_list[0].shape[0]

    X_arr = np.zeros((N, C, Tmax), dtype=np.float32)
    Y_arr = np.zeros((N, C, Tmax), dtype=np.float32)
    for i, (x, y) in enumerate(zip(X_list, Y_list)):
        L = x.shape[1]
        X_arr[i, :, :L] = x
        Y_arr[i, :, :L] = y

    xr_path = os.path.join(out_dir, "Xrerecorded.npy")
    yr_path = os.path.join(out_dir, "Yrerecorded.npy")
    len_path = os.path.join(out_dir, "rerecorded_lengths.npy")
    meta_path = os.path.join(out_dir, "rerecorded_song_order.json")

    np.save(xr_path, X_arr)
    np.save(yr_path, Y_arr)
    np.save(len_path, lengths)
    with open(meta_path, "w") as f:
        json.dump(song_meta, f, indent=2)

    print("[Saved]", xr_path, X_arr.shape)
    print("[Saved]", yr_path, Y_arr.shape)
    print("[Saved]", len_path, lengths.shape, "Tmax =", Tmax)
    print("[Saved]", meta_path)

    infer_T = int(args.full_chunk_T)
    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    def _base_builder():
        return build_base_model(
            T=infer_T,
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
        C=C,
        T=infer_T,
        levels=args.levels,
    )

    Ypred = np.zeros((N, C, Tmax), dtype=np.float32)
    for i, x_full in enumerate(tqdm(X_list, desc="LDWT full-song inference")):
        yp = run_full_inference(
            base_model,
            x_full,
            chunk_T=int(args.full_chunk_T),
            hop_T=int(args.full_hop_T),
            batch=int(args.batch),
        )
        L = yp.shape[1]
        Ypred[i, :, :L] = yp

    ypred_path = os.path.join(out_dir, "Ypred_rerecorded_ldwt.npy")
    np.save(ypred_path, Ypred)

    print("[Saved]", ypred_path, Ypred.shape)
    print("\nDone.")
    print("Common arrays:")
    print("  Xrerecorded.npy")
    print("  Yrerecorded.npy")
    print("  rerecorded_lengths.npy")
    print("  rerecorded_song_order.json")
    print("Prediction:")
    print("  Ypred_rerecorded_ldwt.npy")


if __name__ == "__main__":
    main()
