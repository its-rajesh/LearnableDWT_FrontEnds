#!/usr/bin/env python3
import os
import json
import time
import math
import argparse
import random
from collections import deque

import numpy as np
import soundfile as sf
import scipy.signal as sp
import tensorflow as tf

import iprdwt as ip


# =========================================================
# Constants
# =========================================================
STEMS = ["vocals", "bass", "drums"]


# =========================================================
# General utilities
# =========================================================
def _now_run_dir(root="runs_finetune_recorded"):
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, time.strftime("%Y%m%d_%H%M%S"))


def sanitize_audio(x):
    x = np.asarray(x, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def audio_peak(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return float(np.max(np.abs(x)) + eps)


def audio_rms(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def count_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def _read_train_config_from_pretrained(pretrained_path):
    cfg = {}
    try:
        run_dir = os.path.dirname(os.path.abspath(pretrained_path))
        cfg_path = os.path.join(run_dir, "train_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            print(f"[Config] loaded: {cfg_path}")
        else:
            print(f"[Config] train_config.json not found next to pretrained ({cfg_path}). Using defaults.")
    except Exception as e:
        print(f"[Config] could not read train_config.json: {e}. Using defaults.")
    return cfg


def set_pr_lambda_zero(base_model):
    for layer in base_model.layers:
        if isinstance(layer, ip.PRDWT1D):
            try:
                layer.pr_lambda = 0.0
            except Exception:
                pass


def freeze_dwt_idwt_only(base_model, freeze=True):
    for layer in base_model.layers:
        if isinstance(layer, (ip.PRDWT1D, ip.PRIDWT1D)):
            layer.trainable = (not freeze)


def build_base_like_pretrained(channels, cfg):
    levels = int(cfg.get("levels", 2))
    filter_length = int(cfg.get("filter_length", 101))
    pr_shifts = int(cfg.get("pr_shifts", 32))
    pr_lambda_B = float(cfg.get("pr_lambda_B", 1e-1))
    pr_dc_lambda = float(cfg.get("pr_dc_lambda", 1.0))
    pr_nyq_lambda = float(cfg.get("pr_nyq_lambda", 1.0))

    base = ip.build_pr_dwt_unet(
        time_length=None,
        channels=int(channels),
        levels=levels,
        filter_length=filter_length,
        pr_shifts=pr_shifts,
        pr_lambda=pr_lambda_B,
        pr_dc_lambda=pr_dc_lambda,
        pr_nyq_lambda=pr_nyq_lambda,
        unet_depth=4,
        base_filters=64,
        return_taps=False,
    )
    return base, levels


def load_weights_into_base(base, pretrained_path, build_T, channels):
    dummy = tf.zeros([1, int(channels), int(build_T)], dtype=tf.float32)
    _ = base(dummy, training=False)
    print(f"[Load weights] {pretrained_path}")
    base.load_weights(pretrained_path)
    print("[Loaded] weights into freshly-built base model.")
    return base


# =========================================================
# Wav-tree scanning
# =========================================================
def scan_song_dirs(split_root, expected_src_sr=44100):
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split root not found: {split_root}")

    records = []
    for song in sorted(os.listdir(split_root)):
        song_dir = os.path.join(split_root, song)
        if not os.path.isdir(song_dir):
            continue

        xdir = os.path.join(song_dir, "X")
        ydir = os.path.join(song_dir, "Y")
        if not (os.path.isdir(xdir) and os.path.isdir(ydir)):
            continue

        x_paths = [os.path.join(xdir, f"{stem}.wav") for stem in STEMS]
        y_paths = [os.path.join(ydir, f"{stem}.wav") for stem in STEMS]
        if not all(os.path.isfile(p) for p in x_paths + y_paths):
            continue

        infos = [sf.info(p) for p in x_paths + y_paths]
        srs = [inf.samplerate for inf in infos]
        if len(set(srs)) != 1:
            print(f"[Skip] inconsistent SR in {song_dir}")
            continue

        src_sr = int(srs[0])
        if src_sr != int(expected_src_sr):
            print(f"[Skip] unexpected source SR={src_sr} in {song_dir}, expected {expected_src_sr}")
            continue

        frames = [inf.frames for inf in infos]
        min_frames = int(min(frames))
        if min_frames < 8192:
            print(f"[Skip] too short: {song_dir}")
            continue

        records.append({
            "song": song,
            "song_dir": song_dir,
            "x_paths": x_paths,
            "y_paths": y_paths,
            "src_sr": src_sr,
            "frames_src": min_frames,
        })

    return records


def split_records(records, val_ratio=0.15, seed=1337):
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)

    if len(recs) <= 1:
        return recs, []

    n_val = max(1, int(round(len(recs) * val_ratio)))
    val_records = recs[:n_val]
    train_records = recs[n_val:]

    if len(train_records) == 0:
        train_records = val_records[:1]
        val_records = val_records[1:]

    return train_records, val_records


# =========================================================
# Crop reading + resampling
# =========================================================
def _read_mono_segment(path, start, frames, dtype="float32"):
    x, _ = sf.read(path, start=int(start), frames=int(frames), dtype=dtype, always_2d=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = sanitize_audio(x)
    if len(x) < frames:
        x = np.pad(x, (0, frames - len(x)))
    return x.astype(np.float32)


def _resample_44100_to_22050(x):
    x = sanitize_audio(x)
    y = sp.resample_poly(x, up=1, down=2).astype(np.float32)
    y = sanitize_audio(y)
    return y


def read_pair_crop_resampled(
    record,
    start_tgt,
    crop_T_tgt,
    target_sr=22050,
    pair_peak_norm=None,   # disabled by default now
):
    src_sr = int(record["src_sr"])
    if src_sr % target_sr != 0:
        raise ValueError(f"Unsupported resample ratio: src_sr={src_sr}, target_sr={target_sr}")

    ratio = src_sr // target_sr
    crop_T_src = int(crop_T_tgt * ratio)
    start_src = int(start_tgt * ratio)

    X = np.stack([_read_mono_segment(p, start_src, crop_T_src) for p in record["x_paths"]], axis=0)
    Y = np.stack([_read_mono_segment(p, start_src, crop_T_src) for p in record["y_paths"]], axis=0)

    if ratio == 2:
        X = np.stack([_resample_44100_to_22050(X[c]) for c in range(X.shape[0])], axis=0)
        Y = np.stack([_resample_44100_to_22050(Y[c]) for c in range(Y.shape[0])], axis=0)
    elif ratio == 1:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
    else:
        raise ValueError(f"Unsupported resample ratio: {ratio}")

    X = X[:, :crop_T_tgt]
    Y = Y[:, :crop_T_tgt]
    if X.shape[1] < crop_T_tgt:
        X = np.pad(X, ((0, 0), (0, crop_T_tgt - X.shape[1])))
    if Y.shape[1] < crop_T_tgt:
        Y = np.pad(Y, ((0, 0), (0, crop_T_tgt - Y.shape[1])))

    X = sanitize_audio(X)
    Y = sanitize_audio(Y)

    # disabled unless explicitly requested
    if pair_peak_norm is not None and pair_peak_norm > 0:
        pk = max(audio_peak(X), audio_peak(Y))
        if np.isfinite(pk) and pk > 1e-8:
            g = min(float(pair_peak_norm) / pk, 50.0)
            X = X * g
            Y = Y * g

    X = sanitize_audio(X)
    Y = sanitize_audio(Y)
    return X.astype(np.float32), Y.astype(np.float32)


# =========================================================
# Activity-aware crop filtering
# =========================================================
def stem_activity_mask(Y, rms_thresh=1e-4, peak_thresh=1e-3):
    """
    Returns bool[3] based on TARGET crop activity.
    """
    flags = []
    for c in range(Y.shape[0]):
        rr = audio_rms(Y[c])
        pk = audio_peak(Y[c])
        flags.append((rr > rms_thresh) and (pk > peak_thresh))
    return np.asarray(flags, dtype=bool)


def crop_is_valid(
    X,
    Y,
    min_pair_rms=1e-6,
    stem_rms_thresh=1e-4,
    stem_peak_thresh=1e-3,
    min_active_stems=2,
    require_vocal_active=False,
):
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        return False, np.zeros((3,), dtype=bool)

    xr = audio_rms(X)
    yr = audio_rms(Y)
    if (xr <= min_pair_rms) and (yr <= min_pair_rms):
        return False, np.zeros((3,), dtype=bool)

    act = stem_activity_mask(Y, rms_thresh=stem_rms_thresh, peak_thresh=stem_peak_thresh)
    if int(np.sum(act)) < int(min_active_stems):
        return False, act
    if require_vocal_active and (not bool(act[0])):
        return False, act

    return True, act


class RandomSongCropSequence:
    """
    Activity-aware crop sampler.
    """
    def __init__(
        self,
        records,
        crop_T_tgt,
        aligned_to,
        target_sr=22050,
        seed=1337,
        pair_peak_norm=None,
        max_tries=80,
        min_pair_rms=1e-6,
        stem_rms_thresh=1e-4,
        stem_peak_thresh=1e-3,
        min_active_stems=2,
        require_vocal_every_n=0,
    ):
        self.records = list(records)
        self.crop_T_tgt = int(crop_T_tgt)
        self.aligned_to = int(max(1, aligned_to))
        self.target_sr = int(target_sr)
        self.rng = np.random.RandomState(seed)
        self.pair_peak_norm = pair_peak_norm
        self.max_tries = int(max_tries)

        self.min_pair_rms = float(min_pair_rms)
        self.stem_rms_thresh = float(stem_rms_thresh)
        self.stem_peak_thresh = float(stem_peak_thresh)
        self.min_active_stems = int(min_active_stems)
        self.require_vocal_every_n = int(require_vocal_every_n)
        self.sample_count = 0

        self.records = [r for r in self.records if r["frames_src"] >= (self.crop_T_tgt * (r["src_sr"] // self.target_sr))]
        if len(self.records) == 0:
            raise ValueError("No training records are long enough for crop_T after source->target resampling.")

        # stats
        self.accepted = 0
        self.rejected_inactive = 0
        self.rejected_silent = 0
        self.rejected_nonfinite = 0
        self.last_stats_window = deque(maxlen=200)

    def _need_vocal_now(self):
        if self.require_vocal_every_n <= 0:
            return False
        return (self.sample_count % self.require_vocal_every_n) == 0

    def _sample_candidate(self):
        rec = self.records[self.rng.randint(0, len(self.records))]
        ratio = int(rec["src_sr"] // self.target_sr)
        frames_tgt = rec["frames_src"] // ratio
        max_start_tgt = frames_tgt - self.crop_T_tgt

        if max_start_tgt <= 0:
            start_tgt = 0
        else:
            start_tgt = self.rng.randint(0, max_start_tgt + 1)
            start_tgt = (start_tgt // self.aligned_to) * self.aligned_to

        X, Y = read_pair_crop_resampled(
            rec,
            start_tgt=start_tgt,
            crop_T_tgt=self.crop_T_tgt,
            target_sr=self.target_sr,
            pair_peak_norm=self.pair_peak_norm,
        )
        return X, Y

    def sample_one(self):
        need_vocal = self._need_vocal_now()

        for _ in range(self.max_tries):
            X, Y = self._sample_candidate()

            if (not np.all(np.isfinite(X))) or (not np.all(np.isfinite(Y))):
                self.rejected_nonfinite += 1
                self.last_stats_window.append("nonfinite")
                continue

            ok, act = crop_is_valid(
                X, Y,
                min_pair_rms=self.min_pair_rms,
                stem_rms_thresh=self.stem_rms_thresh,
                stem_peak_thresh=self.stem_peak_thresh,
                min_active_stems=self.min_active_stems,
                require_vocal_active=need_vocal,
            )
            if ok:
                self.accepted += 1
                self.sample_count += 1
                self.last_stats_window.append("ok")
                return X, Y

            # split reject reasons roughly
            if (audio_rms(X) <= self.min_pair_rms) and (audio_rms(Y) <= self.min_pair_rms):
                self.rejected_silent += 1
                self.last_stats_window.append("silent")
            else:
                self.rejected_inactive += 1
                self.last_stats_window.append("inactive")

        # fallback: still return something valid-ish rather than crash
        for _ in range(20):
            X, Y = self._sample_candidate()
            if np.all(np.isfinite(X)) and np.all(np.isfinite(Y)):
                self.sample_count += 1
                self.last_stats_window.append("fallback")
                return X, Y

        raise RuntimeError("Could not sample a usable crop.")

    def stats(self):
        total = self.accepted + self.rejected_inactive + self.rejected_silent + self.rejected_nonfinite
        return {
            "accepted": self.accepted,
            "rejected_inactive": self.rejected_inactive,
            "rejected_silent": self.rejected_silent,
            "rejected_nonfinite": self.rejected_nonfinite,
            "total_seen": total,
            "recent_ok_frac": float(np.mean([1.0 if s == "ok" else 0.0 for s in self.last_stats_window])) if len(self.last_stats_window) else np.nan,
        }


def make_train_ds_from_records(
    records,
    batch_size,
    crop_T,
    aligned_to,
    target_sr=22050,
    seed=1337,
    pair_peak_norm=None,
    min_pair_rms=1e-6,
    stem_rms_thresh=1e-4,
    stem_peak_thresh=1e-3,
    min_active_stems=2,
    require_vocal_every_n=0,
):
    sampler = RandomSongCropSequence(
        records=records,
        crop_T_tgt=crop_T,
        aligned_to=aligned_to,
        target_sr=target_sr,
        seed=seed,
        pair_peak_norm=pair_peak_norm,
        min_pair_rms=min_pair_rms,
        stem_rms_thresh=stem_rms_thresh,
        stem_peak_thresh=stem_peak_thresh,
        min_active_stems=min_active_stems,
        require_vocal_every_n=require_vocal_every_n,
    )

    def gen():
        while True:
            X, Y = sampler.sample_one()
            yield X, Y

    sig = (
        tf.TensorSpec(shape=(3, crop_T), dtype=tf.float32),
        tf.TensorSpec(shape=(3, crop_T), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds, sampler


def deterministic_val_crops(
    records,
    crop_T,
    aligned_to,
    target_sr=22050,
    crops_per_song=2,
    pair_peak_norm=None,
    min_pair_rms=1e-6,
    stem_rms_thresh=1e-4,
    stem_peak_thresh=1e-3,
    min_active_stems=1,   # validation can be looser
):
    out = []
    for rec in records:
        ratio = int(rec["src_sr"] // target_sr)
        frames_tgt = rec["frames_src"] // ratio
        if frames_tgt < crop_T:
            continue

        if crops_per_song <= 1:
            starts = [0 if frames_tgt <= crop_T else (frames_tgt - crop_T) // 2]
        else:
            max_start = frames_tgt - crop_T
            starts = np.linspace(0, max_start, num=crops_per_song)
            starts = [int(s) for s in starts]

        used = []
        for s in starts:
            s = (s // aligned_to) * aligned_to
            if s not in used:
                used.append(s)

        for s in used:
            X, Y = read_pair_crop_resampled(
                rec,
                start_tgt=s,
                crop_T_tgt=crop_T,
                target_sr=target_sr,
                pair_peak_norm=pair_peak_norm,
            )
            ok, _ = crop_is_valid(
                X, Y,
                min_pair_rms=min_pair_rms,
                stem_rms_thresh=stem_rms_thresh,
                stem_peak_thresh=stem_peak_thresh,
                min_active_stems=min_active_stems,
                require_vocal_active=False,
            )
            if ok:
                out.append((X, Y))

    if len(out) == 0:
        raise ValueError("Validation set ended up empty.")

    X = np.stack([a for a, _ in out], axis=0).astype(np.float32)
    Y = np.stack([b for _, b in out], axis=0).astype(np.float32)
    return X, Y


def make_val_ds_from_records(
    records,
    batch_size,
    crop_T,
    aligned_to,
    target_sr=22050,
    crops_per_song=2,
    pair_peak_norm=None,
    min_pair_rms=1e-6,
    stem_rms_thresh=1e-4,
    stem_peak_thresh=1e-3,
    min_active_stems=1,
):
    X, Y = deterministic_val_crops(
        records=records,
        crop_T=crop_T,
        aligned_to=aligned_to,
        target_sr=target_sr,
        crops_per_song=crops_per_song,
        pair_peak_norm=pair_peak_norm,
        min_pair_rms=min_pair_rms,
        stem_rms_thresh=stem_rms_thresh,
        stem_peak_thresh=stem_peak_thresh,
        min_active_stems=min_active_stems,
    )
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds, X.shape[0]


def estimate_steps_per_epoch(records, crop_T, batch_size, target_sr=22050, multiplier=1.0):
    total_frames_tgt = 0.0
    for r in records:
        ratio = int(r["src_sr"] // target_sr)
        total_frames_tgt += float(r["frames_src"] // ratio)
    approx_crops = total_frames_tgt / float(crop_T)
    steps = int(math.ceil((approx_crops * float(multiplier)) / float(batch_size)))
    return max(50, steps)


# =========================================================
# Diagnostics callback
# =========================================================
class CropStatsCallback(tf.keras.callbacks.Callback):
    def __init__(self, sampler, every_n_epochs=1):
        super().__init__()
        self.sampler = sampler
        self.every_n_epochs = int(every_n_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % self.every_n_epochs) != 0:
            return
        st = self.sampler.stats()
        print(
            f"[CropStats] epoch={epoch+1} "
            f"accepted={st['accepted']} "
            f"rej_inactive={st['rejected_inactive']} "
            f"rej_silent={st['rejected_silent']} "
            f"rej_nonfinite={st['rejected_nonfinite']} "
            f"recent_ok_frac={st['recent_ok_frac']:.3f}"
        )


# =========================================================
# Fine-tune
# =========================================================
def fine_tune_two_phase_from_wavs(
    pretrained_path,
    rerec_root,
    train_split,
    crop_T,
    batch,
    phase1_epochs,
    phase2_epochs,
    lr1,
    lr2,
    val_ratio=0.15,
    seed=1337,
    pit=False,
    hf_lambda=0.0,
    clipnorm=0.25,
    pair_peak_norm=None,   # changed default behavior
    steps_per_epoch=None,
    val_crops_per_song=2,
    train_step_multiplier=1.0,
    source_sr=44100,
    target_sr=22050,
    min_pair_rms=1e-6,
    stem_rms_thresh=1e-4,
    stem_peak_thresh=1e-3,
    min_active_stems=2,
    require_vocal_every_n=0,
):
    run_dir = _now_run_dir("runs_finetune_recorded")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Run dir] {run_dir}")

    split_root = os.path.join(rerec_root, train_split)
    print(f"[Scan] {split_root}")
    records = scan_song_dirs(split_root, expected_src_sr=source_sr)
    if len(records) == 0:
        raise ValueError(f"No valid song folders found under {split_root}")

    train_records, val_records = split_records(records, val_ratio=val_ratio, seed=seed)
    if len(train_records) == 0:
        raise ValueError("No training songs after split.")
    if len(val_records) == 0:
        print("[Warn] val split became empty; using one training song for validation.")
        val_records = train_records[:1]

    print(f"[Songs] total={len(records)} train={len(train_records)} val={len(val_records)}")

    C = 3
    cfg = _read_train_config_from_pretrained(pretrained_path)
    base, levels = build_base_like_pretrained(channels=C, cfg=cfg)
    base = load_weights_into_base(base, pretrained_path, build_T=crop_T, channels=C)

    align = 2 ** int(levels)

    tr_ds, tr_sampler = make_train_ds_from_records(
        records=train_records,
        batch_size=batch,
        crop_T=crop_T,
        aligned_to=align,
        target_sr=target_sr,
        seed=seed,
        pair_peak_norm=pair_peak_norm,
        min_pair_rms=min_pair_rms,
        stem_rms_thresh=stem_rms_thresh,
        stem_peak_thresh=stem_peak_thresh,
        min_active_stems=min_active_stems,
        require_vocal_every_n=require_vocal_every_n,
    )

    va_ds, n_val = make_val_ds_from_records(
        records=val_records,
        batch_size=batch,
        crop_T=crop_T,
        aligned_to=align,
        target_sr=target_sr,
        crops_per_song=val_crops_per_song,
        pair_peak_norm=pair_peak_norm,
        min_pair_rms=min_pair_rms,
        stem_rms_thresh=stem_rms_thresh,
        stem_peak_thresh=stem_peak_thresh,
        min_active_stems=1,
    )

    if steps_per_epoch is None:
        steps_per_epoch = estimate_steps_per_epoch(
            train_records,
            crop_T=crop_T,
            batch_size=batch,
            target_sr=target_sr,
            multiplier=train_step_multiplier,
        )

    val_steps = int(math.ceil(n_val / float(batch)))

    print(
        f"[Dataset] source_sr={source_sr} target_sr={target_sr} crop_T={crop_T} "
        f"batch={batch} align={align}"
    )
    print(
        f"[Crop filter] min_active_stems={min_active_stems} "
        f"stem_rms_thresh={stem_rms_thresh} stem_peak_thresh={stem_peak_thresh} "
        f"require_vocal_every_n={require_vocal_every_n}"
    )
    print(
        f"[Normalization] pair_peak_norm={pair_peak_norm} "
        f"(None means disabled; recommended for rerecorded finetune)"
    )
    print(f"[Steps] steps_per_epoch={steps_per_epoch} val_steps={val_steps} n_val_crops={n_val}")

    task_loss_fn = ip.make_task_loss(pit=pit)
    set_pr_lambda_zero(base)

    # -------------------------
    # Phase 1
    # -------------------------
    print("\n===== Phase 1: Freeze DWT+IDWT, train U-Net only =====")
    freeze_dwt_idwt_only(base, freeze=True)

    trainer1 = ip.TwoStageTrainer(
        base_model=base,
        task_loss_fn=task_loss_fn,
        levels=int(levels),
        hf_lambda=float(hf_lambda),
        task_lambda=1.0,
        name="PR_DWT_UNet_ft_phase1",
    )
    opt1 = tf.keras.optimizers.Adam(learning_rate=float(lr1), clipnorm=float(clipnorm))
    trainer1.compile(optimizer=opt1, jit_compile=False)

    print(f"[Trainable params / phase1] {count_trainable_params(base)}")

    best1 = os.path.join(run_dir, "best_phase1.keras")
    last1 = os.path.join(run_dir, "last_phase1.keras")
    cb1 = [
        ip.SaveBestBase(trainer1, best1, monitor="val_loss", mode="min", verbose=1),
        ip.SaveLastBase(trainer1, last1, verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_phase1.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        CropStatsCallback(tr_sampler, every_n_epochs=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    h1 = trainer1.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=int(phase1_epochs),
        steps_per_epoch=int(steps_per_epoch),
        validation_steps=int(val_steps),
        callbacks=cb1,
        verbose=1,
    )

    # -------------------------
    # Phase 2
    # -------------------------
    print("\n===== Phase 2: Unfreeze ALL, train full model =====")
    freeze_dwt_idwt_only(base, freeze=False)

    trainer2 = ip.TwoStageTrainer(
        base_model=base,
        task_loss_fn=task_loss_fn,
        levels=int(levels),
        hf_lambda=float(hf_lambda),
        task_lambda=1.0,
        name="PR_DWT_UNet_ft_phase2",
    )
    opt2 = tf.keras.optimizers.Adam(learning_rate=float(lr2), clipnorm=float(clipnorm))
    trainer2.compile(optimizer=opt2, jit_compile=False)

    print(f"[Trainable params / phase2] {count_trainable_params(base)}")

    best2 = os.path.join(run_dir, "best.keras")
    last2 = os.path.join(run_dir, "last.keras")
    cb2 = [
        ip.SaveBestBase(trainer2, best2, monitor="val_loss", mode="min", verbose=1),
        ip.SaveLastBase(trainer2, last2, verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_phase2.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=20, min_delta=1e-4,
            restore_best_weights=True, verbose=1
        ),
        CropStatsCallback(tr_sampler, every_n_epochs=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    h2 = trainer2.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=int(phase2_epochs),
        steps_per_epoch=int(steps_per_epoch),
        validation_steps=int(val_steps),
        callbacks=cb2,
        verbose=1,
    )

    out_cfg = dict(
        pretrained=pretrained_path,
        rerec_root=rerec_root,
        train_split=train_split,
        crop_T=int(crop_T),
        batch=int(batch),
        phase1_epochs=int(phase1_epochs),
        phase2_epochs=int(phase2_epochs),
        lr1=float(lr1),
        lr2=float(lr2),
        val_ratio=float(val_ratio),
        seed=int(seed),
        pit=bool(pit),
        hf_lambda=float(hf_lambda),
        clipnorm=float(clipnorm),
        pair_peak_norm=None if pair_peak_norm is None else float(pair_peak_norm),
        steps_per_epoch=int(steps_per_epoch),
        val_crops_per_song=int(val_crops_per_song),
        train_step_multiplier=float(train_step_multiplier),
        source_sr=int(source_sr),
        target_sr=int(target_sr),
        min_pair_rms=float(min_pair_rms),
        stem_rms_thresh=float(stem_rms_thresh),
        stem_peak_thresh=float(stem_peak_thresh),
        min_active_stems=int(min_active_stems),
        require_vocal_every_n=int(require_vocal_every_n),
        n_total_songs=int(len(records)),
        n_train_songs=int(len(train_records)),
        n_val_songs=int(len(val_records)),
        levels=int(levels),
        train_songs=[r["song"] for r in train_records],
        val_songs=[r["song"] for r in val_records],
    )
    with open(os.path.join(run_dir, "finetune_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    print(f"\n[Done] Best phase2 model: {best2}")
    return run_dir, h1, h2


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--rerec_root", type=str, required=True)
    ap.add_argument("--train_split", type=str, default="train")

    ap.add_argument("--crop_T", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=2)

    ap.add_argument("--phase1_epochs", type=int, default=50)
    ap.add_argument("--phase2_epochs", type=int, default=200)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=5e-5)

    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--pit", action="store_true")
    ap.add_argument("--hf_lambda", type=float, default=0.0)
    ap.add_argument("--clipnorm", type=float, default=0.25)

    # IMPORTANT: default is disabled now
    ap.add_argument("--pair_peak_norm", type=float, default=0.0,
                    help="<=0 disables per-crop normalization; recommended for rerecorded finetune")
    ap.add_argument("--steps_per_epoch", type=int, default=0)
    ap.add_argument("--val_crops_per_song", type=int, default=2)
    ap.add_argument("--train_step_multiplier", type=float, default=1.0)

    ap.add_argument("--source_sr", type=int, default=22050)
    ap.add_argument("--target_sr", type=int, default=22050)

    # activity-aware crop filtering
    ap.add_argument("--min_pair_rms", type=float, default=1e-6)
    ap.add_argument("--stem_rms_thresh", type=float, default=1e-4)
    ap.add_argument("--stem_peak_thresh", type=float, default=1e-3)
    ap.add_argument("--min_active_stems", type=int, default=2,
                    help="Require at least this many active target stems in a crop")
    ap.add_argument("--require_vocal_every_n", type=int, default=4,
                    help="Every Nth training crop must have active vocals; 0 disables")

    args = ap.parse_args()

    pair_peak_norm = None if args.pair_peak_norm <= 0 else float(args.pair_peak_norm)
    steps_per_epoch = None if args.steps_per_epoch <= 0 else int(args.steps_per_epoch)

    fine_tune_two_phase_from_wavs(
        pretrained_path=args.pretrained,
        rerec_root=args.rerec_root,
        train_split=args.train_split,
        crop_T=args.crop_T,
        batch=args.batch,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        lr1=args.lr1,
        lr2=args.lr2,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pit=args.pit,
        hf_lambda=args.hf_lambda,
        clipnorm=args.clipnorm,
        pair_peak_norm=pair_peak_norm,
        steps_per_epoch=steps_per_epoch,
        val_crops_per_song=args.val_crops_per_song,
        train_step_multiplier=args.train_step_multiplier,
        source_sr=args.source_sr,
        target_sr=args.target_sr,
        min_pair_rms=args.min_pair_rms,
        stem_rms_thresh=args.stem_rms_thresh,
        stem_peak_thresh=args.stem_peak_thresh,
        min_active_stems=args.min_active_stems,
        require_vocal_every_n=args.require_vocal_every_n,
    )


if __name__ == "__main__":
    main()