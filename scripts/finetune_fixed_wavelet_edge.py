#!/usr/bin/env python3
"""Small fixed-wavelet finetune on the disjoint edge distribution.

This intentionally keeps the paper split:
  train/val: finetune_bleed_small_disjoint
  test:      Edgecase_active_matched/Xedge_m9db.npy,Yedge_m9db.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT = Path("/home/rrame12/Desktop/Research/DWT_IR")
PYTHON = Path("/home/rrame12/anaconda3/envs/all/bin/python")


def setup_imports() -> None:
    sys.path.insert(0, str(PROJECT))


def custom_objects():
    from ablation_frontends import (
        FixedDWT1D,
        FixedIDWT1D,
        ISTFTBackend,
        MatchTimeLen,
        SplitChannels,
        STFTFrontend,
        TwoStageTrainer,
        UpsampleTo,
    )

    return {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "FixedDWT1D": FixedDWT1D,
        "FixedIDWT1D": FixedIDWT1D,
        "STFTFrontend": STFTFrontend,
        "ISTFTBackend": ISTFTBackend,
        "TwoStageTrainer": TwoStageTrainer,
    }




def positive_sisdr_pair(y_true, y_pred, eps=1e-8):
    yt = y_true - tf.reduce_mean(y_true, axis=2, keepdims=True)
    yp = y_pred - tf.reduce_mean(y_pred, axis=2, keepdims=True)
    dot = tf.reduce_sum(yt * yp, axis=2, keepdims=True)
    energy = tf.reduce_sum(yt ** 2, axis=2, keepdims=True) + eps
    scale = tf.maximum(dot / energy, eps)
    target = scale * yt
    noise = yp - target
    s_target = tf.reduce_sum(target ** 2, axis=2) + eps
    e_noise = tf.reduce_sum(noise ** 2, axis=2) + eps
    return 10.0 * tf.math.log(s_target / e_noise) / tf.math.log(10.0)


def positive_sisdr_loss_no_pit(y_true, y_pred):
    return -tf.reduce_mean(positive_sisdr_pair(y_true, y_pred))


def choose_task_loss(kind: str):
    setup_imports()
    from ablation_frontends import make_task_loss
    if kind == "sisdr":
        return make_task_loss(pit=False)
    if kind == "positive_sisdr":
        return positive_sisdr_loss_no_pit
    raise ValueError(f"Unknown loss {kind}")


def latest_checkpoint(runs_root: Path, wavelet: str) -> Path:
    candidates = sorted(runs_root.glob(f"*_dwt_fixed_{wavelet}/best.keras"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No pretrained checkpoint found for {wavelet} under {runs_root}")
    return candidates[-1]


def split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = int(np.round(len(x) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], train_idx, val_idx


def make_dataset(x, y, batch_size, crop_len=None, aligned_to=4, shuffle=True, seed=1337):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), reshuffle_each_iteration=True, seed=seed)
    if crop_len is not None:
        crop_len = int(crop_len)
        aligned_to = int(max(1, aligned_to))

        def crop_pair(xb, yb):
            t = tf.shape(xb)[-1]
            max_start = tf.maximum(0, t - crop_len)
            start = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
            start = (start // aligned_to) * aligned_to
            return xb[:, start:start + crop_len], yb[:, start:start + crop_len]

        ds = ds.map(crop_pair, num_parallel_calls=1, deterministic=True)
    return ds.batch(batch_size, drop_remainder=True).prefetch(1)


class SaveBest(tf.keras.callbacks.Callback):
    def __init__(self, model, path: Path):
        super().__init__()
        self.base = model
        self.path = path
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss")
        if val_loss is not None and float(val_loss) < self.best:
            self.best = float(val_loss)
            self.base.save(self.path)
            print(f"\nEpoch {epoch + 1}: saved best -> {self.path}  val_loss={self.best:.6f}", flush=True)


def mean_stems(block: dict) -> float:
    return float(np.mean([v["mean"] for v in block.values()]))


def parse_metrics(metrics_dir: Path, method: str) -> dict:
    std = json.load(open(metrics_dir / "standard_metrics_summary.json"))
    mimo = json.load(open(metrics_dir / "mimo_metrics_summary.json"))
    return {
        "Method": method,
        "SI-SDR": std["sisdr"]["mean"],
        "SIR": std["sir"]["mean"],
        "Delta SIR": std["sir"]["mean"] - std["sir_in"]["mean"],
        "SAR": std["sar"]["mean"],
        "SIR(B)": mean_stems(mimo["output_after"]["SIRB"]),
        "Delta SIR(B)": mean_stems(mimo["output_after"]["SIRB"]) - mean_stems(mimo["mixture_before"]["SIRB"]),
        "EXP (%)": mean_stems(mimo["output_after"]["EXP"]),
    }


def parse_reference(metrics_dir: Path) -> dict:
    std = json.load(open(metrics_dir / "standard_metrics_summary.json"))
    mimo = json.load(open(metrics_dir / "mimo_metrics_summary.json"))
    return {
        "Method": "Reference",
        "SI-SDR": std["sisdr_in"]["mean"],
        "SIR": std["sir_in"]["mean"],
        "Delta SIR": 0.0,
        "SAR": std["sar_in"]["mean"],
        "SIR(B)": mean_stems(mimo["mixture_before"]["SIRB"]),
        "Delta SIR(B)": 0.0,
        "EXP (%)": mean_stems(mimo["mixture_before"]["EXP"]),
    }


def main() -> None:
    setup_imports()
    from ablation_frontends import TwoStageTrainer, make_task_loss

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=PROJECT)
    ap.add_argument("--pretrained_root", type=Path, default=PROJECT / "runs_fixed_wavelet_family")
    ap.add_argument("--out_root", type=Path, default=PROJECT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint")
    ap.add_argument("--wavelets", nargs="+", default=["haar", "db2", "sym4", "coif1"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--train_T", type=int, default=32768)
    ap.add_argument("--eval_T", type=int, default=220448)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--clipnorm", type=float, default=0.25)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--loss", choices=["sisdr", "positive_sisdr"], default="sisdr")
    ap.add_argument("--max_train_examples", type=int, default=0)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_root = args.out_root / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    x_train_path = args.project / "finetune_bleed_small_disjoint" / "Xtrain_finetune_disjoint.npy"
    y_train_path = args.project / "finetune_bleed_small_disjoint" / "Ytrain_finetune_disjoint.npy"
    x_test_path = args.project / "Edgecase_active_matched" / "Xedge_m9db.npy"
    y_test_path = args.project / "Edgecase_active_matched" / "Yedge_m9db.npy"

    x_all = np.load(x_train_path).astype(np.float32)[:, :, :args.eval_T]
    y_all = np.load(y_train_path).astype(np.float32)[:, :, :args.eval_T]
    if args.max_train_examples > 0:
        x_all = x_all[:args.max_train_examples]
        y_all = y_all[:args.max_train_examples]
    x_tr, y_tr, x_va, y_va, train_idx, val_idx = split_train_val(x_all, y_all, args.val_ratio, args.seed)

    x_test = np.load(x_test_path).astype(np.float32)[:, :, :args.eval_T]
    y_test = np.load(y_test_path).astype(np.float32)[:, :, :args.eval_T]

    split_cfg = {
        "train_source": str(x_train_path),
        "target_source": str(y_train_path),
        "test_x": str(x_test_path),
        "test_y": str(y_test_path),
        "train_examples": int(len(x_tr)),
        "val_examples": int(len(x_va)),
        "test_examples": int(len(x_test)),
        "train_indices_within_disjoint_pool": train_idx.tolist(),
        "val_indices_within_disjoint_pool": val_idx.tolist(),
        "note": "Training/validation use finetune_bleed_small_disjoint; held-out test uses Edgecase_active_matched m9db arrays.",
    }
    with open(run_root / "split_config.json", "w") as f:
        json.dump(split_cfg, f, indent=2)

    tr_ds = make_dataset(x_tr, y_tr, args.batch, crop_len=args.train_T, aligned_to=4, shuffle=True, seed=args.seed)
    va_ds = make_dataset(x_va, y_va, args.batch, crop_len=None, aligned_to=4, shuffle=False, seed=args.seed)

    rows = []
    reference_added = False
    for wavelet in args.wavelets:
        ckpt = latest_checkpoint(args.pretrained_root, wavelet)
        out_dir = run_root / wavelet
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== Finetune {wavelet} from {ckpt} =====", flush=True)

        base = tf.keras.models.load_model(ckpt, custom_objects=custom_objects(), compile=False)
        trainer = TwoStageTrainer(base_model=base, task_loss_fn=choose_task_loss(args.loss), hf_lambda=0.0, task_lambda=1.0)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clipnorm), jit_compile=False)

        with open(out_dir / "finetune_config.json", "w") as f:
            json.dump({
                "wavelet": wavelet,
                "pretrained_checkpoint": str(ckpt),
                "epochs": args.epochs,
                "batch": args.batch,
                "train_T": args.train_T,
                "eval_T": args.eval_T,
                "lr": args.lr,
                "clipnorm": args.clipnorm,
                "loss": args.loss,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                **split_cfg,
            }, f, indent=2)

        callbacks = [
            SaveBest(base, out_dir / "best.keras"),
            tf.keras.callbacks.CSVLogger(out_dir / "log.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        trainer.fit(tr_ds, validation_data=va_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
        if not (out_dir / "best.keras").exists():
            base.save(out_dir / "best.keras")

        best = tf.keras.models.load_model(out_dir / "best.keras", custom_objects=custom_objects(), compile=False)
        y_pred = np.zeros_like(y_test, dtype=np.float32)
        for start in range(0, len(x_test), args.batch):
            end = min(len(x_test), start + args.batch)
            y_pred[start:end] = best.predict(x_test[start:end], verbose=0).astype(np.float32)[:, :, :args.eval_T]
        pred_path = out_dir / "Ypred_edge_m9db.npy"
        np.save(pred_path, y_pred)

        metrics_dir = out_dir / "metrics_edge_m9db_corrected_exp"
        cmd = [
            str(PYTHON), str(args.project / "Evaluations" / "eval_all_metrics_single.py"),
            "--x_mix", str(x_test_path),
            "--y_true", str(y_test_path),
            "--y_pred", str(pred_path),
            "--out_dir", str(metrics_dir),
            "--sr", "22050",
            "--T", str(args.eval_T),
            "--mimo_K", "512",
            "--mimo_lam", "1e-3",
            "--std_align", "1",
            "--std_max_lag", "22050",
            "--permute_pred", "1",
            "--stem_names", "Vocal", "Bass", "Drums",
        ]
        subprocess.run(cmd, check=True, cwd=str(args.project))
        if not reference_added:
            rows.append(parse_reference(metrics_dir))
            reference_added = True
        rows.append(parse_metrics(metrics_dir, f"Fixed {wavelet} finetuned"))

    csv_path = run_root / "fixed_wavelet_finetune_edge_m9db_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
