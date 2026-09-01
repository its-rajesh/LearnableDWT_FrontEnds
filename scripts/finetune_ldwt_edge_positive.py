#!/usr/bin/env python3
"""Controlled LDWT L2/F11 finetune for TASLP revision experiments."""

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
sys.path.insert(0, str(PROJECT))
import iprdwt as ip  # noqa: E402


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
    if kind == "sisdr":
        return ip.make_task_loss(pit=False)
    if kind == "positive_sisdr":
        return positive_sisdr_loss_no_pit
    raise ValueError(kind)


def split_train_val(x, y, val_ratio, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = int(np.round(len(x) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], train_idx, val_idx


def make_ds(x, y, batch_size, crop_len=None, aligned_to=4, shuffle=True, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), seed=seed, reshuffle_each_iteration=True)
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


def build_base(pr_lambda, channels=3):
    return ip.build_pr_dwt_unet(
        time_length=None,
        channels=channels,
        levels=2,
        filter_length=11,
        pr_shifts=32,
        pr_lambda=float(pr_lambda),
        pr_dc_lambda=1.0,
        pr_nyq_lambda=1.0,
        unet_depth=4,
        base_filters=64,
        return_taps=False,
    )


def freeze_all_except_dwt(base_model):
    for layer in base_model.layers:
        if isinstance(layer, (ip.PRDWT1D, ip.PRIDWT1D)):
            layer.trainable = True
        else:
            layer.trainable = False


def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=PROJECT)
    ap.add_argument("--pretrained", type=Path, default=PROJECT / "runs_pr_ablation" / "L2_F11" / "20260226_183007" / "best.keras")
    ap.add_argument("--out_root", type=Path, default=PROJECT / "revision_experiments" / "multiseed_ldwt_l2f11")
    ap.add_argument("--phaseA_epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10, help="Stage-B task epochs")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--train_T", type=int, default=32768)
    ap.add_argument("--eval_T", type=int, default=220448)
    ap.add_argument("--lrA", type=float, default=2e-5)
    ap.add_argument("--lrB", type=float, default=5e-5)
    ap.add_argument("--clipnorm", type=float, default=0.25)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["sisdr", "positive_sisdr"], default="positive_sisdr")
    ap.add_argument("--max_train_examples", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    run_root = args.out_root / time.strftime("%Y%m%d_%H%M%S") / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    x_train_path = args.project / "finetune_bleed_small_disjoint" / "Xtrain_finetune_disjoint.npy"
    y_train_path = args.project / "finetune_bleed_small_disjoint" / "Ytrain_finetune_disjoint.npy"
    x_test_path = args.project / "Edgecase_active_matched" / "Xedge_m9db.npy"
    y_test_path = args.project / "Edgecase_active_matched" / "Yedge_m9db.npy"

    X = np.load(x_train_path).astype(np.float32)[:, :, :args.eval_T]
    Y = np.load(y_train_path).astype(np.float32)[:, :, :args.eval_T]
    if args.max_train_examples > 0:
        X = X[:args.max_train_examples]
        Y = Y[:args.max_train_examples]
    x_tr, y_tr, x_va, y_va, train_idx, val_idx = split_train_val(X, Y, args.val_ratio, args.seed)
    tr_ds = make_ds(x_tr, y_tr, args.batch, crop_len=args.train_T, aligned_to=4, shuffle=True, seed=args.seed)
    va_ds = make_ds(x_va[:, :, :args.train_T], y_va[:, :, :args.train_T], args.batch, crop_len=None, aligned_to=4, shuffle=False, seed=args.seed)

    base_A = build_base(pr_lambda=1.0, channels=X.shape[1])
    _ = base_A(tf.zeros((1, X.shape[1], args.train_T), dtype=tf.float32), training=False)
    base_A.load_weights(args.pretrained)
    freeze_all_except_dwt(base_A)
    trainer_A = ip.TwoStageTrainer(base_A, task_loss_fn=choose_task_loss(args.loss), levels=2, hf_lambda=0.0, task_lambda=0.0)
    trainer_A.compile(tf.keras.optimizers.Adam(args.lrA, clipnorm=args.clipnorm), jit_compile=False)
    print(f"[Stage A trainable params] {count_params(base_A)}")
    trainer_A.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.phaseA_epochs,
        callbacks=[
            ip.SaveBestBase(trainer_A, str(run_root / "best_stageA.keras"), monitor="val_loss", mode="min", verbose=1),
            tf.keras.callbacks.CSVLogger(run_root / "log_stageA.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=1,
    )

    base_B = build_base(pr_lambda=0.1, channels=X.shape[1])
    _ = base_B(tf.zeros((1, X.shape[1], args.train_T), dtype=tf.float32), training=False)
    base_B.set_weights(base_A.get_weights())
    for layer in base_B.layers:
        layer.trainable = True
    trainer_B = ip.TwoStageTrainer(base_B, task_loss_fn=choose_task_loss(args.loss), levels=2, hf_lambda=0.0, task_lambda=1.0)
    trainer_B.compile(tf.keras.optimizers.Adam(args.lrB, clipnorm=args.clipnorm), jit_compile=False)
    print(f"[Stage B trainable params] {count_params(base_B)}")
    trainer_B.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.epochs,
        callbacks=[
            ip.SaveBestBase(trainer_B, str(run_root / "best.keras"), monitor="val_loss", mode="min", verbose=1),
            ip.SaveLastBase(trainer_B, str(run_root / "last.keras"), verbose=0),
            tf.keras.callbacks.CSVLogger(run_root / "log_phaseB.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=1,
    )

    x_test = np.load(x_test_path).astype(np.float32)[:, :, :args.eval_T]
    y_test = np.load(y_test_path).astype(np.float32)[:, :, :args.eval_T]
    x_eval_path = run_root / "Xedge_m9db_T220448.npy"
    y_eval_path = run_root / "Yedge_m9db_T220448.npy"
    np.save(x_eval_path, x_test)
    np.save(y_eval_path, y_test)
    ypred = np.zeros((len(x_test), x_test.shape[1], args.eval_T), dtype=np.float32)
    for start in range(0, len(x_test), args.batch):
        end = min(len(x_test), start + args.batch)
        yp = base_B.predict(x_test[start:end], verbose=0)
        ypred[start:end, :, : yp.shape[-1]] = yp[:, :, :args.eval_T]
    pred_path = run_root / "Ypred_edge_m9db.npy"
    np.save(pred_path, ypred)

    metrics_dir = run_root / "metrics_edge_m9db_corrected_exp"
    cmd = [
        str(PYTHON), str(args.project / "Evaluations" / "eval_all_metrics_single.py"),
        "--x_mix", str(x_eval_path), "--y_true", str(y_eval_path), "--y_pred", str(pred_path),
        "--out_dir", str(metrics_dir), "--sr", "22050", "--mimo_K", "512", "--mimo_lam", "0.001",
        "--std_align", "1", "--std_max_lag", "22050", "--permute_pred", "1",
        "--stem_names", "Vocal", "Bass", "Drums",
    ]
    subprocess.run(cmd, check=True, cwd=str(args.project / "Evaluations"))

    cfg = vars(args).copy()
    cfg.update({
        "train_source": str(x_train_path),
        "target_source": str(y_train_path),
        "test_x": str(x_test_path),
        "test_y": str(y_test_path),
        "train_indices_within_disjoint_pool": train_idx.tolist(),
        "val_indices_within_disjoint_pool": val_idx.tolist(),
        "prediction_path": str(pred_path),
        "metrics_dir": str(metrics_dir),
    })
    with open(run_root / "finetune_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    row = parse_metrics(metrics_dir, "LDWT")
    with open(run_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[Done] {run_root}")


if __name__ == "__main__":
    main()
