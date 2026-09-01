#!/usr/bin/env python3
import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf

import iprdwt as ip


def count_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def split_train_val(x, y, val_ratio=0.15, seed=1337):
    N = len(x)
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(round(N * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def make_ds(x, y, batch_size=2, shuffle=True, crop_len=32768, aligned_to=4, seed=1337):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), reshuffle_each_iteration=True, seed=seed)

    crop_len = int(crop_len)
    aligned_to = int(max(1, aligned_to))

    def _crop_pair(xb, yb):
        T = tf.shape(xb)[-1]
        max_start = tf.maximum(0, T - crop_len)
        start = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
        start = (start // aligned_to) * aligned_to
        xb = xb[:, start:start + crop_len]
        yb = yb[:, start:start + crop_len]
        return xb, yb

    ds = ds.map(_crop_pair, num_parallel_calls=1, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


def make_val_ds(x, y, batch_size=2):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds


def freeze_all_except_dwt(base_model, train_dwt_only=True):
    for layer in base_model.layers:
        if isinstance(layer, (ip.PRDWT1D, ip.PRIDWT1D)):
            layer.trainable = bool(train_dwt_only)
        else:
            layer.trainable = (not train_dwt_only)


def build_base_l2_f11(channels=3):
    base = ip.build_pr_dwt_unet(
        time_length=None,
        channels=channels,
        levels=2,
        filter_length=11,
        pr_shifts=32,
        pr_lambda=1e-1,      # Stage B default; Stage A will rebuild stronger
        pr_dc_lambda=1.0,
        pr_nyq_lambda=1.0,
        unet_depth=4,
        base_filters=64,
        return_taps=False,
    )
    return base


def build_base_l2_f11_with_pr(pr_lambda, channels=3):
    base = ip.build_pr_dwt_unet(
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
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=str, required=True)
    ap.add_argument("--y", type=str, required=True)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--train_T", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)

    # slight finetune
    ap.add_argument("--phaseA_epochs", type=int, default=5,
                    help="PR refresh only")
    ap.add_argument("--phaseB_epochs", type=int, default=30,
                    help="task finetune")
    ap.add_argument("--lrA", type=float, default=2e-5)
    ap.add_argument("--lrB", type=float, default=5e-5)
    ap.add_argument("--clipnorm", type=float, default=0.25)

    # keep PR strong
    ap.add_argument("--pr_lambda_A", type=float, default=1.0)
    ap.add_argument("--pr_lambda_B", type=float, default=1e-1)
    ap.add_argument("--hf_lambda_B", type=float, default=0.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    X = np.load(args.x).astype(np.float32)
    Y = np.load(args.y).astype(np.float32)

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expected (N,C,T). Got X={X.shape}, Y={Y.shape}")
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}")

    N, C, Tfull = X.shape
    print(f"[Loaded] X={X.shape}, Y={Y.shape}")

    x_tr, y_tr, x_va, y_va = split_train_val(X, Y, val_ratio=args.val_ratio, seed=args.seed)
    align = 2 ** 2  # levels=2
    tr_ds = make_ds(x_tr, y_tr, batch_size=args.batch, shuffle=True, crop_len=args.train_T, aligned_to=align, seed=args.seed)
    va_ds = make_val_ds(x_va[:, :, :args.train_T], y_va[:, :, :args.train_T], batch_size=args.batch)

    # -------------------------
    # Stage A: PR refresh only
    # train only DWT/IDWT, task_lambda=0
    # -------------------------
    print("\n===== Stage A: PR refresh only (DWT/IDWT only) =====")
    base_A = build_base_l2_f11_with_pr(pr_lambda=args.pr_lambda_A, channels=C)
    _ = base_A(tf.zeros((1, C, args.train_T), dtype=tf.float32), training=False)
    base_A.load_weights(args.pretrained)

    freeze_all_except_dwt(base_A, train_dwt_only=True)

    trainer_A = ip.TwoStageTrainer(
        base_model=base_A,
        task_loss_fn=ip.make_task_loss(pit=False),
        levels=2,
        hf_lambda=0.0,
        task_lambda=0.0,
        name="ft_stageA_pr_refresh",
    )
    trainer_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lrA, clipnorm=args.clipnorm),
        jit_compile=False
    )

    print(f"[Trainable params / phaseA] {count_trainable_params(base_A)}")

    cbA = [
        ip.SaveBestBase(trainer_A, os.path.join(run_dir, "best_stageA.keras"), monitor="val_loss", mode="min", verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_stageA.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    trainer_A.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.phaseA_epochs,
        callbacks=cbA,
        verbose=1,
    )

    # -------------------------
    # Stage B: full finetune
    # task + strong PR
    # -------------------------
    print("\n===== Stage B: full finetune (task + strong PR) =====")
    base_B = build_base_l2_f11_with_pr(pr_lambda=args.pr_lambda_B, channels=C)
    _ = base_B(tf.zeros((1, C, args.train_T), dtype=tf.float32), training=False)
    base_B.set_weights(base_A.get_weights())

    # unfreeze all
    for layer in base_B.layers:
        layer.trainable = True

    trainer_B = ip.TwoStageTrainer(
        base_model=base_B,
        task_loss_fn=ip.make_task_loss(pit=False),
        levels=2,
        hf_lambda=args.hf_lambda_B,
        task_lambda=1.0,
        name="ft_stageB_task",
    )
    trainer_B.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lrB, clipnorm=args.clipnorm),
        jit_compile=False
    )

    print(f"[Trainable params / phaseB] {count_trainable_params(base_B)}")

    cbB = [
        ip.SaveBestBase(trainer_B, os.path.join(run_dir, "best.keras"), monitor="val_loss", mode="min", verbose=1),
        ip.SaveLastBase(trainer_B, os.path.join(run_dir, "last.keras"), verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_phaseB.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10, min_delta=1e-4,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    trainer_B.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.phaseB_epochs,
        callbacks=cbB,
        verbose=1,
    )

    with open(os.path.join(run_dir, "finetune_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[Done] Best model: {os.path.join(run_dir, 'best.keras')}")


if __name__ == "__main__":
    main()