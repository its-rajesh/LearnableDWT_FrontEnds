#!/usr/bin/env python3
import os
import argparse
import itertools
import numpy as np
import tensorflow as tf

from iprdwt import train_two_stage  # uses your code :contentReference[oaicite:1]{index=1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")
    ap.add_argument("--out_root", type=str, default="./runs_pr_ablation")

    # data/time
    ap.add_argument("--full_t", type=int, default=220448)
    ap.add_argument("--train_t", type=int, default=32768)

    # training
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--epochs_A", type=int, default=10)
    ap.add_argument("--epochs_B", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pit", type=int, default=0)

    # optimizer safety
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    # PR + HF
    ap.add_argument("--pr_lambda_A", type=float, default=1.0)
    ap.add_argument("--pr_lambda_B", type=float, default=1e-1)
    ap.add_argument("--pr_dc_lambda", type=float, default=1.0)
    ap.add_argument("--pr_nyq_lambda", type=float, default=1.0)
    ap.add_argument("--hf_lambda_B", type=float, default=0.05)

    # grid
    ap.add_argument("--filters", type=int, nargs="+", default=[11, 101, 1001])
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3, 5])

    # suggestion knobs
    ap.add_argument("--pr_shifts_mode", type=str, default="auto",
                    choices=["auto", "fixed"],
                    help="auto: pr_shifts scales with filter length; fixed: use --pr_shifts_fixed")
    ap.add_argument("--pr_shifts_fixed", type=int, default=32)

    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    print("Loading train files...")
    X = np.load(os.path.join(args.dpath, "Xtrain.npy")).astype(np.float32)
    Y = np.load(os.path.join(args.dpath, "Ytrain.npy")).astype(np.float32)
    X = X[:, :, :args.full_t]
    Y = Y[:, :, :args.full_t]
    C = X.shape[1]
    print("X:", X.shape, "Y:", Y.shape)

    # sanity: train_t must allow alignment by 2**levels (your code crops aligned) :contentReference[oaicite:2]{index=2}
    # We won't force it here, but we'll warn.
    for L in args.levels:
        align = 2 ** int(L)
        if args.train_t % align != 0:
            print(f"[WARN] train_t={args.train_t} not divisible by 2**levels={align} (levels={L}). "
                  f"Your cropping aligns start indices, but segment length divisibility is still recommended.")

    combos = list(itertools.product(args.levels, args.filters))
    print("\nAblation grid (levels, filter_length):")
    for lv, fl in combos:
        print(f"  L={lv}, F={fl}")

    for lv, fl in combos:
        if fl % 2 == 0:
            raise ValueError(f"filter_length must be odd. Got {fl}")

        tag = f"L{lv}_F{fl}"
        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        # --- Suggested PR shifts scaling ---
        # PR autocorr constraints become harder as L grows; scaling helps, but don’t make it huge.
        if args.pr_shifts_mode == "fixed":
            pr_shifts = int(args.pr_shifts_fixed)
        else:
            # auto: cap to avoid massive loop cost for F=1001
            # rule of thumb: about ~L/32, capped between [12, 64]
            pr_shifts = int(np.clip(fl // 32, 12, 64))

        print("\n" + "=" * 80)
        print(f"RUN: {tag}  (pr_shifts={pr_shifts})  -> {out_dir}")
        print("=" * 80)

        # For huge filters/levels, memory can spike; a safe heuristic is to reduce batch.
        # You can override by passing --batch.
        batch = args.batch
        if fl >= 1001 or lv >= 5:
            batch = min(batch, 1)

        trainer, history, run_dir = train_two_stage(
            X, Y,
            T=args.train_t, C=C,
            out_dir=out_dir,
            val_ratio=args.val_ratio,
            epochs_A=args.epochs_A,
            epochs_B=args.epochs_B,
            batch_size=batch,
            pit=bool(args.pit),
            lr=args.lr,
            clipnorm=args.clipnorm,
            early_patience=args.early_patience,
            min_delta=args.min_delta,

            # grid params
            levels=lv,
            filter_length=fl,

            # PR settings
            pr_shifts=pr_shifts,
            pr_lambda_A=args.pr_lambda_A,
            pr_lambda_B=args.pr_lambda_B,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,

            # HF penalty
            hf_lambda_B=args.hf_lambda_B,
        )

        print(f"[DONE] {tag} saved in {run_dir}")

    print("\nALL DONE. Root:", args.out_root)


if __name__ == "__main__":
    # keep TF deterministic-ish like your main script does (optional)
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    except Exception:
        pass
    main()