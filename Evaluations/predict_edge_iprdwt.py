#!/usr/bin/env python3
"""
predict_edge_iprdwt.py

Predict Ypred for edge-case datasets:
  Xedge_0db.npy, Xedge_3db.npy, Xedge_6db.npy, Xedge_9db.npy, Xedge_12db.npy

Outputs:
  Ypred_edge_0db.npy, ..., Ypred_edge_12db.npy

Uses the same robust loading logic as test_iprdwt.py, including fallback
for TwoStageTrainer-style .keras archives.
"""

import os
import json
import argparse
import tempfile
import zipfile
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
)


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)


def build_base_model(T, C, levels, filt_len, pr_shifts,
                     pr_lambda, pr_dc_lambda, pr_nyq_lambda,
                     unet_depth, base_filters):
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
    """
    Try normal tf.keras.models.load_model().
    If it fails, restore weights from inside .keras archive:
      - model.weights.h5
      - variables/variables.* fallback
    Handles weights saved under "base/..." using a wrapper with attribute `.base`.
    """
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
            f"Could not restore weights from {best_model_path}. "
            f"Expected model.weights.h5 or variables/variables.index"
        )

    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def predict_one_file(model, x_path, ypred_path, T, batch_size):
    print(f"\n[Predict] Loading {x_path}")
    X = np.load(x_path).astype(np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,C,T), got {X.shape} for {x_path}")

    X = X[:, :, :T]
    N, C, TT = X.shape
    print(f"[Predict] X shape after crop: {X.shape}")

    Ypred = np.zeros_like(X, dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        print(f"[Predict] {os.path.basename(x_path)} : {s}:{e}/{N}", flush=True)
        yp = model.predict(X[s:e], verbose=0)
        Ypred[s:e] = np.asarray(yp, dtype=np.float32)

        if not np.all(np.isfinite(Ypred[s:e])):
            raise ValueError(f"Non-finite prediction detected in batch {s}:{e} for {x_path}")

    np.save(ypred_path, Ypred.astype(np.float32))
    print(f"[Saved] {ypred_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edge_dir", type=str, required=True,
                   help="Directory containing Xedge_*db.npy and Yedge_*db.npy")
    p.add_argument("--best_model", type=str, required=True,
                   help="Path to trained best.keras")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to save Ypred_edge_*db.npy. Default: <edge_dir>/predictions")
    p.add_argument("--db_levels", type=int, nargs="+", default=[0, 3, 6, 9, 12])

    p.add_argument("--T", type=int, default=220448)
    p.add_argument("--batch", type=int, default=2)

    # Model hyperparams: must match training
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

    edge_dir = os.path.abspath(args.edge_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(edge_dir, "predictions")
    ensure_dir(out_dir)

    # Use first X file to infer channel count
    first_x = os.path.join(edge_dir, f"Xedge_{args.db_levels[0]}db.npy")
    if not os.path.exists(first_x):
        raise FileNotFoundError(f"Missing first edge file: {first_x}")

    X0 = np.load(first_x).astype(np.float32)
    if X0.ndim != 3:
        raise ValueError(f"Expected shape (N,C,T), got {X0.shape}")
    C = X0.shape[1]

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    def _base_builder():
        return build_base_model(
            T=args.T,
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
    model = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=_base_builder,
        C=C,
        T=args.T,
        levels=args.levels,
    )

    run_info = {
        "best_model": os.path.abspath(args.best_model),
        "edge_dir": edge_dir,
        "out_dir": out_dir,
        "db_levels": args.db_levels,
        "T": args.T,
        "batch": args.batch,
        "model_hparams": {
            "levels": args.levels,
            "filter_length": args.filter_length,
            "pr_shifts": args.pr_shifts,
            "pr_lambda": args.pr_lambda,
            "pr_dc_lambda": args.pr_dc_lambda,
            "pr_nyq_lambda": args.pr_nyq_lambda,
            "unet_depth": args.unet_depth,
            "base_filters": args.base_filters,
        }
    }
    with open(os.path.join(out_dir, "predict_edge_config.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    for db in args.db_levels:
        x_path = os.path.join(edge_dir, f"Xedge_{db}db.npy")
        y_path = os.path.join(edge_dir, f"Yedge_{db}db.npy")
        ypred_path = os.path.join(out_dir, f"Ypred_edge_{db}db.npy")

        if not os.path.exists(x_path):
            print(f"[SKIP] Missing {x_path}")
            continue
        if not os.path.exists(y_path):
            print(f"[WARN] Missing matching target {y_path} (prediction still possible)")

        if os.path.exists(ypred_path):
            print(f"[SKIP] Already exists: {ypred_path}")
            continue

        predict_one_file(model, x_path, ypred_path, T=args.T, batch_size=args.batch)

    print("\n[DONE]")
    print("Predictions saved in:", out_dir)


if __name__ == "__main__":
    main()