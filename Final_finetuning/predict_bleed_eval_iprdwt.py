#!/usr/bin/env python3
import os
import json
import argparse
import tempfile
import zipfile
import shutil

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


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)


def build_base_model(T, C):
    base = build_pr_dwt_unet(
        time_length=T,
        channels=C,
        levels=2,
        filter_length=11,
        pr_shifts=32,
        pr_lambda=1e-1,
        pr_dc_lambda=1.0,
        pr_nyq_lambda=1.0,
        unet_depth=4,
        base_filters=64,
        return_taps=False,
    )
    _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(base, 2)
    return base


def load_base_model_from_any(best_model_path, custom_objs, base_builder, C, T):
    try:
        loaded = tf.keras.models.load_model(best_model_path, custom_objects=custom_objs, compile=False)
        base = getattr(loaded, "base", None)
        if base is None:
            base = loaded
        _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
        relink_pridwt_to_prdwt(base, 2)
        print("[Load] load_model() success.")
        return base
    except Exception as e:
        print("[WARN] load_model() failed; using restore fallback.")
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
            _ = base_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)
            relink_pridwt_to_prdwt(base_model, 2)

            try:
                base_model.load_weights(h5_path)
                relink_pridwt_to_prdwt(base_model, 2)
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
            relink_pridwt_to_prdwt(base_model, 2)
            print("[Restore] OK (H5 wrapper).")
            return base_model

        ckpt_prefix = os.path.join(extracted_dir, "variables", "variables")
        if os.path.exists(ckpt_prefix + ".index"):
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, 2)
            print("[Restore] OK (TF checkpoint).")
            return base_model

        raise FileNotFoundError(f"Could not restore weights from {best_model_path}")

    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def tag_from_db(db):
    return f"m{abs(int(db))}db" if db < 0 else f"{int(db)}db"


def predict_one_file(model, x_path, ypred_path, T, batch_size):
    print(f"\n[Predict] {x_path}")
    X = np.load(x_path).astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,C,T), got {X.shape}")

    X = X[:, :, :T]
    N, C, _ = X.shape
    Ypred = np.zeros((N, C, T), dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        print(f"  batch {s}:{e}/{N}", flush=True)
        yp = model.predict(X[s:e], verbose=0)
        yp = np.asarray(yp, dtype=np.float32)

        if yp.shape != (e - s, C, T):
            raise ValueError(f"Unexpected prediction shape {yp.shape}")
        if not np.all(np.isfinite(yp)):
            raise ValueError(f"Non-finite output in batch {s}:{e}")

        Ypred[s:e] = yp

    np.save(ypred_path, Ypred.astype(np.float32))
    print(f"[Saved] {ypred_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_dir", type=str, required=True)
    ap.add_argument("--best_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bleed_levels", type=int, nargs="+",
                    default=[-40, -20, -18, -16, -14, -12, -9, -6, -3, 0])
    ap.add_argument("--T", type=int, default=220448)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    first_tag = tag_from_db(args.bleed_levels[0])
    X0 = np.load(os.path.join(args.edge_dir, f"Xedge_{first_tag}.npy")).astype(np.float32)
    C = X0.shape[1]

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    model = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=lambda: build_base_model(args.T, C),
        C=C,
        T=args.T,
    )

    with open(os.path.join(args.out_dir, "predict_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for db in args.bleed_levels:
        tag = tag_from_db(db)
        x_path = os.path.join(args.edge_dir, f"Xedge_{tag}.npy")
        ypred_path = os.path.join(args.out_dir, f"Ypred_edge_{tag}.npy")
        if not os.path.exists(x_path):
            print(f"[SKIP] missing {x_path}")
            continue
        predict_one_file(model, x_path, ypred_path, args.T, args.batch)

    print("\nDone.")


if __name__ == "__main__":
    main()