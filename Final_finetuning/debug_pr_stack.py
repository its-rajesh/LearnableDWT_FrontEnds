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
    PRDWT1D, PRIDWT1D, MatchTimeLen, SplitChannels, UpsampleTo,
    build_pr_dwt_unet
)

EPS = 1e-12


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
        return_taps=False
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
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(ckpt_prefix).expect_partial()
            relink_pridwt_to_prdwt(base_model, levels)
            print("[Restore] OK (TF checkpoint).")
            return base_model

        raise FileNotFoundError(f"Could not restore weights from {best_model_path}")
    finally:
        if extracted_dir == tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def pr_even_shift_residuals(h0, max_m=20):
    h0 = np.asarray(h0).astype(np.float64)
    L = len(h0)
    ms = np.arange(0, max_m + 1)
    r2m = []
    for m in ms:
        k = 2 * m
        if k == 0:
            r = np.sum(h0 * h0)
        elif k < L:
            r = np.sum(h0[k:] * h0[:-k])
        else:
            r = 0.0
        r2m.append(r)
    return ms, np.array(r2m, dtype=np.float64)


def dc_nyq_stats(h0):
    h0 = np.asarray(h0, dtype=np.float64)
    dc = np.sum(h0)
    alt = np.ones_like(h0)
    alt[1::2] = -1.0
    nyq = np.sum(alt * h0)
    return dc, nyq


def build_pr_analysis_synthesis_model(base_model, T, C, levels, filter_length,
                                      pr_shifts, pr_lambda, pr_dc_lambda, pr_nyq_lambda,
                                      unet_depth, base_filters):
    inp = tf.keras.Input(shape=(C, T), name="x_in")
    x = tf.keras.layers.Permute((2, 1))(inp)  # [B,T,C]

    dwt_layers = []
    for i in range(levels):
        dwt = PRDWT1D(
            filter_length=filter_length,
            pr_shifts=pr_shifts,
            pr_lambda=pr_lambda,
            pr_dc_lambda=pr_dc_lambda,
            pr_nyq_lambda=pr_nyq_lambda,
            name=f"prdwt_{i}",
        )
        dwt_layers.append(dwt)

    approx = x
    details = []
    for i in range(levels):
        a, d = dwt_layers[i](approx)
        details.append(d)
        approx = a

    idwt_layers = []
    for i in range(levels):
        idwt = PRIDWT1D(name=f"pridwt_{i}").set_dwt(dwt_layers[i])
        idwt_layers.append(idwt)

    recon = approx
    for i in reversed(range(levels)):
        recon = idwt_layers[i]([recon, details[i]])

    recon = MatchTimeLen()([recon, x])
    y = tf.keras.layers.Permute((2, 1))(recon)

    pr_model = tf.keras.Model(inp, y, name="PR_AnalysisSynthesis")
    _ = pr_model(tf.zeros((1, C, T), dtype=tf.float32), training=False)

    for i in range(levels):
        pr_model.get_layer(f"prdwt_{i}").set_weights(base_model.get_layer(f"prdwt_{i}").get_weights())
    relink_pridwt_to_prdwt(pr_model, levels)
    return pr_model


def nrmse(x, xhat):
    return np.sqrt(np.mean((x - xhat) ** 2)) / (np.sqrt(np.mean(x ** 2)) + EPS)


def pr_snr_db(x, xhat):
    return 10.0 * np.log10((np.sum(x ** 2) + EPS) / (np.sum((x - xhat) ** 2) + EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_model", type=str, required=True)
    ap.add_argument("--T", type=int, default=32768)
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=11)
    ap.add_argument("--pr_shifts", type=int, default=32)
    ap.add_argument("--pr_lambda", type=float, default=1e-1)
    ap.add_argument("--pr_dc_lambda", type=float, default=1.0)
    ap.add_argument("--pr_nyq_lambda", type=float, default=1.0)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--base_filters", type=int, default=64)
    args = ap.parse_args()

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    base = load_base_model_from_any(
        best_model_path=args.best_model,
        custom_objs=custom_objs,
        base_builder=lambda: build_base_model(
            args.T, args.C, args.levels, args.filter_length,
            args.pr_shifts, args.pr_lambda, args.pr_dc_lambda, args.pr_nyq_lambda,
            args.unet_depth, args.base_filters
        ),
        C=args.C, T=args.T, levels=args.levels,
    )

    print("\n===== FILTER CHECK =====")
    for i in range(args.levels):
        h0 = base.get_layer(f"prdwt_{i}").get_weights()[0].astype(np.float64)
        h0 = h0 / (np.linalg.norm(h0) + EPS)
        ms, r2m = pr_even_shift_residuals(h0, max_m=args.pr_shifts)
        dc, nyq = dc_nyq_stats(h0)
        print(
            f"level={i} | r0={r2m[0]:.6f} | max|r2m|(m>=1)={np.max(np.abs(r2m[1:])):.6e} "
            f"| dc={dc:.6f} | dc_err={dc - np.sqrt(2.0):.6e} | nyq={nyq:.6e}"
        )

    pr_model = build_pr_analysis_synthesis_model(
        base_model=base,
        T=args.T,
        C=args.C,
        levels=args.levels,
        filter_length=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
        unet_depth=args.unet_depth,
        base_filters=args.base_filters,
    )

    print("\n===== RANDOM INPUT PR TEST =====")
    x = np.random.randn(4, args.C, args.T).astype(np.float32)
    xh = pr_model.predict(x, verbose=0)
    for b in range(x.shape[0]):
        print(f"sample {b}: PR-SNR={pr_snr_db(x[b], xh[b]):.3f} dB | NRMSE={nrmse(x[b], xh[b]):.6f}")


if __name__ == "__main__":
    main()