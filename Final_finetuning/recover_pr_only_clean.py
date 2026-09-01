#!/usr/bin/env python3
import os
import json
import time
import math
import argparse
import tempfile
import zipfile
import shutil

import numpy as np
import tensorflow as tf
import iprdwt as ip

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================

def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, ip.PRIDWT1D):
            idwt.set_dwt(dwt)


def build_base_model(T, C, levels=2, filt_len=11,
                     pr_shifts=32, pr_lambda=50.0,
                     pr_dc_lambda=1.0, pr_nyq_lambda=1.0,
                     unet_depth=4, base_filters=64):
    base = ip.build_pr_dwt_unet(
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


def load_base_model_from_any(best_model_path, base_builder, C, T, levels):
    custom_objs = {
        "MatchTimeLen": ip.MatchTimeLen,
        "SplitChannels": ip.SplitChannels,
        "UpsampleTo": ip.UpsampleTo,
        "PRDWT1D": ip.PRDWT1D,
        "PRIDWT1D": ip.PRIDWT1D,
    }

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


def init_haar_like_filter(F):
    """
    Odd-length centered Haar-like init.
    """
    h = np.zeros((F,), dtype=np.float32)
    mid = F // 2
    h[mid] = 1.0 / np.sqrt(2.0)
    if mid + 1 < F:
        h[mid + 1] = 1.0 / np.sqrt(2.0)
    return h


def maybe_reinit_pr_filters(base_model, levels=2, mode="none"):
    """
    mode:
      none          -> keep current weights
      level0_haar   -> reset only level-0 lowpass to Haar-like
      all_haar      -> reset all levels to Haar-like
    """
    if mode == "none":
        return

    for i in range(levels):
        if mode == "level0_haar" and i != 0:
            continue

        layer = base_model.get_layer(f"prdwt_{i}")
        w = layer.get_weights()
        if not w:
            continue
        h0 = w[0].copy()
        F = len(h0)
        w[0] = init_haar_like_filter(F).astype(np.float32)
        layer.set_weights(w)
        print(f"[Reinit] prdwt_{i} lowpass reset to Haar-like init")


def freeze_unet_train_only_dwt(base_model):
    """
    Freeze everything except PRDWT layers.
    PRIDWT has no own weights; it references PRDWT.
    """
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, ip.PRDWT1D):
            layer.trainable = True

    trainable = base_model.trainable_variables
    print(f"[Trainable vars count] {len(trainable)}")
    for v in trainable:
        print("  ", v.name, v.shape)

    assert len(trainable) > 0, "No trainable variables found."


def pr_even_shift_residuals(h0, max_m=20):
    h0 = np.asarray(h0).astype(np.float64)
    h0 = h0 / (np.linalg.norm(h0) + EPS)
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
    h0 = h0 / (np.linalg.norm(h0) + EPS)
    dc = np.sum(h0)
    alt = np.ones_like(h0)
    alt[1::2] = -1.0
    nyq = np.sum(alt * h0)
    return dc, nyq


def build_pr_analysis_synthesis_model(base_model, T, C,
                                      levels=2, filt_len=11,
                                      pr_shifts=32, pr_lambda=50.0,
                                      pr_dc_lambda=1.0, pr_nyq_lambda=1.0):
    inp = tf.keras.Input(shape=(C, T), name="x_in")
    x = tf.keras.layers.Permute((2, 1))(inp)  # [B,T,C]

    dwt_layers = []
    for i in range(levels):
        dwt = ip.PRDWT1D(
            filter_length=filt_len,
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
        idwt = ip.PRIDWT1D(name=f"pridwt_{i}").set_dwt(dwt_layers[i])
        idwt_layers.append(idwt)

    recon = approx
    for i in reversed(range(levels)):
        recon = idwt_layers[i]([recon, details[i]])

    recon = ip.MatchTimeLen()([recon, x])
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


def print_pr_debug(base_model, T, C, levels, filt_len,
                   pr_shifts, pr_lambda, pr_dc_lambda, pr_nyq_lambda):
    print("\n===== FILTER CHECK =====")
    for i in range(levels):
        h0 = base_model.get_layer(f"prdwt_{i}").get_weights()[0].astype(np.float64)
        ms, r2m = pr_even_shift_residuals(h0, max_m=pr_shifts)
        dc, nyq = dc_nyq_stats(h0)
        print(
            f"level={i} | r0={r2m[0]:.6f} | "
            f"max|r2m|(m>=1)={np.max(np.abs(r2m[1:])):.6e} | "
            f"dc={dc:.6f} | dc_err={dc - np.sqrt(2.0):.6e} | "
            f"nyq={nyq:.6e}"
        )

    pr_model = build_pr_analysis_synthesis_model(
        base_model=base_model,
        T=T, C=C,
        levels=levels,
        filt_len=filt_len,
        pr_shifts=pr_shifts,
        pr_lambda=pr_lambda,
        pr_dc_lambda=pr_dc_lambda,
        pr_nyq_lambda=pr_nyq_lambda,
    )

    print("\n===== RANDOM INPUT PR TEST =====")
    x = np.random.randn(4, C, T).astype(np.float32)
    xh = pr_model.predict(x, verbose=0)
    vals = []
    for b in range(x.shape[0]):
        snr = pr_snr_db(x[b], xh[b])
        err = nrmse(x[b], xh[b])
        vals.append((float(snr), float(err)))
        print(f"sample {b}: PR-SNR={snr:.3f} dB | NRMSE={err:.6f}")
    return vals


def split_train_val(X, val_ratio=0.15, seed=1337):
    N = len(X)
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(round(N * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], X[val_idx]


def make_pr_ds(X, batch_size=4, crop_len=32768, aligned_to=4, shuffle=True, seed=1337):
    ds = tf.data.Dataset.from_tensor_slices(X)
    if shuffle:
        ds = ds.shuffle(min(len(X), 4096), reshuffle_each_iteration=True, seed=seed)

    def _crop(xb):
        T = tf.shape(xb)[-1]
        max_start = tf.maximum(0, T - crop_len)
        start = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
        start = (start // aligned_to) * aligned_to
        xb = xb[:, start:start + crop_len]
        return xb

    ds = ds.map(_crop, num_parallel_calls=1, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


# ============================================================
# Custom PR-only trainer
# ============================================================

class PROnlyTrainer(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__(name="PR_Only_Trainer")
        self.base = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pr_loss_tracker = tf.keras.metrics.Mean(name="pr_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.pr_loss_tracker]

    def train_step(self, batch):
        x = batch  # only x is needed

        with tf.GradientTape() as tape:
            _ = self.base(x, training=True)
            pr_loss = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, tf.float32)
            loss = pr_loss

        vars_ = self.base.trainable_variables
        grads = tape.gradient(loss, vars_)
        grads_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]

        if not grads_vars:
            raise ValueError("No gradients found for trainable variables.")

        self.optimizer.apply_gradients(grads_vars)

        self.loss_tracker.update_state(loss)
        self.pr_loss_tracker.update_state(pr_loss)
        return {"loss": self.loss_tracker.result(), "pr_loss": self.pr_loss_tracker.result()}

    def test_step(self, batch):
        x = batch
        _ = self.base(x, training=False)
        pr_loss = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, tf.float32)
        loss = pr_loss

        self.loss_tracker.update_state(loss)
        self.pr_loss_tracker.update_state(pr_loss)
        return {"loss": self.loss_tracker.result(), "pr_loss": self.pr_loss_tracker.result()}


class SaveBestBase(tf.keras.callbacks.Callback):
    def __init__(self, trainer, path, monitor="val_loss", mode="min"):
        super().__init__()
        self.trainer = trainer
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        improved = (val < self.best) if self.mode == "min" else (val > self.best)
        if improved:
            self.best = val
            self.trainer.base.save(self.path)
            print(f"\n[SaveBestBase] epoch={epoch+1} {self.monitor}={val:.6f} -> {self.path}")


class SaveLastBase(tf.keras.callbacks.Callback):
    def __init__(self, trainer, path):
        super().__init__()
        self.trainer = trainer
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.base.save(self.path)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=str, required=True,
                    help="Waveform array (N,C,T), e.g. Xtrain.npy")
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--train_T", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=11)
    ap.add_argument("--pr_shifts", type=int, default=32)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clipnorm", type=float, default=0.25)

    ap.add_argument("--pr_lambda", type=float, default=50.0)
    ap.add_argument("--pr_dc_lambda", type=float, default=1.0)
    ap.add_argument("--pr_nyq_lambda", type=float, default=1.0)

    ap.add_argument("--reinit_mode", type=str, default="level0_haar",
                    choices=["none", "level0_haar", "all_haar"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    X = np.load(args.x).astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,C,T), got {X.shape}")
    N, C, _ = X.shape
    print(f"[Loaded] X={X.shape}")

    x_tr, x_va = split_train_val(X, val_ratio=args.val_ratio, seed=args.seed)
    tr_ds = make_pr_ds(x_tr, batch_size=args.batch, crop_len=args.train_T, aligned_to=2**args.levels,
                       shuffle=True, seed=args.seed)
    va_ds = make_pr_ds(x_va, batch_size=args.batch, crop_len=args.train_T, aligned_to=2**args.levels,
                       shuffle=False, seed=args.seed)

    base_loaded = load_base_model_from_any(
        best_model_path=args.pretrained,
        base_builder=lambda: build_base_model(
            T=args.train_T, C=C,
            levels=args.levels,
            filt_len=args.filter_length,
            pr_shifts=args.pr_shifts,
            pr_lambda=args.pr_lambda,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,
        ),
        C=C, T=args.train_T, levels=args.levels,
    )

    # rebuild same architecture but with strong PR weights
    base = build_base_model(
        T=args.train_T, C=C,
        levels=args.levels,
        filt_len=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
    )
    base.set_weights(base_loaded.get_weights())
    relink_pridwt_to_prdwt(base, args.levels)

    maybe_reinit_pr_filters(base, levels=args.levels, mode=args.reinit_mode)
    freeze_unet_train_only_dwt(base)

    print_pr_debug(
        base_model=base,
        T=args.train_T, C=C,
        levels=args.levels,
        filt_len=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
    )

    trainer = PROnlyTrainer(base)
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clipnorm),
        jit_compile=False,
    )

    callbacks = [
        SaveBestBase(trainer, os.path.join(run_dir, "best_pr.keras"), monitor="val_loss", mode="min"),
        SaveLastBase(trainer, os.path.join(run_dir, "last_pr.keras")),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_pr.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    trainer.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_path = os.path.join(run_dir, "best_pr.keras")
    best = load_base_model_from_any(
        best_model_path=best_path,
        base_builder=lambda: build_base_model(
            T=args.train_T, C=C,
            levels=args.levels,
            filt_len=args.filter_length,
            pr_shifts=args.pr_shifts,
            pr_lambda=args.pr_lambda,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,
        ),
        C=C, T=args.train_T, levels=args.levels,
    )

    final_vals = print_pr_debug(
        base_model=best,
        T=args.train_T, C=C,
        levels=args.levels,
        filt_len=args.filter_length,
        pr_shifts=args.pr_shifts,
        pr_lambda=args.pr_lambda,
        pr_dc_lambda=args.pr_dc_lambda,
        pr_nyq_lambda=args.pr_nyq_lambda,
    )

    with open(os.path.join(run_dir, "recover_pr_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(os.path.join(run_dir, "final_random_pr_debug.json"), "w") as f:
        json.dump(
            [{"pr_snr_db": s, "nrmse": e} for (s, e) in final_vals],
            f, indent=2
        )

    print(f"\n[Done] Best PR model: {best_path}")
    print("If PR is still bad after this, stop learning the wavelet filters and freeze a known PR bank.")


if __name__ == "__main__":
    main()