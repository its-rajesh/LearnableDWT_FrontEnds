# =========================
# PR-DWT-U-Net (Keras 3 safe) + Two-Stage Training (PR warmup -> Task)
# Key changes vs your old file:
#   (1) FIX HF LOSS: do NOT avgpool detail bands to align; instead UPSAMPLE all bands to lowest-res and concat
#   (2) Stronger PR regularizer (even-shift + DC/Nyquist + energy)
#   (3) Optional wavelet-domain HF-preservation penalty (NO STFT anywhere)
#   (4) Two-stage training:
#         Stage A: PR-only warmup (task loss weight = 0)
#         Stage B: task training with small PR and optional HF penalty
#   (5) PiT remains optional (default False)
#
# FIX FOR YOUR CRASH:
#   Keras 3 cannot serialize subclassed TwoStageTrainer (non-serializable args).
#   So we checkpoint trainer.base (Functional model) via SaveBestBase/SaveLastBase
#   instead of ModelCheckpoint on the trainer.
# =========================


# =========================
# PR-DWT-U-Net (Keras 3 safe) + Two-Stage Training (PR warmup -> Task)
#
# Architecture / core logic: UNCHANGED.
# Fixes in this version (implementation-only, NOT architecture changes):
#   (A) Replace conv1d_transpose in PRIDWT1D with exact zero-insertion upsample + conv1d (same math, safer kernels).
#   (B) Add optional random aligned cropping in tf.data pipeline (same model, just trains on segments to avoid cuDNN crash).
#
# NOTE:
#   - If you set TRAIN_T = FULL_T (220448), cropping is effectively disabled.
#   - Recommended: TRAIN_T in {16384, 32768, 65536}. This does NOT change the network architecture.
# =========================

# =========================
# PR-DWT-U-Net (Keras 3 safe) + Two-Stage Training (PR warmup -> Task)
# Key changes vs your old file:
#   (1) FIX HF LOSS: do NOT avgpool detail bands to align; instead UPSAMPLE all bands to lowest-res and concat
#   (2) Stronger PR regularizer (even-shift + DC/Nyquist + energy)
#   (3) Optional wavelet-domain HF-preservation penalty (NO STFT anywhere)
#   (4) Two-stage training:
#         Stage A: PR-only warmup (task loss weight = 0)
#         Stage B: task training with small PR and optional HF penalty
#   (5) PiT remains optional (default False)
#
# FIX FOR YOUR CRASH:
#   Keras 3 cannot serialize subclassed TwoStageTrainer (non-serializable args).
#   So we checkpoint trainer.base (Functional model) via SaveBestBase/SaveLastBase
#   instead of ModelCheckpoint on the trainer.
#
# IMPORTANT ABOUT CROPPING:
#   - We train on aligned crops of length T (random start).
#   - Validation is FULL-length (no crop).
#   - If you set TRAIN_T = FULL_T (220448), cropping is effectively disabled.
#   - Recommended: TRAIN_T in {16384, 32768, 65536}. This does NOT change the network architecture.
# =========================

# =========================
# PR-DWT-U-Net (Keras 3 safe) + Two-Stage Training (PR warmup -> Task)
# Key changes vs your old file:
#   (1) FIX HF LOSS: do NOT avgpool detail bands to align; instead UPSAMPLE all bands to lowest-res and concat
#   (2) Stronger PR regularizer (even-shift + DC/Nyquist + energy)
#   (3) Optional wavelet-domain HF-preservation penalty (NO STFT anywhere)
#   (4) Two-stage training:
#         Stage A: PR-only warmup (task loss weight = 0)
#         Stage B: task training with small PR and optional HF penalty
#   (5) PiT remains optional (default False)
#
# FIX FOR YOUR CRASH:
#   Keras 3 cannot serialize subclassed TwoStageTrainer (non-serializable args).
#   So we checkpoint trainer.base (Functional model) via SaveBestBase/SaveLastBase
#   instead of ModelCheckpoint on the trainer.
#
# IMPORTANT ABOUT CROPPING:
#   - We train on aligned crops of length T (random start).
#   - Validation is FULL-length (no crop).
#   - If you set TRAIN_T = FULL_T (220448), cropping is effectively disabled.
#   - Recommended: TRAIN_T in {16384, 32768, 65536}. This does NOT change the network architecture.
# =========================


import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------
# TF RUNTIME SAFETY (Blackwell / cuDNN stability)
# -------------------------
# Goal: avoid XLA auto-clustering + reduce cuDNN nondeterministic/autotune paths that can crash
# NOTE: does NOT change the model architecture or math.
tf.keras.backend.set_floatx("float32")
try:
    # hard-disable XLA JIT and Grappler meta-optimizer clustering
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# Make GPU errors synchronous (easier debugging; sometimes avoids async kernel failure cascades)
try:
    tf.config.experimental.set_synchronous_execution(True)
except Exception:
    pass

# Enable memory growth (safer allocator behavior)
try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

# Turn off TF32 (sometimes helps new GPU numerical quirks)
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass

# TF_RUNTIME_SAFETY


# -------------------------
# Filterbank helpers
# -------------------------
def _diag_filter_1d(h, channels):
    """Block-diagonal conv filter [K, C, C] applying same kernel per channel."""
    h = tf.reshape(h, [-1, 1, 1])              # [K,1,1]
    h = tf.tile(h, [1, channels, 1])           # [K,C,1]
    eye = tf.eye(channels)
    eye = tf.reshape(eye, [1, channels, channels])
    return h * eye                              # [K,C,C]

def qmf_highpass_from_lowpass(h0):
    """QMF: h1[n] = (-1)^n * h0[L-1-n]"""
    L = tf.shape(h0)[0]
    h0_rev = tf.reverse(h0, axis=[0])
    n = tf.cast(tf.range(L), h0.dtype)
    alt = tf.where(tf.math.floormod(tf.cast(n, tf.int32), 2) == 0,
                   tf.ones_like(n), -tf.ones_like(n))
    return alt * h0_rev

# -------------------------
# Serializables (no Lambda)
# -------------------------
@tf.keras.utils.register_keras_serializable()
class MatchTimeLen(layers.Layer):
    """XLA-safe: crop to min_len then pad to ref length. inputs: [x, ref], both [B,T,F]."""
    def call(self, inputs):
        x, ref = inputs
        tx = tf.shape(x)[1]
        tr = tf.shape(ref)[1]
        min_len = tf.minimum(tx, tr)
        x = x[:, :min_len, :]
        pad_amt = tr - min_len  # >= 0
        return tf.pad(x, [[0, 0], [0, pad_amt], [0, 0]])

@tf.keras.utils.register_keras_serializable()
class SplitChannels(layers.Layer):
    """Split last dim into n_splits chunks. Returns Python list of tensors."""
    def __init__(self, n_splits, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = int(n_splits)

    def call(self, x):
        return tf.split(x, num_or_size_splits=self.n_splits, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_splits": self.n_splits})
        return cfg

@tf.keras.utils.register_keras_serializable()
class UpsampleTo(layers.Layer):
    """
    Upsample x in time to match ref length, using nearest neighbor (repeat),
    then MatchTimeLen to crop/pad exactly to ref.
    inputs: [x, ref] where x/ref are [B,T,F]
    """
    def call(self, inputs):
        x, ref = inputs
        tx = tf.shape(x)[1]
        tr = tf.shape(ref)[1]

        # repeat factor r = ceil(tr / tx), but at least 1
        r = tf.maximum(1, (tr + tx - 1) // tf.maximum(1, tx))  # int32
        x_rep = tf.repeat(x, repeats=r, axis=1)

        # Now crop/pad to exact target length
        x_out = MatchTimeLen()([x_rep, ref])
        return x_out

    def compute_output_shape(self, input_shape):
        # input_shape: [(B, tx, F), (B, tr, F)]
        x_shape, ref_shape = input_shape
        return (x_shape[0], ref_shape[1], x_shape[2])


# -------------------------
# PR-DWT (analysis) + PR-iDWT (synthesis)
# -------------------------
@tf.keras.utils.register_keras_serializable()
class PRDWT1D(layers.Layer):
    """
    Analysis FB:
      - learn ONLY h0_raw (length L)
      - normalize h0 to unit norm
      - derive h1 via QMF
      - diagonal conv stride=2
    Adds near-PR penalty via self.add_loss (Keras 3-safe).
    """
    def __init__(self, filter_length=31, pr_shifts=12, pr_lambda=1e-2,
                 pr_dc_lambda=0.0, pr_nyq_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        if filter_length % 2 == 0:
            raise ValueError("filter_length must be odd (e.g., 31, 51, 101).")
        self.L = int(filter_length)
        self.pr_shifts = int(pr_shifts)
        self.pr_lambda = float(pr_lambda)
        self.pr_dc_lambda = float(pr_dc_lambda)
        self.pr_nyq_lambda = float(pr_nyq_lambda)
        self.channels = None

    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.h0_raw = self.add_weight(
            name="h0_raw",
            shape=(self.L,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )
        super().build(input_shape)

    def h0(self):
        h = self.h0_raw
        return h / (tf.norm(h) + 1e-8)

    def pr_regularizer(self):
        """
        Classic orthonormal/QMF PR (FIR) conditions can be enforced via even-shift autocorrelation:
            r[2m] = sum_n h[n] h[n-2m] = δ[m]
        We enforce:
          - r[0] ~ 1
          - r[2m] ~ 0 for m>=1 (up to pr_shifts)
        Optional:
          - DC gain constraint (lowpass should pass DC): sum h ~ sqrt(2)
          - Nyquist null constraint (lowpass should reject pi): sum (-1)^n h[n] ~ 0
        """
        h = self.h0()

        # even-shift autocorr residuals
        loss = tf.square(tf.reduce_sum(h * h) - 1.0)  # r0 -> 1
        for m in range(1, self.pr_shifts + 1):
            k = 2 * m
            if k < self.L:
                r = tf.reduce_sum(h[k:] * h[:-k])
                loss += tf.square(r)

        # optional DC/Nyquist shaping (doesn't guarantee PR; helps lowpass "look like lowpass")
        if self.pr_dc_lambda > 0.0:
            dc = tf.reduce_sum(h)
            loss += self.pr_dc_lambda * tf.square(dc - tf.sqrt(tf.constant(2.0, h.dtype)))
        if self.pr_nyq_lambda > 0.0:
            n = tf.cast(tf.range(self.L), h.dtype)
            alt = tf.where(tf.math.floormod(tf.cast(n, tf.int32), 2) == 0,
                           tf.ones_like(n), -tf.ones_like(n))
            nyq = tf.reduce_sum(alt * h)
            loss += self.pr_nyq_lambda * tf.square(nyq)

        return loss

    def call(self, x):
        if self.pr_lambda > 0.0:
            self.add_loss(self.pr_lambda * self.pr_regularizer())

        h0 = self.h0()
        h1 = qmf_highpass_from_lowpass(h0)

        H0 = _diag_filter_1d(h0, self.channels)
        H1 = _diag_filter_1d(h1, self.channels)

        a = tf.nn.conv1d(x, H0, stride=2, padding="SAME")
        d = tf.nn.conv1d(x, H1, stride=2, padding="SAME")
        return a, d

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filter_length": self.L,
            "pr_shifts": self.pr_shifts,
            "pr_lambda": self.pr_lambda,
            "pr_dc_lambda": self.pr_dc_lambda,
            "pr_nyq_lambda": self.pr_nyq_lambda,
        })
        return cfg

@tf.keras.utils.register_keras_serializable()
class PRIDWT1D(layers.Layer):
    """
    Synthesis FB:
      g0 = reverse(h0), g1 = reverse(h1)
      use conv1d_transpose (inverse of stride-2 analysis)
    Note: we "tie" to the analysis layer via set_dwt() during model construction.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dwt = None

    def set_dwt(self, dwt_layer: PRDWT1D):
        # must be called BEFORE the layer is built (we do it in model construction)
        self._dwt = dwt_layer
        return self

    def call(self, inputs):
        a, d = inputs
        if self._dwt is None:
            raise ValueError("PRIDWT1D: dwt reference not set. Call set_dwt(dwt_layer) during build.")
        dwt = self._dwt

        h0 = dwt.h0()
        h1 = qmf_highpass_from_lowpass(h0)
        g0 = tf.reverse(h0, axis=[0])
        g1 = tf.reverse(h1, axis=[0])

        # diagonal filters [K, C, C]
        G0 = _diag_filter_1d(g0, dwt.channels)
        G1 = _diag_filter_1d(g1, dwt.channels)

        # ----- exact "stride-2 transpose conv" via zero-insertion + conv1d -----
        # a,d: [B, T2, C] -> upsample to [B, 2*T2, C] with zeros on odd indices
        B = tf.shape(a)[0]
        T2 = tf.shape(a)[1]
        C = tf.shape(a)[2]
        T = T2 * 2

        def upsample_zero_interleave(x):
            x = tf.reshape(x, [B, T2, 1, C])         # [B,T2,1,C]
            z = tf.zeros_like(x)                     # [B,T2,1,C]
            x = tf.concat([x, z], axis=2)            # [B,T2,2,C]
            x = tf.reshape(x, [B, T, C])             # [B,T,C]
            return x

        a_up = upsample_zero_interleave(a)           # [B,T,C]
        d_up = upsample_zero_interleave(d)           # [B,T,C]

        xa = tf.nn.conv1d(a_up, G0, stride=1, padding="SAME")
        xd = tf.nn.conv1d(d_up, G1, stride=1, padding="SAME")
        return xa + xd

# -------------------------
# U-Net core (channels-last)
# -------------------------
def conv_block(x, f, k=9):
    x = layers.Conv1D(f, k, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(f, k, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def unet_1d(x, depth=4, base_filters=64, k=9):
    skips, h = [], x

    for i in range(depth):
        f = base_filters * (2 ** i)
        h = conv_block(h, f, k=k)
        skips.append(h)
        h = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(h)

    h = conv_block(h, base_filters * (2 ** depth), k=k)

    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        h = layers.UpSampling1D(size=2)(h)
        h = MatchTimeLen()([h, skips[i]])
        h = layers.Concatenate(axis=-1)([h, skips[i]])
        h = conv_block(h, f, k=k)

    return layers.Conv1D(x.shape[-1], 1, padding="same")(h)

# -------------------------
# Loss: SI-SDR (+ optional PiT)
# -------------------------
def sisdr_pair(y_true, y_pred, eps=1e-8):
    # y_*: [B,C,T]
    yt = y_true - tf.reduce_mean(y_true, axis=2, keepdims=True)
    yp = y_pred - tf.reduce_mean(y_pred, axis=2, keepdims=True)
    dot = tf.reduce_sum(yt * yp, axis=2, keepdims=True)
    energy = tf.reduce_sum(yt ** 2, axis=2, keepdims=True) + eps
    scale = dot / energy
    target = scale * yt
    noise = yp - target
    s_target = tf.reduce_sum(target ** 2, axis=2) + eps
    e_noise = tf.reduce_sum(noise ** 2, axis=2) + eps
    sisdr = 10.0 * tf.math.log(s_target / e_noise) / tf.math.log(10.0)
    return sisdr  # [B,C]

def sisdr_loss_no_pit(y_true, y_pred):
    return -tf.reduce_mean(sisdr_pair(y_true, y_pred))

def pit_sisdr_loss(y_true, y_pred):
    C = int(y_true.shape[1])
    perms = tf.constant(list(__import__("itertools").permutations(range(C))), dtype=tf.int32)

    def score_for_perm(p):
        yp = tf.gather(y_pred, p, axis=1)
        return tf.reduce_mean(sisdr_pair(y_true, yp), axis=1)  # [B]

    scores = tf.map_fn(score_for_perm, perms, fn_output_signature=tf.float32)  # [P,B]
    best = tf.reduce_max(scores, axis=0)  # [B]
    return -tf.reduce_mean(best)

def make_task_loss(pit=False):
    return pit_sisdr_loss if pit else sisdr_loss_no_pit

# -------------------------
# Wavelet-domain auxiliary penalties (NO STFT)
# -------------------------
def wavelet_hf_preserve_loss(y_pred, y_in, levels=2, filter_length=101):
    """
    Mitigate "HF band loss":
    Encourage the *detail-band energy* of y_pred to not collapse relative to input.
    This is NOT a separation target; it only says: don't annihilate details.

    Uses the model's own learned PR filters? Here we use a fixed simple differentiable proxy:
      - take 1D finite difference as HF proxy at full rate and compare energies
    This avoids needing to instantiate extra DWT layers.

    If you prefer true wavelet details: we can swap this to use PRDWT1D, but that
    couples the penalty to the PR filters strongly. This proxy is stable.
    """
    # y_*: [B,C,T]
    # HF proxy: first difference along time
    dy_p = y_pred[:, :, 1:] - y_pred[:, :, :-1]
    dy_x = y_in[:, :, 1:] - y_in[:, :, :-1]

    ep = tf.reduce_mean(dy_p ** 2, axis=2)  # [B,C]
    ex = tf.reduce_mean(dy_x ** 2, axis=2)  # [B,C]

    # penalize only when pred HF energy is too small (hinge)
    ratio = ep / (ex + 1e-8)
    # want ratio >= r_min (e.g., 0.5); hinge = max(0, r_min - ratio)
    r_min = 0.5
    hinge = tf.nn.relu(r_min - ratio)
    return tf.reduce_mean(hinge)

# -------------------------
# Build model with optional tap points
# -------------------------
def build_pr_dwt_unet(
    time_length: int | None,
    channels: int,
    levels=2,
    filter_length=101,
    pr_shifts=12,
    pr_lambda=1e-2,
    pr_dc_lambda=0.0,
    pr_nyq_lambda=0.0,
    unet_depth=4,
    base_filters=64,
    return_taps=False,
):
    """
    Input/Output: [B, C, T] (channels-first)

    IMPORTANT CHANGE (HF FIX):
      We align all subbands to the LOWEST resolution by UPSAMPLING (not pooling).
      Pooling detail bands destroys HF information systematically.
    """
    inp = layers.Input(shape=(channels, time_length), name="mix")   # [B,C,T]  (T can be None for variable length)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)        # [B,T,C]

    # analysis pyramid
    dwt_layers = [
        PRDWT1D(filter_length=filter_length,
                pr_shifts=pr_shifts,
                pr_lambda=pr_lambda,
                pr_dc_lambda=pr_dc_lambda,
                pr_nyq_lambda=pr_nyq_lambda,
                name=f"prdwt_{i}")
        for i in range(levels)
    ]

    approx = x
    details = []
    taps = {}

    for i, dwt in enumerate(dwt_layers):
        a, d = dwt(approx)
        taps[f"a_{i}"] = a
        taps[f"d_{i}"] = d
        details.append(d)
        approx = a

    # ---- Align to lowest-res by UPSAMPLING everything to approx's length ----
    aligned = []
    low_ref = approx  # [B, T_low, C]
    for i, d in enumerate(details):
        d_up = UpsampleTo(name=f"up_to_low_d_{i}")([d, low_ref])
        aligned.append(d_up)
    aligned.append(approx)
    feat = layers.Concatenate(axis=-1, name="feat_concat")(aligned)
    taps["feat_before_unet"] = feat

    # U-Net at lowest resolution
    feat_hat = unet_1d(feat, depth=unet_depth, base_filters=base_filters, k=9)

    # split channels back into (details_low..., approx_low)
    parts = SplitChannels(levels + 1, name="split_bands")(feat_hat)
    detail_hats_low = parts[:levels]
    approx_hat = parts[-1]
    taps["approx_hat_low"] = approx_hat
    for i in range(levels):
        taps[f"detail_hat_low_{i}"] = detail_hats_low[i]

    # detail_hats are already at low-res matching approx_hat (lowest)
    # we must DOWNSAMPLE them to each pyramid stage input for synthesis? Actually synthesis expects
    # details at each stage native rate. So we need to upsample from low-res to each detail's native length.
    detail_hats = []
    for i in range(levels):
        # native ref detail length is details[i]
        dh = detail_hats_low[i]
        dh_up = UpsampleTo(name=f"up_to_native_d_{i}")([dh, details[i]])
        detail_hats.append(dh_up)
        taps[f"detail_hat_{i}"] = dh_up

    # synthesis pyramid
    idwt_layers = []
    for i in range(levels):
        idwt = PRIDWT1D(name=f"pridwt_{i}").set_dwt(dwt_layers[i])
        idwt_layers.append(idwt)

    recon = approx_hat
    for i in reversed(range(levels)):
        recon = idwt_layers[i]([recon, detail_hats[i]])

    taps["recon_pre_match"] = recon
    recon = MatchTimeLen(name="match_out_len")([recon, x])  # length = input length

    y = layers.Permute((2, 1), name="to_channels_time")(recon)  # [B,C,T]
    taps["y"] = y

    if not return_taps:
        return Model(inp, y, name="PR_DWT_UNet")

    ordered_keys = (
        [f"a_{i}" for i in range(levels)] +
        [f"d_{i}" for i in range(levels)] +
        ["feat_before_unet"] +
        ["approx_hat_low"] + [f"detail_hat_low_{i}" for i in range(levels)] +
        [f"detail_hat_{i}" for i in range(levels)] +
        ["recon_pre_match"]
    )
    outputs = [y] + [taps[k] for k in ordered_keys]
    return Model(inp, outputs, name="PR_DWT_UNet_Taps"), ordered_keys

# -------------------------
# Training helpers
# -------------------------
def split_train_val(x, y, val_ratio=0.15, seed=1337):
    assert len(x) == len(y)
    N = len(x)
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(np.round(N * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

def make_ds(x, y, batch_size=4, shuffle=True, crop_len=None, aligned_to=1, seed=1337, training=True):
    """Dataset builder with OPTIONAL random aligned cropping.
    - x,y are numpy arrays: (N,C,T)
    - crop_len: if not None, take a random crop of length crop_len from BOTH x and y with the same start.
    - aligned_to: enforce start index multiple (e.g., 2**levels) so analysis/synthesis stay aligned.
    - Validation should typically use crop_len=None.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), reshuffle_each_iteration=True, seed=seed)

    if crop_len is not None:
        crop_len = int(crop_len)
        aligned_to = int(max(1, aligned_to))

        def _crop_pair(xb, yb):
            # xb,yb: [C,T]
            T = tf.shape(xb)[-1]
            max_start = tf.maximum(0, T - crop_len)
            # random start in [0, max_start], then align down to multiple
            start = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
            start = (start // aligned_to) * aligned_to
            xb = xb[:, start:start + crop_len]
            yb = yb[:, start:start + crop_len]
            return xb, yb

        ds = ds.map(_crop_pair, num_parallel_calls=1, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=True)

    # Prefetch can sometimes trigger async overlap issues on new stacks; keep small/deterministic
    ds = ds.prefetch(1)
    return ds

# -------------------------
# Two-stage training loop (custom train_step to weight losses)
# -------------------------
@tf.keras.utils.register_keras_serializable()
class TwoStageTrainer(Model):
    """
    Wraps a base model (mix -> yhat) and trains with:
      total = w_task * task_loss + w_hf * hf_preserve + (PR losses already injected via add_loss in PRDWT1D)
    """
    def __init__(self, base_model, task_loss_fn, levels=2, hf_lambda=0.0, task_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.task_loss_fn = task_loss_fn
        self.levels = int(levels)
        self.hf_lambda = float(hf_lambda)
        self.task_lambda = float(task_lambda)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.task_tracker = tf.keras.metrics.Mean(name="task_loss")
        self.hf_tracker = tf.keras.metrics.Mean(name="hf_loss")

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.task_tracker, self.hf_tracker]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            yhat = self.base(x, training=True)
            task = self.task_loss_fn(y, yhat)
            hf = tf.constant(0.0, dtype=task.dtype)
            if self.hf_lambda > 0.0:
                hf = wavelet_hf_preserve_loss(yhat, x, levels=self.levels)

            pr_losses = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, dtype=task.dtype)
            total = self.task_lambda * task + self.hf_lambda * hf + pr_losses

        grads = tape.gradient(total, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        self.loss_tracker.update_state(total)
        self.task_tracker.update_state(task)
        self.hf_tracker.update_state(hf)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        yhat = self.base(x, training=False)
        task = self.task_loss_fn(y, yhat)
        hf = tf.constant(0.0, dtype=task.dtype)
        if self.hf_lambda > 0.0:
            hf = wavelet_hf_preserve_loss(yhat, x, levels=self.levels)
        pr_losses = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, dtype=task.dtype)
        total = self.task_lambda * task + self.hf_lambda * hf + pr_losses

        self.loss_tracker.update_state(total)
        self.task_tracker.update_state(task)
        self.hf_tracker.update_state(hf)
        return {m.name: m.result() for m in self.metrics}


# -------------------------
# Keras 3-safe checkpointing
# -------------------------
class SaveBestBase(tf.keras.callbacks.Callback):
    """Save trainer.base (Functional model) when monitored metric improves.

    Keras 3 cannot serialize subclassed Models that hold non-serializable objects
    unless they implement get_config(). TwoStageTrainer holds function objects,
    so we checkpoint the underlying Functional model instead.
    """
    def __init__(self, trainer, filepath, monitor="val_loss", mode="min", verbose=1):
        super().__init__()
        self.trainer = trainer
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = int(verbose)
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (current < self.best) if self.mode == "min" else (current > self.best)
        if improved:
            self.best = current
            if self.verbose:
                print(
                    f"\nEpoch {epoch+1}: {self.monitor} improved to {float(current):.6f}, "
                    f"saving BASE model to {self.filepath}"
                )
            self.trainer.base.save(self.filepath)


class SaveLastBase(tf.keras.callbacks.Callback):
    """Always save trainer.base at end of every epoch."""
    def __init__(self, trainer, filepath, verbose=0):
        super().__init__()
        self.trainer = trainer
        self.filepath = filepath
        self.verbose = int(verbose)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print(f"\nEpoch {epoch+1}: saving LAST base model to {self.filepath}")
        self.trainer.base.save(self.filepath)


# -------------------------
# Main training entry
# -------------------------
def train_two_stage(
    x_mix, y_target,
    T, C,
    out_dir="./runs_pr",
    val_ratio=0.15,
    epochs_A=25,       # PR warmup epochs
    epochs_B=500,      # task epochs
    batch_size=2,
    pit=False,
    lr=3e-4,
    clipnorm=1.0,
    early_patience=20,
    min_delta=1e-4,

    # Model hyperparams
    levels=2,
    filter_length=101,

    # PR constraints
    pr_shifts=24,
    pr_lambda_A=5e-2,     # Stage A stronger PR
    pr_lambda_B=1e-2,     # Stage B weaker PR
    pr_dc_lambda=1e-2,
    pr_nyq_lambda=1e-2,

    # HF mitigation (no STFT)
    hf_lambda_B=0.05,     # small, only to stop "HF collapse"
):
    os.makedirs(out_dir, exist_ok=True)
    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    x_tr, y_tr, x_va, y_va = split_train_val(x_mix, y_target, val_ratio=val_ratio, seed=1337)
    align = 2 ** int(levels)
    tr_ds = make_ds(x_tr, y_tr, batch_size=batch_size, shuffle=True, crop_len=T, aligned_to=align, seed=1337, training=True)
    va_ds = make_ds(x_va, y_va, batch_size=batch_size, shuffle=False, crop_len=None, aligned_to=align, seed=1337, training=False)

    # ----------------
    # Stage A model (strong PR, no task)
    # ----------------
    base_A = build_pr_dwt_unet(
        time_length=None, channels=C,
        levels=levels, filter_length=filter_length,
        pr_shifts=pr_shifts, pr_lambda=pr_lambda_A,
        pr_dc_lambda=pr_dc_lambda, pr_nyq_lambda=pr_nyq_lambda,
        unet_depth=4, base_filters=64,
        return_taps=False,
    )

    trainer_A = TwoStageTrainer(
        base_model=base_A,
        task_loss_fn=make_task_loss(pit=pit),
        levels=levels,
        hf_lambda=0.0,
        task_lambda=0.0,        # IMPORTANT: PR only warmup (task weight 0)
        name="Trainer_StageA",
    )

    # Optimizer safety:
    # - Stage A can use the provided lr/clipnorm
    # - Stage B is capped to a gentler lr and tighter clipnorm to avoid divergence on recorded data
    lr_A = lr
    lr_B = min(lr, 1e-4)
    clipnorm_A = clipnorm
    clipnorm_B = min(clipnorm, 0.25)

    optA = tf.keras.optimizers.Adam(learning_rate=lr_A, clipnorm=clipnorm_A)
    trainer_A.compile(optimizer=optA, jit_compile=False)

    bestA = os.path.join(run_dir, "best_stageA.keras")
    callbacksA = [
        SaveBestBase(trainer_A, bestA, monitor="val_loss", mode="min", verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log_stageA.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print(f"[Split] train={len(x_tr)}  val={len(x_va)}  (val_ratio={val_ratio})")
    print(f"[Run dir] {run_dir}")
    print("\n===== Stage A: PR warmup (task_lambda=0) =====")
    trainer_A.fit(tr_ds, validation_data=va_ds, epochs=epochs_A, callbacks=callbacksA, verbose=1)

    # ----------------
    # Stage B model (task + soft PR + optional HF)
    # Rebuild with weaker PR lambda, then load weights from Stage A
    # ----------------
    base_B = build_pr_dwt_unet(
        time_length=None, channels=C,
        levels=levels, filter_length=filter_length,
        pr_shifts=pr_shifts, pr_lambda=pr_lambda_B,
        pr_dc_lambda=pr_dc_lambda, pr_nyq_lambda=pr_nyq_lambda,
        unet_depth=4, base_filters=64,
        return_taps=False,
    )
    # Load Stage A weights into Stage B (architecture identical except pr_lambda stored in layer config;
    # weights are compatible)
    base_B.set_weights(base_A.get_weights())

    trainer_B = TwoStageTrainer(
        base_model=base_B,
        task_loss_fn=make_task_loss(pit=pit),
        levels=levels,
        hf_lambda=hf_lambda_B,   # small HF preservation
        task_lambda=1.0,
        name="Trainer_StageB",
    )

    optB = tf.keras.optimizers.Adam(learning_rate=lr_B, clipnorm=clipnorm_B)
    trainer_B.compile(optimizer=optB, jit_compile=False)

    bestB = os.path.join(run_dir, "best.keras")
    lastB = os.path.join(run_dir, "last.keras")

    callbacksB = [
        SaveBestBase(trainer_B, bestB, monitor="val_loss", mode="min", verbose=1),
        SaveLastBase(trainer_B, lastB, verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "tb")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=early_patience, min_delta=min_delta,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    # Save config used
    cfg = dict(
        val_ratio=val_ratio, epochs_A=epochs_A, epochs_B=epochs_B,
        batch_size=batch_size, pit=pit, lr=lr, clipnorm=clipnorm,
        levels=levels, filter_length=filter_length,
        pr_shifts=pr_shifts, pr_lambda_A=pr_lambda_A, pr_lambda_B=pr_lambda_B,
        pr_dc_lambda=pr_dc_lambda, pr_nyq_lambda=pr_nyq_lambda,
        hf_lambda_B=hf_lambda_B,
    )
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n===== Stage B: Task training (task_lambda=1, soft PR, +HF) =====")
    history = trainer_B.fit(tr_ds, validation_data=va_ds, epochs=epochs_B, callbacks=callbacksB, verbose=1)

    print(f"\nSaved best model: {bestB}")
    return trainer_B, history, run_dir

# -------------------------
# Main: your data loader
# -------------------------
if __name__ == "__main__":
    dpath = "/home/rrame12/Desktop/Research/DWT_IR" #"/home/rrame12/Desktop/Datasets/Re-recorded/preprocessed/"
    FULL_T = 220448
    TRAIN_T = 32768  # random aligned training crops; validation stays full length


    print("Loading Files...")
    #Xtrain = np.load(os.path.join(dpath, "mixture_rec_pp.npy")).astype(np.float32)  # (N,3,T)
    #Ytrain = np.load(os.path.join(dpath, "gt_diag_rec_pp.npy")).astype(np.float32)

    Xtrain = np.load(os.path.join(dpath, "Xtrain.npy")).astype(np.float32)  # (N,3,T)
    Ytrain = np.load(os.path.join(dpath, "Ytrain.npy")).astype(np.float32)

    Xtrain = Xtrain[:, :, :FULL_T]
    Ytrain = Ytrain[:, :, :FULL_T]

    print("Training crop length TRAIN_T:", TRAIN_T)
    print("NOTE: training uses random aligned crops; validation uses full-length.")

    print("Final Xtrain shape:", Xtrain.shape)
    print("Final Ytrain shape:", Ytrain.shape)

    # Close-mic labeled tracks => PiT should be False (keep True only as ablation)
    trainer, history, run_dir = train_two_stage(
        Xtrain, Ytrain,
        T=TRAIN_T, C=3,
        out_dir="./runs_pr",
        val_ratio=0.15,
        epochs_A=10,
        epochs_B=500,
        batch_size=2,
        pit=False,
        lr=3e-4,
        clipnorm=1.0,
        early_patience=20,
        min_delta=1e-4,

        levels=2,
        filter_length=101,

        pr_shifts=32, #24
        pr_lambda_A=1.0, #5e-2
        pr_lambda_B=1e-1, #1e-2
        pr_dc_lambda=1.0, #1e-2
        pr_nyq_lambda=1.0, #1e-2

        hf_lambda_B=0.05,
    )

    print("Done.")
