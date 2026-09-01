# ablation_frontends.py
# Frontend ablation built from your PR-DWT-U-Net training script (same trainer/loss/logging),
# only the frontend differs. See original baseline structure in iprdwt.py :contentReference[oaicite:1]{index=1}.

import os, time, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------
# TF runtime safety (same spirit as your file)
# -------------------------
tf.keras.backend.set_floatx("float32")
try:
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.experimental.set_synchronous_execution(True)
except Exception:
    pass
try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass


# -------------------------
# Utilities
# -------------------------
def _diag_filter_1d(h, channels):
    """Block-diagonal conv filter [K, C, C] applying same kernel per channel."""
    h = tf.reshape(h, [-1, 1, 1])
    h = tf.tile(h, [1, channels, 1])
    eye = tf.eye(channels)
    eye = tf.reshape(eye, [1, channels, channels])
    return h * eye

def qmf_highpass_from_lowpass(h0):
    """QMF: h1[n] = (-1)^n * h0[L-1-n]"""
    L = tf.shape(h0)[0]
    h0_rev = tf.reverse(h0, axis=[0])
    n = tf.cast(tf.range(L), h0.dtype)
    alt = tf.where(tf.math.floormod(tf.cast(n, tf.int32), 2) == 0,
                   tf.ones_like(n), -tf.ones_like(n))
    return alt * h0_rev

@tf.keras.utils.register_keras_serializable()
class MatchTimeLen(layers.Layer):
    """Crop to min_len then pad to ref length. inputs: [x, ref], both [B,T,F]."""
    def call(self, inputs):
        x, ref = inputs
        tx = tf.shape(x)[1]
        tr = tf.shape(ref)[1]
        min_len = tf.minimum(tx, tr)
        x = x[:, :min_len, :]
        pad_amt = tr - min_len
        return tf.pad(x, [[0, 0], [0, pad_amt], [0, 0]])

@tf.keras.utils.register_keras_serializable()
class SplitChannels(layers.Layer):
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
    """Nearest-neighbor repeat in time to match ref length, then exact crop/pad."""
    def call(self, inputs):
        x, ref = inputs
        tx = tf.shape(x)[1]
        tr = tf.shape(ref)[1]
        r = tf.maximum(1, (tr + tx - 1) // tf.maximum(1, tx))
        x_rep = tf.repeat(x, repeats=r, axis=1)
        return MatchTimeLen()([x_rep, ref])

# -------------------------
# U-Net core (same as yours)
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
# SI-SDR losses (same as yours)
# -------------------------
def sisdr_pair(y_true, y_pred, eps=1e-8):
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
    return sisdr

def sisdr_loss_no_pit(y_true, y_pred):
    return -tf.reduce_mean(sisdr_pair(y_true, y_pred))

def pit_sisdr_loss(y_true, y_pred):
    C = int(y_true.shape[1])
    perms = tf.constant(list(__import__("itertools").permutations(range(C))), dtype=tf.int32)
    def score_for_perm(p):
        yp = tf.gather(y_pred, p, axis=1)
        return tf.reduce_mean(sisdr_pair(y_true, yp), axis=1)  # [B]
    scores = tf.map_fn(score_for_perm, perms, fn_output_signature=tf.float32)  # [P,B]
    best = tf.reduce_max(scores, axis=0)
    return -tf.reduce_mean(best)

def make_task_loss(pit=False):
    return pit_sisdr_loss if pit else sisdr_loss_no_pit

# -------------------------
# Optional HF-preserve penalty (same proxy as yours)
# -------------------------
def wavelet_hf_preserve_loss(y_pred, y_in):
    dy_p = y_pred[:, :, 1:] - y_pred[:, :, :-1]
    dy_x = y_in[:, :, 1:] - y_in[:, :, :-1]
    ep = tf.reduce_mean(dy_p ** 2, axis=2)
    ex = tf.reduce_mean(dy_x ** 2, axis=2)
    ratio = ep / (ex + 1e-8)
    r_min = 0.5
    hinge = tf.nn.relu(r_min - ratio)
    return tf.reduce_mean(hinge)

# -------------------------
# Dataset (same)
# -------------------------
def split_train_val(x, y, val_ratio=0.15, seed=1337):
    N = len(x)
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(np.round(N * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

def make_ds(x, y, batch_size=4, shuffle=True, crop_len=None, aligned_to=1, seed=1337):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), reshuffle_each_iteration=True, seed=seed)

    if crop_len is not None:
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

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    return ds

# -------------------------
# Trainer (same concept)
# -------------------------
@tf.keras.utils.register_keras_serializable()
class TwoStageTrainer(Model):
    def __init__(self, base_model, task_loss_fn, hf_lambda=0.0, task_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.task_loss_fn = task_loss_fn
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
                hf = wavelet_hf_preserve_loss(yhat, x)
            extra = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, dtype=task.dtype)
            total = self.task_lambda * task + self.hf_lambda * hf + extra

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
            hf = wavelet_hf_preserve_loss(yhat, x)
        extra = tf.add_n(self.base.losses) if self.base.losses else tf.constant(0.0, dtype=task.dtype)
        total = self.task_lambda * task + self.hf_lambda * hf + extra
        self.loss_tracker.update_state(total)
        self.task_tracker.update_state(task)
        self.hf_tracker.update_state(hf)
        return {m.name: m.result() for m in self.metrics}

# -------------------------
# Checkpoint helpers (Keras3-safe)
# -------------------------
class SaveBestBase(tf.keras.callbacks.Callback):
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
                print(f"\nEpoch {epoch+1}: {self.monitor} improved to {float(current):.6f}, saving BASE -> {self.filepath}")
            self.trainer.base.save(self.filepath)

class SaveLastBase(tf.keras.callbacks.Callback):
    def __init__(self, trainer, filepath, verbose=0):
        super().__init__()
        self.trainer = trainer
        self.filepath = filepath
        self.verbose = int(verbose)
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print(f"\nEpoch {epoch+1}: saving LAST base -> {self.filepath}")
        self.trainer.base.save(self.filepath)

# ============================================================
# Frontends
# ============================================================

# (i) Waveform-only: mix -> UNet -> waveform
def build_wave_unet(time_length, channels, unet_depth=4, base_filters=64):
    inp = layers.Input(shape=(channels, time_length), name="mix")  # [B,C,T]
    x = layers.Permute((2, 1), name="to_time_channels")(inp)       # [B,T,C]
    y = unet_1d(x, depth=unet_depth, base_filters=base_filters, k=9)
    y = layers.Permute((2, 1), name="to_channels_time")(y)         # [B,C,T]
    return Model(inp, y, name="WaveUNet")

# (ii) Fixed wavelet frontend: analysis -> low-res UNet -> synthesis.
# Taps are standard orthogonal analysis low-pass filters. We center-pad to the
# requested odd filter length so all fixed families use the same frontend shape.
_FIXED_WAVELET_DEC_LO = {
    "haar": [
        0.7071067811865476, 0.7071067811865476,
    ],
    "db1": [
        0.7071067811865476, 0.7071067811865476,
    ],
    "db2": [
        0.4829629131445341, 0.8365163037378079,
        0.2241438680420134, -0.12940952255126034,
    ],
    "db4": [
        -0.010597401785069032, 0.0328830116668852,
        0.030841381835560764, -0.18703481171888114,
        -0.027983769416859854, 0.6308807679298587,
        0.7148465705529154, 0.23037781330885523,
    ],
    "db8": [
        -0.00011747678412476953, 0.0006754494059985568,
        -0.00039174037337694705, -0.004870352993451574,
        0.008746094047405777, 0.013981027917398282,
        -0.044088253930794755, -0.017369301001807547,
        0.12874742662047847, 0.0004724845739132828,
        -0.2840155429615469, -0.015829105256349305,
        0.5853546836541907, 0.6756307362972898,
        0.31287159091429995, 0.05441584224310401,
    ],
    "sym4": [
        -0.07576571478927333, -0.02963552764599851,
        0.49761866763201545, 0.8037387518059161,
        0.29785779560527736, -0.09921954357684722,
        -0.012603967262037833, 0.0322231006040427,
    ],
    "coif1": [
        -0.01565572813546454, -0.0727326195128539,
        0.38486484686420286, 0.8525720202122554,
        0.3378976624578092, -0.0727326195128539,
    ],
}

def fixed_wavelet_lowpass_padded(wavelet="db2", L=101):
    key = str(wavelet).lower()
    if key not in _FIXED_WAVELET_DEC_LO:
        valid = ", ".join(sorted(_FIXED_WAVELET_DEC_LO))
        raise ValueError(f"Unknown fixed wavelet '{wavelet}'. Valid options: {valid}")
    h = np.array(_FIXED_WAVELET_DEC_LO[key], dtype=np.float32)

    if L < len(h) or (L % 2 == 0):
        raise ValueError(f"filter_length must be odd and >= {len(h)} for wavelet {key}")
    pad_total = L - len(h)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    h = np.pad(h, (pad_left, pad_right), mode="constant")
    h = h / (np.linalg.norm(h) + 1e-8)
    return tf.constant(h, dtype=tf.float32)

def fixed_db2_lowpass_padded(L=101):
    return fixed_wavelet_lowpass_padded("db2", L=L)

@tf.keras.utils.register_keras_serializable()
class FixedDWT1D(layers.Layer):
    def __init__(self, h0_const, **kwargs):
        super().__init__(**kwargs)
        # store as numpy for config-serialization
        h0_np = np.array(h0_const, dtype=np.float32)
        self.h0_list = h0_np.tolist()
        self.h0_const = tf.constant(h0_np, dtype=tf.float32)
        self.channels = None

    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        super().build(input_shape)

    def call(self, x):
        h0 = self.h0_const
        h1 = qmf_highpass_from_lowpass(h0)
        H0 = _diag_filter_1d(h0, self.channels)
        H1 = _diag_filter_1d(h1, self.channels)
        a = tf.nn.conv1d(x, H0, stride=2, padding="SAME")
        d = tf.nn.conv1d(x, H1, stride=2, padding="SAME")
        return a, d

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"h0_const": self.h0_list})
        return cfg

    @classmethod
    def from_config(cls, config):
        h0 = config.pop("h0_const")
        return cls(h0_const=h0, **config)
@tf.keras.utils.register_keras_serializable()
class FixedIDWT1D(layers.Layer):
    def __init__(self, h0_const, **kwargs):
        super().__init__(**kwargs)
        h0_np = np.array(h0_const, dtype=np.float32)
        self.h0_list = h0_np.tolist()
        self.h0_const = tf.constant(h0_np, dtype=tf.float32)

    def call(self, inputs):
        a, d = inputs
        h0 = self.h0_const
        h1 = qmf_highpass_from_lowpass(h0)
        g0 = tf.reverse(h0, axis=[0])
        g1 = tf.reverse(h1, axis=[0])

        C = tf.shape(a)[-1]
        G0 = _diag_filter_1d(g0, C)
        G1 = _diag_filter_1d(g1, C)

        B = tf.shape(a)[0]
        T2 = tf.shape(a)[1]
        T = T2 * 2

        def upsample_zero_interleave(x):
            x = tf.reshape(x, [B, T2, 1, C])
            z = tf.zeros_like(x)
            x = tf.concat([x, z], axis=2)
            x = tf.reshape(x, [B, T, C])
            return x

        a_up = upsample_zero_interleave(a)
        d_up = upsample_zero_interleave(d)
        xa = tf.nn.conv1d(a_up, G0, stride=1, padding="SAME")
        xd = tf.nn.conv1d(d_up, G1, stride=1, padding="SAME")
        return xa + xd

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"h0_const": self.h0_list})
        return cfg

    @classmethod
    def from_config(cls, config):
        h0 = config.pop("h0_const")
        return cls(h0_const=h0, **config)

def build_fixed_dwt_unet(time_length, channels, levels=2, filter_length=101, fixed_wavelet="db2", unet_depth=4, base_filters=64):
    inp = layers.Input(shape=(channels, time_length), name="mix")   # [B,C,T]
    x = layers.Permute((2, 1), name="to_time_channels")(inp)        # [B,T,C]

    h0 = fixed_wavelet_lowpass_padded(fixed_wavelet, L=filter_length).numpy()

    approx = x
    details = []
    dwt_layers = [FixedDWT1D(h0, name=f"fixdwt_{i}") for i in range(levels)]
    for i in range(levels):
        a, d = dwt_layers[i](approx)
        details.append(d)
        approx = a

    # align to lowest-res by UPSAMPLING (same HF-fix idea)
    low_ref = approx
    aligned = []
    for i, d in enumerate(details):
        aligned.append(UpsampleTo(name=f"up_to_low_fixd_{i}")([d, low_ref]))
    aligned.append(approx)
    feat = layers.Concatenate(axis=-1, name="feat_concat")(aligned)

    feat_hat = unet_1d(feat, depth=unet_depth, base_filters=base_filters, k=9)

    parts = SplitChannels(levels + 1, name="split_bands")(feat_hat)
    detail_hats_low = parts[:levels]
    approx_hat = parts[-1]

    # upsample low-res detail estimates back to each native detail length
    detail_hats = []
    for i in range(levels):
        detail_hats.append(UpsampleTo(name=f"up_to_native_fixd_{i}")([detail_hats_low[i], details[i]]))

    idwt_layers = [FixedIDWT1D(h0, name=f"fixidwt_{i}") for i in range(levels)]

    recon = approx_hat
    for i in reversed(range(levels)):
        recon = idwt_layers[i]([recon, detail_hats[i]])

    recon = MatchTimeLen(name="match_out_len")([recon, x])
    y = layers.Permute((2, 1), name="to_channels_time")(recon)
    return Model(inp, y, name="FixedDWT_UNet")

# (iii) STFT frontend: waveform -> STFT -> UNet (1D over frames) -> iSTFT -> waveform
@tf.keras.utils.register_keras_serializable()
class STFTFrontend(layers.Layer):
    def __init__(self, frame_length=1024, frame_step=256, fft_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = int(frame_length)
        self.frame_step = int(frame_step)
        self.fft_length = int(fft_length)

    def call(self, x_time_ch):  # [B,T,C] float32
        # compute STFT per channel, then stack real/imag into features
        # output: feats [B,Frames, C*(Fbins*2)]
        B = tf.shape(x_time_ch)[0]
        T = tf.shape(x_time_ch)[1]
        C = tf.shape(x_time_ch)[2]

        feats = []
        for c in range(x_time_ch.shape[-1]):
            xc = x_time_ch[:, :, c]  # [B,T]
            X = tf.signal.stft(
                xc,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                window_fn=tf.signal.hann_window,
                pad_end=True
            )  # [B,Frames,F]
            feats.append(tf.math.real(X))
            feats.append(tf.math.imag(X))

        F = tf.shape(feats[0])[-1]
        feats = tf.stack(feats, axis=-1)          # [B,Frames,F, 2C]
        feats = tf.reshape(feats, [B, -1, F * tf.shape(feats)[-1]])  # [B,Frames, F*(2C)]
        return feats

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length))
        return cfg

@tf.keras.utils.register_keras_serializable()
class ISTFTBackend(layers.Layer):
    def __init__(self, channels, frame_length=1024, frame_step=256, fft_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.frame_length = int(frame_length)
        self.frame_step = int(frame_step)
        self.fft_length = int(fft_length)

    def call(self, feats):  # [B,Frames, F*(2C)]
        B = tf.shape(feats)[0]
        Frames = tf.shape(feats)[1]

        F = self.fft_length // 2 + 1
        feats = tf.reshape(feats, [B, Frames, F, 2 * self.channels])  # [B,Frames,F,2C]

        outs = []
        for c in range(self.channels):
            re = feats[:, :, :, 2*c + 0]
            im = feats[:, :, :, 2*c + 1]
            X = tf.complex(re, im)  # [B,Frames,F]
            x = tf.signal.inverse_stft(
                X,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                window_fn=tf.signal.hann_window
            )  # [B,T]
            outs.append(x)

        y = tf.stack(outs, axis=-1)  # [B,T,C]
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(channels=self.channels, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length))
        return cfg

def build_stft_unet(time_length, channels, frame_length=1024, frame_step=256, fft_length=1024,
                    unet_depth=4, base_filters=64):
    inp = layers.Input(shape=(channels, time_length), name="mix")  # [B,C,T]
    x = layers.Permute((2, 1), name="to_time_channels")(inp)       # [B,T,C]

    stft = STFTFrontend(frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, name="stft_frontend")
    feats = stft(x)  # [B,Frames, F*(2C)]

    feats_hat = unet_1d(feats, depth=unet_depth, base_filters=base_filters, k=9)  # same 1D U-Net

    istft = ISTFTBackend(channels=channels, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length,
                         name="istft_backend")
    recon = istft(feats_hat)                # [B,T',C]
    recon = MatchTimeLen(name="match_out_len")([recon, x])  # match original T
    y = layers.Permute((2, 1), name="to_channels_time")(recon)      # [B,C,T]
    return Model(inp, y, name="STFT_UNet")


# ============================================================
# Training entry (single-stage; same loop, same callbacks)
# ============================================================
def train_ablation(
    x_mix, y_target,
    T, C,
    frontend="wave",
    out_dir="./runs_ablation",
    val_ratio=0.15,
    epochs=300,
    batch_size=2,
    pit=False,
    lr=1e-4,
    clipnorm=0.25,
    early_patience=20,
    min_delta=1e-4,
    hf_lambda=0.0,
    # fixed dwt params
    levels=2,
    filter_length=101,
    fixed_wavelet="db2",
    # stft params
    frame_length=1024,
    frame_step=256,
    fft_length=1024,
):
    os.makedirs(out_dir, exist_ok=True)
    frontend_tag = frontend if frontend != "dwt_fixed" else f"{frontend}_{fixed_wavelet}"
    run_name = time.strftime("%Y%m%d_%H%M%S") + f"_{frontend_tag}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    x_tr, y_tr, x_va, y_va = split_train_val(x_mix, y_target, val_ratio=val_ratio, seed=1337)

    # alignment for cropping:
    # - wave: 1
    # - fixed_dwt: 2**levels
    # - stft: frame_step (safe) but we still crop on waveform; use lcm-ish => use frame_step
    if frontend == "dwt_fixed":
        align = 2 ** int(levels)
    elif frontend == "stft":
        align = int(frame_step)
    else:
        align = 1

    tr_ds = make_ds(x_tr, y_tr, batch_size=batch_size, shuffle=True, crop_len=T, aligned_to=align, seed=1337)
    va_ds = make_ds(x_va, y_va, batch_size=batch_size, shuffle=False, crop_len=None, aligned_to=align, seed=1337)

    # build base model
    if frontend == "wave":
        base = build_wave_unet(time_length=None, channels=C)
    elif frontend == "dwt_fixed":
        base = build_fixed_dwt_unet(time_length=None, channels=C, levels=levels, filter_length=filter_length, fixed_wavelet=fixed_wavelet)
    elif frontend == "stft":
        base = build_stft_unet(time_length=None, channels=C,
                               frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    else:
        raise ValueError("frontend must be one of: wave, dwt_fixed, stft")

    trainer = TwoStageTrainer(
        base_model=base,
        task_loss_fn=make_task_loss(pit=pit),
        hf_lambda=hf_lambda,
        task_lambda=1.0,
        name=f"Trainer_{frontend}",
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    trainer.compile(optimizer=opt, jit_compile=False)

    best = os.path.join(run_dir, "best.keras")
    last = os.path.join(run_dir, "last.keras")

    callbacks = [
        SaveBestBase(trainer, best, monitor="val_loss", mode="min", verbose=1),
        SaveLastBase(trainer, last, verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "tb")),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=early_patience,
                                         min_delta=min_delta, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    cfg = dict(
        frontend=frontend,
        val_ratio=val_ratio,
        epochs=epochs,
        batch_size=batch_size,
        pit=pit,
        lr=lr,
        clipnorm=clipnorm,
        hf_lambda=hf_lambda,
        levels=levels,
        filter_length=filter_length,
        fixed_wavelet=fixed_wavelet,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        crop_len=T,
        align=align,
    )
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[Split] train={len(x_tr)}  val={len(x_va)}  (val_ratio={val_ratio})")
    print(f"[Run dir] {run_dir}")
    print(f"[Frontend] {frontend}")
    if frontend == "dwt_fixed":
        print(f"[Fixed wavelet] {fixed_wavelet}")
    history = trainer.fit(tr_ds, validation_data=va_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    print(f"\nSaved best model: {best}")
    return trainer, history, run_dir


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")
    ap.add_argument("--out_root", type=str, default="./runs_ablation")
    ap.add_argument("--frontend", type=str, default="wave", choices=["wave", "dwt_fixed", "stft"])
    ap.add_argument("--fixed_wavelet", type=str, default="db2",
                    choices=sorted(_FIXED_WAVELET_DEC_LO.keys()),
                    help="Fixed DWT wavelet family for --frontend dwt_fixed.")
    ap.add_argument("--full_t", type=int, default=220448)
    ap.add_argument("--train_t", type=int, default=32768)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clipnorm", type=float, default=0.25)
    ap.add_argument("--hf_lambda", type=float, default=0.0)

    # fixed dwt
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=101)

    # stft
    ap.add_argument("--frame_length", type=int, default=1024)
    ap.add_argument("--frame_step", type=int, default=256)
    ap.add_argument("--fft_length", type=int, default=1024)

    args = ap.parse_args()

    print("Loading Files...")
    X = np.load(os.path.join(args.dpath, "Xtrain.npy")).astype(np.float32)  # (N,C,T)
    Y = np.load(os.path.join(args.dpath, "Ytrain.npy")).astype(np.float32)

    X = X[:, :, :args.full_t]
    Y = Y[:, :, :args.full_t]

    C = X.shape[1]
    print("Final X shape:", X.shape)
    print("Final Y shape:", Y.shape)
    print("Training crop length TRAIN_T:", args.train_t)

    trainer, hist, run_dir = train_ablation(
        X, Y,
        T=args.train_t, C=C,
        frontend=args.frontend,
        out_dir=args.out_root,
        val_ratio=0.15,
        epochs=args.epochs,
        batch_size=args.batch,
        pit=bool(args.pit),
        lr=args.lr,
        clipnorm=args.clipnorm,
        early_patience=20,
        min_delta=1e-4,
        hf_lambda=args.hf_lambda,
        levels=args.levels,
        filter_length=args.filter_length,
        fixed_wavelet=args.fixed_wavelet,
        frame_length=args.frame_length,
        frame_step=args.frame_step,
        fft_length=args.fft_length,
    )

    print("Done.")