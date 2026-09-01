#!/usr/bin/env python3
"""Phase-1 polarity diagnosis and Phase-4 measured-RIR assignment analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal as sp
import soundfile as sf
import tensorflow as tf


ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
REV = ROOT / "revision_experiments"
WORK = Path("/home/rrame12/Desktop/Research/dbss/asa")
STEMS = ["Vocal", "Bass", "Drums"]
STEM_FILES = ["vocals", "bass", "drums"]
EPS = 1e-12

X_TEST = ROOT / "Edgecase_active_matched" / "Xedge_m9db.npy"
Y_TEST = ROOT / "Edgecase_active_matched" / "Yedge_m9db.npy"
PRED = {
    "Waveform": ROOT / "runs_ablation_finetune_edge_m9db_disjoint_small" / "20260705_074511_waveform" / "Ypred_edge_m9db.npy",
    "Fixed-db4": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4" / "Ypred_edge_m9db.npy",
    "LDWT": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "predictions_edge_m9db" / "Ypred_edge_m9db.npy",
}
LDWT_CKPT = ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "best.keras"
RIR_DIR = ROOT / "measured_rir_synth_noise" / "rir_wavs"
MEASURED_DIAG = ROOT / "measured_rir_synth_matched" / "diagnostics" / "measured_rir_component_diagnostics.csv"


FIXED_WAVELET_DEC_LO = {
    "db4": [
        -0.010597401785069032, 0.0328830116668852,
        0.030841381835560764, -0.18703481171888114,
        -0.027983769416859854, 0.6308807679298587,
        0.7148465705529154, 0.23037781330885523,
    ],
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def out_dir(name: str) -> Path:
    p = REV / f"{timestamp()}_{name}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def qmf(h0: np.ndarray) -> np.ndarray:
    alt = np.ones(len(h0), dtype=np.float64)
    alt[1::2] = -1.0
    return alt * h0[::-1]


def fixed_db4_h0(length: int = 101) -> np.ndarray:
    h = np.asarray(FIXED_WAVELET_DEC_LO["db4"], dtype=np.float64)
    pad = length - len(h)
    h = np.pad(h, (pad // 2, pad - pad // 2))
    return h / (np.linalg.norm(h) + EPS)


def ldwt_h0s(ckpt: Path = LDWT_CKPT) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="ldwt_") as td:
        with zipfile.ZipFile(ckpt, "r") as zf:
            zf.extract("model.weights.h5", td)
        vals = []
        with h5py.File(Path(td) / "model.weights.h5", "r") as h5:
            def visit(name, obj):
                if hasattr(obj, "shape") and name.startswith("layers/prdwt1d") and name.endswith("/vars/0"):
                    vals.append((name, np.asarray(obj, dtype=np.float64)))
            h5.visititems(visit)
    h0s = []
    for _, h in sorted(vals):
        h0s.append(h / (np.linalg.norm(h) + EPS))
    return h0s


def diag_filter(h: np.ndarray, channels: int, dtype=tf.float32):
    ht = tf.reshape(tf.convert_to_tensor(h, dtype=dtype), [-1, 1, 1])
    ht = tf.tile(ht, [1, channels, 1])
    eye = tf.reshape(tf.eye(channels, dtype=dtype), [1, channels, channels])
    return ht * eye


def fb_reconstruct(x_bct: np.ndarray, h0s: list[np.ndarray]) -> np.ndarray:
    x = tf.transpose(tf.convert_to_tensor(x_bct, dtype=tf.float32), [0, 2, 1])
    orig_t = tf.shape(x)[1]
    channels = int(x.shape[-1])
    approx = x
    details = []
    for h0 in h0s:
        h1 = qmf(h0)
        a = tf.nn.conv1d(approx, diag_filter(h0, channels), stride=2, padding="SAME")
        d = tf.nn.conv1d(approx, diag_filter(h1, channels), stride=2, padding="SAME")
        details.append((d, h0, h1))
        approx = a
    recon = approx
    for d, h0, h1 in reversed(details):
        g0 = h0[::-1]
        g1 = h1[::-1]
        b = tf.shape(recon)[0]
        t2 = tf.shape(recon)[1]
        c = tf.shape(recon)[2]
        def up(z):
            z = tf.reshape(z, [b, t2, 1, c])
            z0 = tf.zeros_like(z)
            return tf.reshape(tf.concat([z, z0], axis=2), [b, t2 * 2, c])
        rec_up = up(recon)
        d_up = up(d)
        recon = tf.nn.conv1d(rec_up, diag_filter(g0, channels), stride=1, padding="SAME")
        recon = recon + tf.nn.conv1d(d_up, diag_filter(g1, channels), stride=1, padding="SAME")
    recon = recon[:, :orig_t, :]
    return tf.transpose(recon, [0, 2, 1]).numpy()


def corr_alpha(yhat: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    y0 = np.asarray(y, dtype=np.float64) - np.mean(y)
    x0 = np.asarray(yhat, dtype=np.float64) - np.mean(yhat)
    dot = float(np.sum(x0 * y0))
    ey = float(np.sum(y0 * y0))
    ex = float(np.sum(x0 * x0))
    rho = dot / math.sqrt((ey + EPS) * (ex + EPS))
    alpha = dot / (ey + EPS)
    alpha_sisdr = dot / (ey + 1e-8)
    return rho, alpha, alpha_sisdr


def sdr_plus(yhat: np.ndarray, y: np.ndarray) -> float:
    y0 = np.asarray(y, dtype=np.float64) - np.mean(y)
    x0 = np.asarray(yhat, dtype=np.float64) - np.mean(yhat)
    alpha = max(float(np.sum(x0 * y0) / (np.sum(y0 * y0) + EPS)), 0.0)
    target = alpha * y0
    err = x0 - target
    return float(10.0 * np.log10((np.sum(target * target) + EPS) / (np.sum(err * err) + EPS)))


def permute_by_corr(ytrue: np.ndarray, ypred: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    best = None
    for perm in itertools.permutations(range(ytrue.shape[0])):
        score = 0.0
        for ch in range(ytrue.shape[0]):
            rho, _, _ = corr_alpha(ypred[perm[ch]], ytrue[ch])
            score += abs(rho)
        if best is None or score > best[0]:
            best = (score, perm)
    return ypred[list(best[1])], best[1]


def polarity_diagnosis(args) -> None:
    out = out_dir("polarity_diagnosis")
    X = np.load(X_TEST, mmap_mode="r")
    Y = np.load(Y_TEST, mmap_mode="r")
    n = min(args.max_items, X.shape[0]) if args.max_items else X.shape[0]
    t = min(args.eval_t, X.shape[-1], Y.shape[-1])
    Xn = np.asarray(X[:n, :, :t], dtype=np.float32)
    Yn = np.asarray(Y[:n, :, :t], dtype=np.float32)

    recon = {
        "Waveform": Xn.copy(),
        "Fixed-db4": fb_reconstruct(Xn, [fixed_db4_h0(), fixed_db4_h0()]),
        "LDWT": fb_reconstruct(Xn, ldwt_h0s()),
    }
    rows = []
    for method in ["Waveform", "Fixed-db4", "LDWT"]:
        pred = np.asarray(np.load(PRED[method], mmap_mode="r")[:n, :, :t], dtype=np.float32)
        for i in range(n):
            pred_i, perm = permute_by_corr(Yn[i], pred[i])
            for ch, stem in enumerate(STEMS):
                rho_rec, alpha_rec, alpha_sisdr_rec = corr_alpha(recon[method][i, ch], Xn[i, ch])
                rho_out, alpha_out, alpha_sisdr_out = corr_alpha(pred_i[ch], Yn[i, ch])
                rows.append({
                    "method": method,
                    "idx": i,
                    "output_ch": ch,
                    "stem": stem,
                    "perm": ",".join(map(str, perm)),
                    "frontend_recon_rho_vs_input": rho_rec,
                    "frontend_recon_alpha_vs_input": alpha_rec,
                    "frontend_recon_sisdr_alpha_sign": np.sign(alpha_sisdr_rec),
                    "output_rho_vs_target": rho_out,
                    "output_alpha_vs_target": alpha_out,
                    "output_sisdr_alpha_sign": np.sign(alpha_sisdr_out),
                    "polarity_error": bool(rho_out < 0.0),
                    "sdr_plus": sdr_plus(pred_i[ch], Yn[i, ch]),
                })
    write_csv(out / "polarity_diagnosis_per_output.csv", rows)

    summary = []
    for method in ["Waveform", "Fixed-db4", "LDWT"]:
        for ch, stem in enumerate(STEMS):
            rr = [r for r in rows if r["method"] == method and r["output_ch"] == ch]
            rhos = np.asarray([r["output_rho_vs_target"] for r in rr], dtype=np.float64)
            rec_rhos = np.asarray([r["frontend_recon_rho_vs_input"] for r in rr], dtype=np.float64)
            summary.append({
                "method": method,
                "output_ch": ch,
                "stem": stem,
                "median_frontend_recon_rho": float(np.median(rec_rhos)),
                "min_frontend_recon_rho": float(np.min(rec_rhos)),
                "median_output_rho": float(np.median(rhos)),
                "iqr_output_rho": float(np.percentile(rhos, 75) - np.percentile(rhos, 25)),
                "per_pct": float(100.0 * np.mean(rhos < 0.0)),
                "median_output_alpha": float(np.median([r["output_alpha_vs_target"] for r in rr])),
                "median_sdr_plus": float(np.median([r["sdr_plus"] for r in rr])),
            })
    write_csv(out / "polarity_diagnosis_by_channel.csv", summary)

    # Polarity-preserving loss implementation for controlled runners.
    loss_note = {
        "finding": "Training loss is ordinary SI-SDR; projection scale is unconstrained, so sign inversion is a loss ambiguity rather than a frontend reconstruction bug if frontend rho is positive.",
        "positive_scale_sisdr": "alpha_plus=max(<ypred,ytrue>/(||ytrue||^2+eps), eps); target=alpha_plus*ytrue; loss=-mean(10log10(||target||^2/||ypred-target||^2))",
    }
    (out / "loss_diagnosis.json").write_text(json.dumps(loss_note, indent=2))
    print(f"[Saved] {out}")
    for row in summary:
        print(row)


def read_rir_energy_matrix() -> np.ndarray:
    E = np.zeros((3, 3), dtype=np.float64)
    for si, src in enumerate(STEM_FILES):
        for mic in range(3):
            x, sr = sf.read(RIR_DIR / f"{src}_mic{mic}_ir.wav", dtype="float64")
            if x.ndim == 2:
                x = np.mean(x, axis=1)
            E[si, mic] = float(np.sum(x * x))
    return E


def assignment_stats(E: np.ndarray, assignment: list[int]) -> dict:
    ratios = []
    positive = 0
    for si, mic in enumerate(assignment):
        target = E[si, mic]
        bleed = float(np.sum(E[:, mic]) - target)
        r = 10.0 * np.log10((target + EPS) / (bleed + EPS))
        ratios.append(r)
        positive += int(r > 0)
    ratios = np.asarray(ratios)
    return {
        "assignment": assignment,
        "mean_tbr_db": float(np.mean(ratios)),
        "median_tbr_db": float(np.median(ratios)),
        "min_tbr_db": float(np.min(ratios)),
        "max_tbr_db": float(np.max(ratios)),
        "positive_fraction": float(positive / len(assignment)),
        "per_source_tbr_db": ratios.tolist(),
    }


def measured_assignment(args) -> None:
    out = out_dir("measured_rir_assignment")
    E = read_rir_energy_matrix()
    current = [0, 1, 2]
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-np.log(E + EPS))
    optimal = [None] * 3
    for si, mic in zip(row_ind, col_ind):
        optimal[int(si)] = int(mic)
    independent = np.argmax(E, axis=1).astype(int).tolist()
    rows = []
    for si, src in enumerate(STEM_FILES):
        row = {"source": src}
        for mic in range(3):
            row[f"mic{mic}_energy"] = E[si, mic]
        row["current_mic"] = current[si]
        row["optimal_mic"] = optimal[si]
        row["independent_preferred_mic"] = independent[si]
        rows.append(row)
    write_csv(out / "rir_energy_matrix.csv", rows)
    summary = {
        "sources": STEM_FILES,
        "energy_matrix_source_by_mic": E.tolist(),
        "current_assignment": assignment_stats(E, current),
        "optimal_one_to_one_assignment": assignment_stats(E, optimal),
        "independent_preferred_mics": independent,
        "multiple_sources_prefer_same_mic": len(set(independent)) < len(independent),
    }
    if MEASURED_DIAG.exists():
        diag_rows = list(csv.DictReader(MEASURED_DIAG.open()))
        for key in ["target_to_bleed_db", "target_to_noise_db", "sir_input_db"]:
            vals = np.asarray([float(r[key]) for r in diag_rows], dtype=np.float64)
            summary[f"generated_dataset_{key}"] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    (out / "measured_rir_assignment_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(10 * np.log10(E + EPS), cmap="magma")
    ax.set_xticks(range(3)); ax.set_xticklabels(["mic0", "mic1", "mic2"])
    ax.set_yticks(range(3)); ax.set_yticklabels(STEM_FILES)
    for si in range(3):
        for mic in range(3):
            label = f"{10*np.log10(E[si,mic]+EPS):.1f}"
            ax.text(mic, si, label, ha="center", va="center", color="white")
    ax.set_title("Measured RIR energy (dB)")
    fig.colorbar(im, ax=ax, label="10 log10 energy")
    fig.tight_layout()
    fig.savefig(out / "rir_energy_matrix.png", dpi=300)
    fig.savefig(out / "rir_energy_matrix.pdf")
    plt.close(fig)
    print(f"[Saved] {out}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("polarity")
    p.add_argument("--eval_t", type=int, default=220448)
    p.add_argument("--max_items", type=int, default=100)
    sub.add_parser("assignment")
    args = ap.parse_args()
    if args.cmd == "polarity":
        polarity_diagnosis(args)
    elif args.cmd == "assignment":
        measured_assignment(args)


if __name__ == "__main__":
    main()
