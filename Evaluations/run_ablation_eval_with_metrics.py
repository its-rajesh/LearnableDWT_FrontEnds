#!/usr/bin/env python3
"""
run_ablation_eval_with_metrics.py

Loads:
  Xtest, Ytest from /home/rrame12/Desktop/Research/DWT_IR

Loads ablation models from:
  /home/rrame12/Desktop/Research/DWT_IR/runs_ablation

Models:
  1) waveform only
  2) fixed dwt
  3) stft frontend

For each model:
  - predict Ypred
  - save Ypred.npy
  - run eval_all_metrics_single.py
  - collect summary metrics
  - save one CSV with:
      Reference, SISDR, SIR, SAR, SIR(B), EXP
"""

import os
import sys
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
RUNS = ROOT / "runs_ablation"

X_PATH = ROOT / "Xtest.npy"
Y_PATH = ROOT / "Ytest.npy"

MODELS = {
    "waveform_only": RUNS / "20260225_181051_wave" / "best.keras",
    "fixed_dwt": RUNS / "20260225_222112_dwt_fixed" / "best.keras",
    "stft_frontend": RUNS / "20260226_000354_stft" / "best.keras",
}

OUT_ROOT = RUNS / "results_eval_all"
EVAL_SCRIPT = Path("eval_all_metrics_single.py")  # change if needed


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_custom_objects():
    sys.path.insert(0, str(ROOT))
    from ablation_frontends import (
        MatchTimeLen, SplitChannels, UpsampleTo,
        FixedDWT1D, FixedIDWT1D,
        STFTFrontend, ISTFTBackend,
    )
    return {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "FixedDWT1D": FixedDWT1D,
        "FixedIDWT1D": FixedIDWT1D,
        "STFTFrontend": STFTFrontend,
        "ISTFTBackend": ISTFTBackend,
    }


def predict_model(model_path: Path, X: np.ndarray, batch_size: int = 2) -> np.ndarray:
    custom_objects = load_custom_objects()
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
    )

    Ypred = np.zeros_like(X, dtype=np.float32)
    N = X.shape[0]

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        xb = X[s:e]
        yp = model.predict(xb, verbose=0)
        Ypred[s:e] = np.asarray(yp, dtype=np.float32)

    return Ypred


def run_eval(x_path: Path, y_path: Path, ypred_path: Path, out_dir: Path):
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--x_mix", str(x_path),
        "--y_true", str(y_path),
        "--y_pred", str(ypred_path),
        "--out_dir", str(out_dir),
        "--sr", "22050",
        "--mimo_K", "512",
        "--mimo_lam", "1e-3",
        "--std_align", "1",
        "--std_max_lag", "22050",
        "--permute_pred", "1",
        "--stem_names", "Vocal", "Bass", "Drums",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def mean_of_stems(d: dict):
    means = [float(v["mean"]) for v in d.values()]
    stds = [float(v["std"]) for v in d.values()]
    return float(np.mean(means)), float(np.mean(stds))


def parse_eval_outputs(out_dir: Path):
    with open(out_dir / "standard_metrics_summary.json", "r") as f:
        std = json.load(f)
    with open(out_dir / "mimo_metrics_summary.json", "r") as f:
        mimo = json.load(f)

    sirb_mean, sirb_std = mean_of_stems(mimo["output_after"]["SIRB"])
    exp_mean, exp_std = mean_of_stems(mimo["output_after"]["EXP"])

    return {
        "SISDR_mean": float(std["sisdr"]["mean"]),
        "SISDR_std": float(std["sisdr"]["std"]),
        "SIR_mean": float(std["sir"]["mean"]),
        "SIR_std": float(std["sir"]["std"]),
        "SAR_mean": float(std["sar"]["mean"]),
        "SAR_std": float(std["sar"]["std"]),
        "SIRB_mean": sirb_mean,
        "SIRB_std": sirb_std,
        "EXP_mean": exp_mean,
        "EXP_std": exp_std,
    }


def parse_reference_from_eval(out_dir: Path):
    with open(out_dir / "standard_metrics_summary.json", "r") as f:
        std = json.load(f)
    with open(out_dir / "mimo_metrics_summary.json", "r") as f:
        mimo = json.load(f)

    sirb_mean, sirb_std = mean_of_stems(mimo["mixture_before"]["SIRB"])
    exp_mean, exp_std = mean_of_stems(mimo["mixture_before"]["EXP"])

    return {
        "Method": "Reference",
        "SISDR_mean": float(std["sisdr_in"]["mean"]),
        "SISDR_std": float(std["sisdr_in"]["std"]),
        "SIR_mean": float(std["sir_in"]["mean"]),
        "SIR_std": float(std["sir_in"]["std"]),
        "SAR_mean": float(std["sar_in"]["mean"]),
        "SAR_std": float(std["sar_in"]["std"]),
        "SIRB_mean": sirb_mean,
        "SIRB_std": sirb_std,
        "EXP_mean": exp_mean,
        "EXP_std": exp_std,
    }


def save_summary_csv(rows, csv_path: Path):
    fields = [
        "Method",
        "SISDR_mean", "SISDR_std",
        "SIR_mean", "SIR_std",
        "SAR_mean", "SAR_std",
        "SIRB_mean", "SIRB_std",
        "EXP_mean", "EXP_std",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    print("\n===== ABLATION SUMMARY =====")
    header = f"{'Method':18s} {'SISDR':>18s} {'SIR':>18s} {'SAR':>18s} {'SIR(B)':>18s} {'EXP':>18s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Method']:18s} "
            f"{r['SISDR_mean']:8.2f} ± {r['SISDR_std']:6.2f} "
            f"{r['SIR_mean']:8.2f} ± {r['SIR_std']:6.2f} "
            f"{r['SAR_mean']:8.2f} ± {r['SAR_std']:6.2f} "
            f"{r['SIRB_mean']:8.2f} ± {r['SIRB_std']:6.2f} "
            f"{r['EXP_mean']:8.2f} ± {r['EXP_std']:6.2f}"
        )


def main():
    ensure_dir(OUT_ROOT)

    if not X_PATH.exists():
        raise FileNotFoundError(X_PATH)
    if not Y_PATH.exists():
        raise FileNotFoundError(Y_PATH)
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find {EVAL_SCRIPT}. Put eval_all_metrics_single.py in the current folder "
            "or change EVAL_SCRIPT path inside this script."
        )

    X = np.load(X_PATH).astype(np.float32)
    _ = np.load(Y_PATH).astype(np.float32)  # just to verify it exists / matches eval script usage

    rows = []
    first_eval_dir = None

    for method, ckpt in MODELS.items():
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

        out_dir = ensure_dir(OUT_ROOT / method)
        ypred_path = out_dir / "Ypred.npy"

        print(f"\n===== {method} =====")
        print(f"Loading checkpoint: {ckpt}")
        Ypred = predict_model(ckpt, X, batch_size=2)
        np.save(ypred_path, Ypred)
        print(f"Saved predictions: {ypred_path}")

        run_eval(X_PATH, Y_PATH, ypred_path, out_dir)

        parsed = parse_eval_outputs(out_dir)
        parsed["Method"] = method
        rows.append(parsed)

        if first_eval_dir is None:
            first_eval_dir = out_dir

    ref = parse_reference_from_eval(first_eval_dir)
    rows = [ref] + rows

    csv_path = OUT_ROOT / "ablation_summary_metrics.csv"
    save_summary_csv(rows, csv_path)
    print_table(rows)
    print(f"\nSaved summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()