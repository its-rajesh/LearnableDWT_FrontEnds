#!/usr/bin/env python3
"""
run_pr_grid_eval_with_metrics.py

Evaluate PR-DWT ablation grid:
  Levels = {2,3,5}
  Filter lengths = {11,101,1001}

Uses:
  Xtest.npy, Ytest.npy from /home/rrame12/Desktop/Research/DWT_IR
  model folders from /home/rrame12/Desktop/Research/DWT_IR/runs_pr_ablation

For each tag L{level}_F{filter}:
  - find run dir
  - load best.keras
  - predict Ypred.npy
  - run eval_all_metrics_single.py
  - collect summary metrics:
      Reference, SISDR, SIR, SAR, SIR(B), EXP
  - save combined CSV

Important:
  - crops X/Y to fixed length 220448 before prediction and evaluation
  - saves cropped X/Y inside each result folder
  - skips already-computed runs
"""

import sys
import csv
import json
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------
# Paths / config
# ------------------------------------------------------------

ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
RUNS_ROOT = ROOT / "runs_pr_ablation"
OUT_ROOT = RUNS_ROOT / "results_grid_eval"

X_PATH = ROOT / "Xtest.npy"
Y_PATH = ROOT / "Ytest.npy"

EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_all_metrics_single.py"

LEVELS = [2, 3, 5]
FILTERS = [11, 101, 1001]

FIXED_T = 220448
BATCH_SIZE = 2


# ------------------------------------------------------------
# Import model components
# ------------------------------------------------------------

sys.path.insert(0, str(ROOT))
from iprdwt import (
    PRDWT1D,
    PRIDWT1D,
    MatchTimeLen,
    SplitChannels,
    UpsampleTo,
    build_pr_dwt_unet,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_run_in_dir(group_dir: Path):
    cand = [d for d in group_dir.iterdir() if d.is_dir()]
    if not cand:
        return None
    cand = sorted(cand, key=lambda d: d.stat().st_mtime)
    return cand[-1]


def read_train_config(run_dir: Path):
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    needed = ["levels", "filter_length", "pr_shifts"]
    for k in needed:
        if k not in cfg:
            raise KeyError(f"Missing key '{k}' in {cfg_path}")
    return cfg


def relink_pridwt_to_prdwt(model, levels):
    for i in range(levels):
        dwt = model.get_layer(f"prdwt_{i}")
        idwt = model.get_layer(f"pridwt_{i}")
        if isinstance(idwt, PRIDWT1D):
            idwt.set_dwt(dwt)


def build_base_model(T, C, cfg):
    base = build_pr_dwt_unet(
        time_length=T,
        channels=C,
        levels=int(cfg["levels"]),
        filter_length=int(cfg["filter_length"]),
        pr_shifts=int(cfg["pr_shifts"]),
        pr_lambda=float(cfg.get("pr_lambda_B", cfg.get("pr_lambda", 1e-2))),
        pr_dc_lambda=float(cfg.get("pr_dc_lambda", 1.0)),
        pr_nyq_lambda=float(cfg.get("pr_nyq_lambda", 1.0)),
        unet_depth=int(cfg.get("unet_depth", 4)),
        base_filters=int(cfg.get("base_filters", 64)),
        return_taps=False,
    )
    _ = base(tf.zeros((1, C, T), dtype=tf.float32), training=False)
    relink_pridwt_to_prdwt(base, int(cfg["levels"]))
    return base


def load_base_from_best_keras(best_keras_path: Path, base_model):
    """
    Load weights from a .keras archive into the already-built base model.
    """
    tmpdir = tempfile.mkdtemp(prefix="keras_extract_")
    try:
        if best_keras_path.is_dir():
            extracted_dir = best_keras_path
        else:
            with zipfile.ZipFile(best_keras_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted_dir = Path(tmpdir)

        h5_path = extracted_dir / "model.weights.h5"
        if h5_path.exists():
            base_model.load_weights(str(h5_path))
            return base_model

        ckpt_prefix = extracted_dir / "variables" / "variables"
        if Path(str(ckpt_prefix) + ".index").exists():
            ckpt = tf.train.Checkpoint(model=base_model)
            ckpt.restore(str(ckpt_prefix)).expect_partial()
            return base_model

        raise FileNotFoundError(f"Could not find weights inside {best_keras_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def predict_model(best_path: Path, run_dir: Path, X: np.ndarray, batch_size: int = 2):
    cfg = read_train_config(run_dir)
    N, C, T = X.shape

    base = build_base_model(T=T, C=C, cfg=cfg)
    base = load_base_from_best_keras(best_path, base)
    relink_pridwt_to_prdwt(base, int(cfg["levels"]))

    Ypred = np.zeros_like(X, dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        print(f"[predict] {best_path.parent.name}: {s}:{e}/{N}", flush=True)
        yp = base.predict(X[s:e], verbose=0)
        Ypred[s:e] = np.asarray(yp, dtype=np.float32)

    return Ypred, cfg


def run_eval(x_path: Path, y_path: Path, ypred_path: Path, out_dir: Path):
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--x_mix",
        str(x_path),
        "--y_true",
        str(y_path),
        "--y_pred",
        str(ypred_path),
        "--out_dir",
        str(out_dir),
        "--sr",
        "22050",
        "--mimo_K",
        "512",
        "--mimo_lam",
        "1e-3",
        "--std_align",
        "1",
        "--std_max_lag",
        "22050",
        "--permute_pred",
        "1",
        "--stem_names",
        "Vocal",
        "Bass",
        "Drums",
    ]
    print("Running:", " ".join(cmd), flush=True)
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
        "Level": "",
        "Filter": "",
        "RunDir": "",
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
        "Level",
        "Filter",
        "RunDir",
        "SISDR_mean",
        "SISDR_std",
        "SIR_mean",
        "SIR_std",
        "SAR_mean",
        "SAR_std",
        "SIRB_mean",
        "SIRB_std",
        "EXP_mean",
        "EXP_std",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    print("\n===== PR GRID SUMMARY =====")
    header = (
        f"{'Method':12s} {'Lvl':>4s} {'Filt':>6s} "
        f"{'SISDR':>18s} {'SIR':>18s} {'SAR':>18s} {'SIR(B)':>18s} {'EXP':>18s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Method']:12s} "
            f"{str(r['Level']):>4s} "
            f"{str(r['Filter']):>6s} "
            f"{r['SISDR_mean']:8.2f} ± {r['SISDR_std']:6.2f} "
            f"{r['SIR_mean']:8.2f} ± {r['SIR_std']:6.2f} "
            f"{r['SAR_mean']:8.2f} ± {r['SAR_std']:6.2f} "
            f"{r['SIRB_mean']:8.2f} ± {r['SIRB_std']:6.2f} "
            f"{r['EXP_mean']:8.2f} ± {r['EXP_std']:6.2f}"
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ensure_dir(OUT_ROOT)

    if not X_PATH.exists():
        raise FileNotFoundError(X_PATH)
    if not Y_PATH.exists():
        raise FileNotFoundError(Y_PATH)
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find eval script: {EVAL_SCRIPT}")

    X_full = np.load(X_PATH).astype(np.float32)
    Y_full = np.load(Y_PATH).astype(np.float32)

    if X_full.shape != Y_full.shape:
        raise ValueError(f"Shape mismatch: X={X_full.shape}, Y={Y_full.shape}")

    X = X_full[..., :FIXED_T]
    Y = Y_full[..., :FIXED_T]

    print(f"[INFO] Original X shape: {X_full.shape}", flush=True)
    print(f"[INFO] Original Y shape: {Y_full.shape}", flush=True)
    print(f"[INFO] Using fixed length T={FIXED_T}", flush=True)
    print(f"[INFO] Cropped X shape: {X.shape}", flush=True)
    print(f"[INFO] Cropped Y shape: {Y.shape}", flush=True)

    rows = []
    first_eval_dir = None

    for L in LEVELS:
        for F in FILTERS:
            tag = f"L{L}_F{F}"
            group_dir = RUNS_ROOT / tag

            if not group_dir.exists():
                print(f"[SKIP] missing folder: {group_dir}", flush=True)
                continue

            run_dir = latest_run_in_dir(group_dir)
            if run_dir is None:
                print(f"[SKIP] no runs inside: {group_dir}", flush=True)
                continue

            best_path = run_dir / "best.keras"
            if not best_path.exists():
                print(f"[SKIP] missing best.keras: {best_path}", flush=True)
                continue

            out_dir = ensure_dir(OUT_ROOT / tag)
            ypred_path = out_dir / "Ypred.npy"
            x_eval_path = out_dir / f"Xtest_T{FIXED_T}.npy"
            y_eval_path = out_dir / f"Ytest_T{FIXED_T}.npy"

            std_json = out_dir / "standard_metrics_summary.json"
            mimo_json = out_dir / "mimo_metrics_summary.json"

            if std_json.exists() and mimo_json.exists():
                print(f"[SKIP] already evaluated: {tag}", flush=True)
                parsed = parse_eval_outputs(out_dir)
                parsed["Method"] = tag
                parsed["Level"] = L
                parsed["Filter"] = F
                parsed["RunDir"] = str(run_dir)
                rows.append(parsed)
                if first_eval_dir is None:
                    first_eval_dir = out_dir
                continue

            print("\n" + "=" * 80)
            print(f"Evaluating {tag}")
            print(f"run_dir : {run_dir}")
            print(f"best    : {best_path}")
            print("=" * 80)

            try:
                Ypred, cfg = predict_model(best_path, run_dir, X, batch_size=BATCH_SIZE)
            except Exception as e:
                print(f"[FAIL] {tag}: prediction crashed with error:\n{e}", flush=True)
                continue

            np.save(ypred_path, Ypred)
            np.save(x_eval_path, X)
            np.save(y_eval_path, Y)

            with open(out_dir / "model_config_used.json", "w") as f:
                json.dump(cfg, f, indent=2)

            try:
                run_eval(x_eval_path, y_eval_path, ypred_path, out_dir)
            except Exception as e:
                print(f"[FAIL] {tag}: eval crashed with error:\n{e}", flush=True)
                continue

            parsed = parse_eval_outputs(out_dir)
            parsed["Method"] = tag
            parsed["Level"] = L
            parsed["Filter"] = F
            parsed["RunDir"] = str(run_dir)
            rows.append(parsed)

            if first_eval_dir is None:
                first_eval_dir = out_dir

    if not rows:
        raise RuntimeError("No successful runs found.")

    if first_eval_dir is None:
        raise RuntimeError("Could not determine reference eval directory.")

    ref = parse_reference_from_eval(first_eval_dir)
    rows = [ref] + rows

    ref_row = rows[0]
    model_rows = sorted(
        rows[1:],
        key=lambda r: (int(r["Level"]), int(r["Filter"]))
    )
    rows = [ref_row] + model_rows

    csv_path = OUT_ROOT / "pr_grid_summary_metrics.csv"
    save_summary_csv(rows, csv_path)

    json_path = OUT_ROOT / "pr_grid_summary_metrics.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print_table(rows)
    print(f"\nSaved summary CSV : {csv_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()