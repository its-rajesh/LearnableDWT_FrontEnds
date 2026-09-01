#!/usr/bin/env python3
"""Evaluate fixed-wavelet family ablations with corrected EXP.

Expected checkpoints come from run_fixed_wavelet_family_ablation.py and are named
like YYYYmmdd_HHMMSS_dwt_fixed_db4/best.keras.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
EVAL_SCRIPT = ROOT / "Evaluations" / "eval_all_metrics_single.py"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_run_dir_for_wavelet(runs_root: Path, wavelet: str) -> Path | None:
    candidates = sorted(glob.glob(str(runs_root / f"*_dwt_fixed_{wavelet}")))
    candidates = [Path(c) for c in candidates if (Path(c) / "best.keras").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: d.stat().st_mtime)[-1]


def load_custom_objects():
    sys.path.insert(0, str(ROOT))
    from ablation_frontends import (  # noqa: E402
        FixedDWT1D,
        FixedIDWT1D,
        ISTFTBackend,
        MatchTimeLen,
        SplitChannels,
        STFTFrontend,
        UpsampleTo,
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


def predict_model(model_path: Path, x: np.ndarray, batch_size: int) -> np.ndarray:
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=load_custom_objects(),
        compile=False,
    )
    ypred = np.zeros_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(len(x), start + batch_size)
        ypred[start:end] = np.asarray(model.predict(x[start:end], verbose=0), dtype=np.float32)
    return ypred


def run_eval(x_path: Path, y_path: Path, ypred_path: Path, out_dir: Path, args) -> None:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--x_mix", str(x_path),
        "--y_true", str(y_path),
        "--y_pred", str(ypred_path),
        "--out_dir", str(out_dir),
        "--sr", str(args.sr),
        "--mimo_K", str(args.mimo_K),
        "--mimo_lam", str(args.mimo_lam),
        "--std_align", str(args.std_align),
        "--std_max_lag", str(args.std_max_lag),
        "--permute_pred", str(args.permute_pred),
        "--stem_names", "Vocal", "Bass", "Drums",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT / "Evaluations"))


def mean_of_stems(block: dict) -> tuple[float, float]:
    means = [float(v["mean"]) for v in block.values()]
    stds = [float(v["std"]) for v in block.values()]
    return float(np.mean(means)), float(np.mean(stds))


def parse_eval(out_dir: Path, section: str) -> dict:
    with (out_dir / "standard_metrics_summary.json").open() as f:
        std = json.load(f)
    with (out_dir / "mimo_metrics_summary.json").open() as f:
        mimo = json.load(f)

    if section == "mixture_before":
        keys = {"sisdr": "sisdr_in", "sir": "sir_in", "sar": "sar_in"}
    else:
        keys = {"sisdr": "sisdr", "sir": "sir", "sar": "sar"}

    sirb_mean, sirb_std = mean_of_stems(mimo[section]["SIRB"])
    exp_mean, exp_std = mean_of_stems(mimo[section]["EXP"])
    exp0_mean, exp0_std = mean_of_stems(mimo[section]["EXP0"])
    exp_legacy_mean, exp_legacy_std = mean_of_stems(mimo[section]["EXP_legacy"])

    return {
        "SI-SDR": float(std[keys["sisdr"]]["mean"]),
        "SI-SDR_std": float(std[keys["sisdr"]]["std"]),
        "SIR": float(std[keys["sir"]]["mean"]),
        "SIR_std": float(std[keys["sir"]]["std"]),
        "SAR": float(std[keys["sar"]]["mean"]),
        "SAR_std": float(std[keys["sar"]]["std"]),
        "SIR(B)": sirb_mean,
        "SIR(B)_std": sirb_std,
        "EXP": exp_mean,
        "EXP_std": exp_std,
        "EXP0": exp0_mean,
        "EXP0_std": exp0_std,
        "EXP_legacy": exp_legacy_mean,
        "EXP_legacy_std": exp_legacy_std,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "Method", "run_dir",
        "SI-SDR", "SI-SDR_std", "SIR", "SIR_std", "SAR", "SAR_std",
        "SIR(B)", "SIR(B)_std", "EXP", "EXP_std", "EXP0", "EXP0_std",
        "EXP_legacy", "EXP_legacy_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    print("\n===== FIXED WAVELET FAMILY SUMMARY =====")
    print(f"{'Method':12s} {'SI-SDR':>8s} {'SIR':>8s} {'SAR':>8s} {'SIR(B)':>8s} {'EXP':>8s} {'EXPleg':>8s}")
    for r in rows:
        print(
            f"{r['Method']:12s} {r['SI-SDR']:8.2f} {r['SIR']:8.2f} {r['SAR']:8.2f} "
            f"{r['SIR(B)']:8.2f} {r['EXP']:8.2f} {r['EXP_legacy']:8.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, default=ROOT / "runs_fixed_wavelet_family")
    ap.add_argument("--out_root", type=Path, default=ROOT / "runs_fixed_wavelet_family" / "results_eval_corrected_exp")
    ap.add_argument("--x", type=Path, default=ROOT / "Xtest.npy")
    ap.add_argument("--y", type=Path, default=ROOT / "Ytest.npy")
    ap.add_argument("--wavelets", nargs="+", default=["haar", "db2", "db4", "db8", "sym4", "coif1"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--mimo_K", type=int, default=512)
    ap.add_argument("--mimo_lam", type=float, default=1e-3)
    ap.add_argument("--std_align", type=int, default=1)
    ap.add_argument("--std_max_lag", type=int, default=22050)
    ap.add_argument("--permute_pred", type=int, default=1)
    ap.add_argument("--max_items", type=int, default=-1)
    args = ap.parse_args()

    ensure_dir(args.out_root)
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(EVAL_SCRIPT)
    x = np.load(args.x).astype(np.float32)
    if args.max_items > 0:
        x = x[:args.max_items]
        y_tmp = np.load(args.y).astype(np.float32)[:args.max_items]
        y_eval = args.out_root / f"Ytest_first{args.max_items}.npy"
        x_eval = args.out_root / f"Xtest_first{args.max_items}.npy"
        np.save(x_eval, x)
        np.save(y_eval, y_tmp)
    else:
        x_eval = args.x
        y_eval = args.y

    rows = []
    reference_added = False
    for wavelet in args.wavelets:
        run_dir = latest_run_dir_for_wavelet(args.runs_root, wavelet)
        if run_dir is None:
            print(f"[WARN] no run found for {wavelet} under {args.runs_root}", flush=True)
            continue
        ckpt = run_dir / "best.keras"
        out_dir = ensure_dir(args.out_root / wavelet)
        ypred_path = out_dir / "Ypred.npy"

        print(f"\n===== {wavelet} =====")
        print(f"Checkpoint: {ckpt}")
        ypred = predict_model(ckpt, x, batch_size=args.batch)
        np.save(ypred_path, ypred)
        run_eval(x_eval, y_eval, ypred_path, out_dir, args)

        if not reference_added:
            ref = parse_eval(out_dir, "mixture_before")
            ref["Method"] = "Reference"
            ref["run_dir"] = ""
            rows.append(ref)
            reference_added = True

        row = parse_eval(out_dir, "output_after")
        row["Method"] = wavelet
        row["run_dir"] = str(run_dir)
        rows.append(row)

    csv_path = args.out_root / "fixed_wavelet_family_summary_corrected_exp.csv"
    write_csv(rows, csv_path)
    print_table(rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
