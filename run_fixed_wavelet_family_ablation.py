#!/usr/bin/env python3
"""Run fixed-wavelet frontend ablations with identical training settings.

This is intentionally a thin orchestrator over ablation_frontends.py so each
family uses the same backbone, optimizer, crop length, and split logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PY = Path("/home/rrame12/anaconda3/envs/all/bin/python")


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=Path, default=DEFAULT_PY)
    ap.add_argument("--dpath", type=Path, default=ROOT)
    ap.add_argument("--out_root", type=Path, default=ROOT / "runs_fixed_wavelet_family")
    ap.add_argument("--wavelets", nargs="+", default=["haar", "db2", "db4", "db8", "sym4", "coif1"])
    ap.add_argument("--full_t", type=int, default=220448)
    ap.add_argument("--train_t", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=101)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clipnorm", type=float, default=0.25)
    ap.add_argument("--hf_lambda", type=float, default=0.0)
    ap.add_argument("--pit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": str(ROOT / "ablation_frontends.py"),
        "dpath": str(args.dpath),
        "out_root": str(args.out_root),
        "wavelets": args.wavelets,
        "full_t": args.full_t,
        "train_t": args.train_t,
        "epochs": args.epochs,
        "batch": args.batch,
        "levels": args.levels,
        "filter_length": args.filter_length,
        "lr": args.lr,
        "clipnorm": args.clipnorm,
        "hf_lambda": args.hf_lambda,
        "pit": args.pit,
    }
    with (args.out_root / "fixed_wavelet_family_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    for wavelet in args.wavelets:
        cmd = [
            str(args.python), str(ROOT / "ablation_frontends.py"),
            "--dpath", str(args.dpath),
            "--frontend", "dwt_fixed",
            "--fixed_wavelet", wavelet,
            "--full_t", str(args.full_t),
            "--train_t", str(args.train_t),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--pit", str(args.pit),
            "--lr", str(args.lr),
            "--clipnorm", str(args.clipnorm),
            "--hf_lambda", str(args.hf_lambda),
            "--levels", str(args.levels),
            "--filter_length", str(args.filter_length),
            "--out_root", str(args.out_root),
        ]
        run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
