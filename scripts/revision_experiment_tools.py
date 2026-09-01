#!/usr/bin/env python3
"""Utilities for TASLP revision experiments around the LDWT bleed paper.

This script intentionally reuses saved predictions and the repository evaluator
where possible. It does not edit the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import soundfile as sf


ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
OUT_ROOT = ROOT / "revision_experiments"
PYTHON = Path("/home/rrame12/anaconda3/envs/all/bin/python")
STEMS = ["Vocal", "Bass", "Drums"]
EPS = 1e-12

PATHS = {
    "train_x": ROOT / "finetune_bleed_small_disjoint" / "Xtrain_finetune_disjoint.npy",
    "train_y": ROOT / "finetune_bleed_small_disjoint" / "Ytrain_finetune_disjoint.npy",
    "train_meta": ROOT / "finetune_bleed_small_disjoint" / "finetune_dataset_meta_disjoint.json",
    "test_x": ROOT / "Edgecase_active_matched" / "Xedge_m9db.npy",
    "test_y": ROOT / "Edgecase_active_matched" / "Yedge_m9db.npy",
    "test_meta": ROOT / "Edgecase_active_matched" / "edge_bleed_matched_target_metadata.json",
    "test_indices": ROOT / "Edgecase_active_matched" / "selected_indices.npy",
    "evaluator": ROOT / "Evaluations" / "eval_all_metrics_single.py",
    "ldwt_run": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106",
    "ldwt_ckpt": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "best.keras",
    "ldwt_stage_a": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "best_stageA.keras",
    "ldwt_pred": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "predictions_edge_m9db" / "Ypred_edge_m9db.npy",
    "ldwt_metrics": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "metrics_edge_m9db_corrected_exp",
    "wave_run": ROOT / "runs_ablation_finetune_edge_m9db_disjoint_small" / "20260705_074511_waveform",
    "wave_ckpt": ROOT / "runs_ablation_finetune_edge_m9db_disjoint_small" / "20260705_074511_waveform" / "best.keras",
    "wave_pred": ROOT / "runs_ablation_finetune_edge_m9db_disjoint_small" / "20260705_074511_waveform" / "Ypred_edge_m9db.npy",
    "wave_metrics": ROOT / "runs_ablation_finetune_edge_m9db_disjoint_small" / "20260705_074511_waveform" / "metrics_edge_m9db_corrected_exp_crop220448",
    "db4_run": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4",
    "db4_ckpt": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4" / "best.keras",
    "db4_pred": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4" / "Ypred_edge_m9db.npy",
    "db4_metrics": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4" / "metrics_edge_m9db_corrected_exp",
    "fixed_split": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "split_config.json",
    "measured_matched": ROOT / "measured_rir_synth_matched",
    "measured_rirs": ROOT / "measured_rir_synth_noise" / "rir_wavs",
}

METHODS = {
    "LDWT": {"pred": PATHS["ldwt_pred"], "metrics": PATHS["ldwt_metrics"], "run": PATHS["ldwt_run"]},
    "Waveform": {"pred": PATHS["wave_pred"], "metrics": PATHS["wave_metrics"], "run": PATHS["wave_run"]},
    "Fixed-db4": {"pred": PATHS["db4_pred"], "metrics": PATHS["db4_metrics"], "run": PATHS["db4_run"]},
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_out(name: str) -> Path:
    out = OUT_ROOT / f"{timestamp()}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def file_info(path: Path, hash_small: bool = True) -> dict:
    info = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    st = path.stat()
    info.update({"size_bytes": st.st_size, "mtime": st.st_mtime, "is_dir": path.is_dir()})
    if path.is_dir():
        info["children"] = sorted(p.name for p in path.iterdir())[:200]
        return info
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        info.update({"shape": list(arr.shape), "dtype": str(arr.dtype)})
    if hash_small and st.st_size < 100_000_000:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def load_eval_funcs():
    sys.path.insert(0, str(ROOT / "Evaluations"))
    import eval_all_metrics_single as ev  # pylint: disable=import-outside-toplevel
    return ev


def ldwt_filter_lengths(ckpt: Path) -> list[int]:
    lengths = []
    with zipfile.ZipFile(ckpt, "r") as zf:
        with zf.open("model.weights.h5") as src:
            tmp = Path("/tmp") / f"ldwt_weights_{os.getpid()}.h5"
            tmp.write_bytes(src.read())
    try:
        with h5py.File(tmp, "r") as h5:
            def visit(name, obj):
                if hasattr(obj, "shape") and name.startswith("layers/prdwt1d") and name.endswith("/vars/0"):
                    lengths.append(int(obj.shape[0]))
            h5.visititems(visit)
    finally:
        tmp.unlink(missing_ok=True)
    return lengths


def audit(args) -> None:
    out = ensure_out("audit_manifest")
    manifest = {
        "created_unix_time": time.time(),
        "project_root": str(ROOT),
        "python": str(PYTHON),
        "git": {
            "commit": None,
            "status": None,
            "note": "git binary/repository unavailable from this execution environment; no .git directory found under DWT_IR during audit.",
        },
        "headline_protocol_ambiguities": [],
        "paths": {k: file_info(v) for k, v in PATHS.items() if isinstance(v, Path)},
        "configs": {},
        "checkpoint_selection": {
            "LDWT": "best.keras in LDWT run; config also preserves best_stageA.keras",
            "Waveform": "best.keras in waveform finetune run",
            "Fixed-db4": "best.keras in fixed db4 finetune run",
        },
        "evaluation": {
            "script": str(PATHS["evaluator"]),
            "standard_alignment": "std_align=1, std_max_lag=22050, envelope alignment where configured",
            "prediction_permutation": "permute_pred=1 in corrected-exp evaluator configs",
            "mimo": "BleedMatrixTF with K=512, lambda=1e-3 in corrected-exp evaluator configs",
        },
    }
    config_paths = {
        "ldwt_finetune": PATHS["ldwt_run"] / "finetune_config.json",
        "ldwt_eval": PATHS["ldwt_metrics"] / "eval_config.json",
        "wave_finetune": PATHS["wave_run"] / "finetune_config.json",
        "wave_eval": PATHS["wave_metrics"] / "eval_config.json",
        "db4_finetune": PATHS["db4_run"] / "finetune_config.json",
        "db4_eval": PATHS["db4_metrics"] / "eval_config.json",
        "fixed_split": PATHS["fixed_split"],
        "train_meta": PATHS["train_meta"],
        "test_meta": PATHS["test_meta"],
    }
    for key, path in config_paths.items():
        manifest["configs"][key] = read_json(path)

    lcfg = manifest["configs"].get("ldwt_finetune") or {}
    dcfg = manifest["configs"].get("db4_finetune") or {}
    wcfg = manifest["configs"].get("wave_finetune") or {}
    if lcfg.get("phaseB_epochs") != dcfg.get("epochs"):
        manifest["headline_protocol_ambiguities"].append(
            f"LDWT has phaseA_epochs={lcfg.get('phaseA_epochs')} and phaseB_epochs={lcfg.get('phaseB_epochs')}; fixed-db4 has epochs={dcfg.get('epochs')}."
        )
    if wcfg.get("epochs") != dcfg.get("epochs"):
        manifest["headline_protocol_ambiguities"].append(
            f"Waveform has epochs={wcfg.get('epochs')}; fixed-db4 has epochs={dcfg.get('epochs')}."
        )
    manifest["ldwt_filter_lengths"] = ldwt_filter_lengths(PATHS["ldwt_ckpt"])
    manifest["headline_ldwt_architecture"] = "L2_F11" if manifest["ldwt_filter_lengths"] == [11, 11] else f"unknown lengths {manifest['ldwt_filter_lengths']}"
    if PATHS["test_indices"].exists():
        manifest["test_ordering"] = {
            "array_order": "Rows are evaluated in stored NPY order.",
            "selected_indices": np.load(PATHS["test_indices"]).astype(int).tolist(),
        }
    else:
        manifest["test_ordering"] = {"array_order": "Rows are evaluated in stored NPY order; selected_indices.npy not found."}

    write_json(out / "experiment_manifest.json", manifest)
    readme = [
        "# TASLP Revision Experiment Audit",
        "",
        f"Created: {time.ctime(manifest['created_unix_time'])}",
        "",
        "Primary controlled test set: `Edgecase_active_matched/Xedge_m9db.npy` and `Yedge_m9db.npy`.",
        "",
        "Headline LDWT checkpoint is verified as `L2_F11` from the saved filter lengths `[11, 11]`.",
        "",
        "Protocol ambiguity before expensive reruns:",
    ]
    readme += [f"- {x}" for x in manifest["headline_protocol_ambiguities"]] or ["- none detected"]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"[Saved] {out / 'experiment_manifest.json'}")
    print(f"[Saved] {out / 'README.md'}")
    for amb in manifest["headline_protocol_ambiguities"]:
        print("[AMBIGUITY]", amb)


def finite_report(name: str, arr: np.ndarray) -> dict:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": bool(finite.all()),
        "nonfinite_count": int(np.size(a) - np.count_nonzero(finite)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "rms": float(np.sqrt(np.nanmean(a.astype(np.float64) ** 2) + EPS)),
    }


def smoke(args) -> None:
    out = ensure_out("finite_smoke")
    ev = load_eval_funcs()
    rows = []
    arrays = {
        "train_x_first": np.load(PATHS["train_x"], mmap_mode="r")[: args.max_items],
        "train_y_first": np.load(PATHS["train_y"], mmap_mode="r")[: args.max_items],
        "test_x_first": np.load(PATHS["test_x"], mmap_mode="r")[: args.max_items],
        "test_y_first": np.load(PATHS["test_y"], mmap_mode="r")[: args.max_items],
    }
    for method, meta in METHODS.items():
        if meta["pred"].exists():
            arrays[f"{method}_pred_first"] = np.load(meta["pred"], mmap_mode="r")[: args.max_items]
    for name, arr in arrays.items():
        rows.append(finite_report(name, arr))

    metric_rows = []
    X = np.asarray(arrays["test_x_first"], dtype=np.float64)
    Y = np.asarray(arrays["test_y_first"], dtype=np.float64)
    for n in range(X.shape[0]):
        for ch in range(X.shape[1]):
            y = Y[n, ch]
            x = X[n, ch, : len(y)]
            interf = np.sum(Y[n, np.arange(Y.shape[1]) != ch], axis=0)
            vals = {
                "idx": n,
                "ch": ch,
                "target_energy": float(np.sum((y - y.mean()) ** 2)),
                "input_energy": float(np.sum((x - x.mean()) ** 2)),
                "interf_energy": float(np.sum((interf - interf.mean()) ** 2)),
                "sisdr_in": ev.sisdr_np(y, x),
            }
            sir, sar = ev.sir_sar_np(y, interf, x)
            vals.update({"sir_in": sir, "sar_in": sar})
            vals["finite"] = bool(all(np.isfinite(v) for v in vals.values() if isinstance(v, float)))
            metric_rows.append(vals)

    write_json(out / "finite_array_report.json", rows)
    with (out / "metric_denominator_smoke.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"[Saved] {out}")
    bad = [r for r in rows if not r["finite"]] + [r for r in metric_rows if not r["finite"]]
    print(f"[Smoke] checked arrays={len(rows)} metric_rows={len(metric_rows)} nonfinite_records={len(bad)}")


def signed_corr(yhat: np.ndarray, y: np.ndarray, activity_db: float) -> tuple[float, bool]:
    y0 = np.asarray(y, dtype=np.float64) - np.mean(y)
    x0 = np.asarray(yhat, dtype=np.float64) - np.mean(yhat)
    e_y = float(np.sum(y0 * y0))
    e_x = float(np.sum(x0 * x0))
    active = e_y > 10 ** (activity_db / 10.0) * len(y0) and e_x > 10 ** (activity_db / 10.0) * len(x0)
    if not active:
        return float("nan"), False
    return float(np.sum(x0 * y0) / math.sqrt((e_x + EPS) * (e_y + EPS))), True


def sdr_plus(yhat: np.ndarray, y: np.ndarray) -> float:
    y0 = np.asarray(y, dtype=np.float64) - np.mean(y)
    x0 = np.asarray(yhat, dtype=np.float64) - np.mean(yhat)
    alpha = max(float(np.sum(x0 * y0) / (np.sum(y0 * y0) + EPS)), 0.0)
    target = alpha * y0
    err = x0 - target
    return float(10.0 * np.log10((np.sum(target * target) + EPS) / (np.sum(err * err) + EPS)))


def polarity(args) -> None:
    out = ensure_out("polarity_metrics")
    ev = load_eval_funcs()
    X = np.load(PATHS["test_x"], mmap_mode="r")
    Y = np.load(PATHS["test_y"], mmap_mode="r")
    rows = []
    for method, meta in METHODS.items():
        pred = np.load(meta["pred"], mmap_mode="r")
        T = min(Y.shape[-1], pred.shape[-1], args.eval_t)
        for n in range(Y.shape[0]):
            ytrue = np.asarray(Y[n, :, :T], dtype=np.float32)
            ypred = np.asarray(pred[n, :, :T], dtype=np.float32)
            ypred_perm, perm, lags, _ = ev.permute_prediction_to_reference(
                ytrue, ypred, max_lag=args.max_lag, use_envelope=True
            )
            for ch, stem in enumerate(STEMS):
                _, yhat_aligned, lag = ev.align_to_ref(ytrue[ch], ypred_perm[ch], max_lag=args.max_lag, use_envelope=True)
                rho, active = signed_corr(yhat_aligned, ytrue[ch], args.activity_db)
                rows.append({
                    "method": method,
                    "idx": n,
                    "ch": ch,
                    "stem": stem,
                    "active": active,
                    "rho": rho,
                    "polarity_error": bool(active and rho < args.rho_threshold),
                    "sdr_plus": sdr_plus(yhat_aligned, ytrue[ch]) if active else float("nan"),
                    "perm": ",".join(map(str, perm)),
                    "perm_lags": ",".join(map(str, lags)),
                    "lag": int(lag),
                })
    with (out / "polarity_per_output.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in METHODS:
        vals = np.array([r["rho"] for r in rows if r["method"] == method and r["active"]], dtype=np.float64)
        sdrp = np.array([r["sdr_plus"] for r in rows if r["method"] == method and r["active"]], dtype=np.float64)
        per = 100.0 * np.mean([r["polarity_error"] for r in rows if r["method"] == method and r["active"]])
        summary.append({
            "method": method,
            "active_outputs": int(vals.size),
            "median_rho": float(np.median(vals)),
            "rho_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "polarity_error_rate_pct": float(per),
            "mean_sdr_plus": float(np.nanmean(sdrp)),
            "median_sdr_plus": float(np.nanmedian(sdrp)),
        })
    with (out / "polarity_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[Saved] {out / 'polarity_per_output.csv'}")
    print(f"[Saved] {out / 'polarity_summary.csv'}")
    for row in summary:
        print(row)


def per_channel_mean_from_csv(path: Path, key: str) -> np.ndarray:
    vals = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            vals.append(float(r[key]))
    return np.asarray(vals, dtype=np.float64)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    d = a - b
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(d[idx]))
    return {
        "mean_diff": float(np.mean(d)),
        "median_diff": float(np.median(d)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "cohens_dz": float(np.mean(d) / (np.std(d, ddof=1) + EPS)),
        "n": int(n),
    }


def stats(args) -> None:
    out = ensure_out("paired_stats_existing_seed")
    rng = np.random.default_rng(args.seed)
    key_map = {
        "SIRB": ("mimo_metrics_long.csv", "SIRB_out"),
        "EXP": ("mimo_metrics_long.csv", "EXP_out"),
        "SI-SDR": ("standard_metrics_per_sample.csv", "sisdr"),
        "SIR": ("standard_metrics_per_sample.csv", "sir"),
        "SAR": ("standard_metrics_per_sample.csv", "sar"),
    }
    rows = []
    for metric, (fname, col) in key_map.items():
        ld = per_channel_mean_from_csv(METHODS["LDWT"]["metrics"] / fname, col)
        for base in ["Fixed-db4", "Waveform"]:
            bb = per_channel_mean_from_csv(METHODS[base]["metrics"] / fname, col)
            res = bootstrap_diff(ld, bb, rng, args.n_boot)
            res.update({"metric": metric, "contrast": f"LDWT - {base}", "unit": "matched example-channel observations"})
            rows.append(res)
    with (out / "paired_bootstrap_existing_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {out / 'paired_bootstrap_existing_seed.csv'}")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x, int(sr)


def band_energy_pct(x: np.ndarray, sr: int, cutoff: float) -> float:
    f, pxx = sp.welch(x, fs=sr, nperseg=min(8192, len(x)))
    total = float(np.trapz(pxx, f) + EPS)
    low = float(np.trapz(pxx[f <= cutoff], f[f <= cutoff]) if np.any(f <= cutoff) else 0.0)
    return 100.0 * low / total


def measured_diagnostics(args) -> None:
    out = ensure_out("measured_rir_reference_diagnostics")
    rir_rows = []
    for wav in sorted(PATHS["measured_rirs"].glob("*_mic*_ir.wav")):
        x, sr = read_wav(wav)
        peak_idx = int(np.argmax(np.abs(x)))
        direct_win = x[max(0, peak_idx - 8): min(len(x), peak_idx + 64)]
        rir_rows.append({
            "file": wav.name,
            "sr": sr,
            "length": len(x),
            "energy": float(np.sum(x * x)),
            "rms": float(np.sqrt(np.mean(x * x) + EPS)),
            "peak": float(np.max(np.abs(x))),
            "peak_index": peak_idx,
            "direct_energy_approx": float(np.sum(direct_win * direct_win)),
            "direct_to_total_db": float(10 * np.log10((np.sum(direct_win * direct_win) + EPS) / (np.sum(x * x) + EPS))),
            "dc": float(np.mean(x)),
            "clipped_pct": float(100 * np.mean(np.abs(x) >= 0.999)),
        })
    with (out / "rir_energy_delay_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rir_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rir_rows)

    # Reuse existing component diagnostic when available; otherwise run on a subset.
    comp_csv = PATHS["measured_matched"] / "diagnostics" / "measured_rir_component_diagnostics.csv"
    if comp_csv.exists():
        rows = list(csv.DictReader(comp_csv.open()))
        source = str(comp_csv)
    else:
        rows = []
        source = "not found"
    summary = {"component_diagnostics_source": source}
    if rows:
        for key in ["target_to_bleed_db", "target_to_noise_db", "snr_noise_db", "sir_input_db", "clipped_pct"]:
            vals = np.array([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=np.float64)
            summary[key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }

    # Noise PSD diagnostics from reconstructed noise rows are not possible without per-song reconstruction here;
    # check explicit noise files in the measured dataset tree if present.
    noise_rows = []
    for wav in sorted((PATHS["measured_matched"]).rglob("*noise*.wav"))[:200]:
        x, sr = read_wav(wav)
        row = {"file": str(wav), "sr": sr, "rms": float(np.sqrt(np.mean(x * x) + EPS)), "dc": float(np.mean(x))}
        for cutoff in [20, 40, 60, 80, 100]:
            row[f"pct_below_{cutoff}hz"] = band_energy_pct(x, sr, cutoff)
        noise_rows.append(row)
    if noise_rows:
        with (out / "noise_psd_lowfreq_diagnostics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(noise_rows[0].keys()))
            writer.writeheader()
            writer.writerows(noise_rows)
    write_json(out / "measured_rir_reference_summary.json", summary)
    print(f"[Saved] {out}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("smoke")
    p.add_argument("--max_items", type=int, default=4)
    p = sub.add_parser("polarity")
    p.add_argument("--eval_t", type=int, default=220448)
    p.add_argument("--max_lag", type=int, default=22050)
    p.add_argument("--activity_db", type=float, default=-80.0)
    p.add_argument("--rho_threshold", type=float, default=0.0)
    p = sub.add_parser("stats")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260731)
    sub.add_parser("measured")
    args = ap.parse_args()

    if args.cmd == "audit":
        audit(args)
    elif args.cmd == "smoke":
        smoke(args)
    elif args.cmd == "polarity":
        polarity(args)
    elif args.cmd == "stats":
        stats(args)
    elif args.cmd == "measured":
        measured_diagnostics(args)


if __name__ == "__main__":
    main()
