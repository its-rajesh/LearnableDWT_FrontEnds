#!/usr/bin/env python3
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_BASELINES = [
    ("KAMIR", "kamir"),
    ("CAE", "cae"),
    ("cGANIR", "cganir"),
]


def mean_of_stems(d):
    vals = [float(v["mean"]) for v in d.values()]
    stds = [float(v["std"]) for v in d.values()]
    return float(np.mean(vals)), float(np.mean(stds))


def tag_from_db(db):
    return f"m{abs(int(db))}db" if db < 0 else f"{int(db)}db"


def load_one(metrics_dir):
    with open(os.path.join(metrics_dir, "standard_metrics_summary.json"), "r") as f:
        std = json.load(f)
    with open(os.path.join(metrics_dir, "mimo_metrics_summary.json"), "r") as f:
        mimo = json.load(f)

    sirb_in_mean, sirb_in_std = mean_of_stems(mimo["mixture_before"]["SIRB"])
    sirb_out_mean, sirb_out_std = mean_of_stems(mimo["output_after"]["SIRB"])

    return {
        "sisdr_in_mean": float(std["sisdr_in"]["mean"]),
        "sisdr_in_std": float(std["sisdr_in"]["std"]),
        "sisdr_out_mean": float(std["sisdr"]["mean"]),
        "sisdr_out_std": float(std["sisdr"]["std"]),
        "sirb_in_mean": float(sirb_in_mean),
        "sirb_in_std": float(sirb_in_std),
        "sirb_out_mean": float(sirb_out_mean),
        "sirb_out_std": float(sirb_out_std),
    }


def load_series(metrics_root, bleed_levels, *, strict=True):
    rows = []
    missing = []
    for db in bleed_levels:
        tag = tag_from_db(db)
        metrics_dir = os.path.join(metrics_root, f"metrics_{tag}")
        if not os.path.isdir(metrics_dir):
            missing.append(metrics_dir)
            continue
        d = load_one(metrics_dir)
        d["bleed_nominal_db"] = float(db)
        rows.append(d)

    if strict and missing:
        raise FileNotFoundError("Missing metric directories:\n" + "\n".join(missing))
    return sorted(rows, key=lambda r: r["bleed_nominal_db"]), missing


def parse_baseline_args(items):
    out = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Baseline must be NAME=DIR, got: {item}")
        name, directory = item.split("=", 1)
        name = name.strip()
        directory = directory.strip()
        if not name or not directory:
            raise ValueError(f"Baseline must be NAME=DIR, got: {item}")
        out.append((name, directory))
    return out


def discover_default_baselines(pred_dir, baseline_root):
    root = baseline_root or os.path.join(pred_dir, "baseline_metrics")
    found = []
    for label, subdir in DEFAULT_BASELINES:
        metrics_root = os.path.join(root, subdir)
        if os.path.isdir(metrics_root):
            found.append((label, metrics_root))
    return found


def save_overlay_csv(ldwt_rows, baseline_series, out_csv):
    fields = [
        "method", "bleed_nominal_db",
        "sisdr_in_mean", "sisdr_in_std", "sisdr_out_mean", "sisdr_out_std",
        "sirb_in_mean", "sirb_in_std", "sirb_out_mean", "sirb_out_std",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, rows in [("Mixture/LDWT", ldwt_rows), *baseline_series]:
            for r in rows:
                row = {k: r.get(k) for k in fields if k != "method"}
                row["method"] = method
                w.writerow(row)


def plot_single(rows, x_key, out_png, ylabel, in_mean, in_std, out_mean, out_std, title, baseline_series):
    x = np.array([r[x_key] for r in rows], dtype=float)
    yin = np.array([r[in_mean] for r in rows], dtype=float)
    yin_std = np.array([r[in_std] for r in rows], dtype=float)
    yout = np.array([r[out_mean] for r in rows], dtype=float)
    yout_std = np.array([r[out_std] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    ax.plot(x, yin, marker="o", linewidth=2.2, color="#5f6368", label="Mixture")
    ax.fill_between(x, yin - yin_std, yin + yin_std, color="#5f6368", alpha=0.14, linewidth=0)

    ax.plot(x, yout, marker="s", linewidth=2.8, color="#1f77b4", label="LDWT")
    ax.fill_between(x, yout - yout_std, yout + yout_std, color="#1f77b4", alpha=0.16, linewidth=0)

    baseline_styles = [
        ("#d55e00", "^"),
        ("#009e73", "D"),
        ("#cc79a7", "v"),
        ("#e69f00", "P"),
    ]
    for idx, (name, brow) in enumerate(baseline_series):
        if not brow:
            continue
        bx = np.array([r[x_key] for r in brow], dtype=float)
        by = np.array([r[out_mean] for r in brow], dtype=float)
        color, marker = baseline_styles[idx % len(baseline_styles)]
        ax.plot(
            bx, by,
            marker=marker,
            linewidth=1.7,
            linestyle=":",
            color=color,
            alpha=0.78,
            label=name,
        )

    ax.set_xlabel("Nominal bleed level (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True,
                    help="LDWT prediction folder containing metrics_m40db ... metrics_0db")
    ap.add_argument("--bleed_levels", type=int, nargs="+",
                    default=[-40, -20, -18, -16, -14, -12, -9, -6, -3, 0])
    ap.add_argument("--baseline", action="append", default=[],
                    help="Optional baseline overlay as NAME=DIR, where DIR contains metrics_<tag>/ summaries. Can be repeated.")
    ap.add_argument("--baseline_root", type=str, default=None,
                    help="Root with default subdirs kamir, cae, cganir. Default: <pred_dir>/baseline_metrics if present.")
    args = ap.parse_args()

    rows, _ = load_series(args.pred_dir, args.bleed_levels, strict=True)

    baseline_specs = parse_baseline_args(args.baseline)
    if not baseline_specs:
        baseline_specs = discover_default_baselines(args.pred_dir, args.baseline_root)

    baseline_series = []
    for name, metrics_root in baseline_specs:
        brow, missing = load_series(metrics_root, args.bleed_levels, strict=False)
        if missing:
            print(f"[WARN] {name}: missing {len(missing)} metric directories; plotting {len(brow)} points.")
        if brow:
            baseline_series.append((name, brow))

    overlay_csv = os.path.join(args.pred_dir, "edge_bleed_overlay_summary.csv")
    save_overlay_csv(rows, baseline_series, overlay_csv)

    plot_single(
        rows, "bleed_nominal_db",
        os.path.join(args.pred_dir, "edge_bleed_sisdr_nominal.png"),
        "SI-SDR (dB)",
        "sisdr_in_mean", "sisdr_in_std", "sisdr_out_mean", "sisdr_out_std",
        "SI-SDR vs bleed level",
        baseline_series,
    )

    plot_single(
        rows, "bleed_nominal_db",
        os.path.join(args.pred_dir, "edge_bleed_sirb_nominal.png"),
        "SIR(B) (dB)",
        "sirb_in_mean", "sirb_in_std", "sirb_out_mean", "sirb_out_std",
        "SIR(B) vs bleed level",
        baseline_series,
    )

    print("Saved:")
    print(os.path.join(args.pred_dir, "edge_bleed_sisdr_nominal.png"))
    print(os.path.join(args.pred_dir, "edge_bleed_sirb_nominal.png"))
    print(overlay_csv)
    if baseline_series:
        print("Baselines:", ", ".join(name for name, _ in baseline_series))
    else:
        print("Baselines: none found")


if __name__ == "__main__":
    main()
