#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def mean_stems(block: dict, key: str) -> float:
    vals = [float(v["mean"]) for v in block[key].values()]
    return sum(vals) / len(vals)


def load_method(metric_dir: Path, method: str, reference: bool = False) -> dict:
    std = json.loads((metric_dir / "standard_metrics_summary.json").read_text())
    mimo = json.loads((metric_dir / "mimo_metrics_summary.json").read_text())
    if reference:
        return {
            "Method": method,
            "SI-SDR": float(std["sisdr_in"]["mean"]),
            "SIR": float(std["sir_in"]["mean"]),
            "SAR": float(std["sar_in"]["mean"]),
            "SIR(B)": mean_stems(mimo["mixture_before"], "SIRB"),
            "EXP (%)": mean_stems(mimo["mixture_before"], "EXP"),
        }
    return {
        "Method": method,
        "SI-SDR": float(std["sisdr"]["mean"]),
        "SIR": float(std["sir"]["mean"]),
        "SAR": float(std["sar"]["mean"]),
        "SIR(B)": mean_stems(mimo["output_after"], "SIRB"),
        "EXP (%)": mean_stems(mimo["output_after"], "EXP"),
    }


def fmt_delta(x: float) -> str:
    return f"{x:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--kamir", type=Path, required=True)
    ap.add_argument("--cae", type=Path, required=True)
    ap.add_argument("--cganir", type=Path, required=True)
    ap.add_argument("--htdemucs", type=Path, required=True)
    ap.add_argument("--ldwt", type=Path, required=True)
    args = ap.parse_args()

    rows = [
        load_method(args.reference, "Reference", reference=True),
        load_method(args.kamir, "KAMIR"),
        load_method(args.cae, "CAE"),
        load_method(args.cganir, "cGANIR"),
        load_method(args.htdemucs, "HTDemucs"),
        load_method(args.ldwt, "LDWT (Ours)"),
    ]
    ref_sir = rows[0]["SIR"]
    ref_sirb = rows[0]["SIR(B)"]
    for r in rows:
        r["Delta SIR"] = r["SIR"] - ref_sir
        r["Delta SIR(B)"] = r["SIR(B)"] - ref_sirb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "table2_measured_rir_corrected_all_methods.csv"
    fields = ["Method", "SI-SDR", "SIR", "Delta SIR", "SAR", "SIR(B)", "Delta SIR(B)", "EXP (%)"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & SI-SDR & SIR & $\Delta$SIR & SAR & SIR(B) & $\Delta$SIR(B) & EXP (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['Method']} & {r['SI-SDR']:.2f} & {r['SIR']:.2f} & {fmt_delta(r['Delta SIR'])} & "
            f"{r['SAR']:.2f} & {r['SIR(B)']:.2f} & {fmt_delta(r['Delta SIR(B)'])} & {r['EXP (%)']:.2f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex_path = args.out_dir / "table2_measured_rir_corrected_all_methods.tex"
    tex_path.write_text("\n".join(tex_lines))

    print("Saved:", csv_path)
    print("Saved:", tex_path)
    print()
    for r in rows:
        print(
            f"{r['Method']:12s} SI-SDR={r['SI-SDR']:7.2f} SIR={r['SIR']:7.2f} "
            f"dSIR={r['Delta SIR']:7.2f} SAR={r['SAR']:7.2f} "
            f"SIRB={r['SIR(B)']:7.2f} dSIRB={r['Delta SIR(B)']:7.2f} EXP={r['EXP (%)']:7.2f}"
        )


if __name__ == "__main__":
    main()
