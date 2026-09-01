#!/usr/bin/env python3
"""Paper figures/tables for wavelet-based interpretation of LDWT interference reduction."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import analyze_wavelet_uncertainty as wu


ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
OUT = ROOT / "paper_wavelet_interpretation"
X_PATH = ROOT / "Edgecase_active_matched" / "Xedge_m9db.npy"
Y_PATH = ROOT / "Edgecase_active_matched" / "Yedge_m9db.npy"
RIR_DIR = ROOT / "measured_rir_synth_noise" / "rir_wavs"
LDWT_CKPT = ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "best.keras"
PRED_PATHS = {
    "LDWT": ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "predictions_edge_m9db" / "Ypred_edge_m9db.npy",
    "Fixed-haar": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_094709" / "haar" / "Ypred_edge_m9db.npy",
    "Fixed-db2": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_094709" / "db2" / "Ypred_edge_m9db.npy",
    "Fixed-db4": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db4" / "Ypred_edge_m9db.npy",
    "Fixed-db8": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_095520" / "db8" / "Ypred_edge_m9db.npy",
    "Fixed-sym4": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_094709" / "sym4" / "Ypred_edge_m9db.npy",
    "Fixed-coif1": ROOT / "runs_fixed_wavelet_family_finetune_edge_m9db_disjoint" / "20260715_094709" / "coif1" / "Ypred_edge_m9db.npy",
}


def db(x: float) -> float:
    return 10.0 * math.log10(max(float(x), 1e-30))


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x / (np.linalg.norm(x) + 1e-12)


def max_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = norm(a)
    b = norm(b)
    c = np.correlate(a, b, mode="full")
    return float(np.max(np.abs(c)))


def spectral_distance(a: np.ndarray, b: np.ndarray, n_fft: int = 65536) -> float:
    A = np.abs(np.fft.rfft(norm(a), n=n_fft))
    B = np.abs(np.fft.rfft(norm(b), n=n_fft))
    A = A / (np.linalg.norm(A) + 1e-12)
    B = B / (np.linalg.norm(B) + 1e-12)
    return float(np.linalg.norm(A - B))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], columns: list[str], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.2f}")
            else:
                vals.append(str(val))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


def all_filterbanks() -> dict[str, list[dict[str, np.ndarray]]]:
    banks = {"LDWT": wu.load_ldwt_filters(LDWT_CKPT, levels=2)}
    for wavelet in ["haar", "db2", "db4", "db8", "sym4", "coif1"]:
        h0 = wu.fixed_lowpass_padded(wavelet, 101)
        filt = {"h0": h0, "h1": wu.qmf_highpass_from_lowpass_np(h0)}
        banks[f"Fixed-{wavelet}"] = [filt, filt]
    return banks


def filter_similarity_table(banks: dict[str, list[dict[str, np.ndarray]]]) -> list[dict[str, object]]:
    rows = []
    for method, filters in banks.items():
        if method == "LDWT":
            continue
        fixed = filters
        corr_l1 = max_abs_corr(banks["LDWT"][0]["h0"], fixed[0]["h0"])
        corr_l2 = max_abs_corr(banks["LDWT"][1]["h0"], fixed[1]["h0"])
        spec_l1 = spectral_distance(banks["LDWT"][0]["h0"], fixed[0]["h0"])
        spec_l2 = spectral_distance(banks["LDWT"][1]["h0"], fixed[1]["h0"])
        rows.append({
            "Wavelet": method.replace("Fixed-", ""),
            "L1 corr": corr_l1,
            "L2 corr": corr_l2,
            "L1 spec dist": spec_l1,
            "L2 spec dist": spec_l2,
        })
    return rows


def band_weights(filters: list[dict[str, np.ndarray]], n_fft: int) -> dict[str, np.ndarray]:
    subbands = wu.effective_subbands(filters)
    weights = {}
    for name, h in subbands.items():
        w = np.abs(np.fft.rfft(h, n=n_fft)) ** 2
        weights[name] = w.astype(np.float64)
    return weights


def weighted_fft_energy(arr: np.ndarray, weights: dict[str, np.ndarray], n_fft: int) -> dict[str, float]:
    # arr: [N,C,T]
    out = {k: 0.0 for k in weights}
    for start in range(0, arr.shape[0], 8):
        chunk = arr[start:start + 8].astype(np.float64, copy=False)
        F = np.fft.rfft(chunk, n=n_fft, axis=-1)
        P = np.abs(F) ** 2
        for band, w in weights.items():
            out[band] += float(np.sum(P * w.reshape((1, 1, -1))))
    return out


def align_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-example PIT plus least-squares scalar alignment for diagnostic errors."""
    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    aligned = np.empty_like(y_true, dtype=np.float32)
    for n in range(y_true.shape[0]):
        yt = y_true[n].astype(np.float64, copy=False)
        yp = y_pred[n].astype(np.float64, copy=False)
        best_err = None
        best = None
        for perm in perms:
            cand = yp[list(perm)]
            out = np.empty_like(cand)
            err = 0.0
            for c in range(cand.shape[0]):
                alpha = float(np.dot(cand[c], yt[c]) / (np.dot(cand[c], cand[c]) + 1e-12))
                out[c] = alpha * cand[c]
                err += float(np.sum((out[c] - yt[c]) ** 2))
            if best_err is None or err < best_err:
                best_err = err
                best = out
        aligned[n] = best.astype(np.float32)
    return aligned


def subband_energy_analysis(banks: dict[str, list[dict[str, np.ndarray]]]) -> list[dict[str, object]]:
    X = np.load(X_PATH, mmap_mode="r")
    Y = np.load(Y_PATH, mmap_mode="r")
    T = min(X.shape[-1], Y.shape[-1], *(np.load(p, mmap_mode="r").shape[-1] for p in PRED_PATHS.values() if p.exists()))
    n_fft = 1 << int(math.ceil(math.log2(T)))
    Xc = np.asarray(X[:, :, :T], dtype=np.float32)
    Yc = np.asarray(Y[:, :, :T], dtype=np.float32)
    interference = Xc - Yc

    rows = []
    for frontend, filters in banks.items():
        weights = band_weights(filters, n_fft)
        target_e = weighted_fft_energy(Yc, weights, n_fft)
        interf_e = weighted_fft_energy(interference, weights, n_fft)
        pred_path = PRED_PATHS.get(frontend)
        error_e = None
        if pred_path and pred_path.exists():
            pred = np.asarray(np.load(pred_path, mmap_mode="r")[:, :, :T], dtype=np.float32)
            pred = align_predictions(pred, Yc)
            error_e = weighted_fft_energy(pred - Yc, weights, n_fft)
        for band in ["A2", "D2", "D1"]:
            row = {
                "Frontend": frontend,
                "Subband": band,
                "Input SIR": db(target_e[band] / (interf_e[band] + 1e-30)),
                "Target energy %": 0.0,
            }
            row["Target energy %"] = 100.0 * target_e[band] / (sum(target_e.values()) + 1e-30)
            if error_e is not None:
                row["Output target/error"] = db(target_e[band] / (error_e[band] + 1e-30))
                row["Error energy %"] = 100.0 * error_e[band] / (sum(error_e.values()) + 1e-30)
            rows.append(row)
    return rows


def rir_condition_analysis(banks: dict[str, list[dict[str, np.ndarray]]]) -> list[dict[str, object]]:
    sources = ["vocals", "bass", "drums"]
    n_fft = 65536
    H = []
    sr_ref = None
    for src in sources:
        cols = []
        for mic in range(3):
            sr, data = wavfile.read(RIR_DIR / f"{src}_mic{mic}_ir.wav")
            if sr_ref is None:
                sr_ref = sr
            if data.ndim > 1:
                data = data[:, 0]
            data = data.astype(np.float64)
            if np.issubdtype(data.dtype, np.integer):
                data = data / np.iinfo(data.dtype).max
            cols.append(np.fft.rfft(data, n=n_fft))
        H.append(cols)
    H = np.asarray(H, dtype=np.complex128).transpose(2, 1, 0)  # [F, mic, src]
    cond = np.array([np.linalg.cond(H[k] + 1e-12 * np.eye(3)) for k in range(H.shape[0])])
    cond_db = 20.0 * np.log10(np.clip(cond, 1.0, 1e12))

    rows = []
    for frontend, filters in banks.items():
        weights = band_weights(filters, n_fft)
        for band in ["A2", "D2", "D1"]:
            w = weights[band]
            w = w / (np.sum(w) + 1e-30)
            rows.append({
                "Frontend": frontend,
                "Subband": band,
                "Mean cond (dB)": float(np.sum(w * cond_db)),
                "Median cond (dB)": float(np.interp(0.5, np.cumsum(w[np.argsort(cond_db)]), np.sort(cond_db))),
            })
    return rows


def plot_effective_responses(banks: dict[str, list[dict[str, np.ndarray]]]) -> None:
    n_fft = 65536
    omega = np.linspace(0.0, math.pi, n_fft // 2 + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    colors = {"LDWT": "#111111", "Fixed-db4": "#4C78A8", "Fixed-db8": "#F58518", "Fixed-coif1": "#54A24B"}
    for ax, band in zip(axes, ["A2", "D2", "D1"]):
        for method in ["Fixed-db4", "Fixed-db8", "Fixed-coif1", "LDWT"]:
            h = wu.effective_subbands(banks[method])[band]
            mag = np.abs(np.fft.rfft(h, n=n_fft))
            mag_db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
            lw = 2.5 if method == "LDWT" else 1.4
            ls = "-" if method == "LDWT" else "--"
            ax.plot(omega / math.pi, mag_db, color=colors[method], lw=lw, ls=ls, label=method)
        ax.set_title(band)
        ax.set_xlabel("Normalized frequency ($\\omega/\\pi$)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-80, 3)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "effective_subband_responses.pdf", bbox_inches="tight")
    fig.savefig(OUT / "effective_subband_responses.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subband_bars(rows: list[dict[str, object]]) -> None:
    methods = ["Fixed-db4", "Fixed-db8", "Fixed-coif1", "LDWT"]
    bands = ["A2", "D2", "D1"]
    lookup = {(r["Frontend"], r["Subband"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharex=True)
    width = 0.18
    x = np.arange(len(bands))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#111111"]
    for i, method in enumerate(methods):
        axes[0].bar(x + (i - 1.5) * width, [lookup[(method, b)]["Input SIR"] for b in bands], width, color=colors[i], label=method)
        axes[1].bar(x + (i - 1.5) * width, [lookup[(method, b)].get("Output target/error", np.nan) for b in bands], width, color=colors[i], label=method)
    for ax, title in zip(axes, ["Input subband SIR", "Output target/error ratio"]):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_ylabel("dB")
        ax.grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "subband_interference_error_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT / "subband_interference_error_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_condition(rows: list[dict[str, object]]) -> None:
    methods = ["Fixed-haar", "Fixed-db2", "Fixed-db4", "Fixed-db8", "Fixed-sym4", "Fixed-coif1", "LDWT"]
    bands = ["A2", "D2", "D1"]
    lookup = {(r["Frontend"], r["Subband"]): r for r in rows}
    mat = np.array([[lookup[(m, b)]["Mean cond (dB)"] for b in bands] for m in methods])
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([m.replace("Fixed-", "") for m in methods])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="white" if mat[i, j] > np.nanmean(mat) else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted condition number (dB)")
    ax.set_title("Measured 3x3 RIR mixing condition by subband")
    fig.tight_layout()
    fig.savefig(OUT / "rir_condition_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rir_condition_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    banks = all_filterbanks()
    sim_rows = filter_similarity_table(banks)
    write_csv(OUT / "filter_similarity_table.csv", sim_rows)
    write_latex(
        OUT / "filter_similarity_table.tex",
        sim_rows,
        ["Wavelet", "L1 corr", "L2 corr", "L1 spec dist", "L2 spec dist"],
        "Similarity between learned LDWT low-pass filters and fixed wavelet families.",
        "tab:ldwt_filter_similarity",
    )

    sub_rows = subband_energy_analysis(banks)
    write_csv(OUT / "subband_interference_error_table.csv", sub_rows)
    compact_sub = [r for r in sub_rows if r["Frontend"] in ["Fixed-db4", "Fixed-db8", "Fixed-coif1", "LDWT"]]
    write_latex(
        OUT / "subband_interference_error_table.tex",
        compact_sub,
        ["Frontend", "Subband", "Input SIR", "Output target/error", "Target energy %", "Error energy %"],
        "Subband interference and residual-error energy on the held-out m9 dB test set.",
        "tab:subband_interference_error",
    )

    cond_rows = rir_condition_analysis(banks)
    write_csv(OUT / "rir_condition_table.csv", cond_rows)
    write_latex(
        OUT / "rir_condition_table.tex",
        cond_rows,
        ["Frontend", "Subband", "Mean cond (dB)", "Median cond (dB)"],
        "Measured 3x3 RIR mixing condition summarized by wavelet subband.",
        "tab:rir_condition_subband",
    )

    plot_effective_responses(banks)
    plot_subband_bars(sub_rows)
    plot_condition(cond_rows)

    print(f"[Saved outputs] {OUT}")
    print("[Key files]")
    for name in [
        "effective_subband_responses.pdf",
        "subband_interference_error_bars.pdf",
        "rir_condition_heatmap.pdf",
        "filter_similarity_table.tex",
        "subband_interference_error_table.tex",
        "rir_condition_table.tex",
    ]:
        print(" ", OUT / name)


if __name__ == "__main__":
    main()
