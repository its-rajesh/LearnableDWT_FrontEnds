#!/usr/bin/env python3
"""Quantify time-frequency localization for LDWT and fixed-DWT filters."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR")
DEFAULT_LDWT = ROOT / "runs_ldwt_finetune_edge_m9db_disjoint_small" / "20260705_075106" / "best.keras"

FIXED_WAVELET_DEC_LO = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.12940952255126034],
    "db4": [
        -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
        -0.18703481171888114, -0.027983769416859854, 0.6308807679298587,
        0.7148465705529154, 0.23037781330885523,
    ],
    "db8": [
        -0.00011747678412476953, 0.0006754494059985568, -0.00039174037337694705,
        -0.004870352993451574, 0.008746094047405777, 0.013981027917398282,
        -0.044088253930794755, -0.017369301001807547, 0.12874742662047847,
        0.0004724845739132828, -0.2840155429615469, -0.015829105256349305,
        0.5853546836541907, 0.6756307362972898, 0.31287159091429995,
        0.05441584224310401,
    ],
    "sym4": [
        -0.07576571478927333, -0.02963552764599851, 0.49761866763201545,
        0.8037387518059161, 0.29785779560527736, -0.09921954357684722,
        -0.012603967262037833, 0.0322231006040427,
    ],
    "coif1": [
        -0.01565572813546454, -0.0727326195128539, 0.38486484686420286,
        0.8525720202122554, 0.3378976624578092, -0.0727326195128539,
    ],
}


def qmf_highpass_from_lowpass_np(h0: np.ndarray) -> np.ndarray:
    h0 = np.asarray(h0, dtype=np.float64)
    alt = np.ones(len(h0), dtype=np.float64)
    alt[1::2] = -1.0
    return alt * h0[::-1]


def fixed_lowpass_padded(wavelet: str, length: int) -> np.ndarray:
    h = np.asarray(FIXED_WAVELET_DEC_LO[wavelet], dtype=np.float64)
    if length < len(h) or length % 2 == 0:
        raise ValueError(f"Bad length {length} for {wavelet}")
    pad_total = length - len(h)
    h = np.pad(h, (pad_total // 2, pad_total - pad_total // 2))
    return h / (np.linalg.norm(h) + 1e-12)


def upsample_filter(h: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(h, dtype=np.float64)
    out = np.zeros((len(h) - 1) * factor + 1, dtype=np.float64)
    out[::factor] = h
    return out


def effective_subbands(level_filters: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Return effective two-level analysis filters: D1, D2, A2."""
    h0_0 = level_filters[0]["h0"]
    h1_0 = level_filters[0]["h1"]
    h0_1 = level_filters[1]["h0"]
    h1_1 = level_filters[1]["h1"]
    return {
        "D1": h1_0,
        "D2": np.convolve(h0_0, upsample_filter(h1_1, 2)),
        "A2": np.convolve(h0_0, upsample_filter(h0_1, 2)),
    }


def shortest_width_uniform(prob: np.ndarray, step: float, mass: float = 0.95, circular: bool = False) -> float:
    p = np.asarray(prob, dtype=np.float64)
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    if not circular:
        best = len(p)
        acc = 0.0
        left = 0
        for right, val in enumerate(p):
            acc += val
            while left <= right and acc - p[left] >= mass:
                acc -= p[left]
                left += 1
            if acc >= mass:
                best = min(best, right - left + 1)
        return best * step

    p2 = np.concatenate([p, p])
    n = len(p)
    best = n
    acc = 0.0
    left = 0
    for right, val in enumerate(p2):
        acc += val
        while right - left + 1 > n:
            acc -= p2[left]
            left += 1
        while left <= right and acc - p2[left] >= mass:
            acc -= p2[left]
            left += 1
        if acc >= mass:
            best = min(best, right - left + 1)
    return best * step


def entropy(prob: np.ndarray) -> tuple[float, float]:
    p = np.asarray(prob, dtype=np.float64)
    p = p[p > 0]
    h = float(-np.sum(p * np.log2(p)))
    return h, float(2.0 ** h)


def localization_metrics(h: np.ndarray, n_fft: int = 65536, band: tuple[float, float] | None = None) -> dict[str, float]:
    h = np.asarray(h, dtype=np.float64)
    energy = np.square(np.abs(h))
    pt = energy / (energy.sum() + 1e-30)
    n = np.arange(len(h), dtype=np.float64)
    mu_t = float(np.sum(n * pt))
    sigma_t = float(np.sqrt(np.sum(np.square(n - mu_t) * pt)))
    wt95 = shortest_width_uniform(pt, 1.0, 0.95, circular=False)
    ht, nt_eff = entropy(pt)

    omega = np.linspace(-math.pi, math.pi, n_fft, endpoint=False)
    H = np.fft.fftshift(np.fft.fft(h, n=n_fft))
    pf = np.square(np.abs(H))
    pf = pf / (pf.sum() + 1e-30)
    z = np.sum(pf * np.exp(1j * omega))
    mu_w = float(np.angle(z))
    dist = np.angle(np.exp(1j * (omega - mu_w)))
    sigma_w = float(np.sqrt(np.sum(np.square(dist) * pf)))
    wf95 = shortest_width_uniform(pf, 2 * math.pi / n_fft, 0.95, circular=True)
    hf, nf_eff = entropy(pf)

    out = {
        "support": float(len(h)),
        "mu_t": mu_t,
        "sigma_t_samples": sigma_t,
        "sigma_omega_rad": sigma_w,
        "uncertainty_product": sigma_t * sigma_w,
        "time_width_95_samples": wt95,
        "bandwidth_95_rad": wf95,
        "time_entropy_bits": ht,
        "time_effective_samples": nt_eff,
        "freq_entropy_bits": hf,
        "freq_effective_bins": nf_eff,
    }
    if band is not None:
        lo, hi = band
        mask = (np.abs(omega) >= lo) & (np.abs(omega) <= hi)
        out["intended_band_energy_pct"] = float(100.0 * np.sum(pf[mask]))
    return out


def load_ldwt_filters(checkpoint: Path, levels: int) -> list[dict[str, np.ndarray]]:
    """Extract h0_raw directly from a Keras archive; no full model restore needed."""
    import h5py  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory(prefix="ldwt_weights_") as tmp:
        tmp_path = Path(tmp)
        if checkpoint.is_dir():
            h5_path = checkpoint / "model.weights.h5"
        else:
            with zipfile.ZipFile(checkpoint, "r") as zf:
                zf.extract("model.weights.h5", tmp_path)
            h5_path = tmp_path / "model.weights.h5"

        h0s: list[np.ndarray] = []
        with h5py.File(h5_path, "r") as h5:
            candidates: list[tuple[str, np.ndarray]] = []

            def visit(name, obj):
                if hasattr(obj, "shape") and name.startswith("layers/prdwt1d") and name.endswith("/vars/0"):
                    candidates.append((name, np.asarray(obj, dtype=np.float64)))

            h5.visititems(visit)
            for name, h0_raw in sorted(candidates, key=lambda item: item[0]):
                h0 = h0_raw / (np.linalg.norm(h0_raw) + 1e-12)
                print(f"[LDWT] {name}: length={len(h0)}")
                h0s.append(h0)

    if len(h0s) < levels:
        raise RuntimeError(f"Expected {levels} LDWT filters but found {len(h0s)} in {checkpoint}")
    out = []
    for h0 in h0s[:levels]:
        out.append({"h0": h0, "h1": qmf_highpass_from_lowpass_np(h0)})
    return out


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


def add_rows(rows: list[dict[str, object]], method: str, level_filters: list[dict[str, np.ndarray]]) -> None:
    local_bands = {"h0": (0.0, math.pi / 2), "h1": (math.pi / 2, math.pi)}
    for level, filt in enumerate(level_filters, start=1):
        for name in ("h0", "h1"):
            rows.append({
                "type": "local",
                "method": method,
                "filter": f"L{level}_{name}",
                "band_label": "lowpass" if name == "h0" else "highpass",
                **localization_metrics(filt[name], band=local_bands[name]),
            })

    bands = {"D1": (math.pi / 2, math.pi), "D2": (math.pi / 4, math.pi / 2), "A2": (0.0, math.pi / 4)}
    for name, filt in effective_subbands(level_filters).items():
        rows.append({
            "type": "effective",
            "method": method,
            "filter": name,
            "band_label": {"D1": "detail_1", "D2": "detail_2", "A2": "approx_2"}[name],
            **localization_metrics(filt, band=bands[name]),
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldwt_ckpt", type=Path, default=DEFAULT_LDWT)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "uncertainty_analysis")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=101)
    ap.add_argument("--wavelets", nargs="+", default=["haar", "db2", "db4", "db8", "sym4", "coif1"])
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    ldwt_filters = load_ldwt_filters(args.ldwt_ckpt, levels=args.levels)
    add_rows(rows, "LDWT", ldwt_filters)

    for wavelet in args.wavelets:
        h0 = fixed_lowpass_padded(wavelet, args.filter_length)
        filt = {"h0": h0, "h1": qmf_highpass_from_lowpass_np(h0)}
        add_rows(rows, f"Fixed-{wavelet}", [filt.copy() for _ in range(args.levels)])

    out_csv = args.out_dir / "wavelet_uncertainty_metrics.csv"
    write_csv(out_csv, rows)
    print(f"[Saved] {out_csv}")

    print("\nEffective subband summary")
    print("method,filter,sigma_t,sigma_omega,U,Wt95,Womega95,band_energy_pct")
    for row in rows:
        if row["type"] == "effective":
            print(
                f"{row['method']},{row['filter']},"
                f"{float(row['sigma_t_samples']):.3f},"
                f"{float(row['sigma_omega_rad']):.4f},"
                f"{float(row['uncertainty_product']):.3f},"
                f"{float(row['time_width_95_samples']):.1f},"
                f"{float(row['bandwidth_95_rad']):.4f},"
                f"{float(row['intended_band_energy_pct']):.2f}"
            )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
