#!/usr/bin/env python3
"""Extract measured RIRs using the original deconvolution convention.

This reproduces the old script's full-length regularized inverse and
peak-centered crop, but writes FLOAT WAVs without per-RIR peak normalization.
That is the important correction: downstream dataset generators must see the
same physical source-to-microphone gains that the original in-memory IR bank had.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal as sp
import soundfile as sf

STEMS = (("vocals", 0), ("bass", 1), ("drums", 2))
EPS = 1e-30


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def db10(num: float, den: float) -> float:
    return float(10.0 * np.log10((num + EPS) / (den + EPS)))


def read_audio(path: Path, always_2d: bool = False) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float64", always_2d=always_2d)
    return sanitize(x), int(sr)


def regularized_deconv(excitation: np.ndarray, recording: np.ndarray, reg: float, ir_len: int) -> np.ndarray:
    excitation = sanitize(excitation).reshape(-1)
    recording = sanitize(recording).reshape(-1)
    n = len(excitation) + len(recording) - 1
    nfft = 1 << (n - 1).bit_length()
    x = np.fft.rfft(excitation, nfft)
    y = np.fft.rfft(recording, nfft)
    denom = np.abs(x) ** 2
    h = np.fft.irfft(y * np.conj(x) / (denom + reg * np.max(denom) + 1e-18), nfft)
    return sanitize(h[:ir_len])


def highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    sos = sp.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sanitize(sp.sosfiltfilt(sos, x, axis=0))


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr, subtype="FLOAT")


def dominance(matrix: np.ndarray) -> dict[str, object]:
    vals = []
    for i in range(3):
        vals.append(db10(float(matrix[i, i]), float(sum(matrix[i, j] for j in range(3) if j != i))))
    return {
        "diag_target_to_bleed_db": vals,
        "mean_diag_target_to_bleed_db": float(np.mean(vals)),
        "preferred_mic_per_source": [int(np.argmax(matrix[i])) for i in range(3)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", type=Path, default=Path("/home/rrame12/Desktop/Datasets/Re-recorded/sample_data/_session_room_measurement"))
    ap.add_argument("--out_dir", type=Path, default=Path("/home/rrame12/Desktop/Research/DWT_IR/measured_rir_original_deconv_rawgain"))
    ap.add_argument("--reg", type=float, default=1e-6)
    ap.add_argument("--pre_sec", type=float, default=0.02)
    ap.add_argument("--post_sec", type=float, default=1.0)
    ap.add_argument("--ir_len_sec", type=float, default=2.0)
    ap.add_argument("--target_sr", type=int, default=22050)
    ap.add_argument("--lowcut", type=float, default=0.0)
    args = ap.parse_args()

    wav44 = args.out_dir / "rir_wavs_44100"
    wav22 = args.out_dir / "rir_wavs"
    diag = args.out_dir / "diagnostics"
    wav44.mkdir(parents=True, exist_ok=True)
    wav22.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)

    rows = []
    peak_mat = np.zeros((3, 3), dtype=np.float64)
    energy_mat = np.zeros((3, 3), dtype=np.float64)
    sr0 = None
    for stem, spk_idx in STEMS:
        played, sr = read_audio(args.session_dir / f"sweep_spk{spk_idx}_{stem}.wav")
        rec, sr_rec = read_audio(args.session_dir / f"sweep_rec_spk{spk_idx}_{stem}.wav", always_2d=True)
        if played.ndim == 2:
            played = played.mean(axis=1)
        if rec.shape[1] != 3 and rec.shape[0] == 3:
            rec = rec.T
        if rec.shape[1] != 3:
            raise ValueError(f"Expected 3-channel recording for {stem}, got {rec.shape}")
        if sr != sr_rec:
            raise ValueError(f"Sample-rate mismatch for {stem}: {sr} vs {sr_rec}")
        if sr0 is None:
            sr0 = sr
        elif sr0 != sr:
            raise ValueError("Inconsistent sample rates")
        if args.lowcut > 0:
            rec = highpass(rec, sr, args.lowcut)

        ir_len = int(round(args.ir_len_sec * sr))
        pre = int(round(args.pre_sec * sr))
        post = int(round(args.post_sec * sr))
        for mic in range(3):
            h_full = regularized_deconv(played, rec[:, mic], args.reg, ir_len)
            pk = int(np.argmax(np.abs(h_full)))
            start = max(0, pk - pre)
            stop = min(len(h_full), pk + post)
            h = h_full[start:stop].copy()
            peak_mat[spk_idx, mic] = float(np.max(np.abs(h)))
            energy_mat[spk_idx, mic] = float(np.sum(h * h))
            write_wav(wav44 / f"{stem}_mic{mic}_ir.wav", h, sr)
            if args.target_sr != sr:
                g = math.gcd(sr, args.target_sr)
                h22 = sp.resample_poly(h, args.target_sr // g, sr // g)
                write_wav(wav22 / f"{stem}_mic{mic}_ir.wav", h22, args.target_sr)
            else:
                write_wav(wav22 / f"{stem}_mic{mic}_ir.wav", h, sr)
            rows.append({
                "source": stem,
                "speaker_idx": spk_idx,
                "mic": mic,
                "sample_rate": sr,
                "target_sample_rate": args.target_sr,
                "peak_idx_full": pk,
                "window_start": start,
                "samples": len(h),
                "peak": peak_mat[spk_idx, mic],
                "rms": float(np.sqrt(np.mean(h * h) + EPS)),
                "energy": energy_mat[spk_idx, mic],
            })

    summary = {
        "method": "old full-sweep regularized deconvolution, raw gain preserved in saved WAVs",
        "session_dir": str(args.session_dir),
        "out_dir": str(args.out_dir),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rir_peak_matrix": peak_mat.tolist(),
        "rir_energy_matrix": energy_mat.tolist(),
        "dominance": {
            "rir_peak2": dominance(peak_mat ** 2),
            "rir_energy": dominance(energy_mat),
        },
    }
    with (diag / "rir_extraction_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (diag / "rir_extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Saved] {wav44}")
    print(f"[Saved] {wav22}")
    for key, val in summary["dominance"].items():
        print(key, "diag_dB=", [round(x, 2) for x in val["diag_target_to_bleed_db"]], "mean=", round(val["mean_diag_target_to_bleed_db"], 2), "pref=", val["preferred_mic_per_source"])


if __name__ == "__main__":
    main()
