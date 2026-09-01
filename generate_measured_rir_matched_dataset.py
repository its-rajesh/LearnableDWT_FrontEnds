#!/usr/bin/env python3
"""Generate a matched measured-RIR MUSDB18HQ dataset.

This script rebuilds the measured-RIR dataset so that mixtures and targets are
derived from the same fixed 3x3 measured RIR matrix:

  X_m = sum_s dry_s * h[source=s, mic=m] + optional_noise_m
  Y_s = dry_s * h[source=s, mic=s]

The output layout matches finetune_recorded.py:
  synth_dataset/{train,test}/{song}/X/{vocals,bass,drums}.wav
  synth_dataset/{train,test}/{song}/Y/{vocals,bass,drums}.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import scipy.signal as sp
import soundfile as sf


STEMS = ("vocals", "bass", "drums")
MIC_NAMES = ("mic0", "mic1", "mic2")
EPS = 1e-12


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x))) + EPS)


def db10_ratio(num: float, den: float) -> float:
    return float(10.0 * np.log10((num + EPS) / (den + EPS)))


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return sanitize(x), int(sr)


def resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return sanitize(x)
    g = math.gcd(int(sr), int(target_sr))
    y = sp.resample_poly(x, up=target_sr // g, down=sr // g)
    return sanitize(y)


def read_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    x, sr = read_audio_mono(path)
    return resample_if_needed(x, sr, target_sr)


def convolve_crop(x: np.ndarray, h: np.ndarray, length: int) -> np.ndarray:
    y = sp.fftconvolve(x.astype(np.float32), h.astype(np.float32), mode="full")
    if y.shape[0] < length:
        y = np.pad(y, (0, length - y.shape[0]))
    return sanitize(y[:length])


def load_rirs(rir_dir: Path, target_sr: int) -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, object]]]:
    rirs: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for stem in STEMS:
        for mic in range(3):
            path = rir_dir / f"{stem}_mic{mic}_ir.wav"
            if not path.is_file():
                missing.append(str(path))
                continue
            h, sr = read_audio_mono(path)
            h = resample_if_needed(h, sr, target_sr)
            h = sanitize(h)
            rirs[(stem, mic)] = h
            rows.append({
                "source": stem,
                "mic": mic,
                "path": str(path),
                "source_sr": sr,
                "target_sr": target_sr,
                "samples": int(h.shape[0]),
                "peak": peak(h),
                "rms": rms(h),
            })

    if missing:
        raise FileNotFoundError("Missing measured RIR files:\n" + "\n".join(missing))
    if len(rirs) != 9:
        raise RuntimeError(f"Expected 9 RIRs, found {len(rirs)}")
    return rirs, rows


def song_names_from_old_split(old_dataset: Path, split: str) -> list[str]:
    split_dir = old_dataset / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Old split not found: {split_dir}")
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])


def find_noise_file(root: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*noise*.wav", "*noise*.flac", "*noise*.npy"):
        candidates.extend(root.rglob(pattern))
    candidates = [p for p in candidates if p.is_file() and "synth_dataset" not in p.parts]
    return sorted(candidates)[0] if candidates else None


def load_noise(noise_path: Path | None, target_sr: int, length: int) -> np.ndarray | None:
    if noise_path is None:
        return None
    if noise_path.suffix.lower() == ".npy":
        z = np.load(noise_path)
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = np.stack([z, z, z], axis=0)
        if z.shape[0] != 3 and z.shape[-1] == 3:
            z = z.T
    else:
        z, sr = read_audio_mono(noise_path)
        z = resample_if_needed(z, sr, target_sr)
        z = np.stack([z, z, z], axis=0)

    if z.ndim != 2 or z.shape[0] != 3:
        raise ValueError(f"Noise must have shape (3,T) or mono, got {z.shape}: {noise_path}")
    if z.shape[1] < length:
        reps = int(np.ceil(length / z.shape[1]))
        z = np.tile(z, (1, reps))
    return sanitize(z[:, :length])


def load_sources(song_dir: Path, target_sr: int) -> tuple[dict[str, np.ndarray], int]:
    sources: dict[str, np.ndarray] = {}
    lengths: list[int] = []
    for stem in STEMS:
        path = song_dir / f"{stem}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Missing MUSDB stem: {path}")
        x = read_mono_resampled(path, target_sr)
        sources[stem] = x
        lengths.append(int(x.shape[0]))

    length = min(lengths)
    for stem in STEMS:
        sources[stem] = sanitize(sources[stem][:length])
    return sources, length


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sanitize(x), sr, subtype="FLOAT")


def generate_song(
    song_name: str,
    split: str,
    musdb_root: Path,
    out_root: Path,
    rirs: dict[tuple[str, int], np.ndarray],
    target_sr: int,
    noise_path: Path | None,
    peak_limit: float,
) -> dict[str, object]:
    song_dir = musdb_root / split / song_name
    sources, length = load_sources(song_dir, target_sr)

    components = np.zeros((3, 3, length), dtype=np.float32)  # source, mic, time
    for si, stem in enumerate(STEMS):
        for mic in range(3):
            components[si, mic] = convolve_crop(sources[stem], rirs[(stem, mic)], length)

    X = np.sum(components, axis=0).astype(np.float32)  # mic, time
    noise = load_noise(noise_path, target_sr, length)
    noise_added = noise is not None
    if noise_added:
        X = sanitize(X + noise)

    Y = np.stack([components[i, i] for i in range(3)], axis=0).astype(np.float32)

    max_abs = max(peak(X), peak(Y))
    global_scale = 1.0
    if max_abs > peak_limit:
        global_scale = float(peak_limit / max_abs)
        X *= global_scale
        Y *= global_scale
        components *= global_scale

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError(f"NaN/Inf detected after synthesis: {split}/{song_name}")
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch for {split}/{song_name}: X={X.shape}, Y={Y.shape}")

    out_song = out_root / "synth_dataset" / split / song_name
    for idx, stem in enumerate(STEMS):
        write_wav(out_song / "X" / f"{stem}.wav", X[idx], target_sr)
        write_wav(out_song / "Y" / f"{stem}.wav", Y[idx], target_sr)

    write_wav(out_song / "mix_input.wav", np.mean(X, axis=0), target_sr)
    write_wav(out_song / "mix_target.wav", np.mean(Y, axis=0), target_sr)

    sir_by_mic = {}
    for mic in range(3):
        target_e = float(np.sum(components[mic, mic].astype(np.float64) ** 2))
        bleed_e = float(np.sum((np.sum(components[:, mic], axis=0) - components[mic, mic]).astype(np.float64) ** 2))
        sir_by_mic[f"mic{mic}"] = db10_ratio(target_e, bleed_e)

    row: dict[str, object] = {
        "split": split,
        "song_id": song_name,
        "sample_rate": target_sr,
        "length": int(length),
        "noise_added": bool(noise_added),
        "noise_file": str(noise_path) if noise_path else "",
        "global_scale": global_scale,
        "x_peak": peak(X),
        "y_peak": peak(Y),
        "x_rms_mean": float(np.mean([rms(X[i]) for i in range(3)])),
        "y_rms_mean": float(np.mean([rms(Y[i]) for i in range(3)])),
        "sir_mic0_db": sir_by_mic["mic0"],
        "sir_mic1_db": sir_by_mic["mic1"],
        "sir_mic2_db": sir_by_mic["mic2"],
    }
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path("/home/rrame12/Desktop/Research/DWT_IR"))
    ap.add_argument("--musdb_root", type=Path, default=Path("/home/rrame12/Desktop/Datasets/musdb18hq"))
    ap.add_argument("--old_dataset", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--target_sr", type=int, default=22050)
    ap.add_argument("--peak_limit", type=float, default=0.99)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    args = ap.parse_args()

    project_root = args.project_root
    measured_root = project_root / "measured_rir_synth_noise"
    rir_dir = measured_root / "rir_wavs"
    old_dataset = args.old_dataset or measured_root / "synth_dataset"
    out_dir = args.out_dir or project_root / "measured_rir_synth_matched"
    synth_out = out_dir / "synth_dataset"

    if synth_out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {synth_out}. Use --overwrite.")
        shutil.rmtree(synth_out)

    rirs, rir_rows = load_rirs(rir_dir, args.target_sr)
    print("[RIR matrix] fixed 3 sources x 3 microphones")
    for row in rir_rows:
        print(
            f"  source={row['source']:6s} mic={row['mic']} "
            f"samples={row['samples']} peak={row['peak']:.6g} path={row['path']}"
        )

    noise_path = find_noise_file(measured_root)
    if noise_path is None:
        print("[Warning] No verified recorded noise WAV/FLAC/NPY found; noise was not added.")
    else:
        print(f"[Noise] using {noise_path}")

    all_rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for split, max_items in (("train", args.max_train), ("test", args.max_test)):
        songs = song_names_from_old_split(old_dataset, split)
        if max_items and max_items > 0:
            songs = songs[:max_items]
        print(f"[Generate] split={split} songs={len(songs)}")
        for idx, song_name in enumerate(songs, start=1):
            row = generate_song(
                song_name=song_name,
                split=split,
                musdb_root=args.musdb_root,
                out_root=out_dir,
                rirs=rirs,
                target_sr=args.target_sr,
                noise_path=noise_path,
                peak_limit=args.peak_limit,
            )
            all_rows.append(row)
            if idx == 1 or idx % 10 == 0 or idx == len(songs):
                print(
                    f"  {split} {idx:3d}/{len(songs):3d} {song_name} "
                    f"len={row['length']} x_peak={row['x_peak']:.4f} y_peak={row['y_peak']:.4f}"
                )
        split_counts[split] = len(songs)

    write_csv(out_dir / "metadata.csv", all_rows)
    write_csv(out_dir / "rir_mapping.csv", rir_rows)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump({
            "formula": "X_m=sum_s dry_s*h[source=s,mic=m]+noise_m; Y_s=dry_s*h[source=s,mic=s]",
            "project_root": str(project_root),
            "musdb_root": str(args.musdb_root),
            "old_dataset_split_source": str(old_dataset),
            "out_dir": str(out_dir),
            "target_sr": int(args.target_sr),
            "stems": list(STEMS),
            "mic_names": list(MIC_NAMES),
            "counts": split_counts,
            "noise_added": bool(noise_path is not None),
            "noise_file": str(noise_path) if noise_path else "",
            "peak_limit": float(args.peak_limit),
        }, f, indent=2)

    print("[Done]")
    print(f"  output: {synth_out}")
    print(f"  train examples: {split_counts.get('train', 0)}")
    print(f"  test examples:  {split_counts.get('test', 0)}")
    print(f"  metadata: {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
