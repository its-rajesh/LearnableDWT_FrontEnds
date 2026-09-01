#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "bass", "drums")


def read_mono(path: Path) -> np.ndarray:
    x, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.nan_to_num(np.asarray(x, dtype=np.float32))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x))) + 1e-12)


def activity_score(Y: np.ndarray, min_rms: float, min_peak: float, min_active: int) -> tuple[bool, float, list[float], list[float]]:
    rr = [rms(Y[c]) for c in range(Y.shape[0])]
    pp = [peak(Y[c]) for c in range(Y.shape[0])]
    active = [(rr[c] >= min_rms) and (pp[c] >= min_peak) for c in range(Y.shape[0])]
    score = float(np.min(rr) + 0.01 * np.mean(rr))
    return int(sum(active)) >= int(min_active), score, rr, pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="Path containing train/ and test/ song folders")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--crop_T", type=int, default=0)
    ap.add_argument("--active_crops", type=int, default=0, help="If >0, choose this many deterministic active crops per song")
    ap.add_argument("--scan_hop", type=int, default=22050, help="Hop size at dataset sample rate for active crop search")
    ap.add_argument("--min_active_stems", type=int, default=3)
    ap.add_argument("--stem_rms_thresh", type=float, default=1e-4)
    ap.add_argument("--stem_peak_thresh", type=float, default=1e-3)
    args = ap.parse_args()

    song_dirs = sorted([p for p in (args.dataset / args.split).iterdir() if p.is_dir()])
    if args.max_items > 0:
        song_dirs = song_dirs[: args.max_items]

    xs, ys, meta = [], [], []
    for song_dir in song_dirs:
        x_ch, y_ch, lengths = [], [], []
        for stem in STEMS:
            x = read_mono(song_dir / "X" / f"{stem}.wav")
            y = read_mono(song_dir / "Y" / f"{stem}.wav")
            x_ch.append(x)
            y_ch.append(y)
            lengths.extend([len(x), len(y)])
        full_T = min(lengths)
        crop_T = min(full_T, args.crop_T) if args.crop_T > 0 else full_T
        Xfull = np.stack([c[:full_T] for c in x_ch], axis=0)
        Yfull = np.stack([c[:full_T] for c in y_ch], axis=0)

        if args.active_crops > 0 and crop_T < full_T:
            candidates = []
            hop = max(1, int(args.scan_hop))
            for start in range(0, full_T - crop_T + 1, hop):
                Yc = Yfull[:, start:start + crop_T]
                ok, score, rr, pp = activity_score(
                    Yc,
                    min_rms=args.stem_rms_thresh,
                    min_peak=args.stem_peak_thresh,
                    min_active=args.min_active_stems,
                )
                candidates.append((ok, score, start, rr, pp))
            valid = [c for c in candidates if c[0]]
            chosen = sorted(valid if valid else candidates, key=lambda z: z[1], reverse=True)[: args.active_crops]
            chosen = sorted(chosen, key=lambda z: z[2])
            for ok, score, start, rr, pp in chosen:
                xs.append(Xfull[:, start:start + crop_T])
                ys.append(Yfull[:, start:start + crop_T])
                meta.append({
                    "song": song_dir.name,
                    "split": args.split,
                    "start": int(start),
                    "samples": int(crop_T),
                    "active_ok": bool(ok),
                    "score": float(score),
                    "target_rms": rr,
                    "target_peak": pp,
                })
        else:
            xs.append(Xfull[:, :crop_T])
            ys.append(Yfull[:, :crop_T])
            ok, score, rr, pp = activity_score(Yfull[:, :crop_T], args.stem_rms_thresh, args.stem_peak_thresh, args.min_active_stems)
            meta.append({
                "song": song_dir.name,
                "split": args.split,
                "start": 0,
                "samples": int(crop_T),
                "active_ok": bool(ok),
                "score": float(score),
                "target_rms": rr,
                "target_peak": pp,
            })

    T = min(a.shape[-1] for a in xs + ys)
    X = np.stack([a[:, :T] for a in xs], axis=0).astype(np.float32)
    Y = np.stack([a[:, :T] for a in ys], axis=0).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"X_{args.split}.npy", X)
    np.save(args.out_dir / f"Y_{args.split}.npy", Y)
    np.save(args.out_dir / f"Ypred_reference_{args.split}.npy", X)
    (args.out_dir / f"songs_{args.split}.txt").write_text("\n".join(p.name for p in song_dirs) + "\n")
    (args.out_dir / f"crop_meta_{args.split}.json").write_text(json.dumps(meta, indent=2))
    print(f"X={X.shape} Y={Y.shape}")
    print(f"Saved {args.out_dir}")
    print(f"Active crops ok: {sum(1 for m in meta if m['active_ok'])}/{len(meta)}")


if __name__ == "__main__":
    main()
