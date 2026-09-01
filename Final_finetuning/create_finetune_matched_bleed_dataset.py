#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pyroomacoustics as pra

EPS = 1e-8


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(x * x) + EPS)


def db20(x):
    return 20.0 * np.log10(np.asarray(x) + EPS)


def ensure_nst(y):
    y = np.asarray(y)
    if y.ndim != 3:
        raise ValueError(f"Expected Y shape (N,S,T), got {y.shape}")
    return y.astype(np.float32, copy=False)


def trim_or_pad(x, T):
    x = np.asarray(x)
    L = x.shape[-1]
    if L == T:
        return x
    if L > T:
        return x[..., :T]
    pad = [(0, 0)] * x.ndim
    pad[-1] = (0, T - L)
    return np.pad(x, pad, mode="constant")


def random_point_in_room(room_dim, margin, rng):
    return np.array([
        rng.uniform(margin, room_dim[0] - margin),
        rng.uniform(margin, room_dim[1] - margin),
        rng.uniform(margin, room_dim[2] - margin),
    ], dtype=np.float64)


def pairwise_far_enough(points, p, min_dist):
    for q in points:
        if np.linalg.norm(p - q) < min_dist:
            return False
    return True


def build_linear_mic_array(center, num_mics, spacing):
    offsets = np.arange(num_mics, dtype=np.float64) - (num_mics - 1) / 2.0
    xs = center[0] + offsets * spacing
    ys = np.full(num_mics, center[1], dtype=np.float64)
    zs = np.full(num_mics, center[2], dtype=np.float64)
    return np.vstack([xs, ys, zs])


def sample_geometry(num_sources, num_mics, room_dim, margin, min_src_src,
                    min_src_mic, mic_spacing, rng, max_tries=500):
    for _ in range(max_tries):
        mic_center = np.array([
            rng.uniform(room_dim[0] * 0.35, room_dim[0] * 0.65),
            rng.uniform(room_dim[1] * 0.35, room_dim[1] * 0.65),
            rng.uniform(1.2, 1.6),
        ], dtype=np.float64)

        mic_locs = build_linear_mic_array(mic_center, num_mics, mic_spacing)

        if np.any(mic_locs[0] < margin) or np.any(mic_locs[0] > room_dim[0] - margin):
            continue
        if np.any(mic_locs[1] < margin) or np.any(mic_locs[1] > room_dim[1] - margin):
            continue
        if np.any(mic_locs[2] < margin) or np.any(mic_locs[2] > room_dim[2] - margin):
            continue

        src_points = []
        ok = True
        for _s in range(num_sources):
            found = False
            for _j in range(max_tries):
                p = random_point_in_room(room_dim, margin, rng)
                if not pairwise_far_enough(src_points, p, min_src_src):
                    continue
                d_to_mics = np.sqrt(np.sum((mic_locs.T - p[None, :]) ** 2, axis=1))
                if np.min(d_to_mics) < min_src_mic:
                    continue
                src_points.append(p)
                found = True
                break
            if not found:
                ok = False
                break

        if ok:
            return mic_locs, np.array(src_points, dtype=np.float64)

    raise RuntimeError("Could not sample valid room geometry.")


def simulate_all_paths_and_rirs(y, fs, room_dim, rt60, mic_locs, src_locs, max_order=12):
    """
    Simulate each source separately.
    Returns:
      X_srcs : (S, M, T)
      rirs   : nested list rirs[m][s]
    """
    num_sources, T = y.shape
    num_mics = mic_locs.shape[1]

    e_absorption, sim_max_order = pra.inverse_sabine(rt60, room_dim)
    if max_order is not None:
        sim_max_order = min(sim_max_order, max_order)

    x_srcs = []
    max_len = 0
    saved_rirs = None

    for s in range(num_sources):
        room_s = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=sim_max_order,
            air_absorption=True,
        )
        room_s.add_microphone_array(pra.MicrophoneArray(mic_locs, fs))
        room_s.add_source(src_locs[s], signal=y[s])
        room_s.compute_rir()
        room_s.simulate()

        xs = room_s.mic_array.signals.copy().astype(np.float32)
        x_srcs.append(xs)
        max_len = max(max_len, xs.shape[-1])

        if saved_rirs is None:
            saved_rirs = [[None for _ in range(num_sources)] for _ in range(num_mics)]
        for m in range(num_mics):
            saved_rirs[m][s] = np.asarray(room_s.rir[m][0], dtype=np.float32)

    x_srcs = [trim_or_pad(xs, max_len) for xs in x_srcs]
    X_srcs = np.stack(x_srcs, axis=0)
    X_srcs = trim_or_pad(X_srcs, T)

    return X_srcs.astype(np.float32), saved_rirs


def convolve_same_len(x, h, T):
    y = np.convolve(x.astype(np.float32), h.astype(np.float32), mode="full")
    return trim_or_pad(y, T).astype(np.float32)


def active_indices(Y, rms_thresh_db=-40.0):
    keep = []
    N, S, _ = Y.shape
    for n in range(N):
        rms_db = [db20(rms(Y[n, s])) for s in range(S)]
        if min(rms_db) > rms_thresh_db:
            keep.append(n)
    return np.array(keep, dtype=np.int64)


def parse_level_counts(spec):
    """
    Example:
      "m40:40,m30:40,m20:40,m18:30,m16:30,m14:30,m12:30,m9:20,m6:20"
    """
    out = {}
    for item in spec.split(","):
        tag, cnt = item.strip().split(":")
        out[tag.strip()] = int(cnt)
    return out


def tag_from_db(db):
    return f"m{abs(int(db))}" if db < 0 else f"{int(db)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1234)

    # Finetune only in dominant-assumption-safe range
    ap.add_argument("--bleed_levels", type=int, nargs="+",
                    default=[-40, -30, -20, -18, -16, -14, -12, -9, -6])

    # Oversample clean / low-bleed
    ap.add_argument(
        "--level_counts",
        type=str,
        default="m40:40,m30:40,m20:40,m18:30,m16:30,m14:30,m12:30,m9:20,m6:20"
    )

    ap.add_argument("--rms_thresh_db", type=float, default=-40.0)
    ap.add_argument("--normalize_source_rms", action="store_true")

    # room params
    ap.add_argument("--fs", type=int, default=22050)
    ap.add_argument("--num_mics", type=int, default=3)
    ap.add_argument("--mic_spacing", type=float, default=0.05)
    ap.add_argument("--room_x_min", type=float, default=4.0)
    ap.add_argument("--room_x_max", type=float, default=8.0)
    ap.add_argument("--room_y_min", type=float, default=4.0)
    ap.add_argument("--room_y_max", type=float, default=7.0)
    ap.add_argument("--room_z_min", type=float, default=2.7)
    ap.add_argument("--room_z_max", type=float, default=3.2)
    ap.add_argument("--rt60_min", type=float, default=0.20)
    ap.add_argument("--rt60_max", type=float, default=0.50)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--min_src_src", type=float, default=0.7)
    ap.add_argument("--min_src_mic", type=float, default=1.0)
    ap.add_argument("--max_order", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Y = ensure_nst(np.load(args.y_path))
    N, S, T = Y.shape
    if args.num_mics != S:
        raise ValueError(f"Need num_mics == num_sources. Got {args.num_mics} vs {S}")

    keep_idx = active_indices(Y, rms_thresh_db=args.rms_thresh_db)
    if len(keep_idx) == 0:
        raise ValueError("No active examples found. Lower --rms_thresh_db.")

    level_counts = parse_level_counts(args.level_counts)
    total_needed = sum(level_counts.values())
    print(f"[Active clips] {len(keep_idx)} available")
    print(f"[Requested] total finetune examples = {total_needed}")

    X_all = []
    Y_all = []
    meta = []

    for bdb in args.bleed_levels:
        tag = tag_from_db(bdb)
        if tag not in level_counts:
            raise ValueError(f"Missing count for bleed level {bdb} (tag {tag}) in --level_counts")
        n_this = level_counts[tag]
        alpha = 10.0 ** (bdb / 20.0)

        print(f"\n[Generate] bleed={bdb} dB  alpha={alpha:.6f}  n={n_this}")

        chosen = rng.choice(keep_idx, size=n_this, replace=(n_this > len(keep_idx)))
        for idx in chosen:
            y = Y[idx].copy()

            if args.normalize_source_rms:
                for s in range(S):
                    r = rms(y[s])
                    if r > EPS:
                        y[s] = y[s] / r

            room_dim = np.array([
                rng.uniform(args.room_x_min, args.room_x_max),
                rng.uniform(args.room_y_min, args.room_y_max),
                rng.uniform(args.room_z_min, args.room_z_max),
            ], dtype=np.float64)
            rt60 = float(rng.uniform(args.rt60_min, args.rt60_max))

            mic_locs, src_locs = sample_geometry(
                num_sources=S,
                num_mics=args.num_mics,
                room_dim=room_dim,
                margin=args.margin,
                min_src_src=args.min_src_src,
                min_src_mic=args.min_src_mic,
                mic_spacing=args.mic_spacing,
                rng=rng,
            )

            X_srcs, rirs = simulate_all_paths_and_rirs(
                y=y,
                fs=args.fs,
                room_dim=room_dim,
                rt60=rt60,
                mic_locs=mic_locs,
                src_locs=src_locs,
                max_order=args.max_order,
            )

            x_mix = np.zeros((S, T), dtype=np.float32)
            y_tgt = np.zeros((S, T), dtype=np.float32)

            for m in range(S):
                target = X_srcs[m, m]  # desired diagonal contribution at matched mic
                bleed = np.zeros((T,), dtype=np.float32)
                for k in range(S):
                    if k == m:
                        continue
                    bleed += X_srcs[k, m]

                x_mix[m] = target + alpha * bleed

                h_jj = np.asarray(rirs[m][m], dtype=np.float32)
                y_tgt[m] = convolve_same_len(y[m], h_jj, T)

            X_all.append(x_mix.astype(np.float32))
            Y_all.append(y_tgt.astype(np.float32))
            meta.append({
                "source_index": int(idx),
                "nominal_bleed_db": int(bdb),
                "alpha": float(alpha),
                "rt60": rt60,
                "room_dim": room_dim.tolist(),
            })

    X_all = np.stack(X_all, axis=0).astype(np.float32)
    Y_all = np.stack(Y_all, axis=0).astype(np.float32)

    np.save(os.path.join(args.out_dir, "Xtrain_finetune.npy"), X_all)
    np.save(os.path.join(args.out_dir, "Ytrain_finetune.npy"), Y_all)

    with open(os.path.join(args.out_dir, "finetune_dataset_meta.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "shape_X": list(X_all.shape),
            "shape_Y": list(Y_all.shape),
            "num_examples": int(X_all.shape[0]),
            "examples": meta,
        }, f, indent=2)

    print("\nSaved:")
    print(os.path.join(args.out_dir, "Xtrain_finetune.npy"))
    print(os.path.join(args.out_dir, "Ytrain_finetune.npy"))
    print(f"Final shapes: X={X_all.shape}, Y={Y_all.shape}")


if __name__ == "__main__":
    main()