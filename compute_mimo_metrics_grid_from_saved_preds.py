#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from bleed_matrix_tf import BleedMatrixTF  # provides sir_db, ltr_db, unmodeled_ratio


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x)), float(np.std(x))


def _offdiag_energy_from_E(E: np.ndarray) -> np.ndarray:
    """
    Off-diagonal energy per mic/channel from energy matrix E.

    E: (M, S)
    off_m = sum_s E[m,s] - E[m,m]
    returns: (M,) float64 >= 0
    """
    E = np.asarray(E, dtype=np.float64)
    M, S = E.shape
    tot = np.sum(E, axis=1)  # (M,)
    diag = np.array([E[m, m] if m < S else 0.0 for m in range(M)], dtype=np.float64)
    off = np.maximum(tot - diag, 0.0)
    return off


def _br_db(off_in: np.ndarray, off_out: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Bleed Reduction in dB per mic/channel:
      BR_m = 10 log10(off_in / off_out)
    Positive is good (less bleed after model).
    """
    off_in = np.asarray(off_in, dtype=np.float64)
    off_out = np.asarray(off_out, dtype=np.float64)
    return (10.0 * np.log10(np.maximum(off_in, eps) / np.maximum(off_out, eps))).astype(np.float32)


def compute_mimo_metrics_br(
    bm: BleedMatrixTF,
    y_true: np.ndarray,   # (N, S, T) ground-truth stems (reference sources)
    x_mix:  np.ndarray,   # (N, M, T) mixture channels (input)
    y_pred: np.ndarray,   # (N, S, T) predicted stems (output)
):
    """
    Computes (per example, per channel):
      - LTR_out: LTR on output (sources=y_true, mics=y_pred)  [dB]
      - U_out  : unmodeled ratio on output                     [0..1]
      - BR(dB) : bleed reduction dB comparing mixture -> output

    BR uses off-diagonal energies from E matrices:
      E_in  = bm.compute(y_true[n], x_mix[n]).E
      E_out = bm.compute(y_true[n], y_pred[n]).E
      off = sum_offdiag(E)
      BR = 10log10(off_in/off_out)
    """
    N, S, T = y_true.shape
    N2, M, T2 = x_mix.shape
    N3, S3, T3 = y_pred.shape
    assert N == N2 == N3 and T == T2 == T3 and S == S3, (y_true.shape, x_mix.shape, y_pred.shape)

    ltr_out = np.zeros((N, M), dtype=np.float32)
    u_out   = np.zeros((N, M), dtype=np.float32)
    br_db   = np.zeros((N, M), dtype=np.float32)

    for n in tqdm(range(N), desc="MIMO metrics (BR)", ncols=100):
        # Baseline bleed (mixture)
        r_in = bm.compute(y_true[n], x_mix[n])
        off_in = _offdiag_energy_from_E(r_in.E)  # (M,)

        # Output bleed (after model): treat output stems as "mics"
        r_out = bm.compute(y_true[n], y_pred[n])
        off_out = _offdiag_energy_from_E(r_out.E)

        # Output LTR and U (these correspond to output quality wrt sources)
        ltr_out[n] = r_out.ltr_db
        u_out[n]   = r_out.unmodeled_ratio

        # Bleed reduction
        br_db[n] = _br_db(off_in, off_out)

    return ltr_out, u_out, br_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")

    # mixture channels (input)
    ap.add_argument("--xfile", type=str, default="Xtest.npy")

    # ground-truth stems (reference sources)
    ap.add_argument("--ytrue_file", type=str, default="Ytest.npy")

    ap.add_argument("--T", type=int, default=220448)

    ap.add_argument("--pred_root", type=str, default="./runs_pr_ablation/results_grid/per_model")
    ap.add_argument("--out_dir", type=str, default="./runs_pr_ablation/results_grid/stem_tables")

    ap.add_argument("--levels", type=int, nargs="+", default=[2,3,5])
    ap.add_argument("--filters", type=int, nargs="+", default=[11,101,1001])

    # BleedMatrixTF params
    ap.add_argument("--K", type=int, default=512, help="FIR filter length")
    ap.add_argument("--lam", type=float, default=1e-3, help="Tikhonov regularization")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--max_delay_samples", type=int, default=44100)
    ap.add_argument("--use_envelope", type=int, default=1)

    # stem names / ordering
    ap.add_argument("--stem_names", type=str, nargs="+", default=["Vocal", "Bass", "Drums"])

    # optionally limit for quick debug
    ap.add_argument("--max_items", type=int, default=-1, help="if >0, evaluate only first N items")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load mixture and ground-truth
    X = np.load(os.path.join(args.dpath, args.xfile)).astype(np.float32)[:, :, :args.T]      # (N,M,T)
    Ytrue = np.load(os.path.join(args.dpath, args.ytrue_file)).astype(np.float32)[:, :, :args.T]  # (N,S,T)

    if args.max_items > 0:
        X = X[:args.max_items]
        Ytrue = Ytrue[:args.max_items]

    N, M, T = X.shape
    N2, S, T2 = Ytrue.shape
    assert N == N2 and T == T2, (X.shape, Ytrue.shape)
    assert M == len(args.stem_names) and S == len(args.stem_names), (
        f"Expected {len(args.stem_names)} channels/stems, got M={M}, S={S}"
    )

    bm = BleedMatrixTF(
        K=args.K, lam=args.lam, sr=args.sr,
        max_delay_samples=args.max_delay_samples,
        use_envelope=bool(args.use_envelope),
        verbose=False
    )

    summary_rows = []
    long_rows = []

    for L in args.levels:
        for F in args.filters:
            tag = f"L{L}_F{F}"
            pred_path = os.path.join(args.pred_root, tag, "Ypred.npy")
            if not os.path.exists(pred_path):
                print(f"[SKIP] missing {pred_path}")
                continue

            Ypred = np.load(pred_path).astype(np.float32)[:, :, :args.T]
            if args.max_items > 0:
                Ypred = Ypred[:args.max_items]

            print("\n" + "="*80)
            print(f"{tag}: computing LTR_out, U_out, and BR(dB) (mixture -> output)")
            print(f"Using: E_in=compute(Ytrue, Xmix), E_out=compute(Ytrue, Ypred)")
            print(f"BleedMatrixTF(K={args.K}, lam={args.lam}, use_envelope={bool(args.use_envelope)})")
            print("="*80)

            ltr_out, u_out, br = compute_mimo_metrics_br(bm, Ytrue, X, Ypred)

            stem_stats = {}
            for mi, stem in enumerate(args.stem_names):
                stem_stats[stem] = {
                    "LTR_mean": mean_std(ltr_out[:, mi])[0],
                    "LTR_std":  mean_std(ltr_out[:, mi])[1],
                    "U_mean":   mean_std(u_out[:, mi])[0],
                    "U_std":    mean_std(u_out[:, mi])[1],
                    "BR_mean":  mean_std(br[:, mi])[0],
                    "BR_std":   mean_std(br[:, mi])[1],
                }

                long_rows += [
                    {"tag": tag, "levels": L, "filter_size": F, "stem": stem, "metric": "LTR", "mean": stem_stats[stem]["LTR_mean"], "std": stem_stats[stem]["LTR_std"]},
                    {"tag": tag, "levels": L, "filter_size": F, "stem": stem, "metric": "U",   "mean": stem_stats[stem]["U_mean"],   "std": stem_stats[stem]["U_std"]},
                    {"tag": tag, "levels": L, "filter_size": F, "stem": stem, "metric": "BR",  "mean": stem_stats[stem]["BR_mean"],  "std": stem_stats[stem]["BR_std"]},
                ]

            row = {"Levels": L, "Filter Size": F}
            for stem in args.stem_names:
                row[f"{stem}_LTR"]     = stem_stats[stem]["LTR_mean"]
                row[f"{stem}_LTR_std"] = stem_stats[stem]["LTR_std"]
                row[f"{stem}_U"]       = stem_stats[stem]["U_mean"]
                row[f"{stem}_U_std"]   = stem_stats[stem]["U_std"]
                row[f"{stem}_BR"]      = stem_stats[stem]["BR_mean"]
                row[f"{stem}_BR_std"]  = stem_stats[stem]["BR_std"]
            summary_rows.append(row)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(summary_rows).sort_values(["Levels", "Filter Size"])

    out_long = os.path.join(args.out_dir, "mimo_metrics_long.csv")
    out_wide = os.path.join(args.out_dir, "mimo_metrics_wide.csv")
    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    cfg_out = os.path.join(args.out_dir, "mimo_metrics_config.json")
    with open(cfg_out, "w") as f:
        json.dump({
            "K": args.K, "lam": args.lam, "sr": args.sr,
            "max_delay_samples": args.max_delay_samples,
            "use_envelope": bool(args.use_envelope),
            "T": args.T,
            "pred_root": args.pred_root,
            "xfile": args.xfile,
            "ytrue_file": args.ytrue_file,
            "stems": args.stem_names,
            "max_items": args.max_items,
            "notes": "Computed BR(dB)=10log10(off_in/off_out) with off=sum off-diagonal energies from E. "
                     "E_in=compute(Ytrue,Xmix); E_out=compute(Ytrue,Ypred). Also stores LTR_out and U_out from E_out.",
        }, f, indent=2)

    print("\n[Saved]", out_long)
    print("[Saved]", out_wide)
    print("[Saved]", cfg_out)

    print("\n===== QUICK SUMMARY (mean ± std) =====")
    for _, r in df_wide.iterrows():
        L = int(r["Levels"]); F = int(r["Filter Size"])
        print(f"L{L}_F{F}: ", end="")
        for stem in args.stem_names:
            print(
                f"{stem} LTR={r[f'{stem}_LTR']:.2f}±{r[f'{stem}_LTR_std']:.2f}, "
                f"U={r[f'{stem}_U']:.4f}±{r[f'{stem}_U_std']:.4f}, "
                f"BR={r[f'{stem}_BR']:.2f}±{r[f'{stem}_BR_std']:.2f} dB | ",
                end=""
            )
        print("")

if __name__ == "__main__":
    main()