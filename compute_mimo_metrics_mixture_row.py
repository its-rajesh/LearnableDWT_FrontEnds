#!/usr/bin/env python3
import os, argparse, json
import numpy as np
from tqdm import tqdm
from bleed_matrix_tf import BleedMatrixTF  # sir_db, ltr_db, unmodeled_ratio

def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std())

def _rbr_from_E(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    RBR per mic from energy matrix E (M,S):
      RBR_m = offdiag_energy / total_energy
            = (sum_s E[m,s] - E[m,m]) / sum_s E[m,s]
    returns (M,) in [0,1], NaN-safe.
    """
    E = np.asarray(E, dtype=np.float64)
    M, S = E.shape
    tot = np.sum(E, axis=1)  # (M,)

    diag = np.array([E[m, m] if m < S else 0.0 for m in range(M)], dtype=np.float64)
    off  = np.maximum(tot - diag, 0.0)

    rbr = off / np.maximum(tot, eps)
    return np.clip(rbr, 0.0, 1.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpath", type=str, default="/home/rrame12/Desktop/Research/DWT_IR")
    ap.add_argument("--xfile", type=str, default="Xtest.npy")
    ap.add_argument("--yfile", type=str, default="Ytest.npy")
    ap.add_argument("--T", type=int, default=220448)

    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--max_delay_samples", type=int, default=44100)
    ap.add_argument("--use_envelope", type=int, default=1)

    ap.add_argument("--stem_names", type=str, nargs="+", default=["Vocal","Bass","Drums"])
    ap.add_argument("--out_dir", type=str, default="./runs_pr_ablation/results_grid/stem_tables")
    ap.add_argument("--max_items", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(os.path.join(args.dpath, args.xfile)).astype(np.float32)[:, :, :args.T]
    Y = np.load(os.path.join(args.dpath, args.yfile)).astype(np.float32)[:, :, :args.T]
    if args.max_items > 0:
        X = X[:args.max_items]
        Y = Y[:args.max_items]

    N, M, T = X.shape
    assert Y.shape == (N, M, T)
    assert M == len(args.stem_names)

    bm = BleedMatrixTF(
        K=args.K, lam=args.lam, sr=args.sr,
        max_delay_samples=args.max_delay_samples,
        use_envelope=bool(args.use_envelope),
        verbose=False
    )

    sirb = np.zeros((N, M), np.float32)
    ltr  = np.zeros((N, M), np.float32)
    u    = np.zeros((N, M), np.float32)

    # NEW: RBR (fraction + percent)
    rbr     = np.zeros((N, M), np.float32)
    rbr_pct = np.zeros((N, M), np.float32)

    # Mixture row = use ground-truth sources Y, observed mics X
    for n in tqdm(range(N), desc="Mixture MIMO metrics", ncols=100):
        r = bm.compute(Y[n], X[n])  # sources=Y, mics=X
        sirb[n] = r.sir_db
        ltr[n]  = r.ltr_db
        u[n]    = r.unmodeled_ratio

        # NEW: compute RBR from energy matrix
        rbr_n = _rbr_from_E(r.E)     # (M,)
        rbr[n] = rbr_n
        rbr_pct[n] = 100.0 * rbr_n

    out = {
        "tag":"Mixture",
        "K":args.K, "lam":args.lam, "sr":args.sr, "T":args.T,
        "stems":args.stem_names,
        "values":{}
    }

    for i, stem in enumerate(args.stem_names):
        mu_sirb, sd_sirb = mean_std(sirb[:, i])
        mu_ltr,  sd_ltr  = mean_std(ltr[:, i])
        mu_u,    sd_u    = mean_std(u[:, i])

        mu_rbr,  sd_rbr  = mean_std(rbr[:, i])
        mu_rbrp, sd_rbrp = mean_std(rbr_pct[:, i])

        # explainability in %
        mu_exp = (1.0 - mu_u) * 100.0
        sd_exp = sd_u * 100.0  # std scales linearly

        out["values"][stem] = {
            "SIRB_mean": mu_sirb, "SIRB_std": sd_sirb,
            "LTR_mean":  mu_ltr,  "LTR_std":  sd_ltr,
            "U_mean":    mu_u,    "U_std":    sd_u,
            "EXP_mean":  mu_exp,  "EXP_std":  sd_exp,

            # NEW
            "RBR_mean":      mu_rbr,   "RBR_std":    sd_rbr,
            "RBRpct_mean":   mu_rbrp,  "RBRpct_std": sd_rbrp,
        }

    # Save JSON
    jpath = os.path.join(args.out_dir, "mixture_mimo_metrics.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)

    # Print LaTeX-ready fields (2 decimals)
    def fm(mu, sd): return f"{mu:.2f} $\\pm$ {sd:.2f}"
    def fp(mu, sd): return f"{mu:.2f}\\% $\\pm$ {sd:.2f}\\%"

    print("\n=== Mixture row (paste into table) ===")
    parts = []
    for stem in args.stem_names:
        v = out["values"][stem]

        # If your table order is: SI-SDR, SIR, SAR, LTR, SIR(B), U(Expl%)
        # here we only print: LTR, SIR(B), RBR(%), Expl(%)
        parts.append(fm(v["LTR_mean"],  v["LTR_std"]))        # LTR (dB)
        parts.append(fm(v["SIRB_mean"], v["SIRB_std"]))       # SIR(B) (dB)
        parts.append(fp(v["RBRpct_mean"], v["RBRpct_std"]))   # RBR (% bleed)
        parts.append(fp(v["EXP_mean"],  v["EXP_std"]))        # Explainability (%)

    print("LTR / SIR(B) / RBR(%) / U(Expl%) per stem in order Vocal,Bass,Drums:")
    print(" | ".join(parts))

    print("\n[Saved]", jpath)

if __name__ == "__main__":
    main()