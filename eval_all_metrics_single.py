#!/usr/bin/env python3
"""
Single-script evaluation for multitrack bleed-reduction / interference-reduction models.
Computes standard metrics before/after and MIMO metrics before/after.
"""

import os, csv, json, argparse, itertools
import numpy as np
from tqdm import tqdm
from bleed_matrix_tf import BleedMatrixTF


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def mean_std_med(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan'), float('nan'), float('nan')
    return float(x.mean()), float(np.median(x)), float(x.std())


def estimate_lag(ref, est, max_lag, use_envelope=True):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    T = min(len(ref), len(est))
    ref = ref[:T]
    est = est[:T]
    refp = np.abs(ref) if use_envelope else ref
    estp = np.abs(est) if use_envelope else est
    refp -= refp.mean(); estp -= estp.mean()
    max_lag = int(min(max_lag, T - 1))
    if max_lag <= 0:
        return 0
    n_fft = 1 << (2 * T - 1).bit_length()
    R = np.fft.rfft(refp, n=n_fft)
    E = np.fft.rfft(estp, n=n_fft)
    xcorr = np.fft.irfft(np.conj(R) * E, n=n_fft)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    idx = (lags % n_fft).astype(int)
    return int(lags[np.argmax(xcorr[idx])])


def shift_signal(x, lag, out_len=None):
    x = np.asarray(x)
    T = len(x) if out_len is None else int(out_len)
    y = np.zeros(T, dtype=x.dtype)
    if lag > 0:
        keep = min(len(x) - lag, T)
        if keep > 0:
            y[:keep] = x[lag:lag + keep]
    elif lag < 0:
        d = -lag
        keep = min(len(x), T - d)
        if keep > 0:
            y[d:d + keep] = x[:keep]
    else:
        keep = min(len(x), T)
        y[:keep] = x[:keep]
    return y


def align_to_ref(ref, est, max_lag, use_envelope=True):
    lag = estimate_lag(ref, est, max_lag=max_lag, use_envelope=use_envelope)
    est_al = shift_signal(est, lag, out_len=len(ref))
    return ref, est_al, lag


def sisdr_np(y, yhat, eps=1e-8):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    y0 = y - y.mean(); x0 = yhat - yhat.mean()
    dot = np.sum(y0 * x0)
    den = np.sum(y0 * y0) + eps
    a = dot / den
    s = a * y0
    e = x0 - s
    return float(10.0 * np.log10((np.sum(s * s) + eps) / (np.sum(e * e) + eps)))


def sisnr_np(y, yhat, eps=1e-8):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    y0 = y - y.mean(); x0 = yhat - yhat.mean()
    s_target = np.sum(x0 * y0) * y0 / (np.sum(y0 * y0) + eps)
    e_noise = x0 - s_target
    return float(10.0 * np.log10((np.sum(s_target * s_target) + eps) / (np.sum(e_noise * e_noise) + eps)))


def sir_sar_np(y, interf, yhat, eps=1e-8):
    y0 = np.asarray(y, dtype=np.float64) - np.mean(y)
    v0 = np.asarray(interf, dtype=np.float64) - np.mean(interf)
    x0 = np.asarray(yhat, dtype=np.float64) - np.mean(yhat)
    if np.sum(v0 * v0) < eps:
        sdr = sisdr_np(y0, x0, eps)
        return sdr, sdr
    n1 = np.sqrt(np.sum(y0 * y0) + eps)
    u1 = y0 / n1
    vproj = v0 - np.sum(v0 * u1) * u1
    n2 = np.sqrt(np.sum(vproj * vproj) + eps)
    if n2 < eps:
        sdr = sisdr_np(y0, x0, eps)
        return sdr, sdr
    u2 = vproj / n2
    c1 = np.sum(x0 * u1); c2 = np.sum(x0 * u2)
    s_target = c1 * u1; s_interf = c2 * u2
    s_tot = s_target + s_interf; art = x0 - s_tot
    SIR = 10.0 * np.log10((np.sum(s_target * s_target) + eps) / (np.sum(s_interf * s_interf) + eps))
    SAR = 10.0 * np.log10((np.sum(s_tot * s_tot) + eps) / (np.sum(art * art) + eps))
    return float(SIR), float(SAR)


def permute_prediction_to_reference(Ytrue_ex, Ypred_ex, max_lag, use_envelope):
    C, T = Ytrue_ex.shape
    best_score = -1e18
    best_perm = tuple(range(C))
    best_lags = [0] * C
    best_pred = Ypred_ex.copy()
    for perm in itertools.permutations(range(C)):
        vals = []
        lags = []
        ypred_perm = np.zeros_like(Ypred_ex)
        for ch in range(C):
            ref = Ytrue_ex[ch]
            est = Ypred_ex[perm[ch]]
            ref, est_al, lag = align_to_ref(ref, est, max_lag=max_lag, use_envelope=use_envelope)
            ypred_perm[ch] = est_al
            vals.append(sisdr_np(ref, est_al))
            lags.append(lag)
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score; best_perm = perm; best_lags = lags; best_pred = ypred_perm
    return best_pred, best_perm, best_lags, best_score


def compute_standard_metrics(Xmix, Ytrue, Ypred, stem_names, align_std=True, std_max_lag=22050, std_use_envelope=True, permute_pred=False):
    N, C, T = Ytrue.shape
    rows = []; perm_rows = []
    for n in tqdm(range(N), desc='Standard metrics', ncols=100):
        ypred_ex = Ypred[n]
        if permute_pred:
            ypred_eval, perm, perm_lags, perm_score = permute_prediction_to_reference(Ytrue[n], ypred_ex, std_max_lag, std_use_envelope)
            perm_rows.append({'idx': n, 'perm': ','.join(map(str, perm)), 'perm_mean_aligned_sisdr': float(perm_score), 'lags': ','.join(map(str, perm_lags))})
        else:
            ypred_eval = ypred_ex
        for ch in range(C):
            y = Ytrue[n, ch]; x_in = Xmix[n, ch]; yhat = ypred_eval[ch]
            if align_std:
                _, x_eval, lag_in = align_to_ref(y, x_in, max_lag=std_max_lag, use_envelope=std_use_envelope)
                _, y_eval, lag_out = align_to_ref(y, yhat, max_lag=std_max_lag, use_envelope=std_use_envelope)
            else:
                x_eval = x_in; y_eval = yhat; lag_in = 0; lag_out = 0
            interf = np.sum(Ytrue[n, np.arange(C) != ch], axis=0)
            sisdr_in = sisdr_np(y, x_eval); sisnr_in = sisnr_np(y, x_eval); sir_in, sar_in = sir_sar_np(y, interf, x_eval)
            sisdr = sisdr_np(y, y_eval); sisnr = sisnr_np(y, y_eval); sir, sar = sir_sar_np(y, interf, y_eval)
            rows.append({'idx': n, 'ch': ch, 'stem': stem_names[ch], 'lag_in': int(lag_in), 'lag_out': int(lag_out), 'sisdr_in': sisdr_in, 'sisnr_in': sisnr_in, 'sir_in': sir_in, 'sar_in': sar_in, 'sisdr': sisdr, 'sisnr': sisnr, 'sir': sir, 'sar': sar, 'sisdr_impr': sisdr - sisdr_in, 'sisnr_impr': sisnr - sisnr_in, 'sir_impr': sir - sir_in, 'sar_impr': sar - sar_in})
    summary_keys = ['sisdr_in', 'sisnr_in', 'sir_in', 'sar_in', 'sisdr', 'sisnr', 'sir', 'sar', 'sisdr_impr', 'sisnr_impr', 'sir_impr', 'sar_impr']
    summary = {}
    for k in summary_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        mu, med, sd = mean_std_med(vals)
        summary[k] = {'mean': mu, 'median': med, 'std': sd}
    return rows, summary, perm_rows


def _diag_from_E(E):
    E = np.asarray(E, dtype=np.float64)
    M, S = E.shape
    return np.array([E[m, m] if m < S else 0.0 for m in range(M)], dtype=np.float64)


def _offdiag_energy_from_E(E):
    E = np.asarray(E, dtype=np.float64)
    tot = np.sum(E, axis=1)
    diag = _diag_from_E(E)
    return np.maximum(tot - diag, 0.0)


def _rbr_from_E(E, eps=1e-12):
    E = np.asarray(E, dtype=np.float64)
    tot = np.sum(E, axis=1)
    diag = _diag_from_E(E)
    off = np.maximum(tot - diag, 0.0)
    rbr = off / np.maximum(tot, eps)
    return np.clip(rbr, 0.0, 1.0).astype(np.float32)


def _br_db(off_in, off_out, eps=1e-12):
    off_in = np.asarray(off_in, dtype=np.float64)
    off_out = np.asarray(off_out, dtype=np.float64)
    return (10.0 * np.log10(np.maximum(off_in, eps) / np.maximum(off_out, eps))).astype(np.float32)


def summarize_per_stem(arr, stem_names):
    out = {}
    for i, stem in enumerate(stem_names):
        mu, med, sd = mean_std_med(arr[:, i])
        out[stem] = {'mean': mu, 'median': med, 'std': sd}
    return out


def compute_mimo_all(bm, Xmix, Ytrue, Ypred, stem_names):
    N, C, T = Ytrue.shape
    sirb_in = np.zeros((N, C), np.float32); ltr_in = np.zeros((N, C), np.float32); u_in = np.zeros((N, C), np.float32); u0_in = np.zeros((N, C), np.float32); ulegacy_in = np.zeros((N, C), np.float32); exp_in = np.zeros((N, C), np.float32); exp0_in = np.zeros((N, C), np.float32); explegacy_in = np.zeros((N, C), np.float32); rbr_in = np.zeros((N, C), np.float32); rbrp_in = np.zeros((N, C), np.float32); B_in = np.zeros((N, C, C), np.float32); E_in = np.zeros((N, C, C), np.float32)
    sirb_out = np.zeros((N, C), np.float32); ltr_out = np.zeros((N, C), np.float32); u_out = np.zeros((N, C), np.float32); u0_out = np.zeros((N, C), np.float32); ulegacy_out = np.zeros((N, C), np.float32); exp_out = np.zeros((N, C), np.float32); exp0_out = np.zeros((N, C), np.float32); explegacy_out = np.zeros((N, C), np.float32); rbr_out = np.zeros((N, C), np.float32); rbrp_out = np.zeros((N, C), np.float32); B_out = np.zeros((N, C, C), np.float32); E_out = np.zeros((N, C, C), np.float32)
    br = np.zeros((N, C), np.float32)
    for n in tqdm(range(N), desc='MIMO all metrics', ncols=100):
        rin = bm.compute(Ytrue[n], Xmix[n])
        rout = bm.compute(Ytrue[n], Ypred[n])
        sirb_in[n] = rin.sir_db; ltr_in[n] = rin.ltr_db; u_in[n] = rin.unmodeled_ratio; u0_in[n] = rin.unmodeled_ratio_uncentered; ulegacy_in[n] = rin.legacy_unmodeled_ratio; exp_in[n] = (1.0 - rin.unmodeled_ratio) * 100.0; exp0_in[n] = (1.0 - rin.unmodeled_ratio_uncentered) * 100.0; explegacy_in[n] = (1.0 - rin.legacy_unmodeled_ratio) * 100.0; rbr_in[n] = _rbr_from_E(rin.E); rbrp_in[n] = rbr_in[n] * 100.0; B_in[n] = rin.B; E_in[n] = rin.E
        sirb_out[n] = rout.sir_db; ltr_out[n] = rout.ltr_db; u_out[n] = rout.unmodeled_ratio; u0_out[n] = rout.unmodeled_ratio_uncentered; ulegacy_out[n] = rout.legacy_unmodeled_ratio; exp_out[n] = (1.0 - rout.unmodeled_ratio) * 100.0; exp0_out[n] = (1.0 - rout.unmodeled_ratio_uncentered) * 100.0; explegacy_out[n] = (1.0 - rout.legacy_unmodeled_ratio) * 100.0; rbr_out[n] = _rbr_from_E(rout.E); rbrp_out[n] = rbr_out[n] * 100.0; B_out[n] = rout.B; E_out[n] = rout.E
        br[n] = _br_db(_offdiag_energy_from_E(rin.E), _offdiag_energy_from_E(rout.E))
    summary = {
        'mixture_before': {'SIRB': summarize_per_stem(sirb_in, stem_names), 'LTR': summarize_per_stem(ltr_in, stem_names), 'U': summarize_per_stem(u_in, stem_names), 'U0': summarize_per_stem(u0_in, stem_names), 'U_legacy': summarize_per_stem(ulegacy_in, stem_names), 'EXP': summarize_per_stem(exp_in, stem_names), 'EXP0': summarize_per_stem(exp0_in, stem_names), 'EXP_legacy': summarize_per_stem(explegacy_in, stem_names), 'RBR': summarize_per_stem(rbr_in, stem_names), 'RBRpct': summarize_per_stem(rbrp_in, stem_names), 'bleed_matrix_mean': np.mean(B_in, axis=0).tolist(), 'energy_matrix_mean': np.mean(E_in, axis=0).tolist()},
        'output_after': {'SIRB': summarize_per_stem(sirb_out, stem_names), 'LTR': summarize_per_stem(ltr_out, stem_names), 'U': summarize_per_stem(u_out, stem_names), 'U0': summarize_per_stem(u0_out, stem_names), 'U_legacy': summarize_per_stem(ulegacy_out, stem_names), 'EXP': summarize_per_stem(exp_out, stem_names), 'EXP0': summarize_per_stem(exp0_out, stem_names), 'EXP_legacy': summarize_per_stem(explegacy_out, stem_names), 'RBR': summarize_per_stem(rbr_out, stem_names), 'RBRpct': summarize_per_stem(rbrp_out, stem_names), 'bleed_matrix_mean': np.mean(B_out, axis=0).tolist(), 'energy_matrix_mean': np.mean(E_out, axis=0).tolist()},
        'bleed_reduction': {'BR': summarize_per_stem(br, stem_names)}
    }
    per_example = {'sirb_in': sirb_in, 'ltr_in': ltr_in, 'u_in': u_in, 'u0_in': u0_in, 'ulegacy_in': ulegacy_in, 'exp_in': exp_in, 'exp0_in': exp0_in, 'explegacy_in': explegacy_in, 'rbr_in': rbr_in, 'rbrp_in': rbrp_in, 'B_in': B_in, 'E_in': E_in, 'sirb_out': sirb_out, 'ltr_out': ltr_out, 'u_out': u_out, 'u0_out': u0_out, 'ulegacy_out': ulegacy_out, 'exp_out': exp_out, 'exp0_out': exp0_out, 'explegacy_out': explegacy_out, 'rbr_out': rbr_out, 'rbrp_out': rbrp_out, 'B_out': B_out, 'E_out': E_out, 'br': br}
    return per_example, summary


def save_rows_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]


def save_mimo_long_csv(path, per_ex, stem_names):
    rows = []
    N, C = per_ex['sirb_in'].shape
    for n in range(N):
        for ch, stem in enumerate(stem_names):
            rows.append({'idx': n, 'ch': ch, 'stem': stem, 'SIRB_in': float(per_ex['sirb_in'][n, ch]), 'LTR_in': float(per_ex['ltr_in'][n, ch]), 'U_in': float(per_ex['u_in'][n, ch]), 'U0_in': float(per_ex['u0_in'][n, ch]), 'U_legacy_in': float(per_ex['ulegacy_in'][n, ch]), 'EXP_in': float(per_ex['exp_in'][n, ch]), 'EXP0_in': float(per_ex['exp0_in'][n, ch]), 'EXP_legacy_in': float(per_ex['explegacy_in'][n, ch]), 'RBR_in': float(per_ex['rbr_in'][n, ch]), 'RBRpct_in': float(per_ex['rbrp_in'][n, ch]), 'SIRB_out': float(per_ex['sirb_out'][n, ch]), 'LTR_out': float(per_ex['ltr_out'][n, ch]), 'U_out': float(per_ex['u_out'][n, ch]), 'U0_out': float(per_ex['u0_out'][n, ch]), 'U_legacy_out': float(per_ex['ulegacy_out'][n, ch]), 'EXP_out': float(per_ex['exp_out'][n, ch]), 'EXP0_out': float(per_ex['exp0_out'][n, ch]), 'EXP_legacy_out': float(per_ex['explegacy_out'][n, ch]), 'RBR_out': float(per_ex['rbr_out'][n, ch]), 'RBRpct_out': float(per_ex['rbrp_out'][n, ch]), 'BR': float(per_ex['br'][n, ch])})
    save_rows_csv(path, rows)


def print_standard_summary(summary):
    print('\n===== STANDARD =====')
    order = ['sisdr_in', 'sisnr_in', 'sir_in', 'sar_in', 'sisdr', 'sisnr', 'sir', 'sar', 'sisdr_impr', 'sisnr_impr', 'sir_impr', 'sar_impr']
    for k in order:
        v = summary[k]
        print(f"{k:12s} mean={v['mean']:.4f}  median={v['median']:.4f}  std={v['std']:.4f}")


def print_stem_block(title, block, stem_names):
    print(f'[{title}]')
    for stem in stem_names:
        v = block[stem]
        print(f"  {stem:10s} mean={v['mean']:.4f}  std={v['std']:.4f}")


def print_mimo_summary(summary, stem_names):
    print('\n===== MIMO BEFORE (Mixture vs Ytrue) =====')
    print_stem_block('SIRB_in', summary['mixture_before']['SIRB'], stem_names)
    print_stem_block('LTR_in', summary['mixture_before']['LTR'], stem_names)
    print_stem_block('U_in', summary['mixture_before']['U'], stem_names)
    print_stem_block('EXP_in', summary['mixture_before']['EXP'], stem_names)
    print_stem_block('EXP_legacy_in', summary['mixture_before']['EXP_legacy'], stem_names)
    print_stem_block('RBRpct_in', summary['mixture_before']['RBRpct'], stem_names)
    print('\n===== MIMO AFTER (Ypred vs Ytrue) =====')
    print_stem_block('SIRB_out', summary['output_after']['SIRB'], stem_names)
    print_stem_block('LTR_out', summary['output_after']['LTR'], stem_names)
    print_stem_block('U_out', summary['output_after']['U'], stem_names)
    print_stem_block('EXP_out', summary['output_after']['EXP'], stem_names)
    print_stem_block('EXP_legacy_out', summary['output_after']['EXP_legacy'], stem_names)
    print_stem_block('RBRpct_out', summary['output_after']['RBRpct'], stem_names)
    print('\n===== BLEED REDUCTION =====')
    print_stem_block('BR', summary['bleed_reduction']['BR'], stem_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x_mix', type=str, required=True)
    ap.add_argument('--y_true', type=str, required=True)
    ap.add_argument('--y_pred', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--T', type=int, default=-1)
    ap.add_argument('--max_items', type=int, default=-1)
    ap.add_argument('--std_align', type=int, default=1)
    ap.add_argument('--std_max_lag', type=int, default=22050)
    ap.add_argument('--std_use_envelope', type=int, default=1)
    ap.add_argument('--permute_pred', type=int, default=0)
    ap.add_argument('--mimo_K', type=int, default=512)
    ap.add_argument('--mimo_lam', type=float, default=1e-3)
    ap.add_argument('--max_delay_samples', type=int, default=44100)
    ap.add_argument('--use_envelope', type=int, default=1)
    ap.add_argument('--stem_names', type=str, nargs='+', default=['Vocal', 'Bass', 'Drums'])
    ap.add_argument("--crop_T", type=int, default=None,
                help="Optional common crop length along time axis before metric computation")
    args = ap.parse_args()
    ensure_dir(args.out_dir)


    Xmix  = np.load(args.x_mix).astype(np.float32)
    Ytrue = np.load(args.y_true).astype(np.float32)
    Ypred = np.load(args.y_pred).astype(np.float32)

    if Xmix.ndim != 3 or Ytrue.ndim != 3 or Ypred.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N,C,T). Got {Xmix.shape}, {Ytrue.shape}, {Ypred.shape}")

    # First match N,C
    if Xmix.shape[:2] != Ytrue.shape[:2] or Xmix.shape[:2] != Ypred.shape[:2]:
        raise ValueError(f"N/C mismatch: X={Xmix.shape}, Ytrue={Ytrue.shape}, Ypred={Ypred.shape}")

    # Then crop time
    if args.crop_T is not None:
        T = min(args.crop_T, Xmix.shape[-1], Ytrue.shape[-1], Ypred.shape[-1])
    else:
        T = min(Xmix.shape[-1], Ytrue.shape[-1], Ypred.shape[-1])

    Xmix  = Xmix[..., :T]
    Ytrue = Ytrue[..., :T]
    Ypred = Ypred[..., :T]

    print(f"[Shapes after crop] Xmix={Xmix.shape}, Ytrue={Ytrue.shape}, Ypred={Ypred.shape}")


    N, C, T = Ytrue.shape
    assert Xmix.shape == Ytrue.shape == Ypred.shape
    assert C == len(args.stem_names), (C, args.stem_names)
    print(f'Loaded shapes: Xmix={Xmix.shape}, Ytrue={Ytrue.shape}, Ypred={Ypred.shape}')
    std_rows, std_summary, perm_rows = compute_standard_metrics(Xmix, Ytrue, Ypred, args.stem_names, bool(args.std_align), args.std_max_lag, bool(args.std_use_envelope), bool(args.permute_pred))
    save_rows_csv(os.path.join(args.out_dir, 'standard_metrics_per_sample.csv'), std_rows)
    if perm_rows:
        save_rows_csv(os.path.join(args.out_dir, 'prediction_permutations.csv'), perm_rows)
    with open(os.path.join(args.out_dir, 'standard_metrics_summary.json'), 'w') as f:
        json.dump(std_summary, f, indent=2)
    bm = BleedMatrixTF(K=args.mimo_K, lam=args.mimo_lam, sr=args.sr, max_delay_samples=args.max_delay_samples, use_envelope=bool(args.use_envelope), verbose=False)
    mimo_per_ex, mimo_summary = compute_mimo_all(bm, Xmix, Ytrue, Ypred, args.stem_names)
    save_mimo_long_csv(os.path.join(args.out_dir, 'mimo_metrics_long.csv'), mimo_per_ex, args.stem_names)
    np.savez_compressed(os.path.join(args.out_dir, 'mimo_raw_arrays.npz'), **mimo_per_ex)
    with open(os.path.join(args.out_dir, 'mimo_metrics_summary.json'), 'w') as f:
        json.dump(mimo_summary, f, indent=2)
    with open(os.path.join(args.out_dir, 'eval_config.json'), 'w') as f:
        json.dump({'x_mix': args.x_mix, 'y_true': args.y_true, 'y_pred': args.y_pred, 'sr': args.sr, 'T': args.T, 'max_items': args.max_items, 'std_align': bool(args.std_align), 'std_max_lag': args.std_max_lag, 'std_use_envelope': bool(args.std_use_envelope), 'permute_pred': bool(args.permute_pred), 'mimo_K': args.mimo_K, 'mimo_lam': args.mimo_lam, 'max_delay_samples': args.max_delay_samples, 'use_envelope': bool(args.use_envelope), 'stem_names': args.stem_names}, f, indent=2)
    print_standard_summary(std_summary)
    print_mimo_summary(mimo_summary, args.stem_names)
    print(f'\nSaved to: {args.out_dir}')


if __name__ == '__main__':
    main()
