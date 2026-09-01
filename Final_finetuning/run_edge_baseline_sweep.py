#!/usr/bin/env python3
"""Run KAMIR, CAE, and cGANIR on the matched edge bleed sweep and compute metrics.

Outputs are written as:
  OUT_DIR/baseline_predictions/<method>/Ypred_edge_<tag>.npy
  OUT_DIR/baseline_metrics/<method>/metrics_<tag>/{standard,mimo}_metrics_summary.json

This is intentionally a thin orchestrator around the existing baseline scripts so the
paper figures use the same inference/evaluation code as the main baseline tables.
"""

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_LEVELS = [-40, -20, -18, -16, -14, -12, -9, -6, -3, 0]
PROJECT = Path('/home/rrame12/Desktop/Research/DWT_IR')
BASELINES = Path('/home/rrame12/Desktop/Research/Baselines')
PY_ALL = Path('/home/rrame12/anaconda3/envs/all/bin/python')
EVAL_SCRIPT = PROJECT / 'Evaluations' / 'eval_all_metrics_single.py'


def tag_from_db(db: int) -> str:
    return f"m{abs(int(db))}db" if db < 0 else f"{int(db)}db"


def run(cmd, cwd=None):
    print('\n[RUN]', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def eval_metrics(x_path, y_path, pred_path, out_metrics, t):
    if (out_metrics / 'standard_metrics_summary.json').exists() and (out_metrics / 'mimo_metrics_summary.json').exists():
        print('[SKIP metrics]', out_metrics)
        return
    out_metrics.mkdir(parents=True, exist_ok=True)
    run([
        PY_ALL, EVAL_SCRIPT,
        '--x_mix', x_path,
        '--y_true', y_path,
        '--y_pred', pred_path,
        '--out_dir', out_metrics,
        '--sr', '22050',
        '--T', str(t),
        '--std_align', '1',
        '--std_max_lag', '22050',
        '--std_use_envelope', '1',
        '--permute_pred', '1',
        '--mimo_K', '512',
        '--mimo_lam', '0.001',
        '--max_delay_samples', '44100',
        '--use_envelope', '1',
        '--stem_names', 'Vocal', 'Bass', 'Drums',
    ], cwd=EVAL_SCRIPT.parent)


def run_kamir(x_path, pred_path):
    if pred_path.exists():
        print('[SKIP KAMIR pred]', pred_path)
        return
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        'python3', BASELINES / 'KAMIR' / 'eval_kamir_rerecorded.py',
        '--x', x_path,
        '--out', pred_path,
        '--fs', '22050',
    ], cwd=BASELINES / 'KAMIR')


def run_cae(edge_dir, x_name, y_name, pred_path):
    if pred_path.exists():
        print('[SKIP CAE pred]', pred_path)
        return
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        'python3', BASELINES / 'CAE' / 'eval_cae_plain_pytorch.py',
        '--data_root', edge_dir,
        '--x_name', x_name,
        '--y_name', y_name,
        '--model_root', BASELINES / 'CAE' / 'runs_cae_plain_pt',
        '--out', pred_path,
        '--use_clean_phase', '1',
    ], cwd=BASELINES / 'CAE')


def run_cganir(x_path, y_path, pred_path):
    if pred_path.exists():
        print('[SKIP cGANIR pred]', pred_path)
        return
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        'python3', BASELINES / 'cGANIR' / 'eval_cganir_waveform.py',
        '--x', x_path,
        '--y', y_path,
        '--model', BASELINES / 'cGANIR' / 'Codes and Model' / 'generator_epoch700.pth',
        '--out', pred_path,
    ], cwd=BASELINES / 'cGANIR')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edge_dir', type=Path, default=PROJECT / 'Edgecase_active_matched')
    ap.add_argument('--out_dir', type=Path, default=PROJECT / 'bleed_eval_finetuned')
    ap.add_argument('--bleed_levels', type=int, nargs='+', default=DEFAULT_LEVELS)
    ap.add_argument('--methods', nargs='+', default=['kamir', 'cae', 'cganir'], choices=['kamir', 'cae', 'cganir'])
    ap.add_argument('--T', type=int, default=220448)
    args = ap.parse_args()

    pred_root = args.out_dir / 'baseline_predictions'
    metric_root = args.out_dir / 'baseline_metrics'

    for db in args.bleed_levels:
        tag = tag_from_db(db)
        x_name = f'Xedge_{tag}.npy'
        y_name = f'Yedge_{tag}.npy'
        x_path = args.edge_dir / x_name
        y_path = args.edge_dir / y_name
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f'Missing edge arrays for {tag}: {x_path}, {y_path}')

        print(f'\n===== {tag} ({db} dB nominal) =====', flush=True)

        if 'kamir' in args.methods:
            pred = pred_root / 'kamir' / f'Ypred_edge_{tag}.npy'
            run_kamir(x_path, pred)
            eval_metrics(x_path, y_path, pred, metric_root / 'kamir' / f'metrics_{tag}', args.T)

        if 'cae' in args.methods:
            pred = pred_root / 'cae' / f'Ypred_edge_{tag}.npy'
            run_cae(args.edge_dir, x_name, y_name, pred)
            eval_metrics(x_path, y_path, pred, metric_root / 'cae' / f'metrics_{tag}', args.T)

        if 'cganir' in args.methods:
            pred = pred_root / 'cganir' / f'Ypred_edge_{tag}.npy'
            run_cganir(x_path, y_path, pred)
            eval_metrics(x_path, y_path, pred, metric_root / 'cganir' / f'metrics_{tag}', args.T)

    print('\n[DONE] baseline sweep')
    print('Predictions:', pred_root)
    print('Metrics:', metric_root)


if __name__ == '__main__':
    main()
