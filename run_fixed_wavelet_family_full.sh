#!/usr/bin/env bash
set -euo pipefail
cd /home/rrame12/Desktop/Research/DWT_IR
echo "===== START haar $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet haar --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE haar $(date) ====="
echo "===== START db2 $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet db2 --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE db2 $(date) ====="
echo "===== START db4 $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet db4 --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE db4 $(date) ====="
echo "===== START db8 $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet db8 --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE db8 $(date) ====="
echo "===== START sym4 $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet sym4 --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE sym4 $(date) ====="
echo "===== START coif1 $(date) ====="
/home/rrame12/anaconda3/envs/all/bin/python /home/rrame12/Desktop/Research/DWT_IR/ablation_frontends.py --dpath /home/rrame12/Desktop/Research/DWT_IR --frontend dwt_fixed --fixed_wavelet coif1 --full_t 220448 --train_t 32768 --epochs 300 --batch 2 --pit 0 --lr 0.0001 --clipnorm 0.25 --hf_lambda 0.0 --levels 2 --filter_length 101 --out_root /home/rrame12/Desktop/Research/DWT_IR/runs_fixed_wavelet_family
echo "===== DONE coif1 $(date) ====="
