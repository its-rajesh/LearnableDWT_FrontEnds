#!/usr/bin/env bash
set -euo pipefail

EDGE_DIR="/home/rrame12/Desktop/Research/DWT_IR/Edgecase"
PRED_DIR="${EDGE_DIR}/predictions"
EVAL_SCRIPT="/home/rrame12/Desktop/Research/DWT_IR/eval_all_metrics_single.py"

for DB in 0 3 6 9 12; do
  OUT_DIR="${EDGE_DIR}/metrics_${DB}db"
  mkdir -p "${OUT_DIR}"

  echo "=============================================================="
  echo "Running metrics for ${DB} dB"
  echo "=============================================================="

  python "${EVAL_SCRIPT}" \
    --x_mix "${EDGE_DIR}/Xedge_${DB}db.npy" \
    --y_true "${EDGE_DIR}/Yedge_${DB}db.npy" \
    --y_pred "${PRED_DIR}/Ypred_edge_${DB}db.npy" \
    --out_dir "${OUT_DIR}" \
    --sr 22050 \
    --crop_T 220448 \
    --mimo_K 512 \
    --mimo_lam 1e-3 \
    --std_align 1 \
    --std_max_lag 22050 \
    --permute_pred 1 \
    --stem_names Vocal Bass Drums
done

echo "Done."
