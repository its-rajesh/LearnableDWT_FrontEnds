#!/usr/bin/env bash
set -euo pipefail

EDGE_DIR="/home/rrame12/Desktop/Research/DWT_IR/Edgecase_active_matched"
PRED_DIR="$1"   # pass prediction folder here
EVAL_SCRIPT="/home/rrame12/Desktop/Research/DWT_IR/eval_all_metrics_single.py"

BLEED_LEVELS=(-40 -20 -18 -16 -14 -12 -9 -6 -3 0)

tag_from_db () {
  local db="$1"
  if (( db < 0 )); then
    echo "m$((-db))db"
  else
    echo "${db}db"
  fi
}

for DB in "${BLEED_LEVELS[@]}"; do
  TAG="$(tag_from_db "${DB}")"
  OUT_DIR="${PRED_DIR}/metrics_${TAG}"
  mkdir -p "${OUT_DIR}"

  echo "=============================================================="
  echo "Running metrics for bleed ${DB} dB (${TAG})"
  echo "=============================================================="

  python "${EVAL_SCRIPT}" \
    --x_mix "${EDGE_DIR}/Xedge_${TAG}.npy" \
    --y_true "${EDGE_DIR}/Yedge_${TAG}.npy" \
    --y_pred "${PRED_DIR}/Ypred_edge_${TAG}.npy" \
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
