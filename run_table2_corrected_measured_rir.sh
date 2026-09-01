#!/usr/bin/env bash
set -Eeuo pipefail

DROOT="${DROOT:-/home/rrame12/Desktop/Research/DWT_IR}"
PY="${PY:-/home/rrame12/anaconda3/envs/all/bin/python}"
BASELINES="${BASELINES:-/home/rrame12/Desktop/Research/Baselines}"
TABLE_BUILDER="${TABLE_BUILDER:-/home/rrame12/Desktop/Research/dbss/asa/build_table2_from_metrics.py}"

DATA="${DATA:-$DROOT/measured_rir_original_deconv_rawgain_early50_synth_full/synth_dataset}"
PRE="${PRE:-$DROOT/runs_pr_ablation/L2_F11/20260226_183007/best.keras}"
OUT="${OUT:-$DROOT/table2_corrected_measured_rir_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT"/{logs,markers}
STATUS="$OUT/status.log"
STATE="$OUT/state.env"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"
}

fail_report() {
  local code=$?
  log "FAILED at line $1 with exit code $code"
  log "Check stage logs in: $OUT/logs"
  exit "$code"
}
trap 'fail_report $LINENO' ERR

run_stage() {
  local name="$1"
  shift
  local marker="$OUT/markers/${name}.done"
  local log_file="$OUT/logs/${name}.log"
  if [[ -f "$marker" ]]; then
    log "SKIP $name (already done)"
    return 0
  fi
  log "START $name"
  "$@" > "$log_file" 2>&1
  touch "$marker"
  log "DONE  $name"
}

log "OUT=$OUT"
log "DATA=$DATA"
log "PRE=$PRE"
log "PY=$PY"

run_stage 01_finetune_ldwt "$PY" "$DROOT/finetune_recorded.py" \
  --pretrained "$PRE" \
  --rerec_root "$DATA" \
  --train_split train \
  --crop_T 32768 \
  --batch 2 \
  --phase1_epochs 50 \
  --phase2_epochs 200 \
  --lr1 1e-4 \
  --lr2 5e-5 \
  --val_ratio 0.15 \
  --seed 1337 \
  --clipnorm 0.25 \
  --pair_peak_norm 0 \
  --source_sr 22050 \
  --target_sr 22050 \
  --min_active_stems 2 \
  --require_vocal_every_n 4

if [[ ! -f "$STATE" ]] || ! grep -q '^FT_RUN=' "$STATE"; then
  FT_RUN="$(ls -td "$DROOT"/runs_finetune_recorded/* | head -1)"
  BEST="$FT_RUN/best.keras"
  {
    printf 'FT_RUN=%q\n' "$FT_RUN"
    printf 'BEST=%q\n' "$BEST"
  } > "$STATE"
else
  # shellcheck source=/dev/null
  source "$STATE"
fi

# shellcheck source=/dev/null
source "$STATE"
log "FT_RUN=$FT_RUN"
log "BEST=$BEST"

run_stage 02_test_ldwt "$PY" "$DROOT/test_iprdwt_rerecorded_updated.py" \
  --rerec_root "$DATA" \
  --split test \
  --best_model "$BEST" \
  --x_subdir X \
  --y_subdir Y \
  --target_sr 22050 \
  --crop_T 32768 \
  --n_crops_per_song 8 \
  --min_active_stems 3 \
  --strict_activity 1 \
  --stem_rms_thresh 1e-4 \
  --stem_peak_thresh 1e-3 \
  --seed 1337 \
  --batch 2 \
  --levels 2 \
  --filter_length 11 \
  --pr_shifts 12 \
  --pr_lambda 0.1 \
  --pr_dc_lambda 1.0 \
  --pr_nyq_lambda 1.0

LDWT_OUT="$FT_RUN/test_outputs_iprdwt_rerecorded"
X="$LDWT_OUT/Xtest_crops.npy"
Y="$LDWT_OUT/Ytest_crops.npy"
YP_LDWT="$LDWT_OUT/Ypred_test_crops.npy"

run_stage 03_metrics_reference "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$X" \
  --out_dir "$OUT/metrics_reference" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 04_metrics_ldwt "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$YP_LDWT" \
  --out_dir "$OUT/metrics_ldwt" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 05_predict_kamir python3 "$BASELINES/KAMIR/eval_kamir_rerecorded.py" \
  --x "$X" \
  --out "$OUT/Ypred_kamir.npy"

run_stage 06_metrics_kamir "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$OUT/Ypred_kamir.npy" \
  --out_dir "$OUT/metrics_kamir" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 07_predict_cae python3 "$BASELINES/CAE/eval_cae_plain_pytorch.py" \
  --data_root "$LDWT_OUT" \
  --x_name Xtest_crops.npy \
  --y_name Ytest_crops.npy \
  --model_root "$BASELINES/CAE/runs_cae_plain_pt" \
  --out "$OUT/Ypred_cae.npy"

run_stage 08_metrics_cae "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$OUT/Ypred_cae.npy" \
  --out_dir "$OUT/metrics_cae" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 09_predict_cganir python3 "$BASELINES/cGANIR/eval_cganir_waveform.py" \
  --x "$X" \
  --y "$Y" \
  --model "$BASELINES/cGANIR/Codes and Model/generator_epoch700.pth" \
  --out "$OUT/Ypred_cganir.npy"

run_stage 10_metrics_cganir "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$OUT/Ypred_cganir.npy" \
  --out_dir "$OUT/metrics_cganir" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 11_predict_htdemucs python3 "$BASELINES/HTdemucs/eval_demucs_rerecorded_stemwise.py" \
  --x "$X" \
  --out "$OUT/Ypred_htdemucs.npy" \
  --model_name htdemucs

run_stage 12_metrics_htdemucs "$PY" "$DROOT/eval_all_metrics_single.py" \
  --x_mix "$X" --y_true "$Y" --y_pred "$OUT/Ypred_htdemucs.npy" \
  --out_dir "$OUT/metrics_htdemucs" \
  --sr 22050 --mimo_K 512 --mimo_lam 1e-3 \
  --std_align 1 --std_max_lag 22050 \
  --stem_names Vocal Bass Drums

run_stage 13_build_table "$PY" "$TABLE_BUILDER" \
  --out_dir "$OUT" \
  --reference "$OUT/metrics_reference" \
  --kamir "$OUT/metrics_kamir" \
  --cae "$OUT/metrics_cae" \
  --cganir "$OUT/metrics_cganir" \
  --htdemucs "$OUT/metrics_htdemucs" \
  --ldwt "$OUT/metrics_ldwt"

log "DONE. Final table:"
log "$OUT/table2_measured_rir_corrected_all_methods.csv"
log "$OUT/table2_measured_rir_corrected_all_methods.tex"
