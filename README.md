# LDWT Interference Reduction

This repository contains the code and lightweight reproducibility artifacts for the TASLP LDWT interference-reduction experiments.

The task is multichannel musical bleed reduction. Given three microphone mixtures, the model estimates the dominant target source at each microphone. The main model is a U-Net separator with a learnable discrete wavelet transform (LDWT) front end.

## Repository Contents

- `iprdwt.py`: LDWT analysis/synthesis layers and model components.
- `ablate_learnable_dwt_grid.py`: LDWT grid training over levels and filter lengths.
- `finetune_recorded.py`: measured-RIR fine-tuning.
- `test_iprdwt.py`, `test_iprdwt_rerecorded_updated.py`: LDWT evaluation scripts.
- `eval_all_metrics_single.py`, `bleed_matrix_tf.py`: SI-SDR/SIR/SAR/SIR(B)/EXP evaluation.
- `scripts/`: revision and dataset-generation utilities, including corrected measured-RIR extraction.
- `Evaluations/`: evaluation wrappers used for paper tables.
- `Final_finetuning/`: simulated bleed fine-tuning utilities.
- `results/`: lightweight CSV/JSON/TEX summaries and paper figures.
- `docs/`: RIR sanity summaries and revision notes.

Large artifacts are intentionally excluded: MUSDB18HQ audio, generated `.npy` arrays, WAV datasets, trained `.keras` checkpoints, and baseline prediction arrays.

## Key Reproducibility Artifacts

Final simulated hard-subset table:

```text
results/Edgecase_active_matched/hard_m9db_baselines/hard_m9db_all_methods_summary_table.csv
```

Final corrected measured-RIR table:

```text
results/table2_corrected_measured_rir/table2_measured_rir_corrected_all_methods.csv
results/table2_corrected_measured_rir/table2_measured_rir_corrected_all_methods.tex
```

LDWT interpretation figures and tables:

```text
results/paper_wavelet_interpretation/
```

Uncertainty/localization metrics:

```text
results/uncertainty_analysis/wavelet_uncertainty_metrics.csv
```

## Datasets

This repository does not redistribute MUSDB18HQ or generated audio. To reproduce the experiments, provide local paths to:

- MUSDB18HQ
- raw real-room sweep recordings and room noise
- trained baseline checkpoints, where required
- LDWT pretrained/fine-tuned checkpoints

The corrected real-room MUSDB18HQ dataset is generated from a fixed measured 3 x 3 RIR matrix. Mixtures and targets use the same measured RIR matrix; targets use the diagonal source-to-associated-microphone responses.

## Quick Start

Create an environment with the dependencies in `requirements.txt`, then run scripts from the repository root.

```bash
python scripts/extract_original_sweep_rirs_rawgain.py \
  --out_dir /path/to/measured_rir_original_deconv_rawgain_early50 \
  --reg 1e-6 \
  --post_sec 0.05 \
  --lowcut 50
```

Generate the corrected measured-RIR dataset:

```bash
python scripts/generate_measured_rir_rawgain_dataset.py \
  --rir_dir /path/to/measured_rir_original_deconv_rawgain_early50/rir_wavs \
  --out_dir /path/to/measured_rir_original_deconv_rawgain_early50_synth_full \
  --max_train 0 \
  --max_test 0 \
  --noise_gain 0.05 \
  --play_peak 0 \
  --play_rms 0 \
  --overwrite
```

Fine-tune LDWT on the corrected measured-RIR data:

```bash
python finetune_recorded.py \
  --pretrained /path/to/runs_pr_ablation/L2_F11/20260226_183007/best.keras \
  --rerec_root /path/to/measured_rir_original_deconv_rawgain_early50_synth_full/synth_dataset \
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
  --pair_peak_norm 0
```

For the full measured-RIR table pipeline, see:

```text
run_table2_corrected_measured_rir.sh
```

## Notes

The scripts contain local-path defaults from the original experiment machine. For another system, pass explicit dataset/checkpoint/output paths rather than relying on defaults.

No license has been selected yet. Add a license file before making the repository public.
