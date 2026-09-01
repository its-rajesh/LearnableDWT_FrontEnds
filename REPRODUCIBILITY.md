# Reproducibility Notes

## Corrected Measured-RIR Dataset

Raw measurement folder used in the experiments:

```text
/home/rrame12/Desktop/Datasets/Re-recorded/sample_data/_session_room_measurement
```

Expected raw files:

```text
sweep_spk0_vocals.wav
sweep_rec_spk0_vocals.wav
sweep_spk1_bass.wav
sweep_rec_spk1_bass.wav
sweep_spk2_drums.wav
sweep_rec_spk2_drums.wav
session_room_noise.wav
room_params.csv
```

Final RIR extraction:

```bash
python scripts/extract_original_sweep_rirs_rawgain.py \
  --out_dir /home/rrame12/Desktop/Research/DWT_IR/measured_rir_original_deconv_rawgain_early50 \
  --reg 1e-6 \
  --post_sec 0.05 \
  --lowcut 50
```

Important design choices:

- regularized sweep deconvolution
- raw measured propagation gains preserved
- 50 Hz low-cut filter
- 50 ms post-peak early-RIR crop
- fixed 3 x 3 source-microphone RIR matrix
- diagonal target assignment: vocals-mic0, bass-mic1, drums-mic2

Final dataset generation:

```bash
python scripts/generate_measured_rir_rawgain_dataset.py \
  --rir_dir /home/rrame12/Desktop/Research/DWT_IR/measured_rir_original_deconv_rawgain_early50/rir_wavs \
  --out_dir /home/rrame12/Desktop/Research/DWT_IR/measured_rir_original_deconv_rawgain_early50_synth_full \
  --max_train 0 \
  --max_test 0 \
  --noise_gain 0.05 \
  --play_peak 0 \
  --play_rms 0 \
  --overwrite
```

Dataset properties:

- MUSDB18HQ source: `/home/rrame12/Desktop/Datasets/musdb18hq`
- 100 train songs, 50 test songs
- 22050 Hz
- mono WAV outputs
- recorded room noise added to input mixtures only
- targets generated from the same RIR matrix as the mixtures

## LDWT Pretraining And Fine-Tuning

Best LDWT checkpoint used for measured-RIR fine-tuning:

```text
/home/rrame12/Desktop/Research/DWT_IR/runs_pr_ablation/L2_F11/20260226_183007/best.keras
```

Architecture/training configuration:

```text
levels = 2
filter_length = 11
pr_shifts = 12
pr_lambda_B = 0.1
pr_dc_lambda = 1.0
pr_nyq_lambda = 1.0
hf_lambda_B = 0.05
```

Measured-RIR fine-tuned run:

```text
/home/rrame12/Desktop/Research/DWT_IR/runs_finetune_recorded/20260804_080758
```

Fine-tuning settings:

```text
crop_T = 32768
batch = 2
phase1_epochs = 50
phase2_epochs = 200
lr1 = 1e-4
lr2 = 5e-5
val_ratio = 0.15
seed = 1337
clipnorm = 0.25
pair_peak_norm = 0
source_sr = target_sr = 22050
```

## Main Result Tables

Simulated hard-subset all-method table:

```text
results/Edgecase_active_matched/hard_m9db_baselines/hard_m9db_all_methods_summary_table.csv
```

Corrected measured-RIR all-method table:

```text
results/table2_corrected_measured_rir/table2_measured_rir_corrected_all_methods.csv
```

## LDWT Interpretation

Generated artifacts:

```text
results/paper_wavelet_interpretation/effective_subband_responses.pdf
results/paper_wavelet_interpretation/subband_interference_error_bars.pdf
results/paper_wavelet_interpretation/rir_condition_heatmap.pdf
results/paper_wavelet_interpretation/filter_similarity_table.csv
results/paper_wavelet_interpretation/subband_interference_error_table.csv
results/paper_wavelet_interpretation/rir_condition_table.csv
```

Uncertainty/localization metrics:

```text
results/uncertainty_analysis/wavelet_uncertainty_metrics.csv
```

## EXP Metric

EXP is computed as a residual-energy, R2-like percentage. For output channel `m`, the evaluated signal `z_m` is fitted from the target stems using a linear bleed model, producing `z_hat_m`. The reported quantity is:

```text
EXP_m = 100 * (1 - ||z_m - z_hat_m||_2^2 / ||z_m - mean(z_m)||_2^2)
```

This avoids mixing transfer-function-only quantities with signal-energy quantities.
