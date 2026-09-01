# TASLP LDWT Revision Experiments: Current Finalization Status

Date: 2026-08-01

## What is complete

### Phase 1: Polarity diagnosis

Completed output directory:

`/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/20260801_002130_polarity_diagnosis/`

Finding: polarity reversal is a loss ambiguity, not an implementation bug in the analysis/synthesis front end. Front-end reconstruction correlations are positive for Waveform, Fixed db4, and LDWT. The separator output signs flip because the ordinary SI-SDR loss is sign-invariant.

Regression test result:

| Quantity | Value |
|---|---:|
| ordinary SI-SDR for `y` | 125.135 dB |
| ordinary SI-SDR for `-y` | 125.135 dB |
| positive-scale SI-SDR for `y` | 125.135 dB |
| positive-scale SI-SDR for `-y` | -125.134 dB |

Implemented fix: use positive-scale SI-SDR for controlled retraining, where the projection coefficient is clamped positive. This directly penalizes polarity inversion while preserving the SI-SDR structure.

Scripts updated or added:

- `/home/rrame12/Desktop/Research/dbss/asa/finetune_frontend_edge.py`
- `/home/rrame12/Desktop/Research/dbss/asa/finetune_fixed_wavelet_edge.py`
- `/home/rrame12/Desktop/Research/dbss/asa/finetune_ldwt_edge_positive.py`
- `/home/rrame12/Desktop/Research/dbss/asa/test_positive_sisdr_polarity.py`

Smoke status: one finite train/evaluation pass completed for Waveform, Fixed db4, and LDWT L2/F11. No NaN/Inf failure was observed in the smoke runs.

### Phase 4: Measured-RIR assignment analysis

Completed output directory:

`/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/20260801_002150_measured_rir_assignment/`

RIR energy matrix, source x microphone:

| Source | mic0 | mic1 | mic2 |
|---|---:|---:|---:|
| vocals | 187.613 | 4734.291 | 1049.797 |
| bass | 4626.977 | 793.045 | 5157.682 |
| drums | 1018.514 | 2593.344 | 235.529 |

Current assignment `[0,1,2]`:

| Statistic | Target-to-bleed ratio |
|---|---:|
| mean | -12.883 dB |
| median | -14.209 dB |
| min | -14.784 dB |
| max | -9.657 dB |
| positive assigned pairs | 0.000 |

Optimal one-to-one Hungarian assignment `[1,2,0]`:

| Statistic | Target-to-bleed ratio |
|---|---:|
| mean | +0.248 dB |
| median | +1.455 dB |
| min | -6.746 dB |
| max | +6.034 dB |
| positive assigned pairs | 0.667 |

Independent preferred microphones: `[1,2,1]`, so multiple sources prefer the same microphone. The optimized one-to-one mapping is improved but still not cleanly diagonal-dominant.

Decision: the measured-RIR data should be framed as an assumption-violation stress test or removed from the main quantitative table. It should not be repaired by high-pass filtering as the primary step because target-to-noise was already approximately +130 dB; the limiting issue is target-to-bleed/RIR assignment.

## Prepared full controlled run

Runnable script:

`/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/run_remaining_revision_experiments.sh`

Resolved configuration files:

- `/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/20260801_five_seed_10epoch_positive_sisdr/resolved_run_configs.json`
- `/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/20260801_five_seed_10epoch_positive_sisdr/resolved_run_configs.csv`
- `/home/rrame12/Desktop/Research/DWT_IR/revision_experiments/20260801_five_seed_10epoch_positive_sisdr/prelaunch_estimate.json`

Protocol:

| Method | Seeds | Task epochs | Stage-A PR epochs | Loss |
|---|---:|---:|---:|---|
| Waveform | 0,1,2,3,4 | 10 | 0 | positive-scale SI-SDR |
| Fixed db4 | 0,1,2,3,4 | 10 | 0 | positive-scale SI-SDR |
| LDWT L2/F11 | 0,1,2,3,4 | 10 | 5 | positive-scale SI-SDR |

Common data/protocol: `finetune_bleed_small_disjoint` train/validation data, `Edgecase_active_matched/Xedge_m9db.npy` test input, `Edgecase_active_matched/Yedge_m9db.npy` target, crop length `32768`, evaluation length `220448`, common evaluator `Evaluations/eval_all_metrics_single.py`.

Estimated cost: 15 full runs. Smoke outputs imply a lower-bound storage cost around 11 GB and a recommended free-space target of at least 25 GB. Runtime is expected to be multi-hour to overnight depending on GPU availability.

## Pending after long run

The following deliverables require the 15 full training/evaluation runs to complete:

- per-example CSVs for all five seeds and methods;
- per-seed summary CSV;
- aggregate mean/std/median CSV;
- paired LDWT-minus-fixed-db4 and LDWT-minus-waveform seed-level differences;
- hierarchical bootstrap confidence intervals;
- exact paired seed-level permutation tests where feasible;
- LaTeX controlled-comparison tables;
- seed-wise SIR(B) figure with paired seed connections;
- hierarchical SIR(B) distribution figure;
- LDWT initialization/PR/task trajectory table and filter evolution figures.

## Strongest defensible claim right now

The strongest current claim is that the previous polarity reversals were caused by the known sign ambiguity of ordinary SI-SDR, not by a broken wavelet reconstruction convention. A controlled five-seed comparison with positive-scale SI-SDR is now prepared and smoke-tested; it is the correct next evidence for whether LDWT consistently beats fixed db4 under a fair protocol.

## Remaining reviewer-visible weakness

Until the five-seed controlled run finishes, the LDWT-vs-fixed-db4 claim should not rely on the old single-seed table because the protocols were not identical. The measured-RIR table should also not be treated as a normal main result because the measured RIR matrix is not diagonally dominant for the assumed source-microphone assignment.
