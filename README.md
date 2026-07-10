# NSC114 Physiological Imaging and EEG Fusion - Analysis Code

Code for the 114-participant small-sample binary-classification study. The current analysis combines a separately cross-fitted raw/2D score, fold-local EEG summaries, and fold-local signal-interaction candidates under participant-level out-of-fold evaluation.

## Claim boundary

- The EEG feature selection, EEG model settings, signal family, signal model, retained signal features, and fusion weights are selected using fusion outer-training participants only.
- The `mi_max` input is a fixed OOF score produced by the raw/2D uncertain-band pipeline. Despite its historical column name, it is not a pure image-only maximum score.
- The fixed raw/2D OOF score is not regenerated on each fusion outer fold. The repository therefore does not describe the complete image-plus-fusion system as fully nested.
- Reported results are within-dataset participant-level OOF estimates, not external or prospective validation.
- The paired fused-minus-base increment was not statistically confirmed.

## Current aggregate reference result

| Model | AUROC | AUPRC | TN/FP/FN/TP |
|---|---:|---:|---:|
| Fixed raw/2D score | 0.7835 | 0.7820 | 33/29/10/42 |
| Nested EEG base | 0.7916 | 0.7883 | 35/27/10/42 |
| Nested base + signal-interaction candidate | 0.8077 | 0.8092 | 42/20/13/39 |

Paired fused-minus-base delta AUROC was +0.0161 (95% CI -0.0133 to +0.0484; p=0.31), and delta AUPRC was +0.0209 (95% CI -0.0196 to +0.0679; p=0.35).

The machine-readable aggregate reference is in `expected_outputs/aggregate_metrics_20260710.json`.

## Reproduction

See `README_REPRODUCE.md`. The main scripts are:

- `scripts/nsc_uncertain_band_patch_refinement.py`: creates the separately cross-fitted raw/2D uncertain-band score.
- `scripts/nsc114_airtight_fully_nested_20260710.py`: historical filename for the current nested EEG/signal fusion run; its docstring and manifest explicitly state the fixed-image-score boundary.
- `scripts/nsc114_airtight_figures_and_table2_20260710.py`: rebuilds aggregate metrics, confidence intervals, and manuscript figures from local predictions.

## Data protection

This is a code-only repository. Do not commit participant data, recordings, manifests with identifiers, per-participant predictions, caches, or generated reports. `data/`, `analysis/`, and `reports/` are ignored by Git.
