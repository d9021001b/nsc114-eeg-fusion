# NSC114 Physiological Imaging and EEG Fusion - Analysis Code

Code for the 114-participant small-sample binary-classification study. The current analysis combines a fold-local physiological-image branch, fold-local EEG summaries, and fold-local signal-interaction candidates under repeated participant-level out-of-fold evaluation.

## Claim boundary

- The image classifier, EEG feature selection, EEG model settings, signal family, signal model, retained signal features, and fusion weights are rebuilt or selected using outer-training participants only.
- Image probabilities are aggregated by a fixed top-three mean after fold-local image scoring; the image branch does not read the historical `mi_max` or uncertain-band score.
- The complete image-plus-EEG-plus-interaction pipeline is regenerated inside every outer fold and repeated over 10 random seeds.
- Reported results are within-dataset participant-level OOF estimates, not external or prospective validation.
- A mean-pooling sensitivity analysis is retained because performance depends on how repeated image instances are summarized.

## Current aggregate reference result

| Model | AUROC | AUPRC | TN/FP/FN/TP |
|---|---:|---:|---:|
| Fold-local image branch | 0.7488 | 0.7098 | 4/58/0/52 |
| Nested image + EEG base | 0.7813 | 0.7395 | 43/19/12/40 |
| Nested multimodal fusion | 0.8130 | 0.8000 | 48/14/16/36 |

Paired fused-minus-base delta AUROC was +0.0316 (95% CI +0.0022 to +0.0639; p=0.037), and delta AUPRC was +0.0605 (95% CI +0.0133 to +0.1101; p=0.006).

Machine-readable references are in `expected_outputs/aggregate_metrics_v4_20260710.json` and `expected_outputs/aggregate_metrics_v4_mean_sensitivity_20260710.json`.

## Reproduction

See `README_REPRODUCE.md`. The main scripts are:

- `scripts/nsc114_true_nested_top3_grid8_20260710.py`: rebuilds the image, EEG, interaction, and fusion branches inside each outer fold.
- `scripts/nsc114_true_nested_table2_20260711.py`: recomputes point estimates, confidence intervals, and paired increments from local predictions.
- `run_current_nested_fusion.sh`: validates restricted input paths and runs the current analysis end to end.

## Data protection

This is a code-only repository. Do not commit participant data, recordings, manifests with identifiers, per-participant predictions, caches, or generated reports. `data/`, `analysis/`, and `reports/` are ignored by Git.
