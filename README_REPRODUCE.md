# NSC114 EEG Multimodal Small-Data Reproducibility Package

Created: 2026-06-04 15:46:53

## Purpose

This package contains the data, scripts, frozen analysis outputs, Word reports, and environment helpers needed to reproduce the 114-case NSC EEG multimodal small-data modeling project on another workstation.

## Main Reproduction Target

Primary script:

```bash
python scripts/nsc_restricted_subject_bagging_tail_stats.py \
  --manifest data/nsc_dataset_images/manifest.csv \
  --eeg-root data/eeg-csv-data-by-class \
  --base-predictions analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv \
  --cache-dir analysis/recomputed_eeg_feature_cache \
  --out-dir analysis/recomputed_nsc_restricted_subject_bagging_tail_stats_20260604 \
  --penalty l1 --C 0.25 --top-k 320 --objective min_metric --w-eeg-max 0.25 --w-eeg-step 0.05
```

Windows users can run:

1. `setup_env.bat`
2. `run_final_fusion.bat`

Linux users can run:

```bash
bash run_final_fusion.sh
```

## Included Data

- `data/eeg-csv-data-by-class`: EEG CSV trials by class folder. In this project, folder `0` is class 0; folders `1`, `2`, `3`, `4` are class 1.
- `data/nsc_dataset_images`: 2D time-series image dataset and `manifest.csv`.

## Included Frozen Outputs

- `analysis/nsc_restricted_subject_bagging_tail_stats_corrected_20260521`: final corrected tail-statistics bagged fusion output.
- `analysis/nsc_uncertain_band_patch_refinement_20260520`: frozen 2D branch base predictions used by the final fusion script.
- `analysis/nsc_replicated_literature_models_vs_best_20260520`: literature-method comparison tables and figures used by the paper.
- `analysis/nsc_eeg_shap_neuropsych_20260521`: SHAP/surrogate feature-importance outputs and neuropsychological explanation materials.

## Expected Main Metrics

See `expected_outputs/manifest.json` and `expected_outputs/bagged_metrics_summary.csv` for the frozen corrected run.

## Notes

- The final claim level is patient-aware 10-fold OOF with multi-seed bagging, not an external validation claim.
- Recomputed metrics should be close to the frozen outputs. Small differences can occur from package/library versions.
- Deep baseline scripts may require PyTorch; the main final fusion does not.
