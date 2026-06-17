#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r env/requirements.txt
python scripts/nsc_restricted_subject_bagging_tail_stats.py \
  --manifest data/nsc_dataset_images/manifest.csv \
  --eeg-root data/eeg-csv-data-by-class \
  --base-predictions analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv \
  --cache-dir analysis/recomputed_eeg_feature_cache \
  --out-dir analysis/recomputed_nsc_restricted_subject_bagging_tail_stats_20260604 \
  --penalty l1 --C 0.25 --top-k 320 --objective min_metric --w-eeg-max 0.25 --w-eeg-step 0.05
