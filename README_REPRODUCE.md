# Reproducing the Current NSC114 Nested-Fusion Analysis

## 1. Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r env/requirements.txt
```

## 2. Restricted local inputs

Place locally authorized data under the following ignored paths:

```text
data/
  nsc_dataset_images/
    manifest.csv
    0/.../*.png
    1/.../*.png
  physiology-csv/
    *.csv
  eeg-csv-data-by-class/
    0/.../*.csv
    1/.../*.csv
    2/.../*.csv
    3/.../*.csv
    4/.../*.csv
```

The manifest and output prediction files contain identifiers and must remain outside version control.

## 3. Run the current fully nested multimodal analysis

```bash
.venv/bin/python scripts/nsc114_true_nested_top3_grid8_20260710.py \
  --manifest data/nsc_dataset_images/manifest.csv \
  --physio-root data/physiology-csv \
  --eeg-root data/eeg-csv-data-by-class \
  --images-root data/nsc_dataset_images \
  --eeg-cache analysis/nsc_eeg_feature_cache \
  --image-grid 8 \
  --image-aggregation top3mean \
  --image-top-k 174 \
  --image-model ExtraTrees \
  --out-dir analysis/nsc114_true_nested_top3_grid8_20260710
```

The image branch is regenerated within each outer fold. Training participants receive inner out-of-fold image scores; outer-test participants are scored only after fitting on the outer-training set.

## 4. Rebuild aggregate metrics

```bash
.venv/bin/python scripts/nsc114_true_nested_table2_20260711.py \
  --out-dir analysis/nsc114_true_nested_top3_grid8_20260710
```

Expected aggregate values are recorded in `expected_outputs/aggregate_metrics_v4_20260710.json`; the mean-pooling sensitivity reference is in `expected_outputs/aggregate_metrics_v4_mean_sensitivity_20260710.json`. Per-participant predictions remain ignored and must not be published.

## 5. Minimal code-only checks

These checks do not require restricted data:

```bash
python3 -m compileall -q scripts
python3 scripts/nsc114_true_nested_top3_grid8_20260710.py --help
python3 scripts/nsc114_true_nested_table2_20260711.py --help
```

## Interpretation

The aggregate reference result is a within-dataset participant-level OOF estimate. It does not establish external generalization, and the lower mean-pooling sensitivity result shows that repeated-instance aggregation remains an important design choice.
