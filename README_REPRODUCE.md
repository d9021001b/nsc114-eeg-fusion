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

## 3. Generate the fixed raw/2D OOF input

```bash
.venv/bin/python scripts/nsc_uncertain_band_patch_refinement.py \
  --csv-dir data/physiology-csv \
  --manifest data/nsc_dataset_images/manifest.csv \
  --image-root data/nsc_dataset_images \
  --out-dir analysis/nsc_uncertain_band_patch_refinement_20260520 \
  --report reports/NSC_uncertain_band_patch_family_refinement_20260520.docx \
  --outer-random-state 42 \
  --random-state 20260520
```

The historical `mi_max` column written to `uncertain_band_predictions.csv` is the fold-selected raw/2D uncertain-band output. It must not be interpreted as the pure `max_i g(I_i)` image score.

## 4. Run the current nested EEG/signal fusion

```bash
.venv/bin/python scripts/nsc114_airtight_fully_nested_20260710.py \
  --manifest data/nsc_dataset_images/manifest.csv \
  --physio-root data/physiology-csv \
  --eeg-root data/eeg-csv-data-by-class \
  --mi-predictions analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv \
  --out-dir analysis/nsc114_airtight_fully_nested_20260710
```

The script name is retained for provenance. Its current documentation and generated manifest state the actual boundary: EEG/signal selection is nested on the fusion folds, while the raw/2D OOF input is fixed from a separate cross-fitting run.

## 5. Rebuild aggregate metrics and figures

```bash
.venv/bin/python scripts/nsc114_airtight_figures_and_table2_20260710.py
```

Expected aggregate values are recorded in `expected_outputs/aggregate_metrics_20260710.json`. Per-participant predictions remain ignored and must not be published.

## 6. Minimal code-only checks

These checks do not require restricted data:

```bash
python3 -m compileall -q scripts
python3 scripts/nsc114_airtight_fully_nested_20260710.py --help
python3 scripts/nsc_uncertain_band_patch_refinement.py --help
```

## Interpretation

The aggregate reference result is a within-dataset participant-level OOF estimate. It does not establish external generalization or a statistically confirmed incremental benefit of the signal-interaction branch.
