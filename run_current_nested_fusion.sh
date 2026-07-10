#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

for required in \
  "$ROOT/data/nsc_dataset_images/manifest.csv" \
  "$ROOT/data/physiology-csv" \
  "$ROOT/data/eeg-csv-data-by-class"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing restricted local input: %s\n' "$required" >&2
    exit 2
  fi
done

OUT="$ROOT/analysis/nsc114_true_nested_top3_grid8_20260710"

"$PY" "$ROOT/scripts/nsc114_true_nested_top3_grid8_20260710.py" \
  --manifest "$ROOT/data/nsc_dataset_images/manifest.csv" \
  --physio-root "$ROOT/data/physiology-csv" \
  --eeg-root "$ROOT/data/eeg-csv-data-by-class" \
  --images-root "$ROOT/data/nsc_dataset_images" \
  --eeg-cache "$ROOT/analysis/nsc_eeg_feature_cache" \
  --image-grid 8 \
  --image-aggregation top3mean \
  --image-top-k 174 \
  --image-model ExtraTrees \
  --out-dir "$OUT"

"$PY" "$ROOT/scripts/nsc114_true_nested_table2_20260711.py" --out-dir "$OUT"
