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

"$PY" "$ROOT/scripts/nsc_uncertain_band_patch_refinement.py" \
  --csv-dir "$ROOT/data/physiology-csv" \
  --manifest "$ROOT/data/nsc_dataset_images/manifest.csv" \
  --image-root "$ROOT/data/nsc_dataset_images" \
  --out-dir "$ROOT/analysis/nsc_uncertain_band_patch_refinement_20260520" \
  --report "$ROOT/reports/NSC_uncertain_band_patch_family_refinement_20260520.docx" \
  --outer-random-state 42 \
  --random-state 20260520

"$PY" "$ROOT/scripts/nsc114_airtight_fully_nested_20260710.py" \
  --manifest "$ROOT/data/nsc_dataset_images/manifest.csv" \
  --physio-root "$ROOT/data/physiology-csv" \
  --eeg-root "$ROOT/data/eeg-csv-data-by-class" \
  --mi-predictions "$ROOT/analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv" \
  --out-dir "$ROOT/analysis/nsc114_airtight_fully_nested_20260710"

"$PY" "$ROOT/scripts/nsc114_airtight_figures_and_table2_20260710.py"
