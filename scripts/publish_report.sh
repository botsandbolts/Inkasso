#!/usr/bin/env bash
# Publish only aggregate simulated benchmark numbers, never account records.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f outputs/metrics.csv || ! -f outputs/metadata.json ]]; then
  echo "No experiment results found: run python -m inkasso.experiment first." >&2
  exit 1
fi
mkdir -p reports/latest
cp outputs/metrics.csv reports/latest/metrics.csv
cp outputs/metadata.json reports/latest/metadata.json
if ! grep -q "SIMULATOR-ONLY" reports/latest/metadata.json; then
  echo "Report is missing the required simulator-only provenance label." >&2
  exit 1
fi
git add reports/latest/metrics.csv reports/latest/metadata.json
echo "Aggregate SIMULATOR-ONLY results staged for publication."
echo "Review the changes, then run:"
echo "  git diff --cached -- reports/latest"
echo "  git commit -m 'Publish simulated policy comparison'"
echo "  git push origin main"
echo "The Streamlit Community Cloud app will update after GitHub receives the commit."
