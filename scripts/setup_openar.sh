#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f vendor/openar/generate.py ]]; then
  git clone https://github.com/OpenAR-Collective/simple-mock-collections-data-generator.git vendor/openar
fi
mkdir -p data/openar
python vendor/openar/generate.py --seed "Inkasso-2026" --accounts 10000 --out "$(pwd)/data/openar"
python - <<'PY'
from inkasso.data import load_profiles, source_counts
p, source = load_profiles('data/openar', n=1000)
print(source, len(p), 'profiles')
print(source_counts('data/openar'))
PY
