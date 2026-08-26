#!/usr/bin/env bash
# Runs the full 4 provider x 4 category grid with repeated runs, then
# regenerates every figure and table from the result.
#
# This spends real money on API calls. Run scripts/run_pilot.sh first and
# confirm it passes: the pilot exists precisely so that a scoring bug is
# caught for four calls instead of for the whole grid.
#
#   bash scripts/run_full.sh
#   bash scripts/run_full.sh --categories category_a_codegen   # one category
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.harness.runner --full "$@"
python -m src.analysis.figures
