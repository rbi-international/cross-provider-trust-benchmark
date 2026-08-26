#!/usr/bin/env bash
# Regenerates every figure/table in the paper from experiments/runs/, single command.
#
# Reads nothing but run folders, so the results section can always be rebuilt
# from scratch and no number in the paper is hand-copied. Safe to re-run: it
# overwrites paper/figures/ and paper/tables/ in place.
#
#   bash scripts/generate_figures.sh            # everything
#   bash scripts/generate_figures.sh --figures 3 4
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.analysis.figures "$@"
