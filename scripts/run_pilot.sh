#!/usr/bin/env bash
# Runs a small pilot batch (one task per category, one run each) before the
# full grid, per the Week 4 plan. Catches harness and scoring bugs cheaply:
# a bug found here costs four API calls, the same bug found in run_full.sh
# costs the whole grid.
#
#   bash scripts/run_pilot.sh
#   bash scripts/run_pilot.sh --providers openai      # one provider only
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.harness.runner --pilot "$@"
