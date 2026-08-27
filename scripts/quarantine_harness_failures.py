"""
Moves runs that failed for HARNESS reasons out of experiments/runs/ so they
can be re-run cleanly, instead of standing in the dataset as agent failures.

The distinction this script draws is the important one:

  HARNESS failure  - the benchmark broke, not the model. A UnicodeEncodeError
                     from writing a curly quote on a cp1252 console says
                     nothing about the provider. These runs are invalid data
                     and must be re-run.

  MODEL failure    - GraphRecursionError, a phantom tool call, a wrong answer.
                     These are real findings about that provider and stay in
                     the dataset. Removing them would be cherry-picking.

Quarantined runs are MOVED to experiments/quarantine/, never deleted, so the
record of what went wrong survives and the decision stays auditable. Once
moved, `--resume` sees those provider/task/repeat cells as incomplete and
re-runs exactly them.

    python scripts/quarantine_harness_failures.py --dry-run
    python scripts/quarantine_harness_failures.py
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "experiments", "runs")
QUARANTINE_DIR = os.path.join(REPO_ROOT, "experiments", "quarantine")

# Exception types that mean the harness broke rather than the model.
# Deliberately a strict allowlist: anything not named here is treated as real
# data and left alone, so this script can never quietly delete an inconvenient
# result.
HARNESS_ERROR_TYPES = (
    "UnicodeEncodeError",   # file IO without a pinned encoding (fixed in tools.py)
    "UnicodeDecodeError",
    "TypeError",            # provider content-shape not normalised (fixed in trajectory_logger.py)
    "PermissionError",      # transient file lock on Windows
    "FileNotFoundError",    # working-dir teardown race
)


def classify(result):
    error = result.get("error") or ""
    for error_type in HARNESS_ERROR_TYPES:
        if error.startswith(error_type):
            return error_type
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--quarantine-dir", default=QUARANTINE_DIR)
    args = parser.parse_args()

    doomed = []
    for result_path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "result.json"))):
        try:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            continue
        error_type = classify(result)
        if error_type:
            doomed.append((os.path.dirname(result_path), result, error_type))

    if not doomed:
        print("No harness-caused failures found. Nothing to quarantine.")
        return

    by_provider = collections.Counter(r["provider"] for _, r, _ in doomed)
    by_type = collections.Counter(t for _, _, t in doomed)
    by_cell = collections.Counter((r["provider"], r["category"]) for _, r, _ in doomed)

    print(f"{len(doomed)} run(s) failed for harness reasons:\n")
    print("  by error type:")
    for error_type, count in by_type.most_common():
        print(f"    {error_type:22s} {count}")
    print("  by provider:")
    for provider, count in by_provider.most_common():
        print(f"    {provider:22s} {count}")
    print("  by provider x category:")
    for (provider, category), count in sorted(by_cell.items()):
        print(f"    {provider:10s} {category:20s} {count}")

    if args.dry_run:
        print("\nDry run: nothing moved.")
        return

    os.makedirs(args.quarantine_dir, exist_ok=True)
    for run_dir, _, _ in doomed:
        shutil.move(run_dir, os.path.join(args.quarantine_dir, os.path.basename(run_dir)))

    print(f"\nMoved {len(doomed)} run folder(s) to {args.quarantine_dir}")
    print("Re-run them with:")
    print("  python -m src.harness.runner --full --resume")


if __name__ == "__main__":
    main()
