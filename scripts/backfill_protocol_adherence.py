"""
Recomputes phantom-tool-call detection over run folders that are ALREADY on
disk, and rewrites protocol_adherence in each trajectory.json / result.json.

Why this exists, and why it is legitimate rather than post-hoc fudging:

The Week 5 audit found that the original phantom-call regex required the whole
JSON object to match, capturing the argument object non-greedily as (\\{.*?\\}).
That fails whenever the arguments themselves contain braces - which is most
real calls, since run_python arguments are Python source full of dict and
comprehension literals. The consequence was a false NEGATIVE on the most
severe protocol failures in the run set: runs where the model emitted a
tool call as prose and then ran off into a token-repetition loop were being
recorded as protocol-ADHERENT.

Nothing about the runs themselves is being changed here. The agent's raw
output was captured verbatim in trajectory.json at run time; only our reading
of that text was wrong. Re-deriving a metric from unmodified raw output is
exactly what the trajectory log was written for, and it is what makes this
correctable without spending another API call.

    python scripts/backfill_protocol_adherence.py --dry-run    # report only
    python scripts/backfill_protocol_adherence.py              # apply

A .bak copy of every rewritten file is left beside it.
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.harness.trajectory_logger import (
    _detect_degenerate_repetition,
    _detect_phantom_tool_call,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "experiments", "runs")


def recompute(trajectory):
    """Re-derive phantom calls and degeneracy from the saved final_text.

    Only final_text survives in the log, so a phantom call that appeared in
    an earlier message of a multi-step run cannot be recovered. That makes
    this a LOWER BOUND on the true phantom rate, which is the safe direction:
    it can only under-report protocol failure, never invent it. Runs made
    after this fix detect phantom calls on every message as they happen.
    """
    final_text = trajectory.get("final_text")
    existing = trajectory.get("phantom_tool_calls") or []

    phantom = _detect_phantom_tool_call(final_text) if isinstance(final_text, str) else None
    degenerate = _detect_degenerate_repetition(final_text) if isinstance(final_text, str) else None

    phantom_calls = list(existing)
    if phantom and not any(p.get("raw_text") == phantom.get("raw_text") for p in existing):
        phantom_calls.append(phantom)

    return {
        "phantom_tool_calls": phantom_calls,
        "degenerate_outputs": [degenerate] if degenerate else [],
        "protocol_adherence": len(phantom_calls) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    args = parser.parse_args()

    changed, degenerate_found = [], []
    scanned = 0

    for trajectory_path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "trajectory.json"))):
        run_dir = os.path.dirname(trajectory_path)
        result_path = os.path.join(run_dir, "result.json")
        try:
            with open(trajectory_path) as f:
                trajectory = json.load(f)
        except Exception as e:
            print(f"  SKIP {os.path.basename(run_dir)}: {type(e).__name__}: {e}")
            continue

        scanned += 1
        updated = recompute(trajectory)
        was_adherent = trajectory.get("protocol_adherence", True)

        if updated["degenerate_outputs"]:
            degenerate_found.append((run_dir, updated["degenerate_outputs"][0]))

        needs_write = (
            was_adherent != updated["protocol_adherence"]
            or len(trajectory.get("phantom_tool_calls") or []) != len(updated["phantom_tool_calls"])
            or bool(updated["degenerate_outputs"]) != bool(trajectory.get("degenerate_outputs"))
        )
        if not needs_write:
            continue

        if was_adherent != updated["protocol_adherence"]:
            changed.append((run_dir, was_adherent, updated["protocol_adherence"]))

        if args.dry_run:
            continue

        shutil.copyfile(trajectory_path, trajectory_path + ".bak")
        trajectory.update(updated)
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            shutil.copyfile(result_path, result_path + ".bak")
            result["protocol_adherence"] = updated["protocol_adherence"]
            result["phantom_tool_call_count"] = len(updated["phantom_tool_calls"])
            result["degenerate_output_count"] = len(updated["degenerate_outputs"])
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

    print(f"scanned {scanned} run folders")
    print(f"protocol_adherence flipped on {len(changed)} run(s)"
          f"{' (dry run, nothing written)' if args.dry_run else ''}")

    by_provider = collections.Counter()
    for run_dir, _, _ in changed:
        by_provider[os.path.basename(run_dir).split("_")[1]] += 1
    for provider, count in by_provider.most_common():
        print(f"  {provider:10s} {count}")

    print(f"\ndegenerate repetition loops found: {len(degenerate_found)}")
    for run_dir, info in sorted(degenerate_found, key=lambda x: -x[1]["length"])[:8]:
        print(f"  {os.path.basename(run_dir):48s} {info['length']:>7d} chars, "
              f"{info['dominant_char_share']:.0%} one char")


if __name__ == "__main__":
    main()
