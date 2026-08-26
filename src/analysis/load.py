"""
Single source of truth for reading experiments/runs/ back off disk.

Every metric, statistic, table, and figure in the paper is computed from
what this module returns, so no number in the paper can come from anywhere
except a real run folder. Each run folder contributes result.json (verdict
and metadata) joined with trajectory.json (the tool-call path), which the
metrics need together.

Run folders whose result.json is missing or unparseable are reported as
skipped rather than dropped silently - a run that failed to write its own
result is a data-integrity event the analysis should surface, not absorb.
"""
import glob
import json
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "experiments", "runs")

# Ollama parameter counts are the edge-viability proxy for RQ3. Values are
# the model's advertised parameter count in billions. Closed cloud models
# have no published count; they are recorded as None and excluded from the
# RQ3 correlation, with the exclusion stated in the paper rather than
# papered over with a guessed number.
MODEL_SIZE_B = {
    "llama3.2": 3.0,
    "openai/gpt-oss-120b": 120.0,
    "gpt-4o-mini": None,
    "claude-sonnet-5": None,
}


def load_runs(runs_dir=None, require_trajectory=True):
    """Load every run folder into a flat list of run dicts.

    Each dict is result.json plus a "trajectory" key. Returns
    (runs, skipped) where skipped is a list of (path, reason).
    """
    runs_dir = runs_dir or RUNS_DIR
    runs, skipped = [], []

    for result_path in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        run_dir = os.path.dirname(result_path)
        try:
            with open(result_path) as f:
                result = json.load(f)
        except Exception as e:
            skipped.append((run_dir, f"unreadable result.json: {type(e).__name__}: {e}"))
            continue

        trajectory_path = os.path.join(run_dir, "trajectory.json")
        trajectory = None
        if os.path.exists(trajectory_path):
            try:
                with open(trajectory_path) as f:
                    trajectory = json.load(f)
            except Exception as e:
                skipped.append((run_dir, f"unreadable trajectory.json: {type(e).__name__}: {e}"))
                continue

        if trajectory is None:
            if require_trajectory:
                skipped.append((run_dir, "no trajectory.json"))
                continue
            trajectory = {"tool_calls": [], "final_text": None}

        result["trajectory"] = trajectory
        result["run_dir"] = run_dir
        result["model_size_b"] = MODEL_SIZE_B.get(result.get("model"))
        runs.append(result)

    return runs, skipped


def runs_to_frame(runs):
    """Tabular view for the statistics layer.

    One row per run, trajectory collapsed to the summary columns the models
    actually use (tool-call count, protocol adherence). The full trajectory
    stays in the run dicts for ATA, which needs the ordered sequence.
    """
    rows = []
    for run in runs:
        trajectory = run.get("trajectory") or {}
        rows.append({
            "run_id": run.get("run_id"),
            "provider": run.get("provider"),
            "model": run.get("model"),
            "model_size_b": run.get("model_size_b"),
            "task_id": run.get("task_id"),
            "category": run.get("category"),
            "difficulty": run.get("difficulty"),
            "run_idx": run.get("run_idx"),
            "passed": bool(run.get("passed")),
            "success": int(bool(run.get("passed"))),
            "n_tool_calls": len(trajectory.get("tool_calls") or []),
            "protocol_adherence": bool(run.get("protocol_adherence", trajectory.get("protocol_adherence", True))),
            "phantom_tool_call_count": run.get(
                "phantom_tool_call_count", len(trajectory.get("phantom_tool_calls") or [])
            ),
            "elapsed_seconds": run.get("elapsed_seconds"),
            "error": run.get("error"),
        })
    return pd.DataFrame(rows)


def coverage_report(frame):
    """Runs per provider x category cell, and repeats per provider x task.

    Printed before any analysis so an unbalanced or thin grid is visible up
    front - several statistics below assume a roughly balanced design, and
    the pilot data deliberately is not.
    """
    cells = frame.pivot_table(
        index="provider", columns="category", values="run_id", aggfunc="count", fill_value=0
    )
    repeats = frame.groupby(["provider", "task_id"])["run_id"].count()
    return {
        "cells": cells,
        "repeat_counts": repeats.value_counts().sort_index().to_dict(),
        "n_runs": len(frame),
        "n_tasks": frame["task_id"].nunique(),
        "providers": sorted(frame["provider"].dropna().unique()),
        "categories": sorted(frame["category"].dropna().unique()),
    }
