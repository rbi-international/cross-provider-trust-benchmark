"""
Orchestrates the full grid: for each provider, for each task in the
selected categories, for N repeated runs, materialize the task's setup
files into an isolated working directory, run the fixed agent scaffold,
score the result, and write trajectory/provenance/score into
experiments/runs/<run_id>/.

Usage:
    python -m src.harness.runner --pilot          # small batch, ~1-2 tasks/category
    python -m src.harness.runner --full            # the whole configured grid
    python -m src.harness.runner --pilot --providers openai   # single provider only
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.providers.factory import get_llm
from src.agent.tools import set_working_dir
from src.agent.scaffold import run_agent
from src.harness.trajectory_logger import extract_trajectory, save_trajectory
from src.harness.provenance import record_provenance
from src.harness.scoring import score_task

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")
TASKS_DIR = os.path.join(REPO_ROOT, "data", "tasks")
RUNS_DIR = os.path.join(REPO_ROOT, "experiments", "runs")

# A fixed judge model for category D (llm_judge_rubric) tasks, held constant
# across every provider's outputs, see scoring.py's score_llm_judge()
# docstring for why this isn't provider-neutral and what that costs us.
JUDGE_PROVIDER = "anthropic"
JUDGE_MODEL = "claude-sonnet-5"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_tasks(category_dirs):
    tasks = []
    for cat_dir in category_dirs:
        path = os.path.join(TASKS_DIR, cat_dir, "tasks.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping category.")
            continue
        with open(path) as f:
            tasks.extend(json.load(f))
    return tasks


def build_prompt(task):
    """Category-specific prompt augmentation. Codegen tasks get a solution.py
    instruction and a signature hint so the agent knows the exact expected
    function name/signature (standard practice in code-eval harnesses,
    disclosed in the paper's Experimental Setup, since it does give the
    agent slightly more information than the bare MBPP prompt alone)."""
    prompt = task["prompt"]
    if task["category"] == "codegen":
        hint = task["pass_criteria"].get("signature_hint", "")
        prompt += (
            f"\n\nWrite your solution as a function in a file called "
            f"solution.py using write_file. Your function must satisfy "
            f"this example usage: {hint}"
        )
    return prompt


def run_single(provider_name, model_name, task, run_idx, judge_llm, seed):
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{provider_name}_{task['task_id']}_r{run_idx}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    working_dir = tempfile.mkdtemp(prefix="task_")

    try:
        # materialize the task's setup files into the sandboxed working dir
        for filename, content in task.get("setup", {}).get("files", {}).items():
            with open(os.path.join(working_dir, filename), "w") as f:
                f.write(content)
        set_working_dir(working_dir)

        llm = get_llm(provider_name, model_name)
        prompt = build_prompt(task)

        start = time.time()
        error = None
        try:
            messages = run_agent(llm, prompt)
            trajectory = extract_trajectory(messages)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            trajectory = {"tool_calls": [], "final_text": None}
        elapsed = time.time() - start

        if error:
            passed, reason = False, f"FAIL: agent run raised an error: {error}"
        else:
            passed, reason = score_task(task, working_dir, trajectory, judge_llm=judge_llm)

        save_trajectory(trajectory, run_dir)
        record_provenance(run_dir, config={"task_id": task["task_id"], "provider": provider_name, "model": model_name},
                           seed=seed, task_id=task["task_id"], provider=provider_name)

        result = {
            "run_id": run_id,
            "provider": provider_name,
            "model": model_name,
            "task_id": task["task_id"],
            "category": task["category"],
            "difficulty": task.get("difficulty"),
            "run_idx": run_idx,
            "passed": passed,
            "reason": reason,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        }
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {provider_name:10s} {task['task_id']:8s} run{run_idx}  ({elapsed:.1f}s)  {reason[:80]}")
        return result

    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="small batch: 1 task per category, 1 run each")
    parser.add_argument("--full", action="store_true", help="the whole configured grid")
    parser.add_argument("--providers", nargs="*", default=None, help="restrict to these provider names")
    parser.add_argument("--categories", nargs="*", default=None, help="restrict to these category dir names")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        parser.error("pass --pilot or --full")

    config = load_config()
    provider_configs = config["providers"]
    if args.providers:
        provider_configs = [p for p in provider_configs if p["name"] in args.providers]

    category_dirs = args.categories or config["task_categories"]
    tasks = load_tasks(category_dirs)

    if args.pilot:
        # one task per category, easiest difficulty first, for a cheap smoke test
        seen_categories = set()
        pilot_tasks = []
        for t in sorted(tasks, key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x.get("difficulty"), 1)):
            if t["category"] not in seen_categories:
                pilot_tasks.append(t)
                seen_categories.add(t["category"])
        tasks = pilot_tasks
        runs_per_task = 1
    else:
        runs_per_task = config["runs_per_task_provider"]

    os.makedirs(RUNS_DIR, exist_ok=True)
    judge_llm = get_llm(JUDGE_PROVIDER, JUDGE_MODEL)

    print(f"Running {len(tasks)} tasks x {len(provider_configs)} providers x {runs_per_task} repeat(s) "
          f"= {len(tasks) * len(provider_configs) * runs_per_task} total runs\n")

    all_results = []
    for provider_cfg in provider_configs:
        for task in tasks:
            for run_idx in range(runs_per_task):
                result = run_single(
                    provider_cfg["name"], provider_cfg["model"], task, run_idx,
                    judge_llm, config["seed"],
                )
                all_results.append(result)

    summary_path = os.path.join(RUNS_DIR, f"_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    n_pass = sum(1 for r in all_results if r["passed"])
    print(f"\n{n_pass}/{len(all_results)} runs passed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
