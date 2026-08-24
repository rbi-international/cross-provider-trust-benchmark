"""
Every run folder gets a provenance.json snapshot: the exact config used,
the seed, the current git commit hash, and installed package versions.
This is what lets any number in the paper trace back to one reproducible
run, matching the standard research discipline used across the rest of
this project.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError


TRACKED_PACKAGES = [
    "langchain", "langgraph", "langchain-openai",
    "langchain-groq", "langchain-ollama", "langchain-ibm",
]


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _package_versions() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def record_provenance(run_dir: str, config: dict, seed: int, task_id: str, provider: str):
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "provider": provider,
        "seed": seed,
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "config_snapshot": config,
    }
    path = os.path.join(run_dir, "provenance.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
