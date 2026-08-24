"""
Extracts a clean, comparable tool-call trajectory from a LangGraph message
history. This is the raw material for Action Trajectory Agreement (ATA) in
Week 5, comparing whether different providers solve the same task via
similar or divergent tool-call sequences.
"""
import json
import os
from datetime import datetime, timezone


def extract_trajectory(messages) -> list[dict]:
    """
    Walks a LangGraph result's message list and pulls out an ordered list of
    {tool_name, args} for every tool call made during the run, plus the
    final text response. This is provider-agnostic: LangChain normalizes
    tool_calls onto AIMessage.tool_calls for every provider we use.
    """
    trajectory = []
    final_text = None

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                trajectory.append({
                    "tool_name": call.get("name"),
                    "args": call.get("args"),
                })
        # capture the last AI message with no further tool calls as the final answer
        if getattr(msg, "type", None) == "ai" and not tool_calls:
            final_text = getattr(msg, "content", None)

    return {"tool_calls": trajectory, "final_text": final_text}


def save_trajectory(trajectory: dict, run_dir: str, filename: str = "trajectory.json"):
    """Writes the trajectory to the run's output folder with a timestamp,
    so every run is independently auditable."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **trajectory,
    }
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
