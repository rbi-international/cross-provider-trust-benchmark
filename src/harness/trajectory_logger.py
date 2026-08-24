"""
Extracts a clean, comparable tool-call trajectory from a LangGraph message
history. This is the raw material for Action Trajectory Agreement (ATA) in
Week 5, comparing whether different providers solve the same task via
similar or divergent tool-call sequences.

Also detects "phantom" tool calls: when a model describes a tool call as
plain-text JSON (e.g. {"name": "write_file", "parameters": {...}}) instead
of actually invoking LangChain's structured tool-calling mechanism. This is
a documented, real limitation of some models/providers, not a harness bug,
see the Week 4 pilot run discussion. Tracking it explicitly turns an
otherwise-opaque FAIL into a quantifiable "protocol adherence" signal for
RQ2 and RQ5, whether the model can reliably use the tool-calling contract
at all is itself a trustworthiness dimension.
"""
import json
import os
import re
from datetime import datetime, timezone

_KNOWN_TOOL_NAMES = {"read_file", "write_file", "run_python", "calculator"}

# Matches a JSON-object-shaped string with a "name" key and a
# "parameters" or "arguments" key, the shape models fall back to when they
# describe a tool call in text instead of invoking it structurally.
_PHANTOM_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"([a-zA-Z_]+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


def _detect_phantom_tool_call(text):
    """Returns {'attempted_tool': name, 'known_tool': bool, 'raw_text': text}
    if the text looks like a textual tool-call description rather than real
    content, else None. Flags it whether or not the named tool is one of
    ours: a hallucinated tool name (e.g. "solution" instead of "write_file")
    is an even more severe protocol failure than mis-invoking a real one,
    and both are worth counting separately for RQ2/RQ5."""
    if not text:
        return None
    match = _PHANTOM_PATTERN.search(text)
    if match:
        return {
            "attempted_tool": match.group(1),
            "known_tool": match.group(1) in _KNOWN_TOOL_NAMES,
            "raw_text": text.strip(),
        }
    return None


def extract_trajectory(messages):
    """
    Walks a LangGraph result's message list and pulls out:
    - tool_calls: ordered list of {tool_name, args} for every REAL
      structured tool call made during the run
    - final_text: the agent's last substantive text response
    - phantom_tool_calls: any AI messages where the model appears to have
      described a tool call in text instead of invoking it
    - protocol_adherence: False if any phantom tool calls were detected,
      True otherwise
    """
    tool_calls = []
    phantom_calls = []
    final_text = None

    for msg in messages:
        msg_tool_calls = getattr(msg, "tool_calls", None)
        content = getattr(msg, "content", None)

        if msg_tool_calls:
            for call in msg_tool_calls:
                tool_calls.append({
                    "tool_name": call.get("name"),
                    "args": call.get("args"),
                })

        if getattr(msg, "type", None) == "ai":
            if not msg_tool_calls:
                phantom = _detect_phantom_tool_call(content)
                if phantom:
                    phantom_calls.append(phantom)
                else:
                    # only a genuine final answer if it wasn't a phantom call
                    final_text = content

    return {
        "tool_calls": tool_calls,
        "final_text": final_text,
        "phantom_tool_calls": phantom_calls,
        "protocol_adherence": len(phantom_calls) == 0,
    }


def save_trajectory(trajectory, run_dir, filename="trajectory.json"):
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
