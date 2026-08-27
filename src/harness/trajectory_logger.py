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
import zlib
from collections import Counter
from datetime import datetime, timezone

_KNOWN_TOOL_NAMES = {"read_file", "write_file", "run_python", "calculator"}

# Locates the OPENING of a textual tool-call description: a JSON object whose
# first key is "name". Deliberately does NOT try to match the closing brace.
#
# The previous version of this pattern required the whole object to match,
# with the argument object captured non-greedily as (\{.*?\}). That silently
# failed on any tool call whose arguments themselves contain braces - which is
# most real ones, since run_python arguments are Python source containing dict
# and comprehension literals. The result was a false NEGATIVE on exactly the
# most severe protocol failures in the run set (see the Week 5 audit), so
# those runs were recorded as protocol-adherent. Matching only the opening and
# then finding the extent separately is what makes detection robust.
_PHANTOM_OPEN_PATTERN = re.compile(r'\{\s*"name"\s*:\s*"([a-zA-Z_][\w.-]*)"')
_ARGS_KEY_PATTERN = re.compile(r'"(?:parameters|arguments|args)"\s*:')

# A model stuck in a token-repetition loop emits enormous output and is the
# single biggest driver of wall-clock cost in the run set. Text at or beyond
# this length is checked for degeneracy.
#
# Degeneracy is measured by COMPRESSION RATIO rather than by character
# frequency. A character-frequency test only catches a repeated single
# character: the worst case in the Week 4 run set repeated the six-character
# sequence "–" some 13,600 times, which spreads its mass over six
# characters and slips under any per-character threshold. Compression is
# agnostic to the length of the repeated unit.
#
# The separation measured on the real run set is not marginal: degenerate
# outputs compress to 0.004-0.006 of their size, ordinary agent prose to
# 0.63-0.67. The threshold sits in the two-order-of-magnitude gap between
# them, so it is not a tuned parameter.
_DEGENERATE_MIN_LENGTH = 2000
_DEGENERATE_COMPRESSION_RATIO = 0.05


def _json_object_extent(text, start):
    """Returns the index just past the JSON object beginning at `start`, or
    None if it never closes. Brace-counts while respecting string literals
    and escapes, so nested argument objects are handled correctly."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _detect_degenerate_repetition(text):
    """Detects the runaway-repetition failure mode: a short substring emitted
    thousands of times until generation stops.

    This is tracked separately from phantom calls because it is a distinct
    trust failure with a distinct operational cost. A phantom call is wrong
    but cheap; a repetition loop burned up to 30 minutes of wall clock on a
    single task in the Week 4 run set, and was the entire reason the local
    model's mean latency (79.5s) sat 18x above its median (4.3s).
    """
    if not text or len(text) < _DEGENERATE_MIN_LENGTH:
        return None

    encoded = text.encode("utf-8", "replace")
    ratio = len(zlib.compress(encoded, 6)) / len(encoded)
    if ratio > _DEGENERATE_COMPRESSION_RATIO:
        return None

    dominant_char, count = Counter(text).most_common(1)[0]
    return {
        "length": len(text),
        "compression_ratio": round(ratio, 5),
        "dominant_char": dominant_char,
        "dominant_char_share": round(count / len(text), 4),
    }


def normalize_content(content):
    """Flattens a message's content to a plain string.

    Anthropic returns content as a LIST of typed blocks
    ([{"type": "text", "text": ...}, ...]) rather than a bare string, so
    handing it straight to a regex raises
    "TypeError: expected string or bytes-like object, got 'list'".
    In the Week 5 grid that killed 17 anthropic runs, which were then recorded
    as agent failures - a harness bug charged to a provider's score.

    Every provider's content passes through here before any text analysis, so
    the difference in response shape stays a provider-adapter detail and never
    reaches the metrics.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _detect_phantom_tool_call(text):
    """Returns a dict describing a textual tool-call description, else None.

    Flags the call whether or not the named tool is one of ours: a
    hallucinated tool name (e.g. "solution" instead of "write_file") is an
    even more severe protocol failure than mis-invoking a real one, and both
    are worth counting separately for RQ2/RQ5.

    An object that names a tool but never closes - because the model ran off
    into a repetition loop mid-call - is still a phantom call, and is marked
    with well_formed=False rather than being missed.
    """
    if not text:
        return None

    match = _PHANTOM_OPEN_PATTERN.search(text)
    if not match:
        return None

    end = _json_object_extent(text, match.start())
    candidate = text[match.start():end] if end else text[match.start():]

    # Require an argument-bearing key, so ordinary prose that happens to quote
    # a {"name": ...} object is not mistaken for an attempted tool call.
    if not _ARGS_KEY_PATTERN.search(candidate):
        return None

    parsed = None
    if end:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None

    return {
        "attempted_tool": match.group(1),
        "known_tool": match.group(1) in _KNOWN_TOOL_NAMES,
        "well_formed": parsed is not None,
        "truncated": end is None,
        "raw_text": text.strip(),
    }


def extract_trajectory(messages):
    """
    Walks a LangGraph result's message list and pulls out:
    - tool_calls: ordered list of {tool_name, args} for every REAL
      structured tool call made during the run
    - final_text: the agent's last substantive text response
    - phantom_tool_calls: any AI messages where the model appears to have
      described a tool call in text instead of invoking it
    - degenerate_outputs: any AI messages that ran away into a token
      repetition loop
    - protocol_adherence: False if any phantom tool calls were detected,
      True otherwise
    """
    tool_calls = []
    phantom_calls = []
    degenerate_outputs = []
    final_text = None

    for msg in messages:
        msg_tool_calls = getattr(msg, "tool_calls", None)
        # Normalised once, up front: every downstream check is text analysis
        # and none of them should have to know how a given provider shapes
        # its content field.
        content = normalize_content(getattr(msg, "content", None))

        if msg_tool_calls:
            for call in msg_tool_calls:
                tool_calls.append({
                    "tool_name": call.get("name"),
                    "args": call.get("args"),
                })

        if getattr(msg, "type", None) == "ai":
            degenerate = _detect_degenerate_repetition(content)
            if degenerate:
                degenerate_outputs.append(degenerate)

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
        "degenerate_outputs": degenerate_outputs,
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
