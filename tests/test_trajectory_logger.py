"""
Regression tests for phantom tool-call and degenerate-output detection.

Every case in the "nested braces" group below was a FALSE NEGATIVE under the
original regex, which required the whole JSON object to match with the
argument object captured non-greedily as (\\{.*?\\}). Because a run_python
argument is Python source, it almost always contains braces, so the detector
silently missed exactly the runs it existed to catch and recorded them as
protocol-adherent. These tests pin the fix.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.harness.trajectory_logger import (
    _detect_degenerate_repetition,
    _detect_phantom_tool_call,
    _json_object_extent,
    extract_trajectory,
)


class FakeMessage:
    """Stands in for a LangChain message without importing langchain."""

    def __init__(self, type_, content=None, tool_calls=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls or []


# --------------------------------------------------------------------------
# phantom detection - the cases that used to slip through
# --------------------------------------------------------------------------

def test_detects_phantom_call_with_flat_arguments():
    """The case the original regex did handle; kept so the fix is not a
    regression in the other direction."""
    found = _detect_phantom_tool_call('{"name":"run_python","parameters":{"code":"print(1)"}}')
    assert found is not None
    assert found["attempted_tool"] == "run_python"
    assert found["known_tool"] is True
    assert found["well_formed"] is True


def test_detects_phantom_call_whose_arguments_contain_braces():
    """The original false negative: Python source in the arguments contains a
    dict literal, so the non-greedy brace capture terminated early and the
    whole match failed."""
    text = ('{"name":"run_python","parameters":{"code":"counts = {level: 0 for level '
            'in levels}\\nprint(counts)"}}')
    found = _detect_phantom_tool_call(text)
    assert found is not None, "a tool call with braces in its arguments must still be detected"
    assert found["attempted_tool"] == "run_python"


def test_detects_phantom_call_that_never_closes():
    """A model that names a tool and then runs off into a repetition loop
    never emits the closing brace. That is a protocol failure, not an
    absence of one."""
    text = '{"name":"run_python","parameters":{"code":"import json' + ("–" * 5000)
    found = _detect_phantom_tool_call(text)
    assert found is not None
    assert found["truncated"] is True
    assert found["well_formed"] is False


def test_detects_hallucinated_tool_name():
    found = _detect_phantom_tool_call('{"name": "solution", "parameters": {"filename": "solution.py"}}')
    assert found is not None
    assert found["attempted_tool"] == "solution"
    assert found["known_tool"] is False, "a made-up tool name is a worse failure, not an exempt one"


def test_accepts_the_arguments_spelling_variants():
    for key in ("parameters", "arguments", "args"):
        text = '{"name":"write_file","%s":{"filename":"x.py"}}' % key
        assert _detect_phantom_tool_call(text) is not None, key


def test_no_false_positive_on_ordinary_prose():
    assert _detect_phantom_tool_call("The average is 77.20, written to average_result.txt.") is None
    assert _detect_phantom_tool_call("") is None
    assert _detect_phantom_tool_call(None) is None


def test_no_false_positive_on_a_name_object_without_arguments():
    """Prose that happens to quote a JSON object with a name key is not an
    attempted tool call unless it also carries an argument-bearing key."""
    assert _detect_phantom_tool_call('I wrote {"name": "config"} to the file.') is None


def test_json_object_extent_respects_strings_and_escapes():
    assert _json_object_extent('{"a":1}', 0) == 7
    assert _json_object_extent('{"a":{"b":2}}', 0) == 13
    # a brace inside a string literal must not be counted
    assert _json_object_extent('{"a":"}"}', 0) == 9
    # an escaped quote must not end the string early
    assert _json_object_extent('{"a":"\\""}', 0) == 10
    assert _json_object_extent('{"a":1', 0) is None


# --------------------------------------------------------------------------
# degenerate repetition
# --------------------------------------------------------------------------

def test_detects_a_multi_character_repeated_unit():
    """The worst real case repeated the six-character sequence for an en-dash
    escape ~13,600 times. No single character exceeded 17% of the text, so a
    character-frequency test missed it entirely; compression does not."""
    text = "\\u2013" * 14000
    found = _detect_degenerate_repetition(text)
    assert found is not None
    assert found["compression_ratio"] < 0.05
    assert found["dominant_char_share"] < 0.30, "single-char frequency alone would not have caught this"


def test_detects_a_single_character_loop():
    found = _detect_degenerate_repetition("\t" * 20000)
    assert found is not None
    assert found["length"] == 20000


def test_long_varied_prose_is_not_degenerate():
    """The false-positive guard. Note the threshold was calibrated against
    real outputs that all sat BELOW the 2000-character floor (the longest
    well-formed response in the run set is ~700 chars), so this synthetic
    case is what stands in for long genuine output. Varied text of this
    length compresses to roughly 0.3, far above the 0.05 threshold.

    A single sentence repeated many times is NOT a valid negative case here:
    that text really is degenerate, and the detector is right to say so.
    """
    import random

    rng = random.Random(0)
    vocabulary = ("configuration validated required missing report saved working "
                  "directory parsed inspected threshold rewrote appended verified "
                  "checksum manifest schema fallback retried timeout".split())
    prose = " ".join(rng.choice(vocabulary) for _ in range(700))
    assert len(prose) > 2000
    assert _detect_degenerate_repetition(prose) is None


def test_short_text_is_never_degenerate():
    """Below the length floor, repetition is not yet evidence of a loop."""
    assert _detect_degenerate_repetition("ab" * 50) is None
    assert _detect_degenerate_repetition("") is None
    assert _detect_degenerate_repetition(None) is None


# --------------------------------------------------------------------------
# end-to-end trajectory extraction
# --------------------------------------------------------------------------

def test_extract_trajectory_records_real_calls_and_marks_adherence():
    messages = [
        FakeMessage("human", "do the task"),
        FakeMessage("ai", "", tool_calls=[{"name": "read_file", "args": {"filename": "a.txt"}}]),
        FakeMessage("tool", "contents"),
        FakeMessage("ai", "I read the file and wrote the answer."),
    ]
    trajectory = extract_trajectory(messages)
    assert [c["tool_name"] for c in trajectory["tool_calls"]] == ["read_file"]
    assert trajectory["protocol_adherence"] is True
    assert trajectory["phantom_tool_calls"] == []
    assert trajectory["final_text"] == "I read the file and wrote the answer."


def test_extract_trajectory_flags_a_phantom_call_with_braced_arguments():
    """End-to-end version of the false negative: this trajectory must come
    out as protocol_adherence=False."""
    messages = [
        FakeMessage("human", "do the task"),
        FakeMessage("ai", '{"name":"run_python","parameters":{"code":"d = {\\"k\\": 1}"}}'),
    ]
    trajectory = extract_trajectory(messages)
    assert trajectory["protocol_adherence"] is False
    assert len(trajectory["phantom_tool_calls"]) == 1
    assert trajectory["final_text"] is None, "a phantom call is not a final answer"


def test_extract_trajectory_records_degenerate_output_separately():
    """A repetition loop is reported even when it also parses as a phantom
    call: the two are distinct failure modes with distinct costs."""
    runaway = '{"name":"run_python","parameters":{"code":"x' + ("\\u2013" * 14000)
    trajectory = extract_trajectory([FakeMessage("ai", runaway)])
    assert trajectory["protocol_adherence"] is False
    assert len(trajectory["degenerate_outputs"]) == 1
    assert trajectory["degenerate_outputs"][0]["length"] > 2000


# --------------------------------------------------------------------------
# provider content-shape normalisation
# --------------------------------------------------------------------------

def test_normalize_content_flattens_anthropic_block_lists():
    """Anthropic returns content as a list of typed blocks. Passing that
    straight to a regex raises TypeError, which the harness recorded as an
    agent failure - a harness bug charged to a provider's score (17 runs in
    the Week 5 grid)."""
    from src.harness.trajectory_logger import normalize_content

    assert normalize_content("plain") == "plain"
    assert normalize_content(None) == ""
    assert normalize_content([{"type": "text", "text": "hello "},
                              {"type": "text", "text": "world"}]) == "hello world"
    assert normalize_content(["a", "b"]) == "ab"
    # a block with no text must not crash or inject "None"
    assert normalize_content([{"type": "tool_use", "id": "x"}]) == ""


def test_phantom_detection_works_on_block_list_content():
    """The end-to-end regression: a phantom call delivered as Anthropic-style
    blocks must be detected, not raise."""
    messages = [FakeMessage("ai", [
        {"type": "text", "text": '{"name":"run_python","parameters":'},
        {"type": "text", "text": '{"code":"d = {1: 2}"}}'},
    ])]
    trajectory = extract_trajectory(messages)
    assert trajectory["protocol_adherence"] is False
    assert trajectory["phantom_tool_calls"][0]["attempted_tool"] == "run_python"


def test_block_list_final_text_is_flattened_not_a_list():
    messages = [FakeMessage("ai", [{"type": "text", "text": "The answer is 42."}])]
    assert extract_trajectory(messages)["final_text"] == "The answer is 42."
