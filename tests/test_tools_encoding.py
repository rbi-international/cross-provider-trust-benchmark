"""
Guards the UTF-8 encoding contract on the agent's file tools.

This exists because the fix was silently REVERTED once: an unrelated commit
adding the list_files tool rewrote tools.py from a pre-fix copy, dropping the
explicit encoding without anything failing. The bug it re-introduced had
already invalidated 54 runs (all groq, concentrated in codegen), scoring a
harness failure as an agent failure.

The characters exercised below are ones models emit routinely and that cp1252
- Python's default on Windows - cannot encode. On a UTF-8 platform these tests
pass whether or not the encoding is pinned, so they are a weaker guard there;
the source-level assertion at the bottom holds everywhere.
"""
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tools.py imports langchain, which lives only in the agentic-trust-bench env.
# Skip rather than fail so the offline suite still runs under base python.
pytest.importorskip("langchain_core", reason="agent tools require the project conda env")

from src.agent.tools import _ENCODING, read_file, set_working_dir, write_file

# non-breaking hyphen, en-dash, curly quotes, em-dash, ellipsis, accented text
NON_ASCII = "non‑breaking – dash, “curly” quotes — café … naïve"


@pytest.fixture
def working_dir():
    path = tempfile.mkdtemp(prefix="tools_enc_")
    set_working_dir(path)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_write_file_accepts_non_ascii_content(working_dir):
    """The exact failure mode: UnicodeEncodeError on write, recorded as an
    agent failure when the benchmark is what broke."""
    result = write_file.invoke({"filename": "out.txt", "content": NON_ASCII})
    assert "ERROR" not in result
    on_disk = open(os.path.join(working_dir, "out.txt"), encoding="utf-8").read()
    assert on_disk == NON_ASCII


def test_read_file_round_trips_non_ascii(working_dir):
    """Reading must use the same encoding as writing, or the scorer cannot
    read back what the agent legitimately produced."""
    write_file.invoke({"filename": "rt.txt", "content": NON_ASCII})
    assert read_file.invoke({"filename": "rt.txt"}) == NON_ASCII


def test_read_file_handles_a_utf8_fixture_written_by_the_runner(working_dir):
    """Task setup files are materialised by the runner as UTF-8; the tools
    must be able to read one back."""
    with open(os.path.join(working_dir, "fixture.txt"), "w", encoding="utf-8") as f:
        f.write(NON_ASCII)
    assert read_file.invoke({"filename": "fixture.txt"}) == NON_ASCII


def test_encoding_is_pinned_to_utf8_in_source():
    """Platform-independent guard. The round-trip tests above pass on a UTF-8
    default platform even with the encoding unpinned, so this is the assertion
    that actually catches the regression everywhere."""
    import inspect

    from src.agent import tools

    assert _ENCODING == "utf-8"
    source = inspect.getsource(tools)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("with open(path") and "encoding" not in stripped:
            pytest.fail(f"file opened without an explicit encoding: {stripped!r}")
