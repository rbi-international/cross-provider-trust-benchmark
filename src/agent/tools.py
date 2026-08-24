"""
Tool definitions available to the agent. Identical set, identical
descriptions, regardless of which provider is running underneath. This is
part of what "same scaffold" means, if we let tool sets drift between
providers we'd be measuring two different agents, not one agent under two
backends.

Every tool operates inside a single per-task working directory (passed in
via set_working_dir), which is how each task's setup.files get materialized
and how we read back the final state for state_check pass_criteria.
"""
import io
import contextlib
import os
from langchain_core.tools import tool

_WORKING_DIR = {"path": None}


def set_working_dir(path: str):
    """Called once per task run, before the agent starts, to sandbox all
    file operations to that task's isolated folder."""
    _WORKING_DIR["path"] = path


def _resolve(filename: str) -> str:
    if _WORKING_DIR["path"] is None:
        raise RuntimeError("set_working_dir() must be called before tool use")
    # prevent path traversal outside the sandboxed task folder
    safe_name = os.path.basename(filename)
    return os.path.join(_WORKING_DIR["path"], safe_name)


@tool
def read_file(filename: str) -> str:
    """Read and return the full text contents of a file in the working directory."""
    path = _resolve(filename)
    if not os.path.exists(path):
        return f"ERROR: {filename} does not exist in the working directory."
    with open(path, "r") as f:
        return f.read()


@tool
def write_file(filename: str, content: str) -> str:
    """Write text content to a file in the working directory, overwriting it if it exists."""
    path = _resolve(filename)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {filename}."


@tool
def run_python(code: str) -> str:
    """Execute a snippet of Python code and return whatever it prints to stdout.
    Use print() to see any values you need. Has access to files in the working
    directory via open('filename'), relative paths only."""
    old_cwd = os.getcwd()
    os.chdir(_WORKING_DIR["path"])
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__})
        return stdout.getvalue() or "(no output)"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        os.chdir(old_cwd)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression (e.g. '100 * 0.92') and return the result."""
    try:
        # restricted eval: only arithmetic, no names/attributes
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "ERROR: expression contains disallowed characters."
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


ALL_TOOLS = [read_file, write_file, run_python, calculator]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
