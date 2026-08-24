"""
Validates the scoring engine against all three pass_criteria types using
real task definitions from the actual task suite, with a fake agent output
substituted in (no API calls). Catches scoring-logic bugs cheaply before
they'd silently mis-grade a real, paid pilot run.
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.harness.scoring import score_task

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_task(category_dir, task_id):
    path = os.path.join(REPO_ROOT, "data", "tasks", category_dir, "tasks.json")
    tasks = json.load(open(path))
    return next(t for t in tasks if t["task_id"] == task_id)


def test_unit_test_codegen_pass_and_fail():
    task = load_task("category_a_codegen", "A-111")
    tmp = tempfile.mkdtemp()
    try:
        # correct solution
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write(
                "def common_in_nested_lists(nested_list):\n"
                "    result = set(nested_list[0])\n"
                "    for lst in nested_list[1:]:\n"
                "        result &= set(lst)\n"
                "    return list(result)\n"
            )
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert passed, f"expected PASS, got: {reason}"

        # wrong solution
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write("def common_in_nested_lists(nested_list):\n    return []\n")
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert not passed, "expected FAIL for an empty-list stub, got PASS"
        print("  unit_test (codegen): PASS+FAIL both correctly detected")
    finally:
        shutil.rmtree(tmp)


def test_unit_test_error_recovery():
    task = load_task("category_c_error_recovery", "C-003")
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "divide.py"), "w") as f:
            f.write(
                "def safe_divide(a, b):\n"
                "    if b == 0:\n        return None\n"
                "    return a / b\n"
            )
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert passed, f"expected PASS, got: {reason}"
        print("  unit_test (error_recovery C-003): PASS correctly detected")
    finally:
        shutil.rmtree(tmp)


def test_state_check_substring_and_order():
    task = load_task("category_b_tool_orchestration", "B-005")
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "common_words.txt"), "w") as f:
            f.write("banana\ndate\n")
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert passed, f"expected PASS, got: {reason}"

        # wrong order should fail
        with open(os.path.join(tmp, "common_words.txt"), "w") as f:
            f.write("date\nbanana\n")
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert not passed, "expected FAIL for wrong order, got PASS"
        print("  state_check (order B-005): PASS+FAIL both correctly detected")
    finally:
        shutil.rmtree(tmp)


def test_state_check_json_equality():
    task = load_task("category_c_error_recovery", "C-002")
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "settings.json"), "w") as f:
            f.write('{"host": "localhost", "port": 8080}')
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert passed, f"expected PASS, got: {reason}"
        print("  state_check (json C-002): PASS correctly detected")
    finally:
        shutil.rmtree(tmp)


def test_state_check_exec_stdout():
    task = load_task("category_c_error_recovery", "C-001")
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "greet.py"), "w") as f:
            f.write("def greet():\n    print('Hello, World!')\n\ngreet()")
        passed, reason = score_task(task, tmp, {"final_text": ""})
        assert passed, f"expected PASS, got: {reason}"
        print("  state_check (exec_stdout C-001): PASS correctly detected")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_unit_test_codegen_pass_and_fail()
    test_unit_test_error_recovery()
    test_state_check_substring_and_order()
    test_state_check_json_equality()
    test_state_check_exec_stdout()
    print("\nALL SCORING CHECKS PASSED")
