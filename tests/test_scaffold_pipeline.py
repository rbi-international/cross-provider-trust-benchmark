"""
Validates the agent scaffold, tool wiring, and trajectory logging work
correctly end to end, using a scripted fake model instead of a real
provider. This catches structural bugs (tool binding, message parsing,
trajectory extraction) cheaply, before we spend real API budget on Week 4.
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from src.agent.tools import set_working_dir
from src.agent.scaffold import run_agent
from src.harness.trajectory_logger import extract_trajectory, save_trajectory
from src.harness.provenance import record_provenance


class ScriptedFakeChatModel(BaseChatModel):
    """A minimal fake chat model that plays back a fixed script of
    responses: first a tool call, then a final text answer. Good enough
    to exercise the real LangGraph ReAct loop without hitting any API."""

    script: List[Any] = []
    _step: int = 0

    def bind_tools(self, tools, **kwargs):
        return self  # fake model ignores real tool schemas, just plays the script

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        response = self.script[self._step]
        self._step = min(self._step + 1, len(self.script) - 1)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def test_pipeline():
    tmp_dir = tempfile.mkdtemp()
    try:
        set_working_dir(tmp_dir)
        with open(os.path.join(tmp_dir, "numbers.txt"), "w") as f:
            f.write("5,-2,9,-8,1,0,3")

        script = [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file",
                    "args": {"filename": "numbers.txt"},
                    "id": "call_1",
                }],
            ),
            AIMessage(content="Done, filtered and sorted the numbers."),
        ]
        fake_llm = ScriptedFakeChatModel(script=script)

        messages = run_agent(fake_llm, "Read numbers.txt and describe it.")
        assert len(messages) >= 2, "expected at least the tool call + final response"

        trajectory = extract_trajectory(messages)
        assert trajectory["tool_calls"][0]["tool_name"] == "read_file"
        assert trajectory["final_text"] == "Done, filtered and sorted the numbers."

        run_dir = os.path.join(tmp_dir, "run_output")
        save_trajectory(trajectory, run_dir)
        record_provenance(run_dir, config={"test": True}, seed=42, task_id="TEST-001", provider="fake")

        assert os.path.exists(os.path.join(run_dir, "trajectory.json"))
        assert os.path.exists(os.path.join(run_dir, "provenance.json"))

        print("ALL CHECKS PASSED")
        print("Trajectory extracted:", trajectory)
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_pipeline()
