"""
The ONE agent scaffold. Every provider runs through exactly this function,
same system prompt, same tool set, same LangGraph ReAct loop. Only the `llm`
object passed in differs between providers (see src/providers/factory.py).

This file is the thing RQ5 depends on: if we ever caught ourselves writing
per-provider prompt tweaks or per-provider tool variants, the whole "isolate
backend effect" premise of the paper would collapse.
"""
from langgraph.prebuilt import create_react_agent
from src.agent.tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are a careful engineering assistant. You have access to tools for "
    "listing files in your working directory, reading files, writing files, "
    "running Python, and doing arithmetic. "
    "Before assuming information or content is missing, check whether "
    "relevant files already exist in your working directory using "
    "list_files, and read them if relevant, rather than assuming based on "
    "the instruction text alone. "
    "Use tools as needed to complete the task fully before responding. "
    "If a task's instructions are still ambiguous or underspecified after "
    "checking, say so explicitly and state the assumption you are making, "
    "rather than guessing silently."
)


def build_agent(llm):
    """
    Given any LangChain chat model (from get_llm), returns a compiled
    LangGraph ReAct agent using the fixed tool set and system prompt.
    This function is called identically for all four providers, that
    identical call is the experimental control.
    """
    return create_react_agent(llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(llm, task_prompt: str, max_steps: int = 15):
    """
    Runs the scaffold on a single task prompt and returns the full
    LangGraph message history (used by trajectory_logger.py to extract
    the tool-call sequence for the ATA metric).
    """
    agent = build_agent(llm)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task_prompt}]},
        config={"recursion_limit": max_steps},
    )
    return result["messages"]
