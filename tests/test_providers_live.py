"""
Run this LOCALLY, with your real .env filled in, before Week 4's pilot run.
Sends one trivial message to each of the three confirmed providers to
confirm keys/access actually work. Costs a few cents at most, catches auth
or setup problems cheaply instead of mid-pilot.

Usage: python tests/test_providers_live.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.providers.factory import get_llm

PROVIDERS_TO_TEST = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-sonnet-5"),
    ("groq", "openai/gpt-oss-120b"),
    ("ollama", "llama3.2"),
]


def test_provider(name: str, model: str):
    print(f"\n--- Testing {name} ({model}) ---")
    try:
        llm = get_llm(name, model)
        response = llm.invoke("Reply with exactly the word: OK")
        content = response.content.strip()
        print(f"  Response: {content!r}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = {}
    for name, model in PROVIDERS_TO_TEST:
        results[name] = test_provider(name, model)

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        print("\nFix any FAIL above before starting the Week 4 pilot run.")
        sys.exit(1)
    print("\nAll providers reachable, ready for Week 4.")
