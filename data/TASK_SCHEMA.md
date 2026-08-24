# Task Suite Schema

Every task across all four categories uses this shape, so the harness can
load and score them identically regardless of category.

```json
{
  "task_id": "A-019",
  "category": "codegen | tool_orchestration | error_recovery | ambiguous",
  "difficulty": "easy | medium | hard",
  "prompt": "The instruction given to the agent, verbatim.",
  "tools_available": ["read_file", "write_file", "run_python", "calculator"],
  "setup": {
    "files": { "filename.ext": "initial file content, if any" }
  },
  "pass_criteria": {
    "type": "unit_test | state_check | llm_judge_rubric",
    "detail": "unit test assertions, expected final state, or judge rubric text"
  },
  "source": "mbpp_sanitized_task_id:19 | hand-authored"
}
```

## Notes per category

- **A (codegen):** `pass_criteria.type = unit_test`, `detail` holds the MBPP
  `test_list` assertions run against the agent's generated function.
- **B (tool_orchestration):** `pass_criteria.type = state_check`, `detail`
  describes the expected final file/output state. `tools_available` always
  has 3+ usable tools since the task requires a real sequence.
- **C (error_recovery):** `setup.files` holds the broken artifact.
  `pass_criteria.type = unit_test` or `state_check` depending on the task,
  the agent must detect and repair the artifact, not just report the bug.
- **D (ambiguous):** `pass_criteria.type = llm_judge_rubric`. This category
  is inherently subjective, so scoring uses an LLM-as-judge rubric checking
  whether the agent asked a clarifying question OR explicitly stated and
  followed a reasonable assumption, versus silently guessing. This directly
  supports the Trustworthy AI framing: an untrustworthy agent guesses
  silently, a trustworthy one flags the ambiguity.

## Difficulty proxy (Category A only, for now)

MBPP ships no official difficulty label. We compute a transparent proxy:
`score = code_lines * 2 + code_chars * 0.05 + n_test_asserts * 3`, then split
into tertiles (easy/medium/hard) across all 427 sanitized problems. This is
disclosed as a limitation in the paper's Threats to Validity section, since
it's a proxy, not a validated difficulty measure.
