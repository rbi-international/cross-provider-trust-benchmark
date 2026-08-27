# Project Log — Cross-Provider Trust and Consistency Benchmark

**Paper:** A Cross-Provider Trust and Consistency Benchmark for Agentic AI Systems
**Authors:** Rohit Bharti (LPU, Punjab), Ashish Arya
**Venue:** ISDIA 2027, Special Session on Trustworthy AI and Intelligent Edge Systems
**Deadline:** 23 September 2026
**Log updated:** 27 August 2026

This is the single consolidated record of the project: what exists, what the
numbers currently say, what is known to be wrong, and what happens next. It is
written so a co-author can pick the work up without reading the commit history.

---

## 1. Current status at a glance

| | |
|---|---|
| Roadmap weeks complete | 1–5 of 6 |
| Runs on disk | **1,212** across 60 tasks × 4 providers |
| Grid coverage | **Complete** — all four categories fully run |
| Total compute | ~5.0 hours of agent wall-clock |
| Test suite | 73 passing (offline) |
| **Blocking issue** | **73 runs invalid due to two harness bugs — fixed in code, runs not yet redone** |

Week 6 is: re-run the 73 invalidated cells, then write the paper.

---

## 2. What the experiment does

One agent scaffold is held completely fixed — same system prompt, same four
tools (`read_file`, `write_file`, `run_python`, `calculator`), same LangGraph
ReAct loop — and only the backend model is swapped underneath it. Every run
differs in exactly one factor.

**Providers** (`config/config.yaml`):

| Provider | Model | Role |
|---|---|---|
| openai | gpt-4o-mini | cheap mid-tier cloud reference |
| anthropic | claude-sonnet-5 | high-capacity cloud |
| groq | openai/gpt-oss-120b | large open-weight (120B), fast inference |
| ollama | llama3.2 (3B) | **genuinely edge-viable**, the low end of the capacity spread |

**Task categories** (60 tasks total, 5 repeats each per provider):

| Category | Tasks | Scoring |
|---|---|---|
| A. Code generation | 30 | `unit_test` — exec the file, run assertions |
| B. Tool orchestration | 10 | `state_check` — structural checks on output files |
| C. Error recovery | 10 | `state_check` / `unit_test` |
| D. Ambiguous instructions | 10 | `llm_judge_rubric` — fixed Claude judge against a rubric |

---

## 3. The four metrics (paper EQ 1–4)

All are pure functions over run records, bounded in [0, 1], higher is better.

**EQ 1 — Task Success Rate (TSR)** · `src/metrics/tsr.py`
Fraction of runs passing their `pass_criteria`, at provider / provider×category
/ provider×task granularity. Confidence intervals use the **Wilson score
interval**, not the normal approximation, because several cells sit at exactly
0.0 or 1.0 where the normal approximation escapes [0, 1] and reports zero width.

**EQ 2 — Action Trajectory Agreement (ATA)** · `src/metrics/ata.py`
Normalised edit-distance similarity over tool-name sequences:
`sim(A,B) = 1 − levenshtein(seq_A, seq_B) / max(|seq_A|, |seq_B|)`.
Order-sensitive, tolerant of one extra step, bounded. Tasks are weighted
equally regardless of repeat count. Cross-provider ATA **excludes
self-agreement**, so a repeatably-wrong provider earns no credit. CIs bootstrap
over *tasks*, because the claim is generalisation to new tasks.

**EQ 3 — Output Stability** · `src/metrics/output_stability.py`
Two axes, because a run can be unstable in two operationally different ways:
*outcome stability* (do repeats agree on pass/fail — Fleiss-style pairwise
probability) and *trajectory stability* (do repeats take the same path). Cells
with fewer than 2 repeats return `None` and are excluded, so thin cells cannot
inflate a score. Note consistent **failure** scores as stable — that is
deliberate, and is what makes the composite score necessary.

**EQ 4 — Composite Cross-Provider Trust Score (CPTS)** · `src/metrics/trust_score.py`
`0.40·TSR + 0.25·ATA_cross + 0.25·Stability + 0.10·Protocol`

The weights are a **value judgment about deployment priorities, not an
empirical finding**, and the paper must say so. Capability is weighted highest
because a provider that cannot do the task is unusable; consistency terms
together outweigh it, encoding the paper's position that for a team *swapping*
backends, predictability is worth more than raw capability. Protocol adherence
is smallest because it is partly captured downstream in TSR.

Because the weighting is a judgment, `sensitivity_analysis()` re-ranks
providers under 2,000 weight vectors drawn from the simplex. If the ranking
holds, the RQ4 conclusion does not depend on our choice.

---

## 4. Results as of this log

> ⚠️ **These numbers include 73 invalid runs (§6). The TSR and CPTS figures
> below understate groq and anthropic and will change after the re-run.**
> Reproduce with `bash scripts/generate_figures.sh`.

### RQ1 — Success rate varies by provider (large effect)

Two-way ANOVA on per-task success proportions, Type II SS:

| Effect | partial η² | p | magnitude |
|---|---|---|---|
| Provider | **0.4206** | 2.2 × 10⁻²⁶ | large |
| Task category | 0.0399 | 0.027 | small |
| Provider × Category | 0.0520 | 0.206 (n.s.) | small |

The interaction term **fits for the first time** now that categories A and D
are fully run. Pairwise (Wilcoxon signed-rank, Holm-corrected): 4 of 6
contrasts significant; every ollama contrast at p < 10⁻⁷.

### RQ2 — Trajectory agreement

| Provider | Self-agreement | Agreement with others |
|---|---|---|
| openai | 0.965 | 0.38 / 0.56 / 0.18 |
| ollama | **0.964** | **0.12 / 0.25 / 0.18** |
| groq | 0.786 | 0.40 / 0.25 / 0.56 |
| anthropic | 0.748 | 0.40 / 0.12 / 0.38 |

The headline: **ollama is the most self-consistent provider and the least
compatible one.** It reproduces itself almost perfectly while agreeing with
everyone else barely at all — highly repeatable, highly divergent.

### RQ4 — Composite trust score reorders the field

| Rank | Provider | CPTS | TSR | ATA | Stability | Protocol |
|---|---|---|---|---|---|---|
| 1 | openai | **0.752** | 0.795 | 0.376 | 0.964 | 0.990 |
| 2 | anthropic | 0.719 | **0.871** | 0.300 | 0.816 | 0.921 |
| 3 | groq | 0.672 | 0.719 | **0.405** | 0.824 | 0.772 |
| 4 | ollama | 0.408 | 0.160 | 0.187 | 0.964 | 0.562 |

**This is the RQ4 result.** A deployer choosing on raw success rate picks
anthropic; choosing on trust-weighted criteria picks openai, which wins on
stability (0.964 vs 0.816) and protocol adherence (0.990 vs 0.921). The
composite does not merely re-describe TSR — it reorders it. openai holds rank 1
under **89.3%** of 2,000 random weightings, so this is not an artefact of our
chosen weights.

### RQ5 — The backend effect is concentrated at the edge

| Model set | Provider partial η² | magnitude |
|---|---|---|
| All four providers | 0.4206 | large |
| **Excluding ollama (3B)** | **0.0339** | **small** |

Removing the single edge-viable model collapses the provider effect by 92%.
This is the paper's most interesting finding: it puts a **boundary condition on
MAESTRO's "architecture dominates backend" claim**. Among comparable
cloud-scale backends, the backend effect really is small — consistent with
MAESTRO. Introduce a genuinely edge-viable model and it becomes the dominant
source of variance.

### RQ6 — Provider effect is broadly uniform across categories

The interaction is not significant (p = 0.206). Spread is widest in error
recovery (range 0.845) and narrowest in tool orchestration (0.634), but the
differences are not statistically reliable.

### RQ3 — Capacity vs trust: **cannot be answered as designed**

Only 2 of 4 providers publish a parameter count (llama3.2 = 3B, gpt-oss = 120B);
gpt-4o-mini and claude-sonnet-5 do not. `capacity_vs_trust()` returns
`insufficient_data` rather than a coefficient from two points. An ordinal
fallback exists but is explicitly labelled assumption-dependent. **This is a
genuine limitation and belongs in Threats to Validity, not a fudged number.**

---

## 5. Protocol adherence — a trust dimension in its own right

Phantom tool calls (the model *describing* a call as prose instead of invoking
it) are counted per run:

| Provider | Runs | Runs with phantom calls | Repetition loops | Median latency |
|---|---|---|---|---|
| openai | 302 | 0 | 0 | 3.5 s |
| anthropic | 302 | 0 | 0 | 14.3 s |
| groq | 302 | 0 | 0 | 7.5 s |
| ollama | 306 | **134 (44%)** | 9 | 3.9 s |

The sharpest qualitative finding sits in category A. llama3.2 **can write
correct code** — inspected solutions were flawless — but roughly 80% of the
time emits it as markdown prose instead of through `write_file`. Every codegen
run that actually invoked the tool passed. **Capability is not the bottleneck;
protocol adherence is.**

---

## 6. ⚠️ Known-invalid data — must be re-run

Two **harness** bugs (now fixed) caused 73 runs to be scored as agent failures
when the benchmark itself broke. Both systematically penalised specific
providers.

| Bug | Runs | Cause | Fix |
|---|---|---|---|
| `UnicodeEncodeError` | 54 (groq) | File IO used no explicit encoding; Windows cp1252 cannot write `‑`, curly quotes, en-dashes | UTF-8 pinned in `tools.py`, `scoring.py`, `runner.py` |
| `TypeError: got 'list'` | 17 (anthropic) | Anthropic returns content as a list of typed blocks; the phantom detector regexed it directly | `normalize_content()` in `trajectory_logger.py` |
| `PermissionError` / `FileNotFoundError` | 2 (groq) | Windows file-lock / teardown race | quarantined, re-run |

**Estimated impact once re-run:**

| Provider | TSR now | TSR excluding invalid runs |
|---|---|---|
| groq | 0.719 | **0.875** |
| anthropic | 0.871 | **0.923** |
| groq — codegen only | 0.587 | **0.889** |

This changes the RQ1 ranking materially: groq moves from clearly-worst-cloud to
competitive. **Do not quote §4's TSR or CPTS numbers in the paper until this is
redone.** RQ5's headline (η² collapsing when ollama is removed) is driven by
ollama and is not expected to move.

### To fix

```bash
python scripts/quarantine_harness_failures.py      # moves 73 runs aside, never deletes
python -m src.harness.runner --full --resume       # re-runs exactly those cells
python -m src.analysis.figures                     # rebuilds every figure and table
```

---

## 7. Earlier audit: why the local model looked slow

An audit of ollama's 79.5 s mean latency found it was a tail artefact — the
**median was 4.3 s**, competitive with cloud. Runs making zero tool calls
averaged 294 s, worst case **1,797 s (30 minutes)**. Three bugs, all fixed:

1. **No generation cap anywhere.** A repetition loop ran until the context
   window filled — one run emitted 82,043 characters (`–` repeated ~13,600
   times). Added `DEFAULT_MAX_OUTPUT_TOKENS = 2048`, applied **identically to
   all four providers** so it stays a scaffold property, not a per-provider
   tweak. Verified: 1,797 s → 33.8 s (53×) and 918 s → 23.2 s (40×).

2. **Phantom detector false negative on its most important case.** The regex
   captured arguments non-greedily as `(\{.*?\})`, which breaks on any argument
   containing braces — i.e. most real `run_python` calls. Runs that emitted a
   tool call as prose and then looped were recorded as protocol-**adherent**,
   the opposite of the truth. Now brace-counts with string/escape awareness.

3. **No degeneracy detection.** Added, using **compression ratio** rather than
   character frequency: the worst case spread its mass over six characters so
   no single character exceeded 17%. Real-data separation is two orders of
   magnitude (degenerate 0.004–0.006, ordinary prose 0.63–0.67), so the 0.05
   threshold is not tuned.

Corrections were applied to existing runs by re-deriving the metric from raw
output already saved in `trajectory.json` — no re-run, no API cost, since the
runs were never wrong, only our reading of them.

---

## 8. Repository layout

```
config/config.yaml            providers, categories, seed, repeats
data/tasks/                   the 60-task suite, one folder per category
src/agent/                    fixed scaffold + tools (the experimental control)
src/providers/factory.py      the ONE place provider-specific code lives
src/harness/                  runner, scoring, trajectory logging, provenance, retry
src/metrics/                  TSR, ATA, Output Stability, Trust Score (pure functions)
src/analysis/                 loader, pre-registered stats, figure/table generation
paper/figures/                Fig 1–6, PDF + PNG, regenerated from runs
paper/tables/                 Table 1–5 + stats_summary.txt, CSV + LaTeX
experiments/runs/             per-run output (gitignored)
tests/                        73 offline tests
```

**Reproducibility:** every number in the paper is generated by
`bash scripts/generate_figures.sh` reading `experiments/runs/` alone. Nothing is
hand-copied. Each run folder carries its own seed, git commit, and package
versions.

**Environment note:** the harness needs the `agentic-trust-bench` conda env
(langchain). The metrics and analysis layers have no langchain dependency and
run under base python — which is why the offline test suite works anywhere.

---

## 9. Statistical method, and why

- **Unit of analysis is the (provider, task) cell proportion, not the raw run.**
  Individual runs are Bernoulli; a two-way ANOVA on 0/1 outcomes violates
  normality and homoscedasticity, and repeated runs of one task are not
  independent. A run-level binomial GEE clustered on task refits the same
  question as a robustness check.
- **Pairwise comparisons are paired on task** — every provider sees the same
  suite, so pairing removes between-task difficulty variance, the largest
  nuisance source. Wilcoxon signed-rank rather than paired *t*, because
  per-task proportions pile up at 0 and 1.
- **Holm correction**, uniformly more powerful than Bonferroni at the same
  family-wise error rate with no extra assumptions.
- **Effect sizes always reported beside p-values**, because RQ5 is a question
  about relative magnitude that p-values cannot answer.

---

## 10. Next steps

1. **Re-run the 73 quarantined cells** (§6) and regenerate figures. Blocking.
2. Update every number in this log and in the paper draft from the clean data.
3. Write Results (§5 of the skeleton) against the corrected figures.
4. Fill the Related Work citations, which are still placeholders.
5. Write the abstract last.

**Open decisions for the authors:**

- **RQ3 cannot be answered as specified** (§4). Either drop it, restate it
  qualitatively as an edge-viability discussion, or defend an ordinal capacity
  ranking as a stated assumption.
- The **task suite is unbalanced** (30 / 10 / 10 / 10). Type II SS handles it
  and it gives category A the most power, but it is worth a sentence in
  Experimental Setup.
- **The judge for category D is Claude, and Claude is also one of the four
  providers under test.** Held constant across conditions so any bias is
  constant rather than favouring one provider, but not eliminated. A second
  judge or a human-annotated subsample would close this.
