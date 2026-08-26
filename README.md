# Cross-Provider Trust and Consistency Benchmark for Agentic AI

Research project for ISDIA 2027 (Special Session: Trustworthy AI and Intelligent Edge Systems).

**Authors:** Rohit Bharti, Ashish Arya
**Deadline:** 23rd September 2026

## What this is

We hold one agent scaffold fixed (same prompts, same tools) and swap only the
LLM provider underneath it (OpenAI, Groq, Ollama-local, watsonx), across four
task categories (code generation, tool orchestration, error recovery,
ambiguous instructions). We measure whether task success and behavior stay
consistent across providers, and combine that into a Cross-Provider Trust
Score meant for real deployment decisions, especially when a team considers
swapping to a smaller, edge-viable model.

Full research design lives in `paper/ISDIA2027_CrossProvider_Trust_Skeleton.md`.

## Setup

```bash
conda env create -f environment.yml
conda activate agentic-trust-bench
cp .env.example .env   # fill in your real API keys, never commit this file
```

## Project layout

```
config/             run configuration (providers, task categories, seed)
data/tasks/          the task suite, one folder per category
src/agent/            fixed agent scaffold and tool definitions
src/providers/        one adapter per LLM provider, common interface
src/harness/          orchestrates the full grid, trajectory + provenance logging
src/metrics/           TSR, ATA, Output Stability, Composite Trust Score
src/analysis/          stats (ANOVA, effect sizes) and figure generation
experiments/runs/      auto-created per-run output folders (gitignored)
notebooks/             exploratory analysis
tests/                 pytest suite, mirrors src/ structure
paper/                 the paper skeleton and eventual draft
scripts/               single-command entry points (pilot run, full run, figures)
```

## Reproducing a run

```bash
bash scripts/run_pilot.sh    # small batch first, catches bugs cheaply
bash scripts/run_full.sh     # full 4 provider x 4 category grid
bash scripts/generate_figures.sh   # regenerates every figure/table from experiments/runs/
```

Every run folder under `experiments/runs/` carries its own `config.yaml`
snapshot, seed, git commit hash, and package versions, so any number in the
paper traces back to an exact, reproducible run.

## Status

Weeks 1-5 complete. See `paper/ISDIA2027_CrossProvider_Trust_Skeleton.md` for
the full six-week roadmap.

- **Week 1** task suite, 4 categories
- **Week 2** agent scaffold, tools, provider adapters, trajectory + provenance logging
- **Week 3-4** scoring engine (unit test / state check / LLM judge), harness runner,
  protocol-adherence detection
- **Week 5** the four metrics (TSR, ATA, Output Stability, Composite Trust Score),
  the pre-registered statistics, and generation of every figure and table
- **Week 6** finish the run grid, then write the paper

`bash scripts/generate_figures.sh` rebuilds `paper/figures/` and `paper/tables/`
from `experiments/runs/` alone, so every number in the paper traces back to
real run folders and nothing is hand-copied.

### Known data gap

Categories B (tool orchestration) and C (error recovery) have the full 5
repeats per task per provider. Categories A (code generation) and D
(ambiguous instructions) currently have pilot runs only, which is why the
ANOVA cannot yet fit a Provider x Category interaction term and why RQ6 rests
on per-category spread for now. Finishing those two categories is the first
Week 6 task. The analysis reports this limitation itself rather than relying
on the reader to notice: thin cells are drawn as hollow markers in Fig 6 and
the omitted interaction term is stated in `paper/tables/stats_summary.txt`.
