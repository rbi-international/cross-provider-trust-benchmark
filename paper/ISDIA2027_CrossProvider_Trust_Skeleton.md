# A Cross-Provider Trust and Consistency Benchmark for Agentic AI Systems

**Authors:** Rohit Bharti (Lovely Professional University, Punjab, India), Ashish Arya (affiliation, USA)
**Target venue:** ISDIA 2027, Special Session on Trustworthy AI and Intelligent Edge Systems
**Submission deadline:** 23rd September 2026

---

## Abstract
[PLACEHOLDER: 150-200 words. One line problem, one line gap, one line method, one line headline result, one line implication for edge deployment. Write this last, after Results.]

**Keywords:** Trustworthy AI, Agentic AI, LLM provider consistency, Edge AI deployment, Reliability benchmarking

---

## 1. Introduction
- Motivate with a concrete deployment scenario: a team swaps the LLM behind an agent for cost or latency reasons, does the agent still behave the same way
- State the gap: existing agent benchmarks measure task success against one model, not stability across models
- State contribution list (3 bullets, filled once results are known)
- [FIG 1: One-paragraph teaser figure showing the same agentic task solved differently by two providers, illustrating divergence]

## 2. Related Work
- Agentic AI / ReAct-style tool-use systems
- Agent consistency as a statistical property: prior work has framed consistency formally (U-statistics decomposition into observable/internal dimensions) and shown agents can be erratic under tool-description changes or adversarial conditions [PLACEHOLDER: cite]
- Reliability surfaces across models and architectures (ReliabilityBench: consistency, robustness, fault tolerance combined; found real cost-reliability tradeoffs between models) [PLACEHOLDER: cite]
- Architecture vs backend as the dominant driver of variance (MAESTRO: found agent architecture usually dominates over backend model choice, though systems can be structurally stable yet temporally variable) [PLACEHOLDER: cite]
- Model-agnostic evaluation harnesses (BCAS: fixed scaffold across six LLM families to isolate backend sensitivity) [PLACEHOLDER: cite]
- Edge AI deployment constraints: model size, latency, on-device feasibility [PLACEHOLDER: cite]
- Trustworthy AI and reliability metrics for AI systems generally [PLACEHOLDER: cite]
- **Explicit gap statement:** prior consistency/reliability benchmarks compare models or architectures in the abstract; none specifically frame the result as a practitioner-facing decision signal for swapping to an edge-viable backend, and none test whether the "architecture dominates backend" finding (MAESTRO) still holds once one candidate backend sits near the resource floor for edge deployment.

## 3. Proposed Cross-Provider Trust Benchmark
### 3.0 Research Questions
- **RQ1 (success rate variance):** Does task success rate differ significantly across providers for the same agent scaffold?
- **RQ2 (trajectory consistency):** Do agents solving the same task under different providers take similar tool-call paths, or diverge?
- **RQ3 (edge-viability tradeoff):** Is there a relationship between model size (proxy for edge deployability) and behavioral consistency?
- **RQ4 (composite trust score):** Can TSR, ATA, and Output Stability combine into one Cross-Provider Trust Score usable in a real deployment decision?
- **RQ5 (architecture vs backend, edge framing):** Does agent architecture remain the dominant source of variance (per MAESTRO's finding) once one candidate backend is genuinely edge-viable, or does backend effect grow as capacity shrinks?
- **RQ6 (task-dependence):** Is the provider effect uniform across task categories, or concentrated in specific ones (e.g. multi-step tool orchestration, error recovery) while staying small for plain code generation?

### 3.1 Architecture
- [FIG 2: Architecture diagram — agent scaffold (prompts + tools) held constant, provider swapped underneath, task suite run identically across all four providers × four task categories]
### 3.2 Metric Definitions
- [EQ 1: Task Success Rate (TSR) definition]
- [EQ 2: Action Trajectory Agreement (ATA) definition — similarity measure between tool-call sequences]
- [EQ 3: Output Stability definition — variance across repeated runs, same provider]
- [EQ 4: Composite Cross-Provider Trust Score — weighted combination of TSR, ATA, Output Stability]

## 4. Experimental Setup
- **Design:** two-factor, Provider (4 levels) × Task Category (4 levels), repeated runs per cell for variance estimation
- **Providers:** OpenAI/GPT-class (cloud reference), Groq-hosted Llama (fast cloud), Ollama local small model (edge proxy), watsonx (enterprise cloud) — spans large-cloud to edge-viable
- **Task categories:**
  - A. Pure code generation (MBPP-style subset, expanded from your validated 427-problem set)
  - B. Multi-step tool orchestration (3+ sequential tool calls per task)
  - C. Error recovery (task starts broken, agent must detect and self-correct)
  - D. Ambiguous instruction resolution (underspecified task, agent must infer or ask correctly)
- Task suite size: [PLACEHOLDER: target ~25-30 tasks per category = ~100-120 total, scoped down from full MBPP for cost/time]
- Runs per task per provider: [PLACEHOLDER: e.g. 5, for variance estimation — total run count ≈ tasks × providers × repeats]
- [TABLE 1: Task suite composition — category, count, difficulty, example task]
- Provenance: config.yaml, seed, git commit hash, package versions logged per run (standard practice, applied here too)
- **Pre-registered analysis plan** (fixed before running): two-way ANOVA/mixed-effects model (Provider × Task Category) on TSR; effect sizes (partial η²) alongside p-values; Holm or Bonferroni correction across pairwise provider comparisons; bootstrapped confidence intervals on ATA (non-normal); correlation test for model size vs Composite Trust Score (RQ3)

## 5. Results and Discussion
### 5.1 RQ1 — Success rate variance across providers
- [TABLE 2: TSR per provider, with confidence intervals]
- [FIG 3: Bar chart, TSR by provider]
### 5.2 RQ2 — Trajectory consistency
- [TABLE 3: ATA scores, pairwise between providers]
- [FIG 4: Heatmap of pairwise trajectory agreement]
### 5.3 RQ3 — Model size vs consistency
- [FIG 5: Scatter plot, model size (proxy) vs Composite Trust Score]
- [PLACEHOLDER: correlation coefficient and significance]
### 5.4 RQ4 — Composite Trust Score utility
- [TABLE 4: Final Trust Score ranking across providers]
- [PLACEHOLDER: discussion of what a deployer would do differently knowing this score]
### 5.5 RQ5 — Architecture vs backend at the edge
- [PLACEHOLDER: does provider effect size grow as model capacity shrinks; direct comparison against MAESTRO's architecture-dominance finding]
### 5.6 RQ6 — Task-category dependence
- [TABLE 5: TSR and ATA broken down by task category × provider]
- [FIG 6: Interaction plot, provider effect magnitude across the four task categories]

## 6. Threats to Validity
- Internal validity: [PLACEHOLDER: e.g. prompt sensitivity, stochastic sampling controlled via fixed seeds where provider APIs allow]
- External validity: [PLACEHOLDER: coding-task-only scope, edge simulation as future work, generalization limits]
- Construct validity: [PLACEHOLDER: does ATA actually capture meaningful "trust," discuss]

## 7. Conclusion and Future Work
- Restate contribution and headline result
- Future work: extend to simulated edge/IoT device control tasks (stretch goal noted, becomes its own future paper if not finished in time)

## References
[PLACEHOLDER: to be filled — agentic AI benchmarks, trustworthy AI surveys, edge AI deployment papers, LLM provider comparison studies]
