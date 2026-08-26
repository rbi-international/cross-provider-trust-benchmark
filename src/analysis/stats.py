"""
The pre-registered statistical analysis, exactly as committed in Section 4
of the paper skeleton before any runs were scored:

  - two-way ANOVA (Provider x Task Category) on TSR            -> RQ1, RQ6
  - effect sizes (partial eta squared) reported beside p-values
  - Holm-corrected pairwise provider comparisons                -> RQ1
  - bootstrapped confidence intervals on ATA                    -> RQ2
  - model capacity vs Composite Trust Score correlation         -> RQ3
  - provider effect size as capacity shrinks                    -> RQ5

TWO MODELLING DECISIONS WORTH STATING UP FRONT, because a reviewer will
ask and the answers are not arbitrary:

1. The unit of analysis for the ANOVA is the (provider, task) cell success
   proportion, not the individual run. Individual runs are Bernoulli, and a
   two-way ANOVA on raw 0/1 outcomes violates normality and homoscedasticity
   badly enough to matter. Aggregating to per-task proportions gives a
   bounded, roughly continuous response and - more importantly - respects the
   fact that repeated runs of one task are not independent observations.
   fit_logistic_gee() below refits the same question as a run-level binomial
   GEE clustered on task, as a robustness check on that aggregation. When the
   two disagree, the paper reports both.

2. Pairwise provider comparisons are PAIRED. Every provider runs the same
   task suite, so pairing on task removes between-task difficulty variance,
   which is by far the largest nuisance source here. We use the Wilcoxon
   signed-rank test rather than a paired t-test because per-task success
   proportions pile up at 0.0 and 1.0 and are nowhere near normal.

Multiplicity is controlled with Holm, which is uniformly more powerful than
Bonferroni at the same family-wise error rate and needs no extra assumptions.
"""
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from src.metrics.ata import ata_matrix, bootstrap_ci


# --------------------------------------------------------------------------
# RQ1 / RQ6 - two-way ANOVA on TSR
# --------------------------------------------------------------------------

def cell_success_rates(frame):
    """Collapse runs to one success proportion per (provider, task, category).

    This is the ANOVA response variable. n_runs is carried through so thin
    cells can be inspected or filtered before modelling.
    """
    grouped = (
        frame.groupby(["provider", "category", "task_id"], as_index=False)
        .agg(success_rate=("success", "mean"), n_runs=("success", "size"))
    )
    return grouped


def partial_eta_squared(anova_table):
    """Partial eta^2 per effect: SS_effect / (SS_effect + SS_residual).

    Reported for every effect because p-values alone cannot answer RQ5,
    which is a question about the RELATIVE MAGNITUDE of the provider effect
    versus the task-category effect, not about whether either is nonzero.
    """
    if "Residual" not in anova_table.index:
        return {}
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    return {
        effect: float(row["sum_sq"] / (row["sum_sq"] + ss_residual))
        for effect, row in anova_table.iterrows()
        if effect != "Residual"
    }


def interpret_eta_squared(value):
    """Cohen's conventional benchmarks, for the prose in Section 5."""
    if value is None:
        return "undefined"
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "medium"
    return "large"


def two_way_anova(frame, response="success_rate"):
    """Provider x Task Category ANOVA on per-task success proportions.

    Type II sums of squares: the design is unbalanced (categories A and D
    have far fewer tasks scored than B and C), and Type II is the standard
    choice for unbalanced designs when the interaction is not assumed away.
    Returns the table, partial eta^2 per effect, and the fitted model.
    """
    cells = cell_success_rates(frame) if response == "success_rate" else frame

    n_providers = cells["provider"].nunique()
    n_categories = cells["category"].nunique()
    if n_providers < 2 or n_categories < 2:
        return {
            "insufficient_data": True,
            "reason": (f"need >=2 providers and >=2 categories for a two-way ANOVA, "
                       f"got {n_providers} and {n_categories}"),
            "n_cells": len(cells),
        }

    # An interaction term needs replication within every provider x category
    # combination; with only one task in a category the term is unidentifiable.
    combo_counts = cells.groupby(["provider", "category"]).size()
    can_fit_interaction = combo_counts.min() >= 2

    formula = (f"{response} ~ C(provider) * C(category)" if can_fit_interaction
               else f"{response} ~ C(provider) + C(category)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=cells).fit()
        table = anova_lm(model, typ=2)

    eta = partial_eta_squared(table)
    return {
        "insufficient_data": False,
        "formula": formula,
        "interaction_fitted": can_fit_interaction,
        "table": table,
        "partial_eta_sq": eta,
        "effect_magnitude": {k: interpret_eta_squared(v) for k, v in eta.items()},
        "model": model,
        "n_cells": len(cells),
        "r_squared": float(model.rsquared),
    }


def fit_logistic_gee(frame):
    """Run-level robustness check on the aggregated ANOVA.

    Binomial GEE with an exchangeable working correlation, clustered on
    task_id: models the raw 0/1 outcomes directly while accounting for the
    non-independence of repeated runs of the same task. If this agrees with
    two_way_anova() on which effects matter, the aggregation in the main
    analysis did not drive the conclusion.
    """
    usable = frame.dropna(subset=["provider", "category", "task_id"])
    if usable["provider"].nunique() < 2:
        return {"insufficient_data": True, "reason": "need >=2 providers"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.gee(
                "success ~ C(provider) + C(category)",
                groups="task_id",
                data=usable,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
    except Exception as e:
        return {"insufficient_data": True, "reason": f"GEE failed to converge: {type(e).__name__}: {e}"}

    return {
        "insufficient_data": False,
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "summary": str(model.summary()),
    }


# --------------------------------------------------------------------------
# RQ1 - pairwise provider comparisons
# --------------------------------------------------------------------------

def pairwise_provider_tests(frame, alpha=0.05, method="holm"):
    """Paired per-task comparisons between every provider pair.

    Pairs on task_id (all providers see the same suite), tests with Wilcoxon
    signed-rank, then corrects the family of comparisons with Holm. Also
    reports the median per-task difference and a rank-biserial effect size,
    because a p-value on its own does not tell a deployer how much success
    they would give up by swapping.
    """
    cells = cell_success_rates(frame)
    wide = cells.pivot_table(index="task_id", columns="provider", values="success_rate")
    providers = sorted(wide.columns)

    comparisons = []
    for provider_a, provider_b in combinations(providers, 2):
        paired = wide[[provider_a, provider_b]].dropna()
        differences = paired[provider_a] - paired[provider_b]
        n_pairs = len(paired)
        n_discordant = int((differences != 0).sum())

        if n_discordant == 0:
            p_value, statistic, effect = 1.0, None, 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = stats.wilcoxon(
                    paired[provider_a], paired[provider_b], zero_method="wilcox"
                )
            statistic = float(statistic)
            # rank-biserial correlation: 1 - 2W/(n(n+1)/2) on discordant pairs
            max_w = n_discordant * (n_discordant + 1) / 2
            effect = float(1 - 2 * statistic / max_w) if max_w else 0.0

        comparisons.append({
            "provider_a": provider_a,
            "provider_b": provider_b,
            "n_paired_tasks": n_pairs,
            "n_discordant": n_discordant,
            "mean_success_a": float(paired[provider_a].mean()) if n_pairs else None,
            "mean_success_b": float(paired[provider_b].mean()) if n_pairs else None,
            "median_difference": float(differences.median()) if n_pairs else None,
            "mean_difference": float(differences.mean()) if n_pairs else None,
            "statistic": statistic,
            "p_raw": float(p_value),
            "rank_biserial": effect,
        })

    if not comparisons:
        return pd.DataFrame()

    table = pd.DataFrame(comparisons)
    reject, p_adjusted, _, _ = multipletests(table["p_raw"], alpha=alpha, method=method)
    table["p_adjusted"] = p_adjusted
    table["significant"] = reject
    table["correction"] = method
    return table.sort_values("p_adjusted").reset_index(drop=True)


# --------------------------------------------------------------------------
# RQ2 - ATA confidence intervals
# --------------------------------------------------------------------------

def ata_with_cis(runs, providers=None, include_args=False,
                 n_resamples=10000, confidence=0.95, seed=42):
    """Every pairwise ATA with a task-level bootstrap CI (Table 3 / Fig 4).

    The diagonal (self-agreement across repeats) is kept and labelled, since
    it is the reference point that makes the off-diagonal numbers readable:
    cross-provider agreement is only interpretable relative to how much a
    provider agrees with itself.
    """
    matrix = ata_matrix(runs, providers, include_args)
    rows = []
    for (provider_a, provider_b), cell in matrix.items():
        low, high = bootstrap_ci(
            cell["per_task"].values(), n_resamples=n_resamples,
            confidence=confidence, seed=seed,
        )
        rows.append({
            "provider_a": provider_a,
            "provider_b": provider_b,
            "is_self": provider_a == provider_b,
            "ata": cell["ata"],
            "ci_low": low,
            "ci_high": high,
            "n_tasks": cell["n_tasks"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# RQ3 / RQ5 - capacity vs consistency
# --------------------------------------------------------------------------

def capacity_vs_trust(trust_by_provider, model_sizes, capacity_ranks=None):
    """Spearman correlation between model capacity and Composite Trust Score.

    Uses published parameter counts where they exist. Two of the four
    providers are closed models with no published count, which leaves n=2
    usable points - not enough for a correlation, and this function says so
    explicitly rather than returning a spurious coefficient from two points.

    capacity_ranks is the documented fallback: an explicit ordinal ranking
    of the providers by assumed capacity, supplied by the researcher rather
    than inferred here. It is an ASSUMPTION, and any result computed from it
    is labelled as rank-based and assumption-dependent in the paper.

    Spearman rather than Pearson throughout: parameter counts span two orders
    of magnitude and the hypothesized relationship is monotone, not linear.
    """
    out = {}

    paired = [
        (model_sizes[p], trust_by_provider[p])
        for p in trust_by_provider
        if model_sizes.get(p) is not None and trust_by_provider.get(p) is not None
    ]
    if len(paired) >= 3:
        sizes, scores = zip(*paired)
        rho, p_value = stats.spearmanr(sizes, scores)
        out["parameter_count"] = {
            "insufficient_data": False, "n": len(paired),
            "spearman_rho": float(rho), "p_value": float(p_value),
        }
    else:
        out["parameter_count"] = {
            "insufficient_data": True, "n": len(paired),
            "reason": ("fewer than 3 providers have a published parameter count; "
                       "closed cloud models are excluded rather than guessed"),
        }

    if capacity_ranks:
        ranked = [
            (capacity_ranks[p], trust_by_provider[p])
            for p in trust_by_provider
            if p in capacity_ranks and trust_by_provider.get(p) is not None
        ]
        if len(ranked) >= 3:
            ranks, scores = zip(*ranked)
            rho, p_value = stats.spearmanr(ranks, scores)
            out["ordinal_capacity"] = {
                "insufficient_data": False, "n": len(ranked),
                "spearman_rho": float(rho), "p_value": float(p_value),
                "caveat": "based on an assumed capacity ordering, not published sizes",
            }
        else:
            out["ordinal_capacity"] = {"insufficient_data": True, "n": len(ranked)}

    return out


def provider_effect_by_category(frame):
    """Per-category spread across providers - the RQ6 interaction, tabulated.

    Range and standard deviation of provider TSR within each task category.
    A provider effect concentrated in tool orchestration and error recovery,
    while small in plain code generation, is the shape RQ6 predicts.
    """
    cells = (
        frame.groupby(["category", "provider"], as_index=False)
        .agg(tsr=("success", "mean"), n_runs=("success", "size"))
    )
    rows = []
    for category, group in cells.groupby("category"):
        rows.append({
            "category": category,
            "n_providers": len(group),
            "tsr_min": float(group["tsr"].min()),
            "tsr_max": float(group["tsr"].max()),
            "tsr_range": float(group["tsr"].max() - group["tsr"].min()),
            "tsr_std": float(group["tsr"].std(ddof=1)) if len(group) > 1 else 0.0,
            "n_runs": int(group["n_runs"].sum()),
        })
    return pd.DataFrame(rows).sort_values("tsr_range", ascending=False).reset_index(drop=True)


def edge_sensitivity(frame, edge_provider="ollama"):
    """RQ5, stated as a testable contrast rather than a narrative claim.

    MAESTRO reports that agent architecture dominates backend choice. Our
    scaffold is held fixed, so we cannot measure architecture directly; what
    we CAN measure is whether the backend effect grows once a genuinely
    edge-viable model is in the pool. We refit the provider effect with and
    without the edge provider: if partial eta^2 for Provider collapses when
    the small model is removed, the backend effect is concentrated at the
    low-capacity end, which is precisely the boundary condition RQ5 puts on
    the architecture-dominance finding.
    """
    full = two_way_anova(frame)
    without = two_way_anova(frame[frame["provider"] != edge_provider])

    def provider_eta(result):
        if result.get("insufficient_data"):
            return None
        return result["partial_eta_sq"].get("C(provider)")

    eta_full, eta_without = provider_eta(full), provider_eta(without)
    return {
        "edge_provider": edge_provider,
        "provider_eta_sq_all": eta_full,
        "provider_eta_sq_excluding_edge": eta_without,
        "magnitude_all": interpret_eta_squared(eta_full),
        "magnitude_excluding_edge": interpret_eta_squared(eta_without),
        "attenuation": (float(eta_full - eta_without)
                        if eta_full is not None and eta_without is not None else None),
        "anova_all": full,
        "anova_excluding_edge": without,
    }
