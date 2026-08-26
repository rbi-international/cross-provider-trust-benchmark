"""
Covers the analysis layer: the run loader, the pre-registered statistics, and
the figure/table generator.

The statistics tests do NOT re-verify what statsmodels and scipy already
guarantee. They check the things this project is responsible for and that
would silently corrupt the paper if wrong: that the unit of analysis is the
per-task cell rather than the raw run, that partial eta squared is computed
against the right residual, that Holm correction is actually applied, that a
degenerate grid degrades to a stated "insufficient data" rather than a
confident-looking number, and that the figure pipeline survives the sparse
and unbalanced data it will really be handed.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.load import coverage_report, load_runs, runs_to_frame
from src.analysis.stats import (
    ata_with_cis,
    capacity_vs_trust,
    cell_success_rates,
    edge_sensitivity,
    interpret_eta_squared,
    pairwise_provider_tests,
    partial_eta_squared,
    provider_effect_by_category,
    two_way_anova,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def write_run(root, run_id, result, trajectory="default"):
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f)
    if trajectory == "default":
        trajectory = {"tool_calls": [{"tool_name": "read_file", "args": {}}],
                      "final_text": "done", "protocol_adherence": True}
    if trajectory is not None:
        with open(os.path.join(run_dir, "trajectory.json"), "w") as f:
            json.dump(trajectory, f)
    return run_dir


@pytest.fixture
def runs_dir():
    root = tempfile.mkdtemp(prefix="runs_")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def synthetic_frame(providers=("p1", "p2", "p3"), categories=("catA", "catB"),
                    tasks_per_category=4, repeats=3, seed=0):
    """A balanced grid with a real provider effect built in, so the ANOVA has
    a known-nonzero signal to find."""
    rng = np.random.default_rng(seed)
    provider_skill = {p: 0.9 - 0.3 * i for i, p in enumerate(providers)}
    rows = []
    for category in categories:
        for task_index in range(tasks_per_category):
            task_id = f"{category}-{task_index}"
            for provider in providers:
                for repeat in range(repeats):
                    rows.append({
                        "run_id": f"{provider}-{task_id}-{repeat}",
                        "provider": provider, "category": category, "task_id": task_id,
                        "success": int(rng.random() < provider_skill[provider]),
                        "model": f"model-{provider}", "difficulty": "easy",
                        "n_tool_calls": 2, "protocol_adherence": True,
                    })
    frame = pd.DataFrame(rows)
    frame["passed"] = frame["success"].astype(bool)
    return frame


# --------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------

def test_load_runs_joins_result_and_trajectory(runs_dir):
    write_run(runs_dir, "r1", {"run_id": "r1", "provider": "openai", "task_id": "T1",
                               "category": "codegen", "passed": True, "model": "gpt-4o-mini"})
    runs, skipped = load_runs(runs_dir)
    assert len(runs) == 1 and not skipped
    assert runs[0]["trajectory"]["tool_calls"][0]["tool_name"] == "read_file"
    assert runs[0]["run_dir"].endswith("r1")


def test_loader_reports_broken_folders_instead_of_dropping_them(runs_dir):
    """A run that failed to write its own result is a data-integrity event.
    It must be surfaced, not quietly absorbed into a smaller denominator."""
    write_run(runs_dir, "good", {"run_id": "good", "provider": "p", "task_id": "T1",
                                 "category": "c", "passed": True})
    broken = os.path.join(runs_dir, "broken")
    os.makedirs(broken)
    with open(os.path.join(broken, "result.json"), "w") as f:
        f.write("{not valid json")

    no_traj = os.path.join(runs_dir, "no_traj")
    os.makedirs(no_traj)
    with open(os.path.join(no_traj, "result.json"), "w") as f:
        json.dump({"run_id": "no_traj", "provider": "p", "task_id": "T2",
                   "category": "c", "passed": False}, f)

    runs, skipped = load_runs(runs_dir)
    assert len(runs) == 1
    assert len(skipped) == 2
    assert any("unreadable" in reason for _, reason in skipped)
    assert any("no trajectory" in reason for _, reason in skipped)

    # ...unless the caller explicitly opts into trajectory-less runs
    runs, skipped = load_runs(runs_dir, require_trajectory=False)
    assert len(runs) == 2


def test_model_size_is_attached_and_unknown_models_stay_none(runs_dir):
    write_run(runs_dir, "small", {"run_id": "small", "provider": "ollama", "task_id": "T1",
                                  "category": "c", "passed": True, "model": "llama3.2"})
    write_run(runs_dir, "closed", {"run_id": "closed", "provider": "openai", "task_id": "T1",
                                   "category": "c", "passed": True, "model": "gpt-4o-mini"})
    runs, _ = load_runs(runs_dir)
    sizes = {r["provider"]: r["model_size_b"] for r in runs}
    assert sizes["ollama"] == 3.0
    assert sizes["openai"] is None, "closed models must stay None, never a guessed size"


def test_frame_defaults_protocol_adherence_for_pre_week4_runs(runs_dir):
    write_run(runs_dir, "old", {"run_id": "old", "provider": "p", "task_id": "T1",
                                "category": "c", "passed": True},
              trajectory={"tool_calls": [], "final_text": "x"})
    frame = runs_to_frame(load_runs(runs_dir)[0])
    assert bool(frame.loc[0, "protocol_adherence"]) is True
    assert frame.loc[0, "n_tool_calls"] == 0
    assert frame.loc[0, "success"] == 1


def test_coverage_report_surfaces_the_shape_of_the_grid():
    frame = synthetic_frame()
    report = coverage_report(frame)
    assert report["n_runs"] == len(frame)
    assert report["providers"] == ["p1", "p2", "p3"]
    assert report["repeat_counts"] == {3: 24}   # 8 tasks x 3 providers, 3 repeats each


# --------------------------------------------------------------------------
# ANOVA and effect sizes
# --------------------------------------------------------------------------

def test_cell_success_rates_aggregates_runs_to_one_row_per_provider_task():
    """The documented unit of analysis. If this ever silently reverts to
    per-run rows, every p-value in the paper becomes anticonservative."""
    frame = synthetic_frame(repeats=5)
    cells = cell_success_rates(frame)
    assert len(cells) == frame.groupby(["provider", "task_id"]).ngroups
    assert cells["n_runs"].eq(5).all()
    assert cells["success_rate"].between(0, 1).all()


def test_partial_eta_squared_matches_its_definition():
    table = pd.DataFrame(
        {"sum_sq": [4.0, 1.0, 5.0]},
        index=["C(provider)", "C(category)", "Residual"],
    )
    eta = partial_eta_squared(table)
    assert eta["C(provider)"] == pytest.approx(4 / 9)
    assert eta["C(category)"] == pytest.approx(1 / 6)
    assert "Residual" not in eta


def test_interpret_eta_squared_uses_cohen_benchmarks():
    assert interpret_eta_squared(0.005) == "negligible"
    assert interpret_eta_squared(0.03) == "small"
    assert interpret_eta_squared(0.10) == "medium"
    assert interpret_eta_squared(0.40) == "large"
    assert interpret_eta_squared(None) == "undefined"


def test_two_way_anova_finds_a_planted_provider_effect():
    result = two_way_anova(synthetic_frame(tasks_per_category=8, repeats=5))
    assert not result["insufficient_data"]
    assert result["table"].loc["C(provider)", "PR(>F)"] < 0.05
    assert result["partial_eta_sq"]["C(provider)"] > 0.1


def test_anova_drops_the_interaction_term_when_it_is_unidentifiable():
    """With one task in a category the interaction has no replication. The
    model must fall back to main effects and SAY so, rather than failing or
    reporting a fabricated interaction - this is the current pilot's shape."""
    frame = synthetic_frame(categories=("catA",), tasks_per_category=6)
    thin = synthetic_frame(categories=("catB",), tasks_per_category=1)
    combined = pd.concat([frame, thin], ignore_index=True)

    result = two_way_anova(combined)
    assert not result["insufficient_data"]
    assert result["interaction_fitted"] is False
    assert "*" not in result["formula"]


def test_anova_degrades_explicitly_on_a_single_provider():
    frame = synthetic_frame(providers=("only",))
    result = two_way_anova(frame)
    assert result["insufficient_data"] is True
    assert "provider" in result["reason"]


# --------------------------------------------------------------------------
# pairwise comparisons
# --------------------------------------------------------------------------

def test_pairwise_tests_are_holm_corrected_and_never_less_than_raw():
    table = pairwise_provider_tests(synthetic_frame(tasks_per_category=8, repeats=5))
    assert len(table) == 3                      # 3 providers -> 3 pairs
    assert (table["p_adjusted"] >= table["p_raw"] - 1e-12).all()
    assert (table["correction"] == "holm").all()
    assert table["n_paired_tasks"].gt(0).all()


def test_pairwise_pairs_on_task_so_only_shared_tasks_are_compared():
    frame = synthetic_frame(providers=("p1", "p2"), tasks_per_category=4)
    # p2 never attempts one task; that task must drop out of the pairing
    frame = frame[~((frame["provider"] == "p2") & (frame["task_id"] == "catA-0"))]
    table = pairwise_provider_tests(frame)
    assert table.iloc[0]["n_paired_tasks"] == 7   # 8 tasks, one unshared


def test_identical_providers_are_not_reported_as_different():
    """Two providers with identical per-task results must not produce a
    significant difference - the zero-discordant-pairs guard."""
    base = synthetic_frame(providers=("p1",), tasks_per_category=5)
    clone = base.copy()
    clone["provider"] = "p2"
    clone["run_id"] = clone["run_id"] + "-clone"

    table = pairwise_provider_tests(pd.concat([base, clone], ignore_index=True))
    assert table.iloc[0]["p_raw"] == 1.0
    assert not table.iloc[0]["significant"]
    assert table.iloc[0]["median_difference"] == 0.0


# --------------------------------------------------------------------------
# RQ2 / RQ3 / RQ5 / RQ6
# --------------------------------------------------------------------------

def test_ata_with_cis_is_symmetric_and_flags_the_diagonal():
    runs = []
    for task in range(4):
        for repeat in range(3):
            runs.append({"task_id": f"T{task}", "provider": "p1", "passed": True,
                         "trajectory": {"tool_calls": [{"tool_name": "a", "args": {}}]}})
            runs.append({"task_id": f"T{task}", "provider": "p2", "passed": True,
                         "trajectory": {"tool_calls": [{"tool_name": "b", "args": {}}]}})

    table = ata_with_cis(runs, n_resamples=500)
    assert table["is_self"].sum() == 2
    diagonal = table[table["is_self"]]
    assert (diagonal["ata"] == 1.0).all()

    off = table[~table["is_self"]]
    assert (off["ata"] == 0.0).all()
    assert (off["ci_low"] <= off["ata"]).all() and (off["ata"] <= off["ci_high"]).all()


def test_capacity_correlation_refuses_to_report_on_too_few_points():
    """The real dataset has exactly two providers with published parameter
    counts. Returning a coefficient from two points would be indefensible."""
    result = capacity_vs_trust(
        {"a": 0.9, "b": 0.4, "c": 0.7},
        {"a": 120.0, "b": 3.0, "c": None},
    )
    assert result["parameter_count"]["insufficient_data"] is True
    assert result["parameter_count"]["n"] == 2


def test_capacity_correlation_computes_when_enough_sizes_are_known():
    result = capacity_vs_trust(
        {"a": 0.9, "b": 0.4, "c": 0.7, "d": 0.8},
        {"a": 120.0, "b": 3.0, "c": 30.0, "d": 70.0},
    )
    assert result["parameter_count"]["insufficient_data"] is False
    assert result["parameter_count"]["spearman_rho"] == pytest.approx(1.0)


def test_ordinal_fallback_is_labelled_as_assumption_dependent():
    result = capacity_vs_trust(
        {"a": 0.9, "b": 0.4, "c": 0.7},
        {"a": None, "b": None, "c": None},
        capacity_ranks={"a": 3, "b": 1, "c": 2},
    )
    assert result["parameter_count"]["insufficient_data"] is True
    assert result["ordinal_capacity"]["insufficient_data"] is False
    assert "assumed" in result["ordinal_capacity"]["caveat"]


def test_edge_sensitivity_detects_an_effect_carried_by_the_small_model():
    """RQ5 in miniature: three providers that agree with each other plus one
    that is far worse. Removing the outlier must collapse the provider
    effect, which is exactly the contrast the paper reports."""
    rng = np.random.default_rng(11)
    rows = []
    for category in ("catA", "catB"):
        for task_index in range(8):
            task_id = f"{category}-{task_index}"
            # difficulty varies by task, giving the model real residual
            # variance to work against, but is shared by every provider
            task_difficulty = rng.uniform(0.7, 1.0)
            for provider in ("p1", "p2", "p3", "edge"):
                rate = 0.05 if provider == "edge" else task_difficulty
                for repeat in range(5):
                    rows.append({
                        "run_id": f"{provider}-{task_id}-{repeat}",
                        "provider": provider, "category": category, "task_id": task_id,
                        "success": int(rng.random() < rate),
                    })
    frame = pd.DataFrame(rows)
    frame["passed"] = frame["success"].astype(bool)

    result = edge_sensitivity(frame, edge_provider="edge")
    assert result["provider_eta_sq_all"] > 0.5, "the edge model should dominate the provider effect"
    # The claim is relative, not absolute: partial eta^2 is upward-biased at
    # this sample size, so a small residual effect among the peers is
    # expected noise. What must hold is that removing the edge model
    # collapses the effect rather than merely trimming it.
    assert result["provider_eta_sq_excluding_edge"] < result["provider_eta_sq_all"] / 3
    assert result["attenuation"] > 0.5
    assert result["magnitude_all"] == "large"


def test_provider_effect_by_category_orders_by_spread():
    frame = synthetic_frame(categories=("wide", "narrow"), tasks_per_category=5, repeats=4)
    frame.loc[frame["category"] == "narrow", "success"] = 1      # no spread at all
    table = provider_effect_by_category(frame)
    assert list(table["category"])[0] == "wide"
    assert table.loc[table["category"] == "narrow", "tsr_range"].iloc[0] == 0.0


# --------------------------------------------------------------------------
# figure pipeline
# --------------------------------------------------------------------------

def test_figure_pipeline_runs_end_to_end_on_a_sparse_unbalanced_grid(runs_dir, tmp_path, monkeypatch):
    """The generator will be handed exactly this: some cells with five
    repeats, some with one, one provider that makes no tool calls at all. It
    must produce every figure and table without crashing."""
    from src.analysis import figures

    monkeypatch.setattr(figures, "FIGURES_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(figures, "TABLES_DIR", str(tmp_path / "tables"))

    index = 0
    for provider, model in [("openai", "gpt-4o-mini"), ("ollama", "llama3.2"),
                            ("groq", "openai/gpt-oss-120b")]:
        for category, n_tasks, repeats in [("codegen", 3, 5), ("ambiguous", 1, 1)]:
            for task in range(n_tasks):
                for repeat in range(repeats):
                    index += 1
                    trajectory = ({"tool_calls": [], "final_text": "x"}
                                  if provider == "ollama"
                                  else {"tool_calls": [{"tool_name": "write_file", "args": {}}],
                                        "final_text": "x"})
                    write_run(runs_dir, f"run{index}", {
                        "run_id": f"run{index}", "provider": provider, "model": model,
                        "task_id": f"{category}-{task}", "category": category,
                        "difficulty": "easy", "passed": (repeat % 2 == 0),
                        "protocol_adherence": True,
                    }, trajectory=trajectory)

    figures.apply_style()
    runs, skipped = figures.load_runs(runs_dir)
    assert not skipped
    frame = figures.runs_to_frame(runs)

    for number in sorted(figures.ALL_FIGURES):
        figures.ALL_FIGURES[number][1](runs, frame)
    figures.table_1_task_suite(frame)
    figures.write_stats_summary(runs, frame)

    written = os.listdir(tmp_path / "figures")
    for number in range(1, 7):
        assert any(name.startswith(f"fig{number}") and name.endswith(".pdf") for name in written), \
            f"figure {number} was not written"
    assert os.path.exists(tmp_path / "tables" / "stats_summary.txt")


def test_every_provider_has_a_distinct_fixed_color():
    """Color follows the entity, never its rank: no two providers may share
    a hue, or the heatmap and interaction plot become unreadable."""
    from src.analysis.figures import PROVIDER_COLORS
    assert len(set(PROVIDER_COLORS.values())) == len(PROVIDER_COLORS)
    assert all(c.startswith("#") and len(c) == 7 for c in PROVIDER_COLORS.values())
