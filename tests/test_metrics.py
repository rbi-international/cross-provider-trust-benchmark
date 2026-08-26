"""
Validates the four metric definitions (EQ 1-4) against hand-constructed
cases whose correct answers are known by inspection, plus invariants that
must hold on the real run data.

These metrics produce every headline number in the paper, and unlike the
scoring engine their outputs are not obviously wrong when they are subtly
wrong - a mis-normalized ATA still returns a plausible-looking 0.7. So the
tests pin exact values wherever a value is analytically derivable.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.ata import (
    ata_matrix,
    bootstrap_ci,
    cross_provider_ata,
    levenshtein,
    tool_sequence,
    trajectory_similarity,
)
from src.metrics.output_stability import (
    outcome_stability,
    stability_by_provider,
    stability_for_cell,
    trajectory_stability,
)
from src.metrics.trust_score import (
    DEFAULT_WEIGHTS,
    compute_terms,
    protocol_adherence_rate,
    rank_providers,
    sensitivity_analysis,
    trust_score,
    trust_scores,
)
from src.metrics.tsr import tsr, tsr_by, tsr_with_ci, wilson_interval


def traj(*tool_names, args=None):
    """A trajectory that called the given tools in the given order."""
    return {
        "tool_calls": [
            {"tool_name": name, "args": (args or {}).get(name, {"f": name})}
            for name in tool_names
        ]
    }


def run(provider, task_id, passed, *tool_names, **extra):
    return {"provider": provider, "task_id": task_id, "passed": passed,
            "trajectory": traj(*tool_names), **extra}


# --------------------------------------------------------------------------
# EQ 1 - TSR
# --------------------------------------------------------------------------

def test_tsr_basic_and_empty():
    assert tsr([{"passed": True}, {"passed": False}]) == 0.5
    assert tsr([{"passed": True}] * 3) == 1.0
    assert tsr([]) == 0.0


def test_wilson_interval_stays_inside_unit_interval_at_boundaries():
    """The reason Wilson is used instead of the normal approximation: the
    grid has cells at exactly 0.0 and 1.0, where normal-approx CIs escape
    [0, 1] and collapse to zero width."""
    for n_success, n_total in [(0, 5), (5, 5), (0, 1), (1, 1), (3, 7)]:
        low, high = wilson_interval(n_success, n_total)
        assert 0.0 <= low <= high <= 1.0
        assert high > low, "a boundary cell must still have non-zero width"

    # no data means no information, not a confident zero
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_contains_point_estimate_and_narrows_with_n():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high

    narrow = wilson_interval(500, 1000)
    assert (narrow[1] - narrow[0]) < (high - low)


def test_tsr_by_slices_match_manual_counts():
    results = [
        {"passed": True, "provider": "a", "category": "codegen"},
        {"passed": False, "provider": "a", "category": "codegen"},
        {"passed": True, "provider": "a", "category": "ambiguous"},
        {"passed": True, "provider": "b", "category": "codegen"},
    ]
    by_provider = tsr_by(results, "provider")
    assert by_provider["a"]["tsr"] == pytest.approx(2 / 3)
    assert by_provider["b"]["tsr"] == 1.0
    assert by_provider["a"]["n_total"] == 3

    by_cell = tsr_by(results, "provider", "category")
    assert by_cell[("a", "codegen")]["tsr"] == 0.5
    assert by_cell[("a", "ambiguous")]["tsr"] == 1.0


# --------------------------------------------------------------------------
# EQ 2 - ATA
# --------------------------------------------------------------------------

def test_levenshtein_known_values():
    assert levenshtein([], []) == 0
    assert levenshtein(["a"], []) == 1
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert levenshtein(["a", "b"], ["a", "c"]) == 1
    assert levenshtein(["a", "b"], ["b", "a"]) == 2


def test_trajectory_similarity_exact_values():
    assert trajectory_similarity(traj("a", "b", "c"), traj("a", "b", "c")) == 1.0
    # one deletion out of a length-3 sequence
    assert trajectory_similarity(traj("a", "b", "c"), traj("a", "b")) == pytest.approx(2 / 3)
    # fully disjoint sequences of equal length
    assert trajectory_similarity(traj("a", "b"), traj("x", "y")) == 0.0
    # order matters: same multiset, reversed
    assert trajectory_similarity(traj("a", "b"), traj("b", "a")) == 0.0


def test_similarity_is_symmetric_and_bounded():
    a, b = traj("read_file", "calculator", "write_file"), traj("read_file", "write_file")
    assert trajectory_similarity(a, b) == trajectory_similarity(b, a)
    assert 0.0 <= trajectory_similarity(a, b) <= 1.0


def test_two_empty_trajectories_agree_but_empty_vs_nonempty_does_not():
    """Documented deliberate choice: agreeing that no tool was needed is
    agreement. Whether that was correct is TSR's concern, not ATA's."""
    assert trajectory_similarity(traj(), traj()) == 1.0
    assert trajectory_similarity(traj("a"), traj()) == 0.0


def test_include_args_is_stricter_than_names_alone():
    a = {"tool_calls": [{"tool_name": "write_file", "args": {"filename": "x.py"}}]}
    b = {"tool_calls": [{"tool_name": "write_file", "args": {"filename": "y.py"}}]}
    assert trajectory_similarity(a, b, include_args=False) == 1.0
    assert trajectory_similarity(a, b, include_args=True) == 0.0


def test_tool_sequence_handles_missing_and_null_fields():
    assert tool_sequence({}) == []
    assert tool_sequence({"tool_calls": None}) == []
    assert tool_sequence({"tool_calls": [{"tool_name": "a", "args": None}]}, include_args=True) \
        == [("a", ())]


def test_ata_matrix_is_symmetric_with_a_self_agreement_diagonal():
    runs = [
        run("p", "T1", True, "a", "b"),
        run("p", "T1", True, "a", "b"),
        run("q", "T1", True, "a", "c"),
        run("q", "T1", True, "a", "c"),
    ]
    matrix = ata_matrix(runs)
    assert matrix[("p", "p")]["ata"] == 1.0          # identical repeats
    assert matrix[("p", "q")]["ata"] == 0.5          # one of two steps differs
    assert matrix[("p", "q")]["ata"] == matrix[("q", "p")]["ata"]


def test_self_comparison_needs_two_runs():
    """A single run compared with itself is trivially 1.0; that must not be
    counted, or a one-run cell would look perfectly self-consistent."""
    matrix = ata_matrix([run("p", "T1", True, "a")])
    assert matrix[("p", "p")]["ata"] is None
    assert matrix[("p", "p")]["n_tasks"] == 0


def test_cross_provider_ata_excludes_self_agreement():
    """A provider that is perfectly self-consistent but divergent from
    everyone else must not earn credit on the ATA term."""
    runs = [
        run("p", "T1", True, "a", "b"),
        run("p", "T1", True, "a", "b"),
        run("q", "T1", True, "x", "y"),
        run("q", "T1", True, "x", "y"),
    ]
    assert ata_matrix(runs)[("p", "p")]["ata"] == 1.0
    assert cross_provider_ata(runs, "p") == 0.0


def test_ata_weights_tasks_equally_regardless_of_repeat_count():
    """Task T2 has 3x the repeats of T1; both must still count once."""
    runs = [run("p", "T1", True, "a"), run("q", "T1", True, "a")]           # sim 1.0
    runs += [run("p", "T2", True, "a") for _ in range(3)]                    # sim 0.0
    runs += [run("q", "T2", True, "z") for _ in range(3)]
    assert ata_matrix(runs)[("p", "q")]["ata"] == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_mean_and_is_seed_stable():
    scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    low, high = bootstrap_ci(scores, n_resamples=2000, seed=7)
    assert low <= sum(scores) / len(scores) <= high
    assert bootstrap_ci(scores, n_resamples=2000, seed=7) == (low, high)
    assert bootstrap_ci([]) == (None, None)


# --------------------------------------------------------------------------
# EQ 3 - Output Stability
# --------------------------------------------------------------------------

def test_outcome_stability_known_values():
    assert outcome_stability([{"passed": True}] * 5) == 1.0      # all pass
    assert outcome_stability([{"passed": False}] * 5) == 1.0     # all fail: also stable
    assert outcome_stability([{"passed": True}, {"passed": False}]) == 0.0
    # 3 pass / 2 fail out of 5 -> [3*2 + 2*1] / [5*4] = 8/20
    mixed = [{"passed": True}] * 3 + [{"passed": False}] * 2
    assert outcome_stability(mixed) == pytest.approx(0.4)


def test_stability_requires_at_least_two_runs():
    assert outcome_stability([{"passed": True}]) is None
    assert trajectory_stability([traj("a")]) is None
    assert stability_for_cell([run("p", "T1", True, "a")])["output_stability"] is None


def test_consistently_failing_provider_is_stable_but_not_successful():
    """The headline case the composite score exists to disambiguate: a small
    model that fails every time is maximally STABLE and minimally CAPABLE."""
    runs = [run("edge", "T1", False, "a", "b") for _ in range(5)]
    scored = stability_for_cell(runs)
    assert scored["outcome_stability"] == 1.0
    assert scored["trajectory_stability"] == 1.0
    assert tsr(runs) == 0.0


def test_stability_by_provider_averages_over_tasks_and_skips_thin_cells():
    runs = [run("p", "T1", True, "a"), run("p", "T1", False, "a")]  # outcome 0, traj 1
    runs += [run("p", "T2", True, "a"), run("p", "T2", True, "a")]  # outcome 1, traj 1
    runs += [run("p", "T3", True, "a")]                              # single run: skipped
    summary = stability_by_provider(runs)["p"]
    assert summary["n_tasks"] == 2
    assert summary["outcome_stability"] == pytest.approx(0.5)
    assert summary["trajectory_stability"] == 1.0
    assert summary["output_stability"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# EQ 4 - Composite Trust Score
# --------------------------------------------------------------------------

def test_protocol_adherence_defaults_to_true_for_pre_week4_runs():
    """Runs logged before phantom-call detection existed carry no field;
    treating them as adherent matches the logger's own default."""
    assert protocol_adherence_rate([{"passed": True}]) == 1.0
    assert protocol_adherence_rate([{"protocol_adherence": False}]) == 0.0
    assert protocol_adherence_rate([{"protocol_adherence": True},
                                    {"protocol_adherence": False}]) == 0.5
    assert protocol_adherence_rate([]) is None


def test_trust_score_is_a_plain_weighted_sum_when_all_terms_present():
    terms = {"tsr": 1.0, "ata": 0.0, "stability": 1.0, "protocol": 0.0}
    expected = DEFAULT_WEIGHTS["tsr"] + DEFAULT_WEIGHTS["stability"]
    assert trust_score(terms)["trust_score"] == pytest.approx(expected)


def test_trust_score_renormalizes_around_missing_terms():
    """A provider missing a dimension is scored on what it has, not
    penalized as though the missing dimension were zero."""
    terms = {"tsr": 1.0, "ata": None, "stability": 1.0, "protocol": 1.0}
    scored = trust_score(terms)
    assert scored["trust_score"] == pytest.approx(1.0)
    assert "ata" not in scored["terms_used"]
    assert sum(scored["weights_applied"].values()) == pytest.approx(1.0)

    assert trust_score({"tsr": None, "ata": None,
                        "stability": None, "protocol": None})["trust_score"] is None


def test_trust_score_is_bounded_in_unit_interval():
    assert trust_score({k: 0.0 for k in DEFAULT_WEIGHTS})["trust_score"] == 0.0
    assert trust_score({k: 1.0 for k in DEFAULT_WEIGHTS})["trust_score"] == pytest.approx(1.0)


def test_default_weights_sum_to_one():
    """The [0, 1] boundedness claim in the paper depends on this."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_stable_but_wrong_provider_ranks_below_a_capable_one():
    """End-to-end RQ4 behaviour: perfect stability must not rescue a
    provider that cannot do the task and diverges from everyone else."""
    runs = []
    for i in range(4):
        runs += [run("good", f"T{i}", True, "read_file", "write_file") for _ in range(3)]
        runs += [run("good2", f"T{i}", True, "read_file", "write_file") for _ in range(3)]
        runs += [run("stuck", f"T{i}", False, "calculator") for _ in range(3)]

    scored = trust_scores(runs)
    assert scored["stuck"]["stability"] == 1.0          # perfectly repeatable
    assert scored["stuck"]["tsr"] == 0.0
    ranking = [p for p, _ in rank_providers(scored)]
    assert ranking[-1] == "stuck"


def test_compute_terms_covers_every_requested_provider():
    runs = [run("p", "T1", True, "a"), run("p", "T1", True, "a")]
    terms = compute_terms(runs, providers=["p", "absent"])
    assert terms["absent"]["n_runs"] == 0
    assert terms["absent"]["tsr"] is None


def test_sensitivity_analysis_reports_a_normalized_rank_distribution():
    runs = []
    for i in range(3):
        runs += [run("strong", f"T{i}", True, "a", "b") for _ in range(3)]
        runs += [run("weak", f"T{i}", False, "z") for _ in range(3)]

    sensitivity = sensitivity_analysis(runs, n_samples=200, seed=1)
    assert set(sensitivity) == {"strong", "weak"}
    for provider, stats in sensitivity.items():
        assert sum(stats["rank_distribution"].values()) == pytest.approx(1.0)
        assert 0.0 <= stats["p_rank_1"] <= 1.0
    assert sensitivity["strong"]["p_rank_1"] > sensitivity["weak"]["p_rank_1"]
