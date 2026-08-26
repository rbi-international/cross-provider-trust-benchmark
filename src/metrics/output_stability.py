"""
EQ 3 - Output Stability.

TSR and ATA both compare providers to each other. Output Stability asks a
different question, entirely within one provider: run the SAME provider on
the SAME task N times, does it do the same thing? A provider that passes a
task 3 times out of 5 is not "60% good" from a deployment standpoint - it is
unpredictable, and a deployer needs that distinguished from a provider that
fails all 5 times consistently.

We measure it on two axes, because a run can be unstable in two different
ways and they have different operational consequences:

  Outcome stability   - do the repeated runs agree on pass/fail?
                        Computed as the probability that two runs drawn at
                        random (without replacement) from the k repeats
                        landed on the same verdict:

                            [ m(m-1) + (k-m)(k-m-1) ] / [ k(k-1) ]

                        where m is the number of passes. This is Fleiss-style
                        pairwise agreement, not variance: it is bounded in
                        [0, 1], reads directly as a probability, and is 1.0
                        for all-pass AND for all-fail. Consistently failing
                        IS stable - the composite score handles "stable but
                        wrong" by multiplying stability against TSR, rather
                        than by pretending failure is instability.

  Trajectory stability - do the repeated runs take the same tool-call path?
                        Mean pairwise ATA similarity across the k repeats,
                        i.e. the diagonal of the ATA matrix. A provider can
                        be perfectly outcome-stable while wandering through
                        a different tool sequence every time, which still
                        costs a deployer in latency, tokens, and auditability.

A task needs at least 2 repeats to have any stability at all; cells with
k < 2 return None and are excluded from aggregation rather than defaulted,
so a thin pilot cell cannot silently inflate a provider score.
"""
from collections import defaultdict

import numpy as np

from src.metrics.ata import _mean_pairwise_similarity


def outcome_stability(results):
    """Pairwise pass/fail agreement across repeated runs of one provider+task.

    Returns None for fewer than 2 runs (no pair exists).
    """
    k = len(results)
    if k < 2:
        return None
    m = sum(1 for r in results if r["passed"])
    return (m * (m - 1) + (k - m) * (k - m - 1)) / (k * (k - 1))


def trajectory_stability(trajectories, include_args=False):
    """Mean pairwise trajectory similarity across repeated runs.

    Returns None for fewer than 2 runs.
    """
    if len(trajectories) < 2:
        return None
    return _mean_pairwise_similarity(
        trajectories, trajectories, include_args, same_group=True
    )


def stability_for_cell(runs, include_args=False):
    """Both stability axes plus their mean, for one provider+task cell.

    runs: list of run dicts carrying passed and trajectory.
    """
    outcome = outcome_stability(runs)
    trajectory = trajectory_stability([r["trajectory"] for r in runs], include_args)

    parts = [v for v in (outcome, trajectory) if v is not None]
    return {
        "outcome_stability": outcome,
        "trajectory_stability": trajectory,
        "output_stability": float(np.mean(parts)) if parts else None,
        "n_runs": len(runs),
    }


def stability_by_provider(runs, include_args=False):
    """Output Stability per provider, averaged over tasks.

    Each provider+task cell is scored independently, then averaged across
    tasks - the same equal-weighting-per-task rule ATA uses, so a provider
    that happens to have more repeats on one task cannot dominate its own
    aggregate.
    """
    cells = defaultdict(list)
    for run in runs:
        cells[(run["provider"], run["task_id"])].append(run)

    per_provider = defaultdict(list)
    for (provider, task_id), cell_runs in cells.items():
        scored = stability_for_cell(cell_runs, include_args)
        if scored["output_stability"] is not None:
            per_provider[provider].append((task_id, scored))

    summary = {}
    for provider, scored_cells in per_provider.items():
        summary[provider] = {
            "output_stability": float(np.mean([s["output_stability"] for _, s in scored_cells])),
            "outcome_stability": float(np.mean(
                [s["outcome_stability"] for _, s in scored_cells if s["outcome_stability"] is not None]
            )),
            "trajectory_stability": float(np.mean(
                [s["trajectory_stability"] for _, s in scored_cells if s["trajectory_stability"] is not None]
            )),
            "n_tasks": len(scored_cells),
            "per_task": {task_id: s["output_stability"] for task_id, s in scored_cells},
        }
    return summary
