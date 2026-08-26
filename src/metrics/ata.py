"""
EQ 2 - Action Trajectory Agreement (ATA).

RQ2 asks whether two providers running the SAME task under the SAME scaffold
take similar tool-call paths. TSR cannot answer that: two providers can both
pass a task while getting there by visibly different routes, and a deployer
swapping backends cares about that difference (it changes cost, latency, and
what the audit log looks like).

ATA compares tool-call sequences. Given trajectories A and B, we take the
ordered list of tool names each one invoked and compute a normalized
edit-distance similarity:

    sim(A, B) = 1 - levenshtein(seq_A, seq_B) / max(|seq_A|, |seq_B|)

Levenshtein over tool NAMES (not raw text) is the right granularity: it is
order-sensitive (calling read_file then write_file is not the same plan as
the reverse), it tolerates one extra or missing step without collapsing to
zero, and it is bounded in [0, 1] so it composes into the trust score.

Two trajectories that both made zero tool calls score 1.0. That is deliberate
and it is a real agreement: both providers concluded the task needed no tools.
Whether that conclusion was CORRECT is the job of TSR, not ATA - the metrics
are meant to be orthogonal, and conflating them would double-count success.

A stricter include_args variant additionally requires matching arguments
before two steps count as equal. It is reported alongside the name-only
figure because the gap between them is itself informative: providers that
agree on the plan but disagree on the arguments are a different failure mode
from providers that disagree on the plan.

Aggregation, for the pairwise provider heatmap (Fig 4 / Table 3):

    ATA(p, q) = mean over tasks t of [ mean over run pairs (i, j) of
                sim(traj(p, t, i), traj(q, t, j)) ]

Averaging within a task BEFORE averaging across tasks keeps tasks equally
weighted regardless of how many repeats each cell happens to have, which
matters because repeat counts are not perfectly balanced across the grid.
"""
from collections import defaultdict
from itertools import product

import numpy as np


def tool_sequence(trajectory, include_args=False):
    """Ordered list of steps taken. Each step is a tool name, or a
    (name, canonical-args) pair when include_args is set."""
    steps = []
    for call in trajectory.get("tool_calls", []) or []:
        name = call.get("tool_name")
        if include_args:
            args = call.get("args") or {}
            canonical = tuple(sorted((str(k), str(v)) for k, v in args.items()))
            steps.append((name, canonical))
        else:
            steps.append(name)
    return steps


def levenshtein(seq_a, seq_b):
    """Standard edit distance over two sequences of hashable items."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    previous = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        current = [i]
        for j, b in enumerate(seq_b, start=1):
            current.append(min(
                previous[j] + 1,              # deletion
                current[j - 1] + 1,           # insertion
                previous[j - 1] + (a != b),   # substitution
            ))
        previous = current
    return previous[-1]


def trajectory_similarity(traj_a, traj_b, include_args=False):
    """Normalized edit-distance similarity in [0, 1] between two trajectories."""
    seq_a = tool_sequence(traj_a, include_args)
    seq_b = tool_sequence(traj_b, include_args)

    if not seq_a and not seq_b:
        return 1.0  # both used no tools: agreement, see module docstring
    return 1.0 - levenshtein(seq_a, seq_b) / max(len(seq_a), len(seq_b))


def _mean_pairwise_similarity(trajs_a, trajs_b, include_args, same_group):
    """Mean similarity over pairs drawn from two sets of trajectories.

    When same_group is True the two sets are the same runs, so we take
    unordered pairs i < j and skip the diagonal - a run compared with
    itself is trivially 1.0 and would inflate the score.
    """
    if same_group:
        pairs = [(trajs_a[i], trajs_a[j])
                 for i in range(len(trajs_a)) for j in range(i + 1, len(trajs_a))]
    else:
        pairs = list(product(trajs_a, trajs_b))

    if not pairs:
        return None
    return float(np.mean([trajectory_similarity(a, b, include_args) for a, b in pairs]))


def ata_for_task(trajectories_by_provider, provider_a, provider_b, include_args=False):
    """ATA between two providers on ONE task.

    trajectories_by_provider: {provider: [trajectory, ...]} for that task.
    Returns None when either provider has no usable runs of the task, and
    also when a provider is compared with itself but has only one run (no
    pair exists to compare).
    """
    trajs_a = trajectories_by_provider.get(provider_a, [])
    trajs_b = trajectories_by_provider.get(provider_b, [])
    if not trajs_a or not trajs_b:
        return None
    return _mean_pairwise_similarity(
        trajs_a, trajs_b, include_args, same_group=(provider_a == provider_b)
    )


def index_trajectories(runs):
    """Reshape a flat run list into {task_id: {provider: [trajectory, ...]}}.

    Each run must carry task_id, provider, and trajectory.
    """
    index = defaultdict(lambda: defaultdict(list))
    for run in runs:
        index[run["task_id"]][run["provider"]].append(run["trajectory"])
    return {task: dict(by_provider) for task, by_provider in index.items()}


def ata_matrix(runs, providers=None, include_args=False):
    """Pairwise provider ATA matrix - the numbers behind Table 3 / Fig 4.

    Returns {(provider_a, provider_b): {"ata", "n_tasks", "per_task"}}.
    The diagonal is self-agreement across repeated runs of one provider,
    which is exactly the trajectory half of Output Stability (EQ 3).
    """
    index = index_trajectories(runs)
    if providers is None:
        providers = sorted({run["provider"] for run in runs})

    matrix = {}
    for provider_a in providers:
        for provider_b in providers:
            per_task = {}
            for task_id, by_provider in index.items():
                score = ata_for_task(by_provider, provider_a, provider_b, include_args)
                if score is not None:
                    per_task[task_id] = score
            matrix[(provider_a, provider_b)] = {
                "ata": float(np.mean(list(per_task.values()))) if per_task else None,
                "n_tasks": len(per_task),
                "per_task": per_task,
            }
    return matrix


def cross_provider_ata(runs, provider, providers=None, include_args=False):
    """How well one provider trajectories agree with every OTHER provider.

    This is the ATA term that enters the composite trust score: a provider
    is consistent to the extent that swapping TO it reproduces what the rest
    of the field does. Self-agreement is deliberately excluded - that is the
    job of the Output Stability term, and counting it twice would let a
    provider that is merely self-consistent (repeatably wrong) score well.
    """
    matrix = ata_matrix(runs, providers, include_args)
    others = [
        cell["ata"] for (p_a, p_b), cell in matrix.items()
        if p_a == provider and p_b != provider and cell["ata"] is not None
    ]
    return float(np.mean(others)) if others else None


def bootstrap_ci(per_task_scores, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap CI for an ATA figure, resampling over TASKS.

    Pre-registered as a bootstrap rather than a t-interval because ATA is
    bounded, skewed, and visibly non-normal (heavy mass at exactly 1.0),
    so a normal-theory interval is not defensible here. Resampling over
    tasks - not over run pairs - makes the interval reflect generalization
    to new tasks, which is the claim the paper actually makes.
    """
    scores = np.asarray(list(per_task_scores), dtype=float)
    if scores.size == 0:
        return None, None

    rng = np.random.default_rng(seed)
    means = rng.choice(scores, size=(n_resamples, scores.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
