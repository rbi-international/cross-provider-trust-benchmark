"""
EQ 1 - Task Success Rate (TSR).

TSR is the simplest of the four metrics and the base of the composite score:
the fraction of runs that satisfied their task's pass_criteria. It is
reported at three granularities, because RQ1 (provider effect) and RQ6
(task-category dependence) need different slices of the same quantity:

    TSR(p)        over every run of provider p
    TSR(p, c)     over runs of provider p in task category c
    TSR(p, t)     over the repeated runs of provider p on a single task t

Confidence intervals use the Wilson score interval rather than the normal
approximation. This matters here: several provider x category cells sit at
or near 0.0 or 1.0, where the normal approximation produces intervals that
run outside [0, 1] and understate uncertainty at small n. Wilson stays
inside the unit interval and behaves at the boundaries.
"""
from collections import defaultdict

from scipy.stats import norm


def wilson_interval(n_success, n_total, confidence=0.95):
    """Wilson score interval for a binomial proportion.

    Returns (low, high). An empty cell (n_total == 0) returns (0.0, 1.0):
    no runs means no information, not a point estimate at zero.
    """
    if n_total == 0:
        return 0.0, 1.0

    z = norm.ppf(1 - (1 - confidence) / 2)
    p = n_success / n_total
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    half_width = (z / denominator) * ((p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5)
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def tsr(results):
    """Point-estimate TSR over a list of result dicts (each with 'passed')."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["passed"]) / len(results)


def tsr_with_ci(results, confidence=0.95):
    """TSR plus its Wilson CI and the counts behind it."""
    n_total = len(results)
    n_success = sum(1 for r in results if r["passed"])
    low, high = wilson_interval(n_success, n_total, confidence)
    return {
        "tsr": tsr(results),
        "ci_low": low,
        "ci_high": high,
        "n_success": n_success,
        "n_total": n_total,
    }


def _group(results, key_fields):
    grouped = defaultdict(list)
    for r in results:
        grouped[tuple(r[f] for f in key_fields)].append(r)
    return grouped


def tsr_by(results, *key_fields, confidence=0.95):
    """TSR broken down by any combination of result fields.

    tsr_by(results, "provider")               -> RQ1 (Table 2)
    tsr_by(results, "provider", "category")   -> RQ6 (Table 5)
    tsr_by(results, "provider", "task_id")    -> per-task, feeds Output Stability

    Keys are single values when one field is given, tuples when several are.
    """
    out = {}
    for key, group in _group(results, key_fields).items():
        out[key[0] if len(key_fields) == 1 else key] = tsr_with_ci(group, confidence)
    return out
