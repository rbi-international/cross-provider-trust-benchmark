"""
EQ 4 - Composite Cross-Provider Trust Score (CPTS).

RQ4 asks whether TSR, ATA, and Output Stability combine into ONE number a
practitioner could actually act on when deciding whether to swap the model
behind a shipped agent. This module is that combination.

    CPTS(p) = w_tsr * TSR(p)
            + w_ata * ATA_cross(p)
            + w_stab * Stability(p)
            + w_proto * ProtocolAdherence(p)

All four terms are already bounded in [0, 1] and oriented so that higher is
better, so the weighted sum is itself in [0, 1] when the weights sum to 1 -
no rescaling, no z-scoring against the sample. That matters for the practical
claim: a CPTS is comparable across papers and across future provider sets,
which a sample-normalized score would not be.

Term meanings, and why each is not redundant with the others:

  TSR(p)         Does it get the job done at all.
  ATA_cross(p)   Does it get there the way the rest of the field does -
                 the swap-compatibility term. Self-agreement is excluded
                 (see ata.cross_provider_ata) so a repeatably-wrong provider
                 cannot earn credit here.
  Stability(p)   Does it do the same thing twice in a row.
  Protocol(p)    Does it honor the structured tool-calling contract at all,
                 rather than describing tool calls in prose. Added after the
                 Week 4 pilot surfaced this as a real, provider-specific
                 failure mode; it is a trust dimension in its own right,
                 since an agent that silently stops calling tools is worse
                 than one that calls them and fails.

DEFAULT WEIGHTS AND THEIR JUSTIFICATION - this is the part a reviewer will
push on, so it is stated explicitly rather than buried:

    w_tsr = 0.40, w_ata = 0.25, w_stab = 0.25, w_proto = 0.10

The weights are a VALUE JUDGMENT about deployment priorities, not an
empirical finding, and the paper says so. Capability (TSR) is weighted
highest because a provider that cannot do the task is unusable regardless
of how consistently it fails. Consistency across providers and across
repeats are weighted equally to each other, and together they exceed TSR,
which encodes the position this paper argues for: for a team SWAPPING
backends, predictability is collectively worth more than raw capability.
Protocol adherence gets the smallest weight because it is partly captured
downstream in TSR (a phantom tool call usually also fails the task), so a
larger weight would double-count it.

Because the weighting is a judgment call, sensitivity_analysis() below
re-ranks the providers under many alternative weightings. If the ranking is
stable across them, the RQ4 conclusion does not depend on our specific
choice - and if it is not stable, the paper reports that instead of hiding it.
"""
from collections import defaultdict

import numpy as np

from src.metrics.ata import cross_provider_ata
from src.metrics.output_stability import stability_by_provider
from src.metrics.tsr import tsr

DEFAULT_WEIGHTS = {
    "tsr": 0.40,
    "ata": 0.25,
    "stability": 0.25,
    "protocol": 0.10,
}


def protocol_adherence_rate(results):
    """Fraction of runs that used the structured tool-calling contract.

    Runs logged before protocol detection existed (Week 4 and earlier) have
    no protocol_adherence field. Those are treated as adherent, matching the
    field default in the trajectory logger: absence of a detected phantom
    call is what adherence means.
    """
    if not results:
        return None
    adherent = sum(1 for r in results if r.get("protocol_adherence", True))
    return adherent / len(results)


def compute_terms(runs, providers=None, include_args=False):
    """The four component terms per provider, before weighting.

    runs: flat list of run dicts carrying provider, task_id, passed,
          trajectory, and optionally protocol_adherence.
    """
    if providers is None:
        providers = sorted({run["provider"] for run in runs})

    by_provider = defaultdict(list)
    for run in runs:
        by_provider[run["provider"]].append(run)

    stability = stability_by_provider(runs, include_args)

    terms = {}
    for provider in providers:
        provider_runs = by_provider.get(provider, [])
        terms[provider] = {
            "tsr": tsr(provider_runs) if provider_runs else None,
            "ata": cross_provider_ata(runs, provider, providers, include_args),
            "stability": (stability.get(provider) or {}).get("output_stability"),
            "protocol": protocol_adherence_rate(provider_runs),
            "n_runs": len(provider_runs),
        }
    return terms


def trust_score(terms_for_provider, weights=None):
    """Weighted composite from one provider's component terms.

    Missing terms (None) are dropped and the remaining weights renormalized,
    so a provider missing one dimension is scored on what it does have rather
    than silently penalized as if that dimension were zero. The count of
    terms actually used is returned so the paper can flag any such cell.
    """
    weights = weights or DEFAULT_WEIGHTS

    usable = {k: w for k, w in weights.items()
              if terms_for_provider.get(k) is not None and w > 0}
    total_weight = sum(usable.values())
    if total_weight == 0:
        return {"trust_score": None, "terms_used": [], "weights_applied": {}}

    normalized = {k: w / total_weight for k, w in usable.items()}
    score = sum(normalized[k] * terms_for_provider[k] for k in normalized)
    return {
        "trust_score": float(score),
        "terms_used": sorted(normalized),
        "weights_applied": normalized,
    }


def trust_scores(runs, providers=None, weights=None, include_args=False):
    """CPTS for every provider - the numbers behind Table 4 (RQ4 ranking)."""
    terms = compute_terms(runs, providers, include_args)
    scored = {}
    for provider, provider_terms in terms.items():
        scored[provider] = {**provider_terms, **trust_score(provider_terms, weights)}
    return scored


def rank_providers(scored):
    """Providers ordered best-to-worst by CPTS, unscored ones dropped."""
    ranked = [(p, s["trust_score"]) for p, s in scored.items() if s["trust_score"] is not None]
    return sorted(ranked, key=lambda pair: pair[1], reverse=True)


def sensitivity_analysis(runs, providers=None, include_args=False, n_samples=2000, seed=42):
    """Does the RQ4 ranking survive a different choice of weights?

    Draws weight vectors uniformly from the simplex (Dirichlet(1,1,1,1)) and
    records how often each provider takes each rank. A provider that holds
    rank 1 under nearly every weighting is a robust recommendation; one whose
    rank swings with the weights is reported as weighting-dependent, which is
    an honest and useful result rather than a failed one.

    The component terms are computed ONCE and reweighted, so this is cheap
    regardless of n_samples.
    """
    terms = compute_terms(runs, providers, include_args)
    scorable = [p for p, t in terms.items()
                if any(t.get(k) is not None for k in DEFAULT_WEIGHTS)]
    if not scorable:
        return {}

    rng = np.random.default_rng(seed)
    keys = list(DEFAULT_WEIGHTS)
    draws = rng.dirichlet(np.ones(len(keys)), size=n_samples)

    rank_counts = {p: defaultdict(int) for p in scorable}
    for draw in draws:
        weights = dict(zip(keys, draw))
        scored = {p: trust_score(terms[p], weights)["trust_score"] for p in scorable}
        order = sorted((p for p in scorable if scored[p] is not None),
                       key=lambda p: scored[p], reverse=True)
        for position, provider in enumerate(order, start=1):
            rank_counts[provider][position] += 1

    return {
        provider: {
            "rank_distribution": {rank: count / n_samples for rank, count in sorted(counts.items())},
            "modal_rank": max(counts, key=counts.get) if counts else None,
            "p_rank_1": counts.get(1, 0) / n_samples,
        }
        for provider, counts in rank_counts.items()
    }
