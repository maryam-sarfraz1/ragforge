"""Ranking metrics.

All functions take a ranked list of ids (best first) and the judgements for one
query, and return a value in ``[0, 1]``. They are pure and dependency-free so
they can be unit-tested against worked examples rather than trusted on faith.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence


def _dedupe(ids: Iterable[str]) -> list[str]:
    seen: list[str] = []
    known = set()
    for item in ids:
        if item not in known:
            known.add(item)
            seen.append(item)
    return seen


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the relevant documents that appear in the top ``k``.

    The metric that matters most for RAG: a generator cannot use what retrieval
    never surfaced, so recall sets the ceiling on the whole system.
    """
    if not relevant:
        return 0.0
    top = set(_dedupe(retrieved)[:k])
    found = sum(1 for doc_id in set(relevant) if doc_id in top)
    return found / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    top = _dedupe(retrieved)[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for doc_id in top if doc_id in relevant_set) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top ``k``, else 0.0."""
    top = set(_dedupe(retrieved)[:k])
    return 1.0 if any(doc_id in top for doc_id in relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """``1 / rank`` of the first relevant hit; 0.0 if none was retrieved.

    Averaged over queries this is MRR — the metric to watch when the answer comes
    from a single document and its position in the context window matters.
    """
    relevant_set = set(relevant)
    for position, doc_id in enumerate(_dedupe(retrieved), start=1):
        if doc_id in relevant_set:
            return 1.0 / position
    return 0.0


def average_precision(
    retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None
) -> float:
    """Mean of the precision values measured at each relevant hit."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _dedupe(retrieved)
    if k is not None:
        ranked = ranked[:k]
    hits = 0
    total = 0.0
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_set:
            hits += 1
            total += hits / position
    return total / min(len(relevant_set), k) if k else total / len(relevant_set)


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard ``log2(rank + 1)`` discount."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(
    retrieved: Sequence[str],
    grade_fn: Callable[[str], float],
    k: int,
    all_grades: dict[str, float] | None = None,
) -> float:
    """Normalised DCG — the metric to use when relevance is graded, not binary.

    The ideal ranking is built from ``all_grades`` when supplied, so a run is not
    rewarded for a perfect ordering of the few documents it happened to find.
    """
    ranked = _dedupe(retrieved)[:k]
    gains = [grade_fn(doc_id) for doc_id in ranked]
    actual = dcg(gains)
    if all_grades:
        ideal_gains = sorted((g for g in all_grades.values() if g > 0), reverse=True)[:k]
    else:
        ideal_gains = sorted((g for g in gains if g > 0), reverse=True)
    ideal = dcg(ideal_gains)
    return actual / ideal if ideal > 0 else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 100]``)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[int(position)])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)
