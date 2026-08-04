"""The evaluation harness: run a pipeline over a query set and score it.

The whole point of this package. Retrieval changes — a new chunk size, a new
embedding model, adding a reranker — are cheap to make and impossible to judge by
eyeballing three queries. This turns "it feels better" into a number with a
confidence interval.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import Query, RetrievalResult
from .metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class QueryScore:
    """Per-query metrics, kept so failures can be inspected individually."""

    query_id: str
    query: str
    metrics: dict[str, float]
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return self.metrics.get("mrr", 0.0) > 0.0


@dataclass
class EvalResult:
    """Aggregate scores for one pipeline over one evaluation set."""

    label: str
    metrics: dict[str, float]
    per_query: list[QueryScore]
    latency_ms: dict[str, float]
    index_seconds: float = 0.0
    n_queries: int = 0
    n_chunks: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def primary(self, metric: str = "recall@5") -> float:
        return self.metrics.get(metric, 0.0)

    def failures(self, metric: str = "mrr", threshold: float = 0.0) -> list[QueryScore]:
        """Queries scoring at or below ``threshold`` — where to look first."""
        return [score for score in self.per_query if score.metrics.get(metric, 0.0) <= threshold]

    def confidence_interval(self, metric: str = "recall@5", z: float = 1.96) -> tuple[float, float]:
        """Normal-approximation CI over per-query scores.

        Query sets are usually small; without an interval it is easy to celebrate
        a two-point gain that is well inside the noise.
        """
        values = [score.metrics.get(metric, 0.0) for score in self.per_query]
        if len(values) < 2:
            return (0.0, 0.0)
        mean = statistics.fmean(values)
        margin = z * (statistics.stdev(values) / (len(values) ** 0.5))
        return (max(0.0, mean - margin), min(1.0, mean + margin))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "index_seconds": self.index_seconds,
            "n_queries": self.n_queries,
            "n_chunks": self.n_chunks,
            "config": self.config,
            "per_query": [
                {
                    "query_id": score.query_id,
                    "query": score.query,
                    "metrics": score.metrics,
                    "latency_ms": score.latency_ms,
                    "retrieved": score.retrieved,
                    "relevant": score.relevant,
                }
                for score in self.per_query
            ],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


def score_one(
    query: Query,
    result: RetrievalResult,
    ks: Sequence[int] = DEFAULT_KS,
    granularity: str = "doc",
) -> QueryScore:
    """Score a single retrieval result against a query's judgements.

    ``granularity="doc"`` maps the retrieved chunks back to their source documents
    before scoring, because judgements are almost always written at document
    level. The top-k slice is still taken over *chunks* first — that is what the
    system actually puts in the context window.
    """
    relevant = query.judged_ids
    ranked_all = result.doc_ids if granularity == "doc" else result.chunk_ids

    metrics: dict[str, float] = {}
    for k in ks:
        if granularity == "doc":
            ranked = []
            for hit in result.hits[:k]:
                if hit.doc_id not in ranked:
                    ranked.append(hit.doc_id)
        else:
            ranked = result.chunk_ids[:k]
        metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        metrics[f"hit@{k}"] = hit_rate_at_k(ranked, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, query.grade_of, k, query.grades or None)

    metrics["mrr"] = reciprocal_rank(ranked_all, relevant)
    metrics["map"] = average_precision(ranked_all, relevant, k=max(ks))

    return QueryScore(
        query_id=query.id,
        query=query.text,
        metrics=metrics,
        latency_ms=result.latency_ms,
        retrieved=list(ranked_all[: max(ks)]),
        relevant=list(relevant),
    )


def evaluate(
    pipeline,
    queries: Sequence[Query],
    ks: Sequence[int] = DEFAULT_KS,
    granularity: str = "doc",
    label: str | None = None,
    index_seconds: float = 0.0,
    warmup: bool = True,
) -> EvalResult:
    """Run every query through ``pipeline`` and aggregate the scores.

    Args:
        ks: Cut-offs to report. Pick the one that matches how many chunks you
            actually put in the prompt.
        granularity: ``"doc"`` (default) or ``"chunk"``.
        warmup: Issue one throwaway query first so lazily-loaded models and
            caches do not land in the latency numbers.
    """
    if not queries:
        raise ValueError("Cannot evaluate against an empty query set")
    ks = tuple(sorted({int(k) for k in ks}))
    depth = max(ks)

    if warmup:
        pipeline.retrieve(queries[0].text, k=depth)

    scores: list[QueryScore] = []
    for query in queries:
        started = time.perf_counter()
        result = pipeline.retrieve(query.text, k=depth)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        scores.append(score_one(query, result, ks=ks, granularity=granularity))

    metric_names = list(scores[0].metrics)
    aggregate = {
        name: statistics.fmean(score.metrics.get(name, 0.0) for score in scores)
        for name in metric_names
    }
    latencies = [score.latency_ms for score in scores]

    return EvalResult(
        label=label or getattr(pipeline.config, "label", "pipeline"),
        metrics=aggregate,
        per_query=scores,
        latency_ms={
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies),
        },
        index_seconds=index_seconds,
        n_queries=len(scores),
        n_chunks=len(getattr(pipeline, "chunks", [])),
        config=pipeline.config.to_dict() if hasattr(pipeline, "config") else {},
    )


def compare(
    baseline: EvalResult, candidate: EvalResult, metric: str = "recall@5"
) -> dict[str, Any]:
    """Diff two runs on one metric, including a paired per-query breakdown.

    The paired view is the useful one: an average that moved +0.02 might be four
    queries improving and three regressing, which is a very different story from
    a uniform small gain.
    """
    before = {score.query_id: score.metrics.get(metric, 0.0) for score in baseline.per_query}
    after = {score.query_id: score.metrics.get(metric, 0.0) for score in candidate.per_query}
    shared = [qid for qid in before if qid in after]

    improved = [qid for qid in shared if after[qid] > before[qid]]
    regressed = [qid for qid in shared if after[qid] < before[qid]]
    delta = candidate.metrics.get(metric, 0.0) - baseline.metrics.get(metric, 0.0)

    return {
        "metric": metric,
        "baseline": baseline.metrics.get(metric, 0.0),
        "candidate": candidate.metrics.get(metric, 0.0),
        "delta": delta,
        "delta_pct": (delta / baseline.metrics[metric] * 100.0)
        if baseline.metrics.get(metric)
        else None,
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(shared) - len(improved) - len(regressed),
    }
