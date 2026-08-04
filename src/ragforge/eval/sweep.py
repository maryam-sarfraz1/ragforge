"""Grid-search several retrieval configurations and rank them.

Retrieval tuning is a search problem with a small, cheap search space. Running it
properly — same corpus, same queries, same cut-offs — takes minutes and routinely
beats intuition about chunk sizes and fusion weights.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import PipelineConfig
from ..pipeline import RagPipeline
from ..types import Document, Query
from .harness import DEFAULT_KS, EvalResult, evaluate


@dataclass
class SweepRow:
    """One configuration's line on the leaderboard."""

    label: str
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    index_seconds: float
    n_chunks: int
    config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def get(self, metric: str) -> float:
        return self.metrics.get(metric, 0.0)


@dataclass
class SweepReport:
    """All rows plus the ranking, ready to print or render."""

    rows: list[SweepRow]
    primary: str = "recall@5"
    n_queries: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def ranked(self) -> list[SweepRow]:
        healthy = [row for row in self.rows if row.error is None]
        broken = [row for row in self.rows if row.error is not None]
        return sorted(healthy, key=lambda row: row.get(self.primary), reverse=True) + broken

    @property
    def best(self) -> SweepRow | None:
        ranked = self.ranked
        return ranked[0] if ranked and ranked[0].error is None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "n_queries": self.n_queries,
            "rows": [
                {
                    "label": row.label,
                    "metrics": row.metrics,
                    "latency_ms": row.latency_ms,
                    "index_seconds": row.index_seconds,
                    "n_chunks": row.n_chunks,
                    "config": row.config,
                    "error": row.error,
                }
                for row in self.ranked
            ],
        }


def run_sweep(
    configs: Sequence[PipelineConfig],
    documents: Sequence[Document],
    queries: Sequence[Query],
    ks: Sequence[int] = DEFAULT_KS,
    primary: str = "recall@5",
    granularity: str = "doc",
    on_progress: Callable[[int, int, str], None] | None = None,
    keep_results: bool = False,
) -> SweepReport:
    """Index and evaluate every configuration against the same corpus and queries.

    A configuration that blows up is recorded with its error rather than
    aborting the sweep — one missing optional dependency should not throw away
    the other eleven results.
    """
    if not configs:
        raise ValueError("Nothing to sweep: configs is empty")

    rows: list[SweepRow] = []
    results: list[EvalResult] = []

    for position, config in enumerate(configs, start=1):
        if on_progress:
            on_progress(position, len(configs), config.label)
        try:
            started = time.perf_counter()
            pipeline = RagPipeline.from_config(config)
            pipeline.index(documents)
            index_seconds = time.perf_counter() - started

            result = evaluate(
                pipeline,
                queries,
                ks=ks,
                granularity=granularity,
                label=config.label,
                index_seconds=index_seconds,
            )
            if keep_results:
                results.append(result)
            rows.append(
                SweepRow(
                    label=config.label,
                    metrics=result.metrics,
                    latency_ms=result.latency_ms,
                    index_seconds=index_seconds,
                    n_chunks=result.n_chunks,
                    config=config.to_dict(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a broken config must not kill the sweep
            rows.append(
                SweepRow(
                    label=config.label,
                    metrics={},
                    latency_ms={},
                    index_seconds=0.0,
                    n_chunks=0,
                    config=config.to_dict(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if on_progress:
                detail = f"{config.label} FAILED\n{traceback.format_exc()}"
                on_progress(position, len(configs), detail)

    return SweepReport(rows=rows, primary=primary, n_queries=len(queries), results=results)


def ablation_grid() -> dict[str, list[Any]]:
    """A sensible starting grid: the four knobs that move the numbers most."""
    return {
        "chunker": ["fixed", "sentence", "recursive", "markdown"],
        "retriever": ["bm25", "dense", "hybrid"],
    }
