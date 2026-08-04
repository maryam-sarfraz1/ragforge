"""Evaluation harness: metrics, per-query scoring, and configuration sweeps."""

from .harness import (
    DEFAULT_KS,
    EvalResult,
    QueryScore,
    compare,
    evaluate,
    score_one,
)
from .metrics import (
    average_precision,
    dcg,
    hit_rate_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .sweep import SweepReport, SweepRow, ablation_grid, run_sweep

__all__ = [
    "DEFAULT_KS",
    "EvalResult",
    "QueryScore",
    "SweepReport",
    "SweepRow",
    "ablation_grid",
    "average_precision",
    "compare",
    "dcg",
    "evaluate",
    "hit_rate_at_k",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_sweep",
    "score_one",
]
