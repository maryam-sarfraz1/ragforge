"""Fusing dense and sparse result lists.

Dense and sparse retrievers fail in different directions — dense misses exact
identifiers, sparse misses paraphrases — so combining them is close to free
accuracy. The hard part is that their scores are not comparable, which is why the
default fusion works on **ranks**.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..stores.base import MetadataFilter
from ..types import Chunk, ScoredChunk
from .base import Retriever


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    weights: Sequence[float] | None = None,
    smoothing: int = 60,
) -> list[ScoredChunk]:
    """Combine ranked lists with RRF: ``score = Σ wᵢ / (smoothing + rankᵢ)``.

    Rank-based fusion needs no score calibration, which is what makes it robust
    when one retriever returns cosine similarities in ``[0, 1]`` and another
    returns unbounded BM25 scores. ``smoothing=60`` is the value from the original
    Cormack et al. paper and is a sane default; lower values weight the very top
    of each list more aggressively.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must have one entry per result list")

    totals: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    components: dict[str, dict[str, float]] = {}

    for weight, results in zip(weights, result_lists):
        for hit in results:
            rank = hit.rank if hit.rank > 0 else 1
            contribution = float(weight) / (smoothing + rank)
            totals[hit.id] = totals.get(hit.id, 0.0) + contribution
            chunks[hit.id] = hit.chunk
            components.setdefault(hit.id, {})[hit.source or "unknown"] = hit.score

    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=score,
            rank=position,
            source="rrf",
            components=components.get(chunk_id, {}),
        )
        for position, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def weighted_score_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Min-max normalise each list, then take a weighted sum.

    Keeps score *margins* that RRF throws away, which helps when one retriever is
    clearly more trustworthy. The cost is sensitivity to outliers and to how many
    candidates each retriever returned — normalisation shifts when the list length
    changes. Prefer RRF unless a sweep says otherwise.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must have one entry per result list")

    totals: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    components: dict[str, dict[str, float]] = {}

    for weight, results in zip(weights, result_lists):
        normalised = _min_max([hit.score for hit in results])
        for hit, value in zip(results, normalised):
            totals[hit.id] = totals.get(hit.id, 0.0) + float(weight) * value
            chunks[hit.id] = hit.chunk
            components.setdefault(hit.id, {})[hit.source or "unknown"] = hit.score

    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=score,
            rank=position,
            source="weighted",
            components=components.get(chunk_id, {}),
        )
        for position, (chunk_id, score) in enumerate(ordered, start=1)
    ]


FUSIONS = {"rrf": reciprocal_rank_fusion, "weighted": weighted_score_fusion}


class HybridRetriever(Retriever):
    """Runs several retrievers and fuses their ranked lists.

    Args:
        retrievers: The retrievers to combine, usually one dense and one sparse.
        weights: Per-retriever weight. ``(1.0, 1.0)`` is a reasonable start;
            sweep it before hand-tuning.
        fusion: ``"rrf"`` (rank-based, default) or ``"weighted"`` (score-based).
        candidate_multiplier: Each retriever is asked for ``k * multiplier``
            candidates. Fusion can only promote documents that at least one
            retriever surfaced, so too small a pool caps the achievable recall.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        weights: Sequence[float] | None = None,
        fusion: str = "rrf",
        candidate_multiplier: int = 4,
        rrf_smoothing: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        if fusion not in FUSIONS:
            raise ValueError(f"Unknown fusion {fusion!r}. Choose from {', '.join(FUSIONS)}")
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("weights must have one entry per retriever")
        self.fusion = fusion
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.rrf_smoothing = int(rrf_smoothing)

    def index(self, chunks: Sequence[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        depth = k * self.candidate_multiplier
        lists: list[list[ScoredChunk]] = [
            retriever.retrieve(query, k=depth, where=where) for retriever in self.retrievers
        ]
        if self.fusion == "rrf":
            fused = reciprocal_rank_fusion(lists, self.weights, smoothing=self.rrf_smoothing)
        else:
            fused = weighted_score_fusion(lists, self.weights)
        return fused[:k]

    def contributions(self, query: str, k: int = 10) -> list[tuple[str, list[str]]]:
        """Which retriever surfaced each fused hit — useful when debugging a miss."""
        depth = k * self.candidate_multiplier
        lists = [retriever.retrieve(query, k=depth) for retriever in self.retrievers]
        origin: dict[str, list[str]] = {}
        for retriever, results in zip(self.retrievers, lists):
            for hit in results:
                origin.setdefault(hit.id, []).append(retriever.name)
        fused = self.retrieve(query, k=k)
        return [(hit.id, origin.get(hit.id, [])) for hit in fused]

    def describe(self) -> str:
        inner = " + ".join(retriever.describe() for retriever in self.retrievers)
        return f"hybrid[{self.fusion}]({inner})"
