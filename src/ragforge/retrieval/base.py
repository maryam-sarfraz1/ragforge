"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..stores.base import MetadataFilter
from ..types import Chunk, ScoredChunk


class Retriever(ABC):
    """Ranks chunks for a query string."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``k`` chunks, best first, with ``rank`` starting at 1."""

    def index(self, chunks: Sequence[Chunk]) -> None:  # noqa: B027
        """Ingest chunks. Intentionally a no-op rather than abstract: retrievers
        backed by a shared vector store have nothing of their own to build."""

    def describe(self) -> str:
        return self.name


def rank_and_tag(
    pairs: Sequence[tuple],
    source: str,
    k: int | None = None,
) -> list[ScoredChunk]:
    """Turn ``(chunk, score)`` pairs into ranked :class:`ScoredChunk` objects."""
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    if k is not None:
        ordered = ordered[:k]
    return [
        ScoredChunk(chunk=chunk, score=float(score), rank=position, source=source)
        for position, (chunk, score) in enumerate(ordered, start=1)
    ]
