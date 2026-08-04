"""Vector store interface and the metadata filter language shared by backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..types import Chunk

# A deliberately small filter dialect, chosen because it maps cleanly onto Chroma's
# `where` clauses and Qdrant's field conditions without a translation layer that
# would inevitably drift between backends.
OPERATORS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte")

MetadataFilter = dict[str, Any]


def matches_filter(metadata: dict[str, Any], where: MetadataFilter | None) -> bool:
    """Evaluate a filter against one chunk's metadata (used by in-process backends)."""
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if not isinstance(condition, dict):
            if value != condition:
                return False
            continue
        for operator, operand in condition.items():
            if operator not in OPERATORS:
                raise ValueError(f"Unsupported filter operator {operator!r}")
            if operator == "$eq" and value != operand:
                return False
            if operator == "$ne" and value == operand:
                return False
            if operator == "$in" and value not in operand:
                return False
            if operator == "$nin" and value in operand:
                return False
            if operator in ("$gt", "$gte", "$lt", "$lte"):
                if value is None:
                    return False
                try:
                    left, right = float(value), float(operand)
                except (TypeError, ValueError):
                    return False
                if operator == "$gt" and not left > right:
                    return False
                if operator == "$gte" and not left >= right:
                    return False
                if operator == "$lt" and not left < right:
                    return False
                if operator == "$lte" and not left <= right:
                    return False
    return True


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata to the scalar types every backend accepts."""
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = ", ".join(str(item) for item in value)
        else:
            flat[key] = str(value)
    return flat


class VectorStore(ABC):
    """Persistence and nearest-neighbour search over chunk vectors.

    Every backend returns **cosine similarity in ``[-1, 1]``, higher is better**.
    Backends that natively speak distance convert before returning, so retrievers
    and fusion never have to know which store they are talking to.
    """

    name: str = "store"

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Insert or overwrite chunks with their vectors."""

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the ``k`` nearest chunks as ``(chunk, cosine_similarity)`` pairs."""

    @abstractmethod
    def get(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id, skipping ids that are not present."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def reset(self) -> None:
        """Drop everything in the collection."""

    def vectors_for(self, ids: Sequence[str]) -> np.ndarray:
        """Vectors for the given ids, used by MMR re-ranking.

        Backends that cannot return stored vectors should raise
        :class:`NotImplementedError`; MMR then degrades gracefully.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot return stored vectors")

    def persist(self) -> None:  # noqa: B027
        """Flush to disk. Intentionally a no-op rather than abstract: backends that
        write through on every add have nothing to flush."""

    def describe(self) -> str:
        return self.name

    def __len__(self) -> int:
        return self.count()


_REGISTRY: dict[str, type[VectorStore]] = {}


def register_store(cls: type[VectorStore]) -> type[VectorStore]:
    _REGISTRY[cls.name] = cls
    return cls


def available_stores() -> list[str]:
    return sorted(_REGISTRY)


def build_store(name: str, **kwargs) -> VectorStore:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown store {name!r}. Available: {', '.join(available_stores())}")
    return _REGISTRY[name](**kwargs)
