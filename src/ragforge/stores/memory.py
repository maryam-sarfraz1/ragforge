"""Exact in-process vector store backed by a single NumPy matrix.

Brute force is the right default below roughly a million vectors: it is exact, it
has no index-build step, and on a corpus of a few thousand chunks a full scan
costs well under a millisecond. It also serves as the ground truth that the
approximate backends are measured against.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np

from ..types import Chunk
from .base import MetadataFilter, VectorStore, matches_filter, register_store


@register_store
class InMemoryStore(VectorStore):
    """Brute-force cosine search with optional npz persistence."""

    name = "memory"

    def __init__(self, path: str | None = None, normalize: bool = True) -> None:
        self.path = path
        self.normalize = normalize
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        if path and os.path.exists(self._data_file(path)):
            self.load(path)

    @staticmethod
    def _data_file(path: str) -> str:
        return os.path.join(path, "vectors.npz")

    @staticmethod
    def _meta_file(path: str) -> str:
        return os.path.join(path, "chunks.jsonl")

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Expected a ({len(chunks)}, dim) matrix, got {vectors.shape} — "
                "vectors and chunks must line up row for row."
            )
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

        fresh_rows: list[np.ndarray] = []
        fresh_chunks: list[Chunk] = []
        for chunk, vector in zip(chunks, vectors):
            existing = self._index.get(chunk.id)
            if existing is not None:
                # Overwrite in place so re-indexing is idempotent.
                self._chunks[existing] = chunk
                if self._matrix is not None:
                    self._matrix[existing] = vector
                continue
            self._index[chunk.id] = len(self._chunks) + len(fresh_chunks)
            fresh_chunks.append(chunk)
            fresh_rows.append(vector)

        if fresh_chunks:
            self._chunks.extend(fresh_chunks)
            block = np.vstack(fresh_rows)
            self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Query dim {query.shape[0]} != index dim {self._matrix.shape[1]}. "
                "The index was probably built with a different embedder."
            )
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        scores = self._matrix @ query

        if where:
            allowed = np.array(
                [matches_filter(chunk.metadata, where) for chunk in self._chunks],
                dtype=bool,
            )
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        # argpartition finds the top-k without sorting the whole array, then we sort
        # only that slice — the difference shows up on corpora past ~100k chunks.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk_id in ids:
            position = self._index.get(chunk_id)
            if position is not None:
                out.append(self._chunks[position])
        return out

    def vectors_for(self, ids: Sequence[str]) -> np.ndarray:
        if self._matrix is None:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [self._index[i] for i in ids if i in self._index]
        if not rows:
            return np.zeros((0, self._matrix.shape[1]), dtype=np.float32)
        return self._matrix[rows]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks = []
        self._index = {}
        self._matrix = None

    def persist(self) -> None:
        if self.path:
            self.save(self.path)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(self._data_file(path), vectors=matrix)
        with open(self._meta_file(path), "w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def load(self, path: str) -> InMemoryStore:
        with np.load(self._data_file(path), allow_pickle=False) as payload:
            matrix = payload["vectors"]
        self._matrix = matrix if matrix.size else None
        self._chunks = []
        with open(self._meta_file(path), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self._chunks.append(Chunk.from_dict(json.loads(line)))
        self._index = {chunk.id: i for i, chunk in enumerate(self._chunks)}
        return self

    def describe(self) -> str:
        return f"memory({self.count()} chunks)"
