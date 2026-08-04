"""Chroma backend — an embedded, persistent vector database.

Chroma is the pragmatic default when the in-memory store stops fitting: it
persists to a local directory, needs no server, and speaks the same metadata
filter dialect this package already uses.

Install with ``pip install "ragforge[chroma]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..types import Chunk
from .base import MetadataFilter, VectorStore, flatten_metadata, register_store


@register_store
class ChromaStore(VectorStore):
    """Persistent Chroma collection.

    Vectors are supplied by ragforge rather than by Chroma's own embedding
    function, so the same embedder drives every backend and cross-store
    comparisons stay apples to apples.
    """

    name = "chroma"

    def __init__(
        self,
        path: str | None = ".ragforge/chroma",
        collection: str = "ragforge",
        distance: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'chromadb is not installed. Run `pip install "ragforge[chroma]"`, '
                "or use the built-in `memory` store."
            ) from exc

        self.path = path
        self.collection_name = collection
        self.distance = distance
        self._client = (
            chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": distance},
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"Expected {len(chunks)} vectors, got {vectors.shape[0]}")

        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            payload = flatten_metadata(chunk.metadata)
            payload["doc_id"] = chunk.doc_id
            payload["position"] = chunk.position
            metadatas.append(payload)

        # upsert keeps re-indexing idempotent, matching InMemoryStore semantics.
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[vector.tolist() for vector in vectors],
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

    def _to_chunk(self, chunk_id: str, text: str, metadata: dict[str, Any] | None) -> Chunk:
        metadata = dict(metadata or {})
        doc_id = str(metadata.pop("doc_id", chunk_id))
        position = int(metadata.pop("position", 0) or 0)
        return Chunk(id=chunk_id, doc_id=doc_id, text=text or "", position=position,
                     metadata=metadata)

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        if k <= 0 or self.count() == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        result = self._collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[tuple[Chunk, float]] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma returns cosine *distance*; the rest of ragforge speaks similarity.
            score = 1.0 - float(distance) if self.distance == "cosine" else -float(distance)
            hits.append((self._to_chunk(chunk_id, text, metadata), score))
        return hits

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        result = self._collection.get(ids=list(ids), include=["documents", "metadatas"])
        return [
            self._to_chunk(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]

    def vectors_for(self, ids: Sequence[str]) -> np.ndarray:
        if not ids:
            return np.zeros((0, 0), dtype=np.float32)
        result = self._collection.get(ids=list(ids), include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )

    def describe(self) -> str:
        return f"chroma({self.collection_name}, {self.count()} chunks)"
