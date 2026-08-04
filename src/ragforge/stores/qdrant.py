"""Qdrant backend — the production step up from an embedded store.

Qdrant runs as a service (``docker compose up qdrant`` from this repo), does HNSW
approximate search, and supports payload filtering server-side. Point at an
in-memory instance with ``location=":memory:"`` to exercise the adapter in tests
without a container.

Install with ``pip install "ragforge[qdrant]"``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..types import Chunk
from .base import MetadataFilter, VectorStore, flatten_metadata, register_store

# Qdrant point ids must be a UUID or an unsigned integer, but ragforge chunk ids are
# content hashes. Deriving a UUIDv5 from the chunk id keeps the mapping deterministic
# and therefore keeps upserts idempotent across runs.
_NAMESPACE = uuid.UUID("6f2b6a3e-6e5f-5f9c-9a1d-1f7d0c4b8a52")


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


@register_store
class QdrantStore(VectorStore):
    """Qdrant collection with cosine distance."""

    name = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        location: str | None = None,
        collection: str = "ragforge",
        dim: int = 512,
        api_key: str | None = None,
        recreate: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'qdrant-client is not installed. Run `pip install "ragforge[qdrant]"`, '
                "or use the built-in `memory` store."
            ) from exc

        self._models = qmodels
        self.collection_name = collection
        self.dim = int(dim)

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(location=location or ":memory:")

        exists = self._client.collection_exists(collection)
        if exists and recreate:
            self._client.delete_collection(collection)
            exists = False
        if not exists:
            self._create_collection()

    def _create_collection(self) -> None:
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._models.VectorParams(
                size=self.dim,
                distance=self._models.Distance.COSINE,
            ),
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"Expected {len(chunks)} vectors, got {vectors.shape[0]}")

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload: dict[str, Any] = flatten_metadata(chunk.metadata)
            payload.update(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "position": chunk.position,
                    "text": chunk.text,
                }
            )
            points.append(
                self._models.PointStruct(
                    id=point_id_for(chunk.id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self.collection_name, points=points)

    def _build_filter(self, where: MetadataFilter | None):
        if not where:
            return None
        models = self._models
        must = []
        for key, condition in where.items():
            if not isinstance(condition, dict):
                must.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=condition))
                )
                continue
            for operator, operand in condition.items():
                if operator == "$eq":
                    must.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=operand))
                    )
                elif operator == "$in":
                    must.append(
                        models.FieldCondition(key=key, match=models.MatchAny(any=list(operand)))
                    )
                elif operator in ("$gt", "$gte", "$lt", "$lte"):
                    bounds = {operator.lstrip("$"): float(operand)}
                    must.append(models.FieldCondition(key=key, range=models.Range(**bounds)))
                else:
                    raise ValueError(f"Qdrant backend does not support {operator!r}")
        return models.Filter(must=must)

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> Chunk:
        payload = dict(payload or {})
        chunk_id = str(payload.pop("chunk_id", ""))
        doc_id = str(payload.pop("doc_id", chunk_id))
        text = str(payload.pop("text", ""))
        position = int(payload.pop("position", 0) or 0)
        return Chunk(id=chunk_id, doc_id=doc_id, text=text, position=position, metadata=payload)

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        if k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel().tolist()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query,
            limit=k,
            query_filter=self._build_filter(where),
            with_payload=True,
        )
        # Qdrant's cosine score is already a similarity, so no conversion is needed.
        return [(self._to_chunk(point.payload), float(point.score)) for point in response.points]

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id_for(i) for i in ids],
            with_payload=True,
        )
        return [self._to_chunk(record.payload) for record in records]

    def vectors_for(self, ids: Sequence[str]) -> np.ndarray:
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id_for(i) for i in ids],
            with_vectors=True,
        )
        vectors = [record.vector for record in records if record.vector is not None]
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def count(self) -> int:
        return int(self._client.count(self.collection_name, exact=True).count)

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._create_collection()

    def describe(self) -> str:
        return f"qdrant({self.collection_name}, {self.count()} chunks)"
