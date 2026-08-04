"""Dense (embedding) retrieval over a vector store."""

from __future__ import annotations

from ..embedding.base import Embedder
from ..stores.base import MetadataFilter, VectorStore
from ..types import ScoredChunk
from .base import Retriever


class DenseRetriever(Retriever):
    """Embeds the query and asks the vector store for nearest neighbours.

    Strong on paraphrase and concept matching, weak on exact identifiers — a
    dense-only stack will happily miss ``ERR_4021`` while returning three chunks
    about error handling in general. That failure mode is why
    :class:`~ragforge.retrieval.hybrid.HybridRetriever` exists.
    """

    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        if not query.strip() or k <= 0:
            return []
        vector = self.embedder.encode_query(query)
        hits = self.store.search(vector, k=k, where=where)
        return [
            ScoredChunk(chunk=chunk, score=float(score), rank=position, source="dense")
            for position, (chunk, score) in enumerate(hits, start=1)
        ]

    def describe(self) -> str:
        return f"dense({self.embedder.describe()} -> {self.store.name})"
