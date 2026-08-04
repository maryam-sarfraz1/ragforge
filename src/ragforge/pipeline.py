"""Assembles a configuration into a working retrieval pipeline."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

from .chunking import build_chunker
from .config import PipelineConfig
from .embedding.base import CachedEmbedder, Embedder, build_embedder
from .retrieval.base import Retriever
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import CrossEncoderReranker, MMRReranker
from .stores.base import MetadataFilter, VectorStore, build_store
from .types import Chunk, Document, RetrievalResult

_STATE_FILE = "pipeline.json"


class RagPipeline:
    """Chunk → embed → store → retrieve, wired up from a :class:`PipelineConfig`.

    Example:
        >>> pipeline = RagPipeline.from_config(PipelineConfig())
        >>> pipeline.index(load_documents("examples/corpus"))       # doctest: +SKIP
        >>> hits = pipeline.retrieve("how do I rotate an API key?") # doctest: +SKIP
    """

    def __init__(
        self,
        config: PipelineConfig,
        chunker=None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self._chunks: list[Chunk] = []
        self._doc_count = 0

    # ------------------------------------------------------------------ build

    @classmethod
    def from_config(cls, config: PipelineConfig) -> RagPipeline:
        chunker = build_chunker(config.chunker, **config.chunker_args)
        embedder = build_embedder(
            config.embedder, cache=config.embedder_cache, **config.embedder_args
        )

        store_args = dict(config.store_args)
        if config.store == "qdrant" and "dim" not in store_args:
            # Qdrant needs the vector size up front; everything else infers it.
            store_args["dim"] = embedder.dim
        store = build_store(config.store, **store_args)

        return cls(config=config, chunker=chunker, embedder=embedder, store=store)

    def _build_retriever(self) -> Retriever:
        config = self.config
        dense = DenseRetriever(self.store, self.embedder)

        if config.retriever == "dense":
            retriever: Retriever = dense
        elif config.retriever == "bm25":
            retriever = BM25Retriever(**config.bm25_args)
            retriever.index(self._chunks)
        else:
            sparse = BM25Retriever(**config.bm25_args)
            sparse.index(self._chunks)
            retriever = HybridRetriever([dense, sparse], **config.hybrid_args)

        if config.rerank == "mmr":
            retriever = MMRReranker(
                retriever, self.embedder, store=self.store, **config.rerank_args
            )
        elif config.rerank == "cross-encoder":
            retriever = CrossEncoderReranker(retriever, **config.rerank_args)
        return retriever

    # ------------------------------------------------------------------ index

    def index(self, documents: Sequence[Document], batch_size: int = 256) -> RagPipeline:
        """Chunk, embed and store ``documents``, then build the retriever."""
        if not documents:
            raise ValueError("Cannot index an empty document list")

        self._chunks = self.chunker.split_all(documents)
        self._doc_count = len(documents)
        if not self._chunks:
            raise ValueError(
                "Chunking produced no chunks. Check that the documents have text "
                "and that the chunk size is not larger than every document."
            )

        if self.config.fit_embedder:
            # Corpus calibration (IDF) has to happen before anything is encoded.
            self.embedder.fit(chunk.text for chunk in self._chunks)

        for start in range(0, len(self._chunks), batch_size):
            batch = self._chunks[start : start + batch_size]
            vectors = self.embedder.encode([chunk.text for chunk in batch])
            self.store.add(batch, vectors)

        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.save()

        self.retriever = self._build_retriever()
        return self

    # --------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: MetadataFilter | None = None,
    ) -> RetrievalResult:
        """Retrieve the top ``k`` chunks, with wall-clock latency attached."""
        if self.retriever is None:
            raise RuntimeError("Pipeline is not indexed yet — call index() or load() first.")
        started = time.perf_counter()
        hits = self.retriever.retrieve(query, k=k, where=where)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(query=query, hits=hits, latency_ms=elapsed_ms)

    def search(self, query: str, k: int = 5, **kwargs) -> list:
        """Convenience alias returning just the hits."""
        return self.retrieve(query, k=k, **kwargs).hits

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Persist the config, chunks and vectors so ``load`` can skip re-indexing."""
        os.makedirs(path, exist_ok=True)
        self.store.persist()
        if hasattr(self.store, "save"):
            self.store.save(path)
        with open(os.path.join(path, _STATE_FILE), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": self.config.to_dict(),
                    "doc_count": self._doc_count,
                    "chunks": [chunk.to_dict() for chunk in self._chunks],
                },
                handle,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> RagPipeline:
        state_path = os.path.join(path, _STATE_FILE)
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"No index at {path!r} (missing {_STATE_FILE}). Run `ragforge index` first."
            )
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)

        config = PipelineConfig.from_dict(state["config"])
        store_args = dict(config.store_args)
        if config.store == "memory":
            store_args.setdefault("path", path)
        pipeline = cls.from_config(config.merged(store_args=store_args))
        pipeline._chunks = [Chunk.from_dict(item) for item in state.get("chunks", [])]
        pipeline._doc_count = int(state.get("doc_count", 0))

        if config.fit_embedder:
            # The IDF table is derived from the corpus, so recompute rather than
            # persist it — cheap, and it cannot go stale relative to the chunks.
            pipeline.embedder.fit(chunk.text for chunk in pipeline._chunks)
        if pipeline.store.count() == 0 and pipeline._chunks:
            vectors = pipeline.embedder.encode([chunk.text for chunk in pipeline._chunks])
            pipeline.store.add(pipeline._chunks, vectors)

        pipeline.retriever = pipeline._build_retriever()
        return pipeline

    # -------------------------------------------------------------------- misc

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def stats(self) -> dict[str, Any]:
        lengths = [len(chunk.text) for chunk in self._chunks]
        return {
            "config": self.config.name,
            "documents": self._doc_count,
            "chunks": len(self._chunks),
            "chunks_per_doc": round(len(self._chunks) / self._doc_count, 2)
            if self._doc_count
            else 0,
            "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "vectors": self.store.count() if self.store else 0,
            "dim": self.embedder.dim if self.embedder else 0,
            "retriever": self.retriever.describe() if self.retriever else None,
        }

    def describe(self) -> str:
        return (
            f"{self.config.label} | {self._doc_count} docs -> {len(self._chunks)} chunks"
        )
