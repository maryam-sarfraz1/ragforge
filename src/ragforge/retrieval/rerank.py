"""Re-ranking stages that run over an already-retrieved candidate set."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..embedding.base import Embedder, l2_normalize
from ..stores.base import MetadataFilter, VectorStore
from ..types import ScoredChunk
from .base import Retriever


class MMRReranker(Retriever):
    """Maximal Marginal Relevance — trades a little relevance for less redundancy.

    Top-k by pure similarity often returns five near-identical chunks from the
    same section, which wastes context window and hides the one document that
    would have answered the question. MMR greedily picks the candidate that
    maximises ``λ · sim(query, c) − (1 − λ) · max sim(c, already_selected)``.

    ``lambda_mult=1.0`` is plain relevance; ``0.0`` is pure diversity. Values
    around 0.6–0.8 are usually where answer quality peaks.
    """

    name = "mmr"

    def __init__(
        self,
        base: Retriever,
        embedder: Embedder,
        lambda_mult: float = 0.7,
        candidate_multiplier: int = 4,
        store: VectorStore | None = None,
    ) -> None:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be in [0, 1]")
        self.base = base
        self.embedder = embedder
        self.lambda_mult = float(lambda_mult)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.store = store

    def index(self, chunks) -> None:
        self.base.index(chunks)

    def _vectors(self, hits: Sequence[ScoredChunk]) -> np.ndarray:
        if self.store is not None:
            try:
                vectors = self.store.vectors_for([hit.id for hit in hits])
                if vectors.shape[0] == len(hits):
                    return l2_normalize(vectors)
            except NotImplementedError:
                pass
        return l2_normalize(self.embedder.encode([hit.text for hit in hits]))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        candidates = self.base.retrieve(query, k=k * self.candidate_multiplier, where=where)
        if len(candidates) <= 1:
            return candidates[:k]

        vectors = self._vectors(candidates)
        query_vector = l2_normalize(self.embedder.encode_query(query).reshape(1, -1))[0]
        relevance = vectors @ query_vector
        similarity = vectors @ vectors.T

        selected: list[int] = [int(np.argmax(relevance))]
        remaining = set(range(len(candidates))) - set(selected)

        while len(selected) < min(k, len(candidates)) and remaining:
            pool = np.fromiter(remaining, dtype=np.int32, count=len(remaining))
            redundancy = similarity[np.ix_(pool, selected)].max(axis=1)
            mmr = self.lambda_mult * relevance[pool] - (1.0 - self.lambda_mult) * redundancy
            winner = int(pool[int(np.argmax(mmr))])
            selected.append(winner)
            remaining.discard(winner)

        return [
            ScoredChunk(
                chunk=candidates[i].chunk,
                score=float(relevance[i]),
                rank=position,
                source="mmr",
                components=candidates[i].components,
            )
            for position, i in enumerate(selected, start=1)
        ]

    def describe(self) -> str:
        return f"mmr(lambda={self.lambda_mult}) over {self.base.describe()}"


class CrossEncoderReranker(Retriever):
    """Second-stage re-ranking with a cross-encoder.

    A bi-encoder has to embed the query and the document independently, so it can
    never model their interaction. A cross-encoder reads the pair together and is
    reliably more accurate — and far too slow to run over a whole corpus, which is
    exactly why it belongs here, over the top ~50 candidates only.

    Install with ``pip install "ragforge[st]"``.
    """

    name = "cross-encoder"

    def __init__(
        self,
        base: Retriever,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_multiplier: int = 5,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.base = base
        self.model_id = model
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.batch_size = int(batch_size)
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "sentence-transformers is not installed. "
                    'Run `pip install "ragforge[st]"` to use the cross-encoder, or '
                    "use MMRReranker, which needs no extra dependencies."
                ) from exc
            self._model = CrossEncoder(self.model_id, device=self.device)
        return self._model

    def index(self, chunks) -> None:
        self.base.index(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        candidates = self.base.retrieve(query, k=k * self.candidate_multiplier, where=where)
        if not candidates:
            return []
        model = self._ensure_model()
        scores = model.predict(
            [(query, hit.text) for hit in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        order = np.argsort(-np.asarray(scores, dtype=np.float32))[:k]
        return [
            ScoredChunk(
                chunk=candidates[i].chunk,
                score=float(scores[i]),
                rank=position,
                source="cross-encoder",
                components={**candidates[i].components, "first_stage": candidates[i].score},
            )
            for position, i in enumerate(order, start=1)
        ]

    def describe(self) -> str:
        return f"cross-encoder({self.model_id}) over {self.base.describe()}"
