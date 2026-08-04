"""Core data types shared across chunking, storage, retrieval and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def stable_id(*parts: Any) -> str:
    """Deterministic short id, so re-indexing the same corpus yields the same ids."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    """A source document before it is split into chunks."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document.id must be a non-empty string")

    @classmethod
    def from_text(cls, text: str, **metadata: Any) -> Document:
        return cls(id=stable_id(text), text=text, metadata=dict(metadata))


@dataclass
class Chunk:
    """A retrievable unit of text carved out of a :class:`Document`."""

    id: str
    doc_id: str
    text: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            id=payload["id"],
            doc_id=payload["doc_id"],
            text=payload["text"],
            position=int(payload.get("position", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class ScoredChunk:
    """A chunk returned by a retriever, with the score that got it there.

    ``score`` is only comparable within a single retriever's result list —
    BM25 scores and cosine similarities live on different scales, which is
    exactly why :class:`~ragforge.retrieval.hybrid.HybridRetriever` fuses on
    rank rather than on raw score.
    """

    chunk: Chunk
    score: float
    rank: int = 0
    source: str = ""
    components: dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


@dataclass
class Query:
    """An evaluation query and the judgements attached to it."""

    id: str
    text: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    grades: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def grade_of(self, doc_id: str) -> float:
        """Graded relevance for ``doc_id`` (0.0 when the doc is not judged relevant)."""
        if doc_id in self.grades:
            return float(self.grades[doc_id])
        return 1.0 if doc_id in self.relevant_doc_ids else 0.0

    @property
    def judged_ids(self) -> list[str]:
        seen = list(self.relevant_doc_ids)
        for key, grade in self.grades.items():
            if grade > 0 and key not in seen:
                seen.append(key)
        return seen

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Query:
        grades = {str(k): float(v) for k, v in (payload.get("grades") or {}).items()}
        relevant = [str(x) for x in (payload.get("relevant_doc_ids") or [])]
        if not relevant and grades:
            relevant = [k for k, v in grades.items() if v > 0]
        return cls(
            id=str(payload.get("id") or stable_id(payload["query"])),
            text=payload.get("query") or payload["text"],
            relevant_doc_ids=relevant,
            grades=grades,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "query": self.text,
            "relevant_doc_ids": self.relevant_doc_ids,
        }
        if self.grades:
            out["grades"] = self.grades
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class RetrievalResult:
    """Everything a single ``retrieve()`` call produced, including timing."""

    query: str
    hits: list[ScoredChunk]
    latency_ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __getitem__(self, index: int) -> ScoredChunk:
        return self.hits[index]

    @property
    def doc_ids(self) -> list[str]:
        """Ranked doc ids, de-duplicated, preserving first-seen order."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.doc_id not in seen:
                seen.append(hit.doc_id)
        return seen

    @property
    def chunk_ids(self) -> list[str]:
        return [hit.id for hit in self.hits]

    def top(self, k: int) -> list[ScoredChunk]:
        return self.hits[:k]

    def as_context(self, k: int | None = None, separator: str = "\n\n---\n\n") -> str:
        """Concatenate the top hits into a prompt-ready context block."""
        hits = self.hits if k is None else self.hits[:k]
        return separator.join(hit.text for hit in hits)
