"""Okapi BM25, implemented directly on NumPy arrays.

Written out rather than pulled from a dependency for three reasons: it is about
sixty lines of real logic, it lets the tokeniser be shared with the rest of the
package, and BM25 is the baseline every dense retriever should have to beat —
worth being able to read the code that produces it.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence

import numpy as np

from ..stores.base import MetadataFilter, matches_filter
from ..types import Chunk, ScoredChunk
from .base import Retriever

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# A short, safe stoplist. Anything longer starts removing terms that carry signal in
# technical corpora ("no", "not", "all" all matter in a policy document).
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "will", "with",
})


def _simple_stem(token: str) -> str:
    """Conservative suffix stripping — plurals and common verb endings only.

    Deliberately not a full Porter stemmer: aggressive stemming conflates distinct
    technical terms, and on short corpora that costs more precision than the recall
    it buys.
    """
    if len(token) > 4:
        for suffix in ("ies",):
            if token.endswith(suffix):
                return token[: -len(suffix)] + "y"
        for suffix in ("sses", "ches", "shes", "xes"):
            if token.endswith(suffix):
                return token[:-2]
        for suffix in ("ing", "ed"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stem:
        tokens = [_simple_stem(token) for token in tokens]
    return tokens


class BM25Retriever(Retriever):
    """Sparse lexical retrieval with the Okapi BM25 weighting scheme.

    Args:
        k1: Term-frequency saturation. Higher lets repeated terms keep adding
            score; 1.2–1.5 is the usual range.
        b: Length normalisation, 0 disables it and 1 applies it fully.
    """

    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
        stem: bool = True,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._chunks: list[Chunk] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(0, dtype=np.float32)
        self._avgdl = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return tokenize(text, self.remove_stopwords, self.stem)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the inverted index. Replaces any previously indexed content."""
        self._chunks = list(chunks)
        n = len(self._chunks)
        if n == 0:
            self._postings, self._idf = {}, {}
            self._doc_len = np.zeros(0, dtype=np.float32)
            self._avgdl = 0.0
            return

        raw: dict[str, dict[int, int]] = {}
        lengths = np.zeros(n, dtype=np.float32)
        for position, chunk in enumerate(self._chunks):
            tokens = self._tokenize(chunk.text)
            lengths[position] = len(tokens)
            for token in tokens:
                raw.setdefault(token, {})
                raw[token][position] = raw[token].get(position, 0) + 1

        self._doc_len = lengths
        self._avgdl = float(lengths.mean()) if n else 0.0
        self._postings = {}
        self._idf = {}
        for term, postings in raw.items():
            doc_ids = np.fromiter(postings.keys(), dtype=np.int32, count=len(postings))
            term_freqs = np.fromiter(postings.values(), dtype=np.float32, count=len(postings))
            self._postings[term] = (doc_ids, term_freqs)
            document_frequency = len(postings)
            # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at a small
            # positive value so a term appearing in every chunk cannot score negative.
            idf = math.log(
                1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            self._idf[term] = max(idf, 1e-6)

    def scores_for(self, query: str) -> np.ndarray:
        """Raw BM25 score per indexed chunk. Exposed so fusion can reuse it."""
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        if not self._chunks or self._avgdl == 0:
            return scores
        norm = 1.0 - self.b + self.b * (self._doc_len / self._avgdl)
        for term in self._tokenize(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, term_freqs = posting
            denominator = term_freqs + self.k1 * norm[doc_ids]
            scores[doc_ids] += self._idf[term] * (term_freqs * (self.k1 + 1.0)) / denominator
        return scores

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        if not query.strip() or k <= 0 or not self._chunks:
            return []
        scores = self.scores_for(query)

        candidates = np.nonzero(scores > 0)[0]
        if where:
            candidates = np.array(
                [i for i in candidates if matches_filter(self._chunks[i].metadata, where)],
                dtype=np.int32,
            )
        if candidates.size == 0:
            return []

        top = candidates[np.argsort(-scores[candidates])[:k]]
        return [
            ScoredChunk(
                chunk=self._chunks[i],
                score=float(scores[i]),
                rank=position,
                source="bm25",
            )
            for position, i in enumerate(top, start=1)
        ]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bm25.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "remove_stopwords": self.remove_stopwords,
                    "stem": self.stem,
                    "chunks": [chunk.to_dict() for chunk in self._chunks],
                },
                handle,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> BM25Retriever:
        """Rebuild from disk.

        Only the chunks and parameters are persisted; the inverted index is
        recomputed on load. It is fast to rebuild and this keeps the on-disk
        format small and human-readable.
        """
        with open(os.path.join(path, "bm25.json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        retriever = cls(
            k1=payload.get("k1", 1.5),
            b=payload.get("b", 0.75),
            remove_stopwords=payload.get("remove_stopwords", True),
            stem=payload.get("stem", True),
        )
        retriever.index([Chunk.from_dict(item) for item in payload.get("chunks", [])])
        return retriever

    def describe(self) -> str:
        return f"bm25(k1={self.k1}, b={self.b})"
