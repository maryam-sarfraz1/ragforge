"""A dependency-free embedder built on the hashing trick.

Why ship this at all when real sentence encoders exist? Two reasons:

1. **Hermetic tests.** CI runs the full pipeline — index, retrieve, evaluate —
   without downloading a single model weight.
2. **An honest baseline.** A TF-IDF-weighted hashed bag of n-grams is a genuinely
   strong lexical baseline. If a 400 MB transformer cannot beat it on your corpus,
   that is a finding worth having, and this makes the comparison one flag away.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

from .base import Embedder, l2_normalize, register_embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a(text: str) -> int:
    """FNV-1a 64-bit.

    Python's built-in ``hash()`` is salted per process, which would make vectors
    non-reproducible across runs — the one thing an index must never be.
    """
    digest = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    return digest


@register_embedder
class HashingEmbedder(Embedder):
    """Hashed n-gram TF-IDF vectors.

    Args:
        dim: Output dimensionality. Higher means fewer hash collisions.
        word_ngrams: Word n-gram orders to emit, e.g. ``(1, 2)`` for unigrams
            and bigrams.
        char_ngrams: Character n-gram order applied within each word. Adds
            robustness to morphology and typos; set to 0 to disable.
        sublinear_tf: Use ``1 + log(tf)`` instead of raw counts, which stops one
            repeated word from dominating a chunk's vector.
    """

    name = "hashing"

    def __init__(
        self,
        dim: int = 512,
        word_ngrams: Sequence[int] = (1, 2),
        char_ngrams: int = 4,
        sublinear_tf: bool = True,
        use_idf: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = int(dim)
        self.word_ngrams = tuple(int(n) for n in word_ngrams if int(n) > 0)
        self.char_ngrams = int(char_ngrams)
        self.sublinear_tf = bool(sublinear_tf)
        self.use_idf = bool(use_idf)
        self._idf: np.ndarray | None = None
        self._idf_tag = "raw"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return (
            f"hashing:{self._dim}:w{'.'.join(map(str, self.word_ngrams))}"
            f":c{self.char_ngrams}:tf{int(self.sublinear_tf)}:{self._idf_tag}"
        )

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features: list[str] = []
        for order in self.word_ngrams:
            if order == 1:
                features.extend(tokens)
                continue
            for i in range(len(tokens) - order + 1):
                features.append("_".join(tokens[i : i + order]))
        if self.char_ngrams > 0:
            n = self.char_ngrams
            for token in tokens:
                if len(token) <= n:
                    features.append(f"#{token}#")
                    continue
                padded = f"#{token}#"
                for i in range(len(padded) - n + 1):
                    features.append(padded[i : i + n])
        return features

    def _counts(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for feature in self._features(text):
            digest = fnv1a(feature)
            index = digest % self._dim
            # One bit of the hash picks the sign, which keeps collisions from
            # systematically inflating a dimension.
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vector[index] += sign
        return vector

    def fit(self, corpus: Iterable[str]) -> HashingEmbedder:
        """Learn IDF weights from the corpus. Optional but usually worth it."""
        if not self.use_idf:
            return self
        document_frequency = np.zeros(self._dim, dtype=np.float64)
        total = 0
        signature = hashlib.sha1()
        for text in corpus:
            total += 1
            signature.update(text.encode("utf-8", "ignore"))
            seen = {fnv1a(feature) % self._dim for feature in self._features(text)}
            for index in seen:
                document_frequency[index] += 1.0
        if total == 0:
            return self
        self._idf = np.log((total + 1.0) / (document_frequency + 1.0)).astype(np.float32) + 1.0
        self._idf_tag = f"idf{total}-{signature.hexdigest()[:8]}"
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        matrix = np.vstack([self._counts(text) for text in texts])
        if self.sublinear_tf:
            matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
        if self._idf is not None:
            matrix = matrix * self._idf
        return l2_normalize(matrix)

    def describe(self) -> str:
        return f"hashing(dim={self._dim}, idf={self._idf is not None})"
