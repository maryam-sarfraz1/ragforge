"""Embedder interface, on-disk caching, and the model registry."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np


def l2_normalize(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation, so cosine similarity reduces to a dot product."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, epsilon)


class Embedder(ABC):
    """Maps text to a dense vector space.

    Implementations must return **L2-normalised** float32 rows so every vector
    store in this package can treat inner product and cosine as the same thing.
    """

    name: str = "embedder"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a ``(len(texts), dim)`` float32 matrix."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def encode_query(self, text: str) -> np.ndarray:
        """Hook for asymmetric models that prefix queries differently to passages."""
        return self.encode_one(text)

    def fit(self, corpus: Iterable[str]) -> Embedder:
        """Optional corpus-level calibration. No-op for pretrained models."""
        return self

    @property
    def fingerprint(self) -> str:
        """Identity used for cache keys — must change when output vectors change."""
        return f"{self.name}:{self.dim}"

    def describe(self) -> str:
        return self.name


class CachedEmbedder(Embedder):
    """Memoises embeddings on disk, keyed by model fingerprint plus text hash.

    Sweeps re-encode the same corpus once per configuration; without this, a grid
    of a dozen configs pays the encoder cost a dozen times over.
    """

    name = "cached"

    def __init__(self, inner: Embedder, path: str = ".ragforge/embed-cache") -> None:
        self.inner = inner
        self.path = path
        self._memory: dict[str, np.ndarray] = {}
        self._loaded = False
        self._dirty = False

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def _cache_file(self) -> str:
        digest = hashlib.sha1(self.fingerprint.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.path, f"{digest}.npz")

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        target = self._cache_file()
        if not os.path.exists(target):
            return
        try:
            with np.load(target, allow_pickle=False) as payload:
                keys = json.loads(str(payload["keys"]))
                vectors = payload["vectors"]
            self._memory = {key: vectors[i] for i, key in enumerate(keys)}
        except (OSError, ValueError, KeyError):
            # A corrupt or partially written cache is never worth failing a run over.
            self._memory = {}

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        keys = [self._key(text) for text in texts]
        missing = [i for i, key in enumerate(keys) if key not in self._memory]
        if missing:
            fresh = self.inner.encode([texts[i] for i in missing])
            for slot, i in enumerate(missing):
                self._memory[keys[i]] = np.asarray(fresh[slot], dtype=np.float32)
            self._dirty = True
        return np.vstack([self._memory[key] for key in keys]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        # Queries are one-off; skip the cache round-trip.
        return self.inner.encode_query(text)

    def fit(self, corpus: Iterable[str]) -> CachedEmbedder:
        self.inner.fit(corpus)
        # Calibration changes the fingerprint, which invalidates anything cached.
        self._memory = {}
        self._loaded = False
        return self

    def save(self) -> None:
        if not self._dirty or not self._memory:
            return
        os.makedirs(self.path, exist_ok=True)
        keys = list(self._memory)
        vectors = np.vstack([self._memory[key] for key in keys]).astype(np.float32)
        np.savez_compressed(self._cache_file(), keys=json.dumps(keys), vectors=vectors)
        self._dirty = False

    def describe(self) -> str:
        return f"cached({self.inner.describe()})"


_REGISTRY: dict[str, type[Embedder]] = {}


def register_embedder(cls: type[Embedder]) -> type[Embedder]:
    _REGISTRY[cls.name] = cls
    return cls


def available_embedders() -> list[str]:
    return sorted(_REGISTRY)


def build_embedder(name: str, cache: str | None = None, **kwargs) -> Embedder:
    """Instantiate a registered embedder, optionally wrapped in a disk cache."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown embedder {name!r}. Available: {', '.join(available_embedders())}")
    embedder = _REGISTRY[name](**kwargs)
    if cache:
        return CachedEmbedder(embedder, path=cache)
    return embedder
