"""Text embedders. The core install ships a dependency-free baseline."""

from .base import (
    CachedEmbedder,
    Embedder,
    available_embedders,
    build_embedder,
    l2_normalize,
    register_embedder,
)
from .hashing import HashingEmbedder, fnv1a
from .transformer import SentenceTransformerEmbedder

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "available_embedders",
    "build_embedder",
    "fnv1a",
    "l2_normalize",
    "register_embedder",
]
