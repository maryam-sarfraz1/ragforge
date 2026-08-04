"""Vector store backends. All of them return cosine similarity, higher is better."""

from .base import (
    MetadataFilter,
    VectorStore,
    available_stores,
    build_store,
    flatten_metadata,
    matches_filter,
    register_store,
)
from .chroma import ChromaStore
from .memory import InMemoryStore
from .qdrant import QdrantStore

__all__ = [
    "ChromaStore",
    "InMemoryStore",
    "MetadataFilter",
    "QdrantStore",
    "VectorStore",
    "available_stores",
    "build_store",
    "flatten_metadata",
    "matches_filter",
    "register_store",
]
