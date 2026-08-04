from __future__ import annotations

import numpy as np
import pytest

from ragforge.embedding import CachedEmbedder, HashingEmbedder, build_embedder
from ragforge.embedding.hashing import fnv1a
from ragforge.stores import InMemoryStore
from ragforge.stores.base import matches_filter
from ragforge.types import Chunk

TEXTS = [
    "rotate an api key before it expires",
    "rate limits return status 429",
    "webhook signatures use hmac sha256",
]


def _chunks(n=3):
    return [
        Chunk(id=f"c{i}", doc_id=f"d{i}", text=TEXTS[i % len(TEXTS)], position=i,
              metadata={"category": "even" if i % 2 == 0 else "odd", "n": i})
        for i in range(n)
    ]


# ------------------------------------------------------------------- embedding


def test_fnv1a_is_deterministic_and_process_independent():
    # The literal is the FNV-1a 64-bit digest of "a"; a mismatch means the hash
    # changed and every previously built index is silently invalid.
    assert fnv1a("a") == 0xAF63DC4C8601EC8C
    assert fnv1a("hello") == fnv1a("hello")


def test_embeddings_are_l2_normalised():
    vectors = HashingEmbedder(dim=128).encode(TEXTS)
    assert vectors.shape == (3, 128)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic():
    a = HashingEmbedder(dim=64).encode(TEXTS)
    b = HashingEmbedder(dim=64).encode(TEXTS)
    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder(dim=512)
    vectors = embedder.encode(["rotate an api key", "api key rotation steps", "backup schedule"])
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_fit_changes_the_fingerprint_so_caches_invalidate():
    embedder = HashingEmbedder(dim=64)
    before = embedder.fingerprint
    embedder.fit(TEXTS)
    assert embedder.fingerprint != before


def test_encoding_an_empty_batch_returns_an_empty_matrix():
    assert HashingEmbedder(dim=32).encode([]).shape == (0, 32)


def test_cached_embedder_round_trips_through_disk(tmp_path):
    inner = HashingEmbedder(dim=64)
    cache = CachedEmbedder(inner, path=str(tmp_path / "cache"))
    first = cache.encode(TEXTS)
    cache.save()

    reloaded = CachedEmbedder(HashingEmbedder(dim=64), path=str(tmp_path / "cache"))
    assert np.allclose(reloaded.encode(TEXTS), first)


def test_cached_embedder_tolerates_a_corrupt_cache(tmp_path):
    path = tmp_path / "cache"
    cache = CachedEmbedder(HashingEmbedder(dim=32), path=str(path))
    cache.encode(TEXTS)
    cache.save()
    # Truncate the cache file; a bad cache must never break a run.
    next(path.glob("*.npz")).write_bytes(b"not an npz")
    assert CachedEmbedder(HashingEmbedder(dim=32), path=str(path)).encode(TEXTS).shape == (3, 32)


def test_build_embedder_rejects_unknown_names():
    with pytest.raises(KeyError, match="Unknown embedder"):
        build_embedder("nope")


# ---------------------------------------------------------------------- stores


def test_store_add_and_search_returns_the_nearest_chunk():
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=256)
    chunks = _chunks()
    store.add(chunks, embedder.encode([chunk.text for chunk in chunks]))

    hits = store.search(embedder.encode_query("api key rotation"), k=2)
    assert len(hits) == 2
    assert hits[0][0].id == "c0"
    assert hits[0][1] >= hits[1][1]


def test_store_scores_are_cosine_similarities():
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=256)
    chunks = _chunks()
    store.add(chunks, embedder.encode([chunk.text for chunk in chunks]))
    for _, score in store.search(embedder.encode_query("api key"), k=3):
        assert -1.0001 <= score <= 1.0001


def test_re_adding_the_same_chunk_overwrites_rather_than_duplicates():
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=64)
    chunks = _chunks(2)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store.add(chunks, vectors)
    store.add(chunks, vectors)
    assert store.count() == 2


def test_search_rejects_a_dimension_mismatch():
    store = InMemoryStore()
    chunks = _chunks(1)
    store.add(chunks, HashingEmbedder(dim=64).encode([chunks[0].text]))
    with pytest.raises(ValueError, match="different embedder"):
        store.search(np.zeros(128, dtype=np.float32), k=1)


def test_add_rejects_mismatched_vector_count():
    store = InMemoryStore()
    with pytest.raises(ValueError, match="row for row"):
        store.add(_chunks(3), np.zeros((2, 64), dtype=np.float32))


def test_metadata_filter_restricts_results():
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=128)
    chunks = _chunks(4)
    store.add(chunks, embedder.encode([chunk.text for chunk in chunks]))
    hits = store.search(embedder.encode_query("api key"), k=4, where={"category": "odd"})
    assert hits
    assert all(chunk.metadata["category"] == "odd" for chunk, _ in hits)


def test_store_persists_and_reloads(tmp_path):
    embedder = HashingEmbedder(dim=64)
    chunks = _chunks(3)
    store = InMemoryStore()
    store.add(chunks, embedder.encode([chunk.text for chunk in chunks]))
    store.save(str(tmp_path / "idx"))

    reloaded = InMemoryStore(path=str(tmp_path / "idx"))
    assert reloaded.count() == 3
    assert reloaded.get(["c1"])[0].text == chunks[1].text
    assert reloaded.search(embedder.encode_query("api key"), k=1)


def test_empty_store_searches_cleanly():
    assert InMemoryStore().search(np.zeros(8, dtype=np.float32), k=5) == []


def test_vectors_for_returns_matching_rows():
    store = InMemoryStore()
    chunks = _chunks(3)
    store.add(chunks, HashingEmbedder(dim=32).encode([chunk.text for chunk in chunks]))
    assert store.vectors_for(["c0", "c2"]).shape == (2, 32)


@pytest.mark.parametrize(
    "where,expected",
    [
        ({"n": 2}, True),
        ({"n": 3}, False),
        ({"n": {"$gte": 2}}, True),
        ({"n": {"$lt": 2}}, False),
        ({"n": {"$in": [1, 2]}}, True),
        ({"n": {"$nin": [2]}}, False),
        ({"category": {"$ne": "odd"}}, True),
        (None, True),
    ],
)
def test_filter_operators(where, expected):
    assert matches_filter({"n": 2, "category": "even"}, where) is expected


def test_unknown_filter_operator_raises():
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        matches_filter({"n": 1}, {"n": {"$regex": ".*"}})
