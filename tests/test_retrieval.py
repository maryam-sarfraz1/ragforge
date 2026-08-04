from __future__ import annotations

import pytest

from ragforge.embedding import HashingEmbedder
from ragforge.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MMRReranker,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from ragforge.retrieval.bm25 import tokenize
from ragforge.stores import InMemoryStore
from ragforge.types import Chunk, ScoredChunk

CORPUS = [
    ("c0", "auth", "Rotate an API key by creating a replacement and revoking the old one."),
    ("c1", "limits", "Throttled requests return status 429 with the code ERR_4029."),
    ("c2", "hooks", "Verify the webhook signature with an HMAC over the raw body."),
    ("c3", "hooks", "Failed webhook deliveries retry five times with exponential backoff."),
    ("c4", "billing", "Invoices are generated on the first of the month."),
]


@pytest.fixture
def chunks():
    return [
        Chunk(id=cid, doc_id=doc, text=text, position=i, metadata={"doc": doc})
        for i, (cid, doc, text) in enumerate(CORPUS)
    ]


@pytest.fixture
def dense(chunks):
    embedder = HashingEmbedder(dim=512).fit([c.text for c in chunks])
    store = InMemoryStore()
    store.add(chunks, embedder.encode([c.text for c in chunks]))
    return DenseRetriever(store, embedder), store, embedder


# ------------------------------------------------------------------------ bm25


def test_tokenize_removes_stopwords_and_stems():
    tokens = tokenize("The requests are retrying and the deliveries failed")
    assert "the" not in tokens and "are" not in tokens
    assert "request" in tokens and "delivery" in tokens


def test_stemmer_leaves_short_and_irregular_words_alone():
    assert tokenize("status", remove_stopwords=False) == ["status"]
    assert tokenize("gas", remove_stopwords=False) == ["gas"]


def test_bm25_finds_an_exact_identifier(chunks):
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.retrieve("ERR_4029", k=3)
    assert hits[0].id == "c1"


def test_bm25_ranks_are_dense_and_ordered(chunks):
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.retrieve("webhook retry backoff", k=5)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_bm25_returns_nothing_for_out_of_vocabulary_queries(chunks):
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert retriever.retrieve("zzzz nonexistent term", k=5) == []


def test_bm25_scores_are_positive(chunks):
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert all(hit.score > 0 for hit in retriever.retrieve("webhook", k=5))


def test_bm25_respects_a_metadata_filter(chunks):
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.retrieve("webhook", k=5, where={"doc": "hooks"})
    assert hits and all(hit.doc_id == "hooks" for hit in hits)


def test_bm25_handles_an_empty_index():
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.retrieve("anything", k=3) == []


def test_bm25_survives_a_save_load_round_trip(chunks, tmp_path):
    retriever = BM25Retriever(k1=1.3, b=0.6)
    retriever.index(chunks)
    retriever.save(str(tmp_path))

    reloaded = BM25Retriever.load(str(tmp_path))
    assert reloaded.k1 == 1.3 and reloaded.b == 0.6
    assert reloaded.retrieve("ERR_4029", k=1)[0].id == "c1"


def test_longer_documents_are_not_unfairly_favoured():
    """Length normalisation: padding a document with filler must not raise its score."""
    short = Chunk(id="s", doc_id="s", text="webhook signature")
    padded = Chunk(id="l", doc_id="l", text="webhook signature " + "filler word " * 60)
    retriever = BM25Retriever()
    retriever.index([short, padded])
    hits = retriever.retrieve("webhook signature", k=2)
    assert hits[0].id == "s"


# ----------------------------------------------------------------------- dense


def test_dense_retriever_returns_ranked_hits(dense):
    retriever, _, _ = dense
    hits = retriever.retrieve("how do I rotate an api key", k=3)
    assert hits[0].doc_id == "auth"
    assert all(hit.source == "dense" for hit in hits)


def test_dense_retriever_ignores_an_empty_query(dense):
    retriever, _, _ = dense
    assert retriever.retrieve("   ", k=3) == []


# ---------------------------------------------------------------------- fusion


def _list(ids, source, start=1.0):
    return [
        ScoredChunk(chunk=Chunk(id=i, doc_id=i, text=i), score=start - n, rank=n + 1, source=source)
        for n, i in enumerate(ids)
    ]


def test_rrf_promotes_a_document_both_lists_agree_on():
    """Consensus is the property RRF actually buys: appearing in both lists beats
    appearing once, even when the single appearance is at a better rank."""
    fused = reciprocal_rank_fusion([_list(["a", "b"], "x"), _list(["b"], "y")])
    assert fused[0].id == "b"


def test_rrf_favours_the_extremes_over_the_middle():
    """1/(k+rank) is convex, so rank 1 + rank 3 outscores rank 2 + rank 2. This is
    inherent to RRF rather than a quirk of this implementation."""
    fused = reciprocal_rank_fusion([_list(["a", "b", "c"], "x"), _list(["c", "b", "a"], "y")])
    assert fused[-1].id == "b"


def test_rrf_scores_match_the_formula():
    fused = reciprocal_rank_fusion([_list(["a"], "x")], smoothing=60)
    assert fused[0].score == pytest.approx(1.0 / 61.0)


def test_rrf_honours_weights():
    lists = [_list(["a", "b"], "x"), _list(["b", "a"], "y")]
    assert reciprocal_rank_fusion(lists, weights=[5.0, 1.0])[0].id == "a"
    assert reciprocal_rank_fusion(lists, weights=[1.0, 5.0])[0].id == "b"


def test_rrf_records_each_retriever_contribution():
    fused = reciprocal_rank_fusion([_list(["a"], "dense"), _list(["a"], "bm25")])
    assert set(fused[0].components) == {"dense", "bm25"}


def test_rrf_reranks_consecutively_from_one():
    fused = reciprocal_rank_fusion([_list(["a", "b", "c"], "x")])
    assert [hit.rank for hit in fused] == [1, 2, 3]


def test_weighted_fusion_normalises_incomparable_scales():
    dense_hits = _list(["a", "b"], "dense", start=0.9)          # ~0.9, -0.1
    sparse_hits = [                                              # BM25-scale scores
        ScoredChunk(chunk=Chunk(id="b", doc_id="b", text="b"), score=25.0, rank=1, source="bm25"),
        ScoredChunk(chunk=Chunk(id="a", doc_id="a", text="a"), score=1.0, rank=2, source="bm25"),
    ]
    fused = weighted_score_fusion([dense_hits, sparse_hits])
    assert {hit.id for hit in fused} == {"a", "b"}
    assert all(0.0 <= hit.score <= 2.0 for hit in fused)


def test_fusion_rejects_a_weight_count_mismatch():
    with pytest.raises(ValueError, match="one entry per result list"):
        reciprocal_rank_fusion([_list(["a"], "x")], weights=[1.0, 2.0])


# ---------------------------------------------------------------------- hybrid


def test_hybrid_recovers_both_lexical_and_semantic_hits(chunks, dense):
    dense_retriever, _, _ = dense
    sparse = BM25Retriever()
    sparse.index(chunks)
    hybrid = HybridRetriever([dense_retriever, sparse])

    assert hybrid.retrieve("ERR_4029", k=3)[0].id == "c1"
    assert hybrid.retrieve("how do I rotate an api key", k=3)[0].doc_id == "auth"


def test_hybrid_reports_which_retriever_found_each_hit(chunks, dense):
    dense_retriever, _, _ = dense
    sparse = BM25Retriever()
    sparse.index(chunks)
    hybrid = HybridRetriever([dense_retriever, sparse])
    contributions = hybrid.contributions("webhook signature", k=2)
    assert contributions
    assert all(sources for _, sources in contributions)


def test_hybrid_requires_at_least_one_retriever():
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridRetriever([])


def test_hybrid_rejects_an_unknown_fusion(chunks, dense):
    dense_retriever, _, _ = dense
    with pytest.raises(ValueError, match="Unknown fusion"):
        HybridRetriever([dense_retriever], fusion="magic")


def test_hybrid_returns_at_most_k(chunks, dense):
    dense_retriever, _, _ = dense
    sparse = BM25Retriever()
    sparse.index(chunks)
    assert len(HybridRetriever([dense_retriever, sparse]).retrieve("webhook", k=2)) <= 2


# ------------------------------------------------------------------------- mmr


def test_mmr_diversifies_near_duplicate_results():
    duplicates = [
        Chunk(id=f"dup{i}", doc_id="same", text="webhook signature verification hmac")
        for i in range(4)
    ]
    outlier = Chunk(id="other", doc_id="other", text="webhook retry backoff schedule")
    all_chunks = duplicates + [outlier]

    embedder = HashingEmbedder(dim=256).fit([c.text for c in all_chunks])
    store = InMemoryStore()
    store.add(all_chunks, embedder.encode([c.text for c in all_chunks]))
    base = DenseRetriever(store, embedder)

    plain = {hit.doc_id for hit in base.retrieve("webhook", k=2)}
    reranker = MMRReranker(base, embedder, lambda_mult=0.3)
    diverse = {hit.doc_id for hit in reranker.retrieve("webhook", k=2)}
    assert len(diverse) >= len(plain)


def test_mmr_with_lambda_one_keeps_the_top_relevance_hit(chunks, dense):
    retriever, store, embedder = dense
    reranked = MMRReranker(retriever, embedder, lambda_mult=1.0, store=store)
    assert reranked.retrieve("rotate api key", k=1)[0].doc_id == "auth"


def test_mmr_validates_lambda(dense):
    retriever, _, embedder = dense
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MMRReranker(retriever, embedder, lambda_mult=1.5)
