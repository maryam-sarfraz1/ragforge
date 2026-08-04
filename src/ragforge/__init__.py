"""ragforge — hybrid retrieval with a built-in evaluation harness.

Quick start::

    from ragforge import PipelineConfig, RagPipeline, load_documents

    pipeline = RagPipeline.from_config(PipelineConfig(chunker="sentence", retriever="hybrid"))
    pipeline.index(load_documents("examples/corpus"))
    for hit in pipeline.search("how do I rotate an API key?", k=3):
        print(hit.rank, hit.doc_id, hit.text[:80])

Then measure it, which is the part most stacks skip::

    from ragforge import evaluate, load_queries

    result = evaluate(pipeline, load_queries("examples/evalset.jsonl"))
    print(result.metrics["recall@5"])
"""

from .chunking import (
    Chunker,
    FixedTokenChunker,
    MarkdownSectionChunker,
    RecursiveCharacterChunker,
    SentenceWindowChunker,
    build_chunker,
)
from .config import PipelineConfig, expand_grid, load_grid
from .embedding import Embedder, HashingEmbedder, SentenceTransformerEmbedder, build_embedder
from .eval import EvalResult, compare, evaluate, run_sweep
from .loaders import load_documents, load_queries, validate_eval_set
from .pipeline import RagPipeline
from .retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    MMRReranker,
    Retriever,
    reciprocal_rank_fusion,
)
from .stores import ChromaStore, InMemoryStore, QdrantStore, VectorStore, build_store
from .types import Chunk, Document, Query, RetrievalResult, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "ChromaStore",
    "Chunk",
    "Chunker",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Document",
    "Embedder",
    "EvalResult",
    "FixedTokenChunker",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryStore",
    "MMRReranker",
    "MarkdownSectionChunker",
    "PipelineConfig",
    "QdrantStore",
    "Query",
    "RagPipeline",
    "RecursiveCharacterChunker",
    "RetrievalResult",
    "Retriever",
    "ScoredChunk",
    "SentenceTransformerEmbedder",
    "SentenceWindowChunker",
    "VectorStore",
    "__version__",
    "build_chunker",
    "build_embedder",
    "build_store",
    "compare",
    "evaluate",
    "expand_grid",
    "load_documents",
    "load_grid",
    "load_queries",
    "reciprocal_rank_fusion",
    "run_sweep",
    "validate_eval_set",
]
