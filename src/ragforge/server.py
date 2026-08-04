"""Optional FastAPI service exposing a saved index over HTTP.

Install with ``pip install "ragforge[serve]"`` and run ``ragforge serve --index .ragforge``.
"""

from __future__ import annotations

from typing import Any

from .pipeline import RagPipeline


def create_app(index_path: str = ".ragforge"):
    """Build a FastAPI app serving the index at ``index_path``."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi import Query as QueryParam
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'fastapi is not installed. Run `pip install "ragforge[serve]"`.'
        ) from exc

    pipeline = RagPipeline.load(index_path)
    app = FastAPI(
        title="ragforge",
        version="0.1.0",
        description="Hybrid retrieval over a ragforge index.",
    )

    class Hit(BaseModel):
        chunk_id: str
        doc_id: str
        score: float
        rank: int
        text: str
        metadata: dict[str, Any] = {}

    class SearchResponse(BaseModel):
        query: str
        latency_ms: float
        hits: list[Hit]

    class SearchRequest(BaseModel):
        query: str
        k: int = 5
        where: dict[str, Any] | None = None

    def _run(query: str, k: int, where: dict[str, Any] | None) -> SearchResponse:
        if not query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if k < 1 or k > 100:
            raise HTTPException(status_code=400, detail="k must be between 1 and 100")
        result = pipeline.retrieve(query, k=k, where=where)
        return SearchResponse(
            query=query,
            latency_ms=round(result.latency_ms, 2),
            hits=[
                Hit(
                    chunk_id=hit.id,
                    doc_id=hit.doc_id,
                    score=round(hit.score, 6),
                    rank=hit.rank,
                    text=hit.text,
                    metadata=hit.chunk.metadata,
                )
                for hit in result.hits
            ],
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **pipeline.stats()}

    @app.get("/search", response_model=SearchResponse)
    def search_get(q: str, k: int = QueryParam(5, ge=1, le=100)) -> SearchResponse:
        return _run(q, k, None)

    @app.post("/search", response_model=SearchResponse)
    def search_post(request: SearchRequest) -> SearchResponse:
        return _run(request.query, request.k, request.where)

    return app


def serve(index_path: str = ".ragforge", host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'uvicorn is not installed. Run `pip install "ragforge[serve]"`.'
        ) from exc
    uvicorn.run(create_app(index_path), host=host, port=port)
