"""Reading corpora and evaluation sets off disk."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any

from .types import Document, Query

TEXT_EXTENSIONS = (".md", ".markdown", ".txt", ".rst")


def _iter_files(root: str, extensions: Sequence[str]) -> Iterator[str]:
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.lower().endswith(tuple(extensions)):
                yield os.path.join(directory, filename)


def _front_matter(text: str) -> tuple:
    """Split a minimal ``key: value`` front-matter block off the top of a document."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end], text[end + 4 :]
    metadata: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body.lstrip("\n")


def load_documents(
    path: str,
    extensions: Sequence[str] = TEXT_EXTENSIONS,
    id_field: str = "id",
    text_field: str = "text",
) -> list[Document]:
    """Load documents from a directory of text files or a JSONL file.

    Directory mode uses the path relative to ``path`` (without extension) as the
    document id, so ids stay stable and readable — which matters because the
    evaluation set refers to documents by id.
    """
    if os.path.isfile(path):
        return _load_jsonl_documents(path, id_field=id_field, text_field=text_field)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such corpus path: {path}")

    documents: list[Document] = []
    for filepath in _iter_files(path, extensions):
        with open(filepath, encoding="utf-8") as handle:
            raw = handle.read()
        metadata, body = _front_matter(raw)
        relative = os.path.relpath(filepath, path).replace(os.sep, "/")
        doc_id = str(metadata.pop("id", None) or os.path.splitext(relative)[0])
        metadata.setdefault("source", relative)
        documents.append(Document(id=doc_id, text=body.strip(), metadata=metadata))

    if not documents:
        raise ValueError(
            f"No documents found under {path!r} with extensions {', '.join(extensions)}"
        )
    return documents


def _load_jsonl_documents(path: str, id_field: str, text_field: str) -> list[Document]:
    documents: list[Document] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if text_field not in payload:
                raise ValueError(f"{path}:{line_number} has no {text_field!r} field")
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in (id_field, text_field)
            }
            documents.append(
                Document(
                    id=str(payload.get(id_field) or f"doc-{line_number}"),
                    text=str(payload[text_field]),
                    metadata=metadata,
                )
            )
    return documents


def load_queries(path: str) -> list[Query]:
    """Load an evaluation set from JSONL (one query per line) or a JSON list."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        raise ValueError(f"{path} is empty")

    if text.startswith("["):
        payload = json.loads(text)
    else:
        payload = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc

    queries = [Query.from_dict(item) for item in payload]
    unjudged = [query.id for query in queries if not query.judged_ids]
    if unjudged:
        raise ValueError(
            f"{len(unjudged)} queries in {path} have no relevance judgements "
            f"(e.g. {unjudged[0]!r}). Every query needs at least one relevant doc id."
        )
    return queries


def validate_eval_set(queries: Sequence[Query], documents: Sequence[Document]) -> list[str]:
    """Report judgements that point at documents the corpus does not contain.

    A silent id typo here makes recall look broken for reasons that have nothing
    to do with retrieval, and it is a genuinely easy mistake to make by hand.
    """
    known = {document.id for document in documents}
    problems: list[str] = []
    for query in queries:
        for doc_id in query.judged_ids:
            if doc_id not in known:
                problems.append(f"query {query.id!r} references unknown doc {doc_id!r}")
    return problems


def write_queries(queries: Sequence[Query], path: str, indent: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_dict(), ensure_ascii=False, indent=indent) + "\n")
