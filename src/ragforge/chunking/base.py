"""Chunker interface plus the tokenisation helpers every strategy shares."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from ..types import Chunk, Document, stable_id

_WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")

# Splits on sentence-ending punctuation followed by whitespace, while trying not to
# break on common abbreviations or decimal numbers.
_SENTENCE_RE = re.compile(
    r"(?<![A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)"
    r"(?<!\bvs\.)(?<!\bNo\.)(?<!\bFig\.)(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])"
)


def word_tokens(text: str) -> list[str]:
    """Cheap, dependency-free tokeniser used as the default unit of length."""
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, falling back to line breaks for prose-free text."""
    blocks = [block for block in re.split(r"\n{2,}", text) if block.strip()]
    sentences: list[str] = []
    for block in blocks:
        parts = [part.strip() for part in _SENTENCE_RE.split(block) if part.strip()]
        sentences.extend(parts or [block.strip()])
    return sentences


def tiktoken_length(encoding: str = "cl100k_base") -> Callable[[str], int]:
    """Return a token-length function backed by ``tiktoken``.

    Kept optional on purpose: the default word tokeniser keeps the core install at
    ``numpy`` only, and chunk-size *relative* comparisons hold either way.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised only without tiktoken
        raise ImportError(
            "tiktoken is not installed. Run `pip install tiktoken` or leave "
            "`length_fn` unset to use the built-in word tokeniser."
        ) from exc

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text))


class Chunker(ABC):
    """Turns a :class:`Document` into retrievable :class:`Chunk` objects."""

    name: str = "chunker"

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunk strings. Implemented by each strategy."""

    def split(self, document: Document) -> list[Chunk]:
        """Split a document, attaching provenance metadata to every chunk."""
        chunks: list[Chunk] = []
        for position, piece in enumerate(self.split_text(document.text)):
            piece = piece.strip()
            if not piece:
                continue
            metadata = dict(document.metadata)
            metadata["chunker"] = self.name
            chunks.append(
                Chunk(
                    id=stable_id(document.id, position, piece),
                    doc_id=document.id,
                    text=piece,
                    position=position,
                    metadata=metadata,
                )
            )
        return chunks

    def split_all(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split(document))
        return chunks

    def describe(self) -> str:
        return self.name


_REGISTRY: dict[str, type[Chunker]] = {}


def register_chunker(cls: type[Chunker]) -> type[Chunker]:
    """Class decorator that makes a chunker constructible from config by name."""
    _REGISTRY[cls.name] = cls
    return cls


def available_chunkers() -> list[str]:
    return sorted(_REGISTRY)


def build_chunker(name: str, **kwargs) -> Chunker:
    """Instantiate a registered chunker, e.g. ``build_chunker("recursive", size=400)``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown chunker {name!r}. Available: {', '.join(available_chunkers())}")
    return _REGISTRY[name](**kwargs)


def merge_with_overlap(
    units: list[str],
    size: int,
    overlap: int,
    length_fn: Callable[[str], int],
    joiner: str = " ",
    min_size: int | None = None,
) -> list[str]:
    """Greedily pack ``units`` into windows of ``size``, sliding back by ``overlap``.

    Shared by the sentence and recursive strategies so overlap semantics stay identical
    across them — a difference there would quietly bias any A/B comparison.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    index = 0

    while index < len(units):
        unit = units[index]
        unit_len = length_fn(unit)

        if current and current_len + unit_len > size:
            windows.append(joiner.join(current))
            # Walk backwards to build the overlap tail for the next window.
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                previous_len = length_fn(previous)
                if tail_len + previous_len > overlap:
                    break
                tail.insert(0, previous)
                tail_len += previous_len
            if len(tail) == len(current):
                # The whole window fit inside the overlap budget, so carrying it
                # forward would rebuild the identical window and loop forever.
                # Dropping the tail costs one overlap and guarantees progress.
                tail, tail_len = [], 0
            current = tail
            current_len = tail_len
            continue

        current.append(unit)
        current_len += unit_len
        index += 1

    if current:
        text = joiner.join(current)
        if min_size and windows and length_fn(text) < min_size:
            windows[-1] = windows[-1] + joiner + text
        else:
            windows.append(text)

    return windows
