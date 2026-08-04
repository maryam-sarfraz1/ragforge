"""Concrete chunking strategies.

Chunking is the single highest-leverage knob in a retrieval stack and the one
most often set by folklore. Each strategy here is deliberately small so that
``ragforge sweep`` can put numbers on the folklore.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .base import (
    Chunker,
    merge_with_overlap,
    register_chunker,
    split_sentences,
    word_tokens,
)


@register_chunker
class FixedTokenChunker(Chunker):
    """Fixed-width sliding window over tokens.

    The baseline everyone starts with. Fast and predictable, but happily cuts
    sentences — and often facts — in half.
    """

    name = "fixed"

    def __init__(
        self,
        size: int = 256,
        overlap: int = 32,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.tokenizer = tokenizer or word_tokens

    def split_text(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        if not tokens:
            return []
        step = self.size - self.overlap
        windows: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.size]
            if not window:
                break
            windows.append(" ".join(window))
            if start + self.size >= len(tokens):
                break
        return windows

    def describe(self) -> str:
        return f"fixed(size={self.size}, overlap={self.overlap})"


@register_chunker
class RecursiveCharacterChunker(Chunker):
    """Split on the most semantic separator that still fits the budget.

    Tries paragraph breaks first, then line breaks, sentences, and finally raw
    characters — so a chunk boundary lands on a heading or a blank line whenever
    the text gives us that option.
    """

    name = "recursive"

    DEFAULT_SEPARATORS: Sequence[str] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")

    def __init__(
        self,
        size: int = 800,
        overlap: int = 120,
        separators: Sequence[str] | None = None,
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.separators = tuple(separators) if separators else tuple(self.DEFAULT_SEPARATORS)
        self.length_fn = length_fn or len

    def _split_recursive(self, text: str, separators: Sequence[str]) -> list[str]:
        if self.length_fn(text) <= self.size:
            return [text]
        if not separators:
            return [text]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            # Last resort: hard character slices.
            return [text[i : i + self.size] for i in range(0, len(text), self.size)]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_recursive(text, rest)

        # Re-attach the separator so headings and punctuation survive the split.
        pieces = [parts[0]] + [separator + part for part in parts[1:]]
        out: list[str] = []
        for piece in pieces:
            if self.length_fn(piece) <= self.size:
                out.append(piece)
            else:
                out.extend(self._split_recursive(piece, rest))
        return out

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        pieces = [p for p in self._split_recursive(text, self.separators) if p.strip()]
        return merge_with_overlap(
            pieces,
            size=self.size,
            overlap=self.overlap,
            length_fn=self.length_fn,
            joiner="",
            min_size=max(1, self.size // 8),
        )

    def describe(self) -> str:
        return f"recursive(size={self.size}, overlap={self.overlap})"


@register_chunker
class SentenceWindowChunker(Chunker):
    """Pack whole sentences up to a token budget, never mid-sentence.

    Usually the best precision/recall trade-off on prose, because every chunk is
    a self-contained statement rather than a fragment.
    """

    name = "sentence"

    def __init__(
        self,
        size: int = 180,
        overlap: int = 40,
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.length_fn = length_fn or (lambda text: len(word_tokens(text)))

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        return merge_with_overlap(
            sentences,
            size=self.size,
            overlap=self.overlap,
            length_fn=self.length_fn,
            joiner=" ",
            min_size=max(1, self.size // 6),
        )

    def describe(self) -> str:
        return f"sentence(size={self.size}, overlap={self.overlap})"


@register_chunker
class MarkdownSectionChunker(Chunker):
    """One chunk per markdown section, with the heading trail prepended.

    Prepending ``# Handbook > ## Deployment`` to the body gives an embedding model
    the topic words it would otherwise have to guess, which measurably helps on
    short, keyword-like queries.
    """

    name = "markdown"

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(self, max_size: int = 1200, overlap: int = 100, include_trail: bool = True):
        self.max_size = max_size
        self.overlap = overlap
        self.include_trail = include_trail
        self._fallback = RecursiveCharacterChunker(size=max_size, overlap=overlap)

    def split_text(self, text: str) -> list[str]:
        lines = text.splitlines()
        trail: list[str] = []
        buffer: list[str] = []
        sections: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if not body:
                return
            if self.include_trail and trail:
                header = " > ".join(trail)
                body = f"{header}\n\n{body}"
            if len(body) > self.max_size:
                sections.extend(self._fallback.split_text(body))
            else:
                sections.append(body)

        for line in lines:
            match = self._HEADING_RE.match(line)
            if match:
                flush()
                buffer = []
                level = len(match.group(1))
                title = match.group(2).strip()
                trail = trail[: level - 1]
                while len(trail) < level - 1:
                    trail.append("")
                trail.append(title)
            else:
                buffer.append(line)
        flush()
        return [section for section in sections if section.strip()]

    def describe(self) -> str:
        return f"markdown(max_size={self.max_size})"
