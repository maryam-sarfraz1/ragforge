from __future__ import annotations

import pytest

from ragforge.chunking import (
    FixedTokenChunker,
    MarkdownSectionChunker,
    RecursiveCharacterChunker,
    SentenceWindowChunker,
    available_chunkers,
    build_chunker,
    split_sentences,
    word_tokens,
)
from ragforge.chunking.base import merge_with_overlap
from ragforge.types import Document

PROSE = (
    "Rotation is a two step process. First create a replacement key with the same "
    "scopes. Then revoke the original once traffic has moved. Revocation applies "
    "within thirty seconds across every region. A revoked key cannot be restored."
)


def test_word_tokens_keeps_hyphenated_words():
    assert word_tokens("well-known e-mail") == ["well-known", "e-mail"]


def test_split_sentences_does_not_break_on_abbreviations():
    text = "Use e.g. a backoff. Then retry. See Fig. 2 for the curve."
    assert len(split_sentences(text)) == 3


def test_split_sentences_falls_back_to_blocks_without_punctuation():
    assert split_sentences("no terminator here") == ["no terminator here"]


@pytest.mark.parametrize("name", available_chunkers())
def test_every_chunker_produces_non_empty_chunks(name):
    document = Document(id="d", text=PROSE)
    chunks = build_chunker(name).split(document)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.doc_id == "d" for chunk in chunks)


@pytest.mark.parametrize("name", available_chunkers())
def test_chunk_ids_are_stable_across_runs(name):
    document = Document(id="d", text=PROSE)
    first = build_chunker(name).split(document)
    second = build_chunker(name).split(document)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_chunk_positions_are_sequential():
    chunks = SentenceWindowChunker(size=20, overlap=5).split(Document(id="d", text=PROSE))
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


def test_fixed_chunker_respects_size():
    chunker = FixedTokenChunker(size=10, overlap=2)
    for text in chunker.split_text(PROSE):
        assert len(word_tokens(text)) <= 10


def test_fixed_chunker_overlaps_consecutive_windows():
    chunks = FixedTokenChunker(size=10, overlap=4).split_text(PROSE)
    first_tail = word_tokens(chunks[0])[-4:]
    second_head = word_tokens(chunks[1])[:4]
    assert first_tail == second_head


def test_sentence_chunker_never_splits_mid_sentence():
    chunks = SentenceWindowChunker(size=25, overlap=5).split_text(PROSE)
    for chunk in chunks:
        assert chunk.strip().endswith((".", "!", "?"))


def test_recursive_chunker_prefers_paragraph_boundaries():
    text = "para one is here.\n\npara two is here.\n\npara three is here."
    chunks = RecursiveCharacterChunker(size=30, overlap=0).split_text(text)
    assert len(chunks) >= 2
    assert all("para" in chunk for chunk in chunks)


def test_markdown_chunker_prepends_the_heading_trail():
    text = "# Handbook\n\nintro line\n\n## Deployment\n\ndeploy details here\n"
    chunks = MarkdownSectionChunker().split_text(text)
    deployment = [chunk for chunk in chunks if "deploy details" in chunk]
    assert deployment and "Handbook > Deployment" in deployment[0]


def test_markdown_chunker_falls_back_when_a_section_is_too_long():
    text = "# Big\n\n" + ("sentence filler here. " * 200)
    chunks = MarkdownSectionChunker(max_size=300).split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 600 for chunk in chunks)


def test_empty_text_produces_no_chunks():
    for name in available_chunkers():
        assert build_chunker(name).split(Document(id="d", text="   ")) == []


@pytest.mark.parametrize(
    "chunker,kwargs",
    [(FixedTokenChunker, {}), (SentenceWindowChunker, {}), (RecursiveCharacterChunker, {})],
)
def test_overlap_must_be_smaller_than_size(chunker, kwargs):
    with pytest.raises(ValueError):
        chunker(size=10, overlap=10, **kwargs)


def test_build_chunker_rejects_unknown_names():
    with pytest.raises(KeyError, match="Unknown chunker"):
        build_chunker("does-not-exist")


def test_merge_with_overlap_terminates_when_a_unit_exceeds_the_budget():
    """Regression: a unit that fits entirely inside the overlap budget used to make
    the packer rebuild the same window forever."""
    units = ["a", "b", "c" * 50, "d"]
    windows = merge_with_overlap(
        units, size=10, overlap=8, length_fn=len, joiner=" ", min_size=None
    )
    assert windows
    assert any("c" * 50 in window for window in windows)


def test_merge_with_overlap_keeps_every_unit():
    units = [f"unit{i}" for i in range(20)]
    windows = merge_with_overlap(units, size=20, overlap=5, length_fn=len, joiner=" ")
    joined = " ".join(windows)
    assert all(unit in joined for unit in units)
