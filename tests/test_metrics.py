"""Metric tests use worked examples with hand-computed expected values.

A metric that is merely self-consistent is worthless — the point is that these
numbers match what the textbook definition produces.
"""

from __future__ import annotations

import math

import pytest

from ragforge.eval.metrics import (
    average_precision,
    dcg,
    hit_rate_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = ["b", "e", "z"]  # "z" was never retrieved


def test_recall_counts_unretrieved_relevant_documents():
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 2)


def test_hit_rate_is_binary():
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_average_precision_matches_the_worked_example():
    # Hits at ranks 2 and 5 → precisions 1/2 and 2/5, averaged over 3 relevant docs.
    expected = (0.5 + 0.4) / 3
    assert average_precision(RANKED, RELEVANT) == pytest.approx(expected)


def test_dcg_applies_the_log2_discount():
    # gains 3, 2 at ranks 1, 2 → 3/log2(2) + 2/log2(3)
    assert dcg([3.0, 2.0]) == pytest.approx(3.0 + 2.0 / math.log2(3))


def test_ndcg_is_one_for_a_perfect_ranking():
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], lambda d: grades.get(d, 0.0), 3, grades) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking():
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    reversed_score = ndcg_at_k(["c", "b", "a"], lambda d: grades.get(d, 0.0), 3, grades)
    assert 0.0 < reversed_score < 1.0


def test_ndcg_ideal_uses_all_known_grades_not_just_retrieved_ones():
    grades = {"a": 3.0, "unretrieved": 3.0}
    # Retrieving only "a" cannot score 1.0 when another highly relevant doc was missed.
    assert ndcg_at_k(["a"], lambda d: grades.get(d, 0.0), 2, grades) < 1.0


def test_metrics_handle_empty_inputs():
    assert recall_at_k([], ["a"], 5) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0
    assert precision_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0
    assert ndcg_at_k([], lambda d: 0.0, 5) == 0.0


def test_duplicates_are_collapsed_before_scoring():
    """Two chunks from the same document must not count as two retrieved documents."""
    assert precision_at_k(["a", "a", "b"], ["b"], 2) == pytest.approx(1 / 2)


@pytest.mark.parametrize(
    "q,expected",
    [(0, 1.0), (50, 3.0), (100, 5.0), (25, 2.0)],
)
def test_percentile_interpolates(q, expected):
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], q) == pytest.approx(expected)


def test_percentile_of_a_single_value():
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) == 0.0
