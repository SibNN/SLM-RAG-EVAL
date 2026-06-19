import pytest

from src.rag_eval.metrics.non_llm import calculate_recall_at_k


def test_recall_at_k_basic() -> None:
    predicted = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    golden = [
        [2, 4],
        [6, 9],
    ]

    result = calculate_recall_at_k(predicted, golden, ks=[1, 2, 4])

    # query 1: recall@1 = 0/2, recall@2 = 1/2, recall@4 = 2/2
    # query 2: recall@1 = 0/2, recall@2 = 1/2, recall@4 = 1/2

    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 0.5
    assert result["recall@4"] == 0.75


def test_recall_at_k_full_recall() -> None:
    predicted = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    golden = [
        [1, 2],
        [4, 5],
    ]

    result = calculate_recall_at_k(predicted, golden, ks=[2])

    assert result["recall@2"] == 1.0


def test_recall_at_k_no_hits() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [10, 11],
    ]

    result = calculate_recall_at_k(predicted, golden, ks=[3])

    assert result["recall@3"] == 0.0


def test_recall_at_k_length_mismatch() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [1],
        [2],
    ]

    with pytest.raises(ValueError, match="Length of predicted and golden should match"):
        calculate_recall_at_k(predicted, golden)
