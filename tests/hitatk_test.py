import pytest

from src.rag_eval.metrics.non_llm import calculate_hit_at_k


def test_hit_at_k_basic() -> None:
    predicted = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    golden = [
        [3],
        [10],
    ]

    result = calculate_hit_at_k(predicted, golden, ks=[1, 3, 5])

    assert result["hit@1"] == 0.0
    assert result["hit@3"] == 0.5
    assert result["hit@5"] == 0.5


def test_hit_at_k_multiple_relevant() -> None:
    predicted = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    golden = [
        [2, 4],
        [6, 7],
    ]

    result = calculate_hit_at_k(predicted, golden, ks=[1, 2])

    assert result["hit@1"] == 0.0
    assert result["hit@2"] == 1.0


def test_hit_at_k_no_hits() -> None:
    predicted = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    golden = [
        [10],
        [11],
    ]

    result = calculate_hit_at_k(predicted, golden, ks=[3])

    assert result["hit@3"] == 0.0


def test_hit_at_k_length_mismatch() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [1],
        [2],
    ]

    with pytest.raises(ValueError, match="Length of predicted and golden should match"):
        calculate_hit_at_k(predicted, golden)
