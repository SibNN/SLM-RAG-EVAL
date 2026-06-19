import pytest

from src.rag_eval.metrics.non_llm import calculate_mrr


def test_mrr_basic() -> None:
    predicted = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    golden = [
        [2],
        [6],
    ]

    # ranks: 2 -> 1/2, 3 -> 1/3
    expected_mrr = (1 / 2 + 1 / 3) / 2

    result = calculate_mrr(predicted, golden)

    assert result == pytest.approx(expected_mrr)


def test_mrr_multiple_relevant() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [3, 1],  # первый релевантный — rank 1
    ]

    result = calculate_mrr(predicted, golden)

    assert result == 1.0


def test_mrr_no_relevant() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [10],
    ]

    result = calculate_mrr(predicted, golden)

    assert result == 0.0


def test_mrr_length_mismatch() -> None:
    predicted = [
        [1, 2, 3],
    ]
    golden = [
        [1],
        [2],
    ]

    with pytest.raises(ValueError, match="Length of predicted and golden should match"):
        calculate_mrr(predicted, golden)
