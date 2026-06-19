import numpy as np
import pytest

from src.bench_eval.evaluation import ngram_diversity


def test_simple() -> None:
    sentences = ["a"]
    statistics = ngram_diversity(sentences)
    assert statistics["mean"] == 1.0


def test_basic() -> None:
    texts = ["a b a"]
    result = ngram_diversity(texts, n_gram=2)

    # unigrams: [a, b, a] -> unique = {a, b} => 2 / 3
    assert pytest.approx(result[1]) == 2 / 3

    # bigrams: [(a,b), (b,a)] -> unique = 2 / 2
    assert result[2] == 1.0

    expected_mean = np.mean([result[1], result[2]])
    assert result["mean"] == expected_mean


def test_multiple_texts() -> None:
    texts = ["a b c", "a b c"]
    result = ngram_diversity(texts, n_gram=2)

    # unigrams: 6 total, 3 unique
    assert result[1] == 3 / 6

    # bigrams: 4 total, 2 unique
    assert result[2] == 2 / 4
