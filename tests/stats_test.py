from src.bench_eval.evaluation import stats


def test_simple() -> None:
    sentences = ["a", "b", "c"]
    statistics = stats(sentences)
    assert statistics["min"] == 1.0
    assert statistics["max"] == 1.0
    assert statistics["mean"] == 1.0
    assert statistics["median"] == 1.0


def test_big_words() -> None:
    sentences = ["aaaaaaaaaaaaaaaaa", "bd", "c", "e"]
    statistics = stats(sentences)
    assert statistics["min"] == 1.0
    assert statistics["max"] == 1.0
    assert statistics["mean"] == 1
    assert statistics["median"] == 1.0


def test_multiple_words() -> None:
    sentences = ["aaaa aaaaa aaaaa aaa", "b d", "c", "e"]
    statistics = stats(sentences)
    print(statistics)
    assert statistics["min"] == 1
    assert statistics["max"] == 4
    assert statistics["mean"] == 2
    assert statistics["median"] == 1.5
