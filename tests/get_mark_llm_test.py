from src.bench_eval.evaluation import get_mark_llm


def test_correct() -> None:
    llm_answer = """{"mark": 2.0}"""
    mark = get_mark_llm(llm_answer)
    assert mark == 2.0


def test_non_float() -> None:
    llm_answer = """{"mark": 2}"""
    mark = get_mark_llm(llm_answer)
    assert mark == 2.0


def test_bad_json() -> None:
    llm_answer = """{"mark: 2}"""
    mark = get_mark_llm(llm_answer)
    assert mark is None
