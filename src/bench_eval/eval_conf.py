"""Benchmark evaluation config, setting params and functions to evaluate."""

from src.bench_eval.evaluation import (
    hs_evaluation,
    intra_inter_diversity,
    llm_rank_answer,
    llm_rank_question,
    ngram_diversity,
    question_type_stats,
    stats,
)
from src.shared.schemas import EvalTestConfig

EVALUATORS: dict[str, EvalTestConfig] = {
    "mean_hs": EvalTestConfig(
        func=hs_evaluation,
        columns=["question", "answer"],
        LLM=False,
    ),
    "domain_diversity": EvalTestConfig(
        func=intra_inter_diversity,
        columns=[("context", "dataset")],
        LLM=False,
    ),
    "types_diversity": EvalTestConfig(
        func=intra_inter_diversity,
        columns=[("context", "question_type")],
        LLM=False,
    ),
    "NGramDiversity": EvalTestConfig(
        func=ngram_diversity,
        columns=["context", "question", "answer"],
        LLM=False,
    ),
    "stats": EvalTestConfig(
        func=stats,
        columns=["question", "answer"],
        LLM=False,
    ),
    "rank_question": EvalTestConfig(
        func=llm_rank_question,
        columns=["question"],
        LLM=True,
    ),
    "matching_answer": EvalTestConfig(
        func=llm_rank_answer,
        columns=[("question", "answer")],
        LLM=True,
    ),
    "class_support": EvalTestConfig(
        func=question_type_stats,
        columns=[("dataset", "question_type")],
        LLM=False,
    ),
}
