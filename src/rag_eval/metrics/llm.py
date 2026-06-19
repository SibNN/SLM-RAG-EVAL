"""LLM metrics description."""

from deepeval.metrics import GEval
from deepeval.models.llms.local_model import LocalModel
from deepeval.test_case import LLMTestCaseParams


def get_llm_metrics(local_model: LocalModel) -> dict[str, GEval]:
    """Get metrics for llm evaluation.

    Args:
        local_model: model to evaluate on

    Returns:
        dict[str, GEval]: metrics to evaluate on

    """
    metrics: dict[str, GEval] = {}
    correctness = GEval(
        name="Correctness",
        criteria="""Determine whether the actual output is factually correct based on the expected output.
                    It's fine, if actual output is more detailed""",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=local_model,
    )
    answer_relevance = GEval(
        name="AnswerRelevance",
        criteria="Determine if given answer is relevant to the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=local_model,
    )
    context_relevance = GEval(
        name="ContextRelevance",
        criteria="Determine if given retrieval context is relevant to the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=local_model,
    )
    faithfulness = GEval(
        name="Faithfulness",
        criteria="Determine if statements in given actual output are based on retrieved context and not groundless",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=local_model,
    )
    metrics["Correctness"] = correctness
    metrics["AnswerRelevance"] = answer_relevance
    metrics["ContextRelevance"] = context_relevance
    metrics["Faithfulness"] = faithfulness
    return metrics
