"""Paths variables."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
BENCHMARK_FOLDER = DATA_FOLDER / "bench"
RAG_ANSWERS_FOLDER = BENCHMARK_FOLDER / "model_answered"
LLM_JUDGE_FOLDER = BENCHMARK_FOLDER / "llm_as_judge"
METRIC_RESULTS_FOLDER = BENCHMARK_FOLDER / "metric_results"

JUDGE_EVAL_FOLDER = DATA_FOLDER / "judge_eval"
