"""LLM as judge description module."""

import json
from logging import getLogger
from pathlib import Path
from typing import Literal

from dacite import from_dict
from deepeval.models.llms.local_model import LocalModel
from deepeval.test_case import LLMTestCase
from rich.progress import track

from src.rag_eval.metrics.llm import get_llm_metrics
from src.shared.load_config import config
from src.shared.schemas import RAGAnswerRow

logger = getLogger(__name__)


class LLMEvaluator:
    """Evaluate RAG pipeline with LLM as judge method.

    Attributes:
        local_model: deepeval model that compatible to deepeval evaluation library

    """

    def __init__(self, judge_model: str, judge_type: Literal["vllm", "api"]) -> None:
        """Initialize evaluator with openai compatible model."""
        if judge_type == "vllm":
            model = config["llm_judge"]["name"]
            base_url = f"http://{config['llm_judge']['host']}:{config['llm_judge']['port']}/v1"
            api_key = "None"
            enable_thinking = config["llm_judge"]["thinking"]
        else:
            model = judge_model or config["api_llm_judge"]["name"]
            base_url = config["api_llm_judge"]["base_url"]
            api_key = config["api_llm_judge"]["api_key"]
            enable_thinking = config["api_llm_judge"]["thinking"]

        local_model = LocalModel(
            model=model,
            base_url=base_url,
            api_key=api_key,
            generation_kwargs={
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    },
                },
            },
        )
        self.metrics = get_llm_metrics(local_model)

    def run_eval(self, test_cases: list[LLMTestCase], out_jsonl: Path) -> None:
        """Evaluate RAG pipeline answers with LLM-as-judge and save to JSONL.

        Args:
            test_cases: parsed test cases
            out_jsonl: path to save evaluation results (.jsonl)

        """
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with out_jsonl.open("w", encoding="utf-8") as f:
            for case in track(test_cases, total=len(test_cases)):
                result = {
                    "question_type": case.additional_metadata.get("question_type", "unknown"),
                    "dataset": case.additional_metadata.get("dataset", "unkwnown"),
                    "input": case.input,
                    "actual_output": case.actual_output,
                    "expected_output": case.expected_output,
                    "retrieval_context": case.retrieval_context,
                    "metrics": {},
                }

                max_tries = 10

                for metric_name in self.metrics:
                    metric = self.metrics[metric_name]

                    for attempt in range(max_tries):
                        try:
                            metric.measure(case, _show_indicator=False)
                            break
                        except Exception as e:
                            logger.debug(
                                "Metric failed metric=%s attempt=%d error=%s",
                                metric_name,
                                attempt,
                                e,
                            )

                    result["metrics"][metric_name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                    }

                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def json_to_testcases(self, generated_answers: Path) -> list[LLMTestCase]:
        """Prepare llm answers for evaluation from flat JSONL file.

        Args:
            generated_answers: path to the .jsonl file

        Returns:
            list[LLMTestCase]

        """
        test_cases: list[LLMTestCase] = []

        with generated_answers.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    test_case = self.dict_row_to_testcase(row)
                    test_cases.append(test_case)

                except Exception as e:
                    raise RuntimeError(
                        f"Failed parsing JSONL at line {line_num}: {e}",
                    ) from e

        return test_cases

    def list_row_to_testcases(self, rows: list[dict]) -> list[LLMTestCase]:
        """Prepare llm answers for evaluation from loaded dict.

        Args:
            rows: list of dicts

        Returns:
            list[LLMTestCase]

        """
        test_cases: list[LLMTestCase] = []

        for row in rows:
            test_case = self.dict_row_to_testcase(row)
            test_cases.append(test_case)

        return test_cases

    def dict_row_to_testcase(self, row: dict) -> LLMTestCase:
        """Load LLMTestCase from given dict.

        Args:
            row: dict with case

        Returns:
            LLMTestCase: loaded case

        """
        rag_answer = from_dict(data_class=RAGAnswerRow, data=row)

        test_case = LLMTestCase(
            input=rag_answer.question,
            actual_output=rag_answer.answer,
            expected_output=rag_answer.golden_answer,
            retrieval_context=rag_answer.doc_contents,
            additional_metadata={"question_type": rag_answer.question_type, "dataset": rag_answer.dataset},
        )
        return test_case
