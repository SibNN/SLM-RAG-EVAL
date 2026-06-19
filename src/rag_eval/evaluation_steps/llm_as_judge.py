"""Module for starting LLM as judge evaluation."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from src.rag_eval.llm_judge.local_judge import LLMEvaluator
from src.shared.paths import LLM_JUDGE_FOLDER, RAG_ANSWERS_FOLDER


def main(run_name: str, save_in_diff: str | None = None) -> None:
    """Evaluate RAG pipeline on LLM based methods.

    Args:
        run_name: name of the {run_name}.jsonl file with generated answers
            in the generated answers directory
        save_in_diff: name of test run

    """
    result_dir = LLM_JUDGE_FOLDER
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / f"{save_in_diff}.jsonl" if save_in_diff else result_dir / f"{run_name}.jsonl"
    dataset_path = RAG_ANSWERS_FOLDER / f"{run_name}.jsonl"

    llmeval = LLMEvaluator()
    test_cases = llmeval.json_to_testcases(dataset_path)
    llmeval.run_eval(test_cases, save_path)


def eval_part(
    part: list[tuple[int, dict]],
    run_name: str,
    idx: int,
    judge_model: str | None = None,
    judge_type: Literal["vllm", "api"] = "vllm",
) -> Path:
    """Evaluate one partition and save to jsonl.

    Returns:
        Path to saved file

    """
    thread_dir = LLM_JUDGE_FOLDER / run_name
    thread_dir.mkdir(parents=True, exist_ok=True)

    save_path = thread_dir / f"{idx:03d}.jsonl"
    llmeval = LLMEvaluator(judge_model, judge_type)
    # convert rows → testcases
    test_cases = llmeval.list_row_to_testcases([row for _, row in part])
    # run evaluation
    llmeval.run_eval(test_cases, save_path)  # assume returns list[dict]

    return save_path


def evaluate_answers(  # noqa: PLR0913
    run_name: str,
    judge_model: str | None = None,
    judge_type: Literal["vllm", "api"] = "vllm",
    conc: int = 32,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """Parallel evaluation with merge preserving original order.

    Args:
        run_name: name of the run with corresponding rag answers
        judge_model: name of judge model
        judge_type: llm-as-a-judge backend - 'api' to use OpenAI-compatible API,
            or 'vllm' to run generation through a local vLLM server
        conc: number of concurrent threads
        dataset_path: optional path to the dataset bypassing default folders
        output_path: optional path the the result bypassing default folders

    """
    input_path = dataset_path or RAG_ANSWERS_FOLDER / f"{run_name}.jsonl"

    base_dir = LLM_JUDGE_FOLDER / run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    indexed_rows: list[tuple[int, dict]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                indexed_rows.append((idx, json.loads(line)))

    total = len(indexed_rows)

    chunk_size = (total + conc - 1) // conc

    parts = [indexed_rows[i : i + chunk_size] for i in range(0, total, chunk_size)]

    part_paths: list[Path] = []

    with ThreadPoolExecutor(max_workers=conc) as executor:
        futures = {
            executor.submit(eval_part, part, run_name, i, judge_model, judge_type): i for i, part in enumerate(parts)
        }

        for future in as_completed(futures):
            part_paths.append(future.result())

    merged = []
    for path in sorted(part_paths):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))

    final_path = output_path or base_dir.parent / f"{run_name}.jsonl"

    with final_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    shutil.rmtree(base_dir)

    print(f"Merged file saved to: {final_path}")
