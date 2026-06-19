"""Concurrent calls of LLM as judge."""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.rag_eval.evaluation_steps.llm_as_judge import evaluate_answers as eval_multithread
from src.rag_eval.llm_judge.local_judge import LLMEvaluator
from src.services.vllm import VLLMRunner
from src.shared.load_config import config
from src.shared.paths import JUDGE_EVAL_FOLDER

USE_VLLM = True
ONE_JUDGE_ONE_OUT = True  # Whether should judge evaluate only once or multiple times 👧👧🍵


logger = logging.getLogger(__name__)


def launch_judge(dataset_path: Path, save_path: Path) -> None:
    """Evaluate given json input and save results to output.

    Args:
        dataset_path: input path to the jsonl file
        save_path: output path to the jsonl file

    """
    llmeval = LLMEvaluator()
    test_cases = llmeval.json_to_testcases(dataset_path)
    llmeval.run_eval(test_cases, save_path)


def _run_single(dataset: Path, save_path: Path) -> None:
    """Single parallel execution wrapper.

    Args:
        dataset: source json/jsonl file
        save_path: path to the evaluation results

    """
    try:
        launch_judge(
            dataset,
            save_path,
        )
    except Exception:
        raise


def evaluate_answers(
    dataset: Path,
    run_name: str,
    conc: int = 32,
) -> None:
    """Run same evaluation multiple times in parallel.

    Output structure:
        JUDGE_EVAL_FOLDER/<run_name>/<worker_id>.jsonl

    Args:
        dataset: name of the path to the dataset
        run_name: name of run folder
        conc: number of parallel executions
            defaults to 32

    """
    logger.info(
        "Starting parallel evaluation run_name=%s concurrency=%d input=%s",
        run_name,
        conc,
        dataset,
    )

    if ONE_JUDGE_ONE_OUT:
        out_file = JUDGE_EVAL_FOLDER / f"{run_name}.jsonl"
        eval_multithread(run_name, dataset_path=dataset, output_path=out_file)
        return

    base_dir = JUDGE_EVAL_FOLDER / run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="judge") as executor:
        futures = [executor.submit(_run_single, dataset, base_dir / f"{i}.jsonl") for i in range(conc)]

        for future in as_completed(futures):
            future.result()

    logger.info("All workers finished run_name=%s", run_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the mixed dataset",
    )
    parser.add_argument(
        "name",
        type=Path,
        help="output run name, will be stored in a LLM_JUDGE_FOLDER as folder",
    )
    args = parser.parse_args()

    dataset = Path(args.input)
    run_name = args.name
    if USE_VLLM:
        runner = VLLMRunner()
        runner.start(config["llm_judge"])

        evaluate_answers(
            dataset=dataset,
            run_name=run_name,
            conc=32,
        )

        runner.stop()
    else:
        evaluate_answers(
            dataset=dataset,
            run_name=run_name,
            conc=32,
        )
