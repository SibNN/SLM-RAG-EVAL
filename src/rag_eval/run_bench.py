"""Run RAG pipeline evaluation on given dataset."""

import argparse
from pathlib import Path

from src.rag_eval.evaluation_steps import (
    gen_answers,
    llm_as_judge,
    process_answers,
)
from src.rag_eval.metrics_aggregation import non_llm_metrics, show_llm_result
from src.services.vllm import VLLMRunner
from src.shared.load_config import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running evaluation on benchmark")
    parser.add_argument("input", type=Path, help="Path to the benchmark jsonl file")
    parser.add_argument("run", type=str, help="Name of run, name should be without dots and slashes")
    parser.add_argument(
        "--generator_type",
        type=str,
        choices=["api", "vllm"],
        default=None,
        help="Select generation backend: 'api' to use OpenAI-compatible API, "
        "or 'vllm' to run generation through a local vLLM server.",
    )
    parser.add_argument(
        "--judge_type",
        type=str,
        choices=["api", "vllm"],
        default="vllm",
        help="Select llm-as-a-judge backend: 'api' to use OpenAI-compatible API, "
        "or 'vllm' to run generation through a local vLLM server.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Name of judge model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_context", "golden_context"],
        default="golden_context",
        help="Name of the generation mode (no_context, golden_context)",
    )

    args = parser.parse_args()
    input = args.input
    run = args.run
    generator_type = args.generator_type
    judge_type = args.judge_type
    judge_model = args.judge_model
    mode = args.mode

    # Generate answers using pipeline
    if generator_type == "vllm":
        runner = VLLMRunner()
        runner.start(config["generator"])
        gen_answers.main(input, run, mode, generator_type)
        runner.stop()
    elif generator_type == "api":
        gen_answers.main(input, run, mode, generator_type)
    else:
        # Process generated answers
        process_answers.main(input, run)

    # Evaluate using LLM
    if judge_type == "vllm":
        runner = VLLMRunner()
        runner.start(config["llm_judge"])
        llm_as_judge.evaluate_answers(run, judge_model, judge_type)
        runner.stop()
    elif judge_type == "api":
        llm_as_judge.evaluate_answers(run, judge_model, judge_type)

    print("=" * 120)
    print(run)
    print("=" * 120)
    # Show LLM evaluation results and save them to csv
    show_llm_result.main(run)

    # Show non LLM metrics
    non_llm_metrics.main(run)
    print("=" * 120)
