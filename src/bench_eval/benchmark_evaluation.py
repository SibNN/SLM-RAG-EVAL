"""Main benchmark evaluation module."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.progress import track

from src.bench_eval.eval_conf import EVALUATORS
from src.model.vllm import VllmModel
from src.shared.schemas import EvalTestConfig


def evaluate(benchmark: pd.DataFrame, config: dict[str, EvalTestConfig]) -> dict:
    """Evaluate benchmark with given config.

    Runs different tests mentioned in the config.

    Args:
        benchmark: pandas DataFrame with the following columns:
            - question: str
            - answer: str
            - context: str
            - dataset: str
        config: mapping of evaluator name to EvalTestConfig

    Returns:
        dict: metrics from each test

    """
    model = VllmModel()
    results = {}

    for name, params in track(config.items()):
        func = params.func
        cols_list = params.columns
        use_model = params.LLM

        for columns in cols_list:
            key = f"{name}({', '.join(columns) if isinstance(columns, tuple) else columns})"

            if isinstance(columns, str):
                data = benchmark[columns].astype(str).tolist()
                results[key] = func(model, data) if use_model else func(data)
            elif isinstance(columns, tuple):
                data = [benchmark[col].astype(str).tolist() for col in columns]
                results[key] = func(model, *data) if use_model else func(*data)
            else:
                raise ValueError("not supported type of columns")

    return results


def save_results(results: dict, out: Path) -> None:
    """Save evaluation results to the json file.

    Args:
        results (dict): evaluation results
        out (Path): save path

    """
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Evaluate benchmark on different metrics."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="input path to benchmark",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path for results",
        required=True,
    )

    args = parser.parse_args()
    source = Path(args.input)
    bench = pd.read_json(source, lines=True)
    results = evaluate(bench, EVALUATORS)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
