"""Print evaluation summary to the stdout."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.rag_eval.metrics_aggregation.aggregation import (
    compute_aggregates,
    print_table,
    save_tables,
)
from src.shared.paths import LLM_JUDGE_FOLDER, METRIC_RESULTS_FOLDER


def load_rows(file_path: Path, metric_names: list[str]) -> tuple[list[dict], dict[str, list[float]]]:
    """Load jsonl and extract metrics.

    Args:
        file_path: path to the jsonl file with evaluation results
        metric_names: names of the key metrics

    Returns:
        tuple with loaded json and metrics for each sample

    """
    rows = []
    all_metrics: dict[str, list[float]] = defaultdict(list)

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)

            metrics = {}

            for metric_name in metric_names:
                metric_data = raw.get("metrics", {}).get(metric_name)

                if metric_data and metric_data.get("score") is not None:
                    score = float(metric_data["score"])
                    metrics[metric_name] = score
                    all_metrics[metric_name].append(score)

            row = {
                **raw,
                "_metrics": metrics,
            }

            rows.append(row)

    return rows, all_metrics


def process_jsonl(
    file_path: Path,
    metric_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Process given json with predefined fields.

    Args:
        file_path: path to the llm evaluation results file
        metric_names: names of metrics
        save_path: output path

    """
    rows, all_metrics = load_rows(file_path, metric_names)

    tables = {}

    for field in ["question_type", "dataset"]:
        table = compute_aggregates(
            rows,
            metric_names,
            group_field=field,
            all_metrics=all_metrics,
        )

        tables[field] = table

        print_table(table, f"Results by {field.upper()}")

    if save_path:
        save_tables(save_path, tables, mode="w")


def main(name: str) -> None:
    """Print evaluation results.

    Args:
        name: name of evaluation run

    """
    result_dir = METRIC_RESULTS_FOLDER
    result_dir.mkdir(parents=True, exist_ok=True)

    jsonl_file = LLM_JUDGE_FOLDER / f"{name}.jsonl"
    save_file = result_dir / f"{name}.jsonl"

    metric_names = [
        "Correctness",
        "ContextRelevance",
        "Faithfulness",
        "AnswerRelevance",
    ]

    process_jsonl(jsonl_file, metric_names, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Displaying and saving LLM-as-judge metrics",
    )

    parser.add_argument(
        "name",
        type=str,
        help="name of output file",
    )

    args = parser.parse_args()

    main(args.name)
