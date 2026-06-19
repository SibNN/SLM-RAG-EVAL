"""Metric aggregation utils."""

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tabulate import tabulate


def compute_aggregates(
    rows: Iterable[dict],
    metric_names: list[str],
    group_field: str,
    all_metrics: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Compute aggregates by arbitrary field.

    Args:
        rows: list of dicts with evaluation sample
        metric_names: list of names of metrics
        group_field: name of aggregation field to group by
        all_metrics: metrics per all entities

    Returns:
        list[dict]: table

    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        key = row.get(group_field, "unknown")
        grouped[key].append(row["_metrics"])

    table = []

    # per-group avg
    for key, metrics_list in grouped.items():
        result_row = {group_field: key}

        for metric_name in metric_names:
            vals = [m[metric_name] for m in metrics_list if metric_name in m]

            result_row[metric_name] = round(sum(vals) / len(vals), 3) if vals else None

        table.append(result_row)

    # macro avg
    macro_row = {group_field: "macro_avg"}

    for metric_name in metric_names:
        vals = [row[metric_name] for row in table if row[metric_name] is not None]

        macro_row[metric_name] = round(sum(vals) / len(vals), 3) if vals else None

    table.append(macro_row)

    # micro avg (optional)
    if all_metrics is not None:
        micro_row = {group_field: "micro_avg"}

        for metric_name in metric_names:
            vals = all_metrics[metric_name]

            micro_row[metric_name] = round(sum(vals) / len(vals), 3) if vals else None

        table.append(micro_row)

    return table


def print_table(table: list[dict], title: str) -> None:
    """Print pretty table.

    Args:
        table: table to print
        title: name of the table

    """
    print(f"\n=== {title} ===")
    print(tabulate(table, headers="keys", tablefmt="pretty"))


def save_tables(save_path: Path, tables: dict[str, list[dict]], mode: Literal["a", "w"] = "a") -> None:
    """Save evaluation summary results.

    Args:
        save_path: output path
        tables: evaluation results
        mode: if table should overwrite non-empty file contents

    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open(mode, encoding="utf-8") as f:
        if mode == "w":
            current_utc_datetime = datetime.now(UTC)
            f.write(f"# {current_utc_datetime}\n")
        for name, table in tables.items():
            f.write(f"# {name}\n")
            for row in table:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            f.write("\n")
