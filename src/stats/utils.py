"""Utility functions for statistics and evaluation helpers."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.shared.constants import WITH_EMPTY_TAG


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Read a .jsonl file and yield parsed JSON objects."""
    with path.open("r", encoding="utf-8") as f:
        for line_n, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_n}: {e}") from e


def to_float_or_none(x: str | float | None) -> float | None:
    """Convert a value to float if possible."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ms_to_seconds(ms: int | float | None) -> float | None:
    """Convert milliseconds to seconds."""
    if ms is None:
        return None
    try:
        return float(ms) / 1000.0
    except (TypeError, ValueError):
        return None


def parse_model_and_mode_from_name(filename: str, prefix: str) -> tuple[str, str]:
    """Extract (model, mode) from supported filename patterns.

    Examples:
    - dataset_500_filtered_answers_<MODEL>_with_empty.jsonl
    - dataset_500_filtered_answers_<MODEL>.jsonl
    - results_<MODEL>_with_empty.csv
    - results_<MODEL>.csv

    """
    name = Path(filename).name

    if not name.startswith(prefix):
        raise ValueError(f"Unexpected file name (missing prefix '{prefix}'): {name}")

    stem = Path(name).stem
    rest = stem[len(prefix) :]

    if rest.endswith(WITH_EMPTY_TAG):
        model = rest[: -len(WITH_EMPTY_TAG)]
        mode = "with_empty_context"
    else:
        model = rest
        mode = "without_empty_context"

    if not model:
        raise ValueError(f"Could not parse model from filename: {name}")

    return model, mode
