"""Processing of the generated answers."""

import json
import shutil
from pathlib import Path

from src.shared.paths import RAG_ANSWERS_FOLDER


def ensure_columns(jsonl_path: Path) -> None:
    """Ensure that all records in a JSONL file contain required columns "doc_contents".

    Missing columns will be added with value None.
    """
    new_lines = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            for col in ["doc_contents"]:
                if col not in record:
                    record[col] = None

            new_lines.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in new_lines:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(input_path: Path, run_name: str) -> None:
    """Process model answers.

    Args:
        input_path: Path to the input dataset file.
        run_name: Name of run

    """
    output_dir = RAG_ANSWERS_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{run_name}.jsonl"

    rename_map = {
        "id": "question_id",
        "context": "doc_contents",
        "context_ids": "doc_ids",
    }

    with input_path.open("r", encoding="utf-8") as f:
        first_row = json.loads(f.readline())

    # if the dataset is already in the new format, copy the file
    if "context_ids" not in first_row:
        shutil.copy(input_path, output_path)
        ensure_columns(output_path)
        return

    # otherwise convert the old format to the new names
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)

            # preserve original context ids as golden_doc_ids
            row["golden_doc_ids"] = row["context_ids"]

            for old, new in rename_map.items():
                if old in row:
                    row[new] = row.pop(old)

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    ensure_columns(output_path)
