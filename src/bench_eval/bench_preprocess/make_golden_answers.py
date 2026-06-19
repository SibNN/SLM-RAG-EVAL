"""Read and add new collumn golden_answer and fill it with answer column.

All inplace.
"""

import argparse
from pathlib import Path

import pandas as pd


def process_file(path: Path) -> None:
    """Add golden_answer column, if exists.

    Args:
        path: path to the jsonl file

    """
    print(f"Processing: {path.name}")

    df = pd.read_json(path, lines=True)

    if "answer" not in df.columns:
        print(f"Skipping {path.name}: no 'answer' column")
        return

    df["golden_answer"] = df["answer"]

    df.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def main(folder: Path) -> None:
    """Copy answer column into the golden_answer.

    Args:
        folder: path with jsonl files

    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.jsonl"))

    if not files:
        print("No jsonl files found")
        return

    for file in files:
        process_file(file)

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Copy 'answer' column to 'golden_answer' in all jsonl files inplace",
    )
    p.add_argument("input", help="folder with jsonl files")

    args = p.parse_args()

    main(Path(args.input))
