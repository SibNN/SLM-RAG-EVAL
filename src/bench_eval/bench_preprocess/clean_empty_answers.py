"""Module for removing dataset instances if there is no any mandatory field is empty."""

import argparse
from pathlib import Path

import pandas as pd

mandatory_fields = ["answer", "id", "context_ids", "question"]


def get_jsonl_file(parent_fold: Path, split_name: str) -> Path:
    """Read jsonl file in given directory.

    Args:
        parent_fold: main folder
        split_name: name of folder with jsonl file

    Returns:
        Path: path to the first jsonl file

    Raises:
        ValueError: if there is no jsonl file inside parent_fold/split_name

    """
    file_dir = parent_fold / split_name
    file = next(file_dir.glob("*.jsonl"))
    if file is None:
        raise ValueError(f"Input directory should contain {split_name} folder with jsonl file within")
    return file


def clean_empty(input_path: Path, output_path: Path) -> None:
    """Delete row from the dataset, if any nessesary field not given.

    Args:
        input_path: source dataset
        output_path: output dataset

    """
    output_path.mkdir(exist_ok=True)
    output_data = output_path / "data"
    output_data.mkdir(exist_ok=True)
    output_questions = output_path / "questions"
    output_questions.mkdir(exist_ok=True)

    knowledge = get_jsonl_file(input_path, "data")
    df = pd.read_json(knowledge, lines=True)
    out_kn_path = output_data / knowledge.name
    df.to_json(out_kn_path, orient="records", lines=True, force_ascii=False)

    questions = get_jsonl_file(input_path, "questions")

    df = pd.read_json(questions, lines=True)
    print(f"Было строк: {len(df)}")
    for ent in mandatory_fields:
        df_filtered = df[
            df[ent].apply(
                lambda m: m.strip() != "" and m is not None,
            )
        ]
        df = df_filtered
    out_kn_path = output_questions / questions.name
    df.to_json(out_kn_path, orient="records", lines=True, force_ascii=False)
    print(f"Сохранено строк: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="input path",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path",
        required=True,
    )
    args = parser.parse_args()

    clean_empty(Path(args.input), Path(args.output))
