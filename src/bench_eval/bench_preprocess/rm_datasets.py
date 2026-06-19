"""Module for including all datasets, except one mentioned."""

import argparse
from pathlib import Path

import pandas as pd

datasets_to_remove = ["TAPE_MultiQ", "ontico", "croc"]
folded_datasets_to_remove = [[x] for x in datasets_to_remove]


def get_jsonl_file(parent_fold: Path, split_name: str) -> Path:
    """Find a path for jsonl file in given directory.

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


def delete_except(input_path: Path, output_path: Path) -> None:
    """Delete all questions and data files if they are from aforementioned dataset.

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
    print(f"Изначальное колличество строк в {knowledge}: {len(df)}")
    df_filtered = df[
        df["source_dataset"].apply(
            lambda m: m not in folded_datasets_to_remove,
        )
    ]
    out_kn_path = output_data / knowledge.name
    df_filtered.to_json(out_kn_path, orient="records", lines=True, force_ascii=False)
    print(f"Сохранено строк: {len(df_filtered)}")

    questions = get_jsonl_file(input_path, "questions")

    df = pd.read_json(questions, lines=True)
    print(f"Изначальное колличество строк в {questions}: {len(df)}")
    df_filtered = df[
        df["meta"].apply(
            lambda m: isinstance(m, dict) and m.get("source_dataset") not in datasets_to_remove,
        )
    ]
    out_kn_path = output_questions / questions.name
    df_filtered.to_json(out_kn_path, orient="records", lines=True, force_ascii=False)
    print(f"Сохранено строк: {len(df_filtered)}")


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

    delete_except(Path(args.input), Path(args.output))
