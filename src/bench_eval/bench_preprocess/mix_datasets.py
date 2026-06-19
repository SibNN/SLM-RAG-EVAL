"""Combine base and negative datasets with according labels."""

import argparse
from pathlib import Path

import pandas as pd


def read_jsonl_df(path: Path) -> pd.DataFrame:
    """Read jsonl into pandas DataFrame.

    Args:
        path: input path

    Returns:
        opened dataframe

    """
    return pd.read_json(path, lines=True)


def write_jsonl_df(path: Path, df: pd.DataFrame) -> None:
    """Write pandas DataFrame to jsonl.

    Args:
        path: save path
        df: dataframe to save

    """
    df.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def main(input_folder: Path) -> None:
    """Combine two datasets.

    Combine datasets from base and negative folder.
    Assign items from base "golden_score":1
    and items from negative set "golden_score":0

    Save resulting file with the same name.

    Args:
        input_folder: path to the parenting folder
            with base and negative folders inside

    """
    base_dir = input_folder / "base"
    negative_dir = input_folder / "negative"
    mixed_dir = input_folder / "mixed"

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing folder: {base_dir}")

    if not negative_dir.exists():
        raise FileNotFoundError(f"Missing folder: {negative_dir}")

    mixed_dir.mkdir(exist_ok=True)

    base_files = list(base_dir.glob("*.jsonl"))

    if not base_files:
        print("No jsonl files found in base folder")
        return

    for base_file in base_files:
        name = base_file.name
        negative_file = negative_dir / name

        if not negative_file.exists():
            print(f"Skipping {name}: not found in negative/")
            continue

        print(f"Processing: {name}")

        base_df = read_jsonl_df(base_file)
        negative_df = read_jsonl_df(negative_file)

        base_df["golden_score"] = 1
        negative_df["golden_score"] = 0

        mixed_df = pd.concat(
            [base_df, negative_df],
            axis=0,
            ignore_index=True,
        )

        output_file = mixed_dir / name
        write_jsonl_df(output_file, mixed_df)

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mix RAG jsonl datasets as described.")
    p.add_argument("input", help="input_folder")

    args = p.parse_args()
    main(Path(args.input))
