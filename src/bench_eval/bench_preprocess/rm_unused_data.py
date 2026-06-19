"""Removes not used data in the benchmark."""

import argparse
from pathlib import Path

import pandas as pd


def rm_not_used_docs(source_kn: Path, dataset: Path) -> None:
    """Remove not used documents with respect to the given dataset.

    Args:
        source_kn: path to the folder with /data folder and json with knowledge
        dataset: path to the jsonl with dataset

    """
    knowledge_dir = source_kn / "data"
    knowledge_path = next(knowledge_dir.glob("*.jsonl"))
    if knowledge_path is None:
        raise ValueError(f"Bad input folder {source_kn}")
    if not dataset.is_file():
        raise ValueError(f"Bad dataset file {dataset}")
    df = pd.read_json(dataset, lines=True)
    used_ids = set()
    for row in df.itertuples():
        used_ids.update(row.context_ids)

    knowledge = pd.read_json(knowledge_path, lines=True)
    kn_filtered = knowledge[
        knowledge["context_id"].apply(
            lambda m: m in used_ids,
        )
    ]

    save_path = dataset.parent / f"{dataset.stem}_knowledge_base.jsonl"
    kn_filtered.to_json(save_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--knowledge",
        type=Path,
        help="input path to knowledge",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        help="input path to dataset",
        required=True,
    )
    args = parser.parse_args()
    rm_not_used_docs(Path(args.knowledge), Path(args.dataset))
