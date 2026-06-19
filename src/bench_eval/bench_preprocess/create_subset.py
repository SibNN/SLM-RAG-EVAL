"""Create usable subset from different datasets."""

import argparse
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42


def load_subset_by_dataset(
    qa_df: pd.DataFrame,
    max_entities: int = 100,
) -> pd.DataFrame:
    """Sample QA data per source dataset.

    Args:
        qa_df: full QA dataframe with meta.source_dataset
        max_entities: max samples per dataset

    Returns:
        pd.DataFrame: sampled QA dataframe

    """
    qa_df = qa_df.copy()
    qa_df["dataset"] = qa_df["meta"].apply(lambda x: x["source_dataset"])

    sampled_parts = []
    for _, group in qa_df.groupby("dataset"):
        n = min(len(group), max_entities)
        sampled = group.sample(n=n, random_state=RANDOM_SEED)
        sampled_parts.append(sampled)

    return pd.concat(sampled_parts, ignore_index=True)


def enrich_with_data(qa_data: pd.DataFrame, kn_df: pd.DataFrame) -> pd.DataFrame:
    """Attach contexts to QA dataset using context_ids.

    Args:
        qa_data: QA dataset with context_ids
        kn_df: knowledge dataframe with context_id and context

    Returns:
        pd.DataFrame: enriched QA dataset

    """
    kn_df = kn_df.set_index("context_id")

    contexts = []
    for row in qa_data.itertuples():
        row_contexts = [kn_df.at[cid, "context"] for cid in row.context_ids]
        contexts.append(row_contexts)

    qa_data = qa_data.copy()
    qa_data["context"] = contexts
    return qa_data


def create_subset(path: Path, max_ent_per_dataset: int = 100) -> pd.DataFrame:
    """Create QA subset grouped by meta.source_dataset.

    Directory structure:
        path/
            questions/qa.jsonl
            data/data.jsonl

    Args:
        path: root directory
        max_ent_per_dataset: max samples per dataset

    Returns:
        pd.DataFrame: final benchmark dataset

    """
    qa_dir = path / "questions"
    knowledge_dir = path / "data"

    qa_files = list(qa_dir.glob("*.jsonl"))
    data_files = list(knowledge_dir.glob("*.jsonl"))

    if len(qa_files) != 1 or len(data_files) != 1:
        raise ValueError("Expected exactly one QA file and one data file.")

    qa_df = pd.read_json(qa_files[0], lines=True)
    kn_df = pd.read_json(data_files[0], lines=True)

    qa_sampled = load_subset_by_dataset(
        qa_df,
        max_entities=max_ent_per_dataset,
    )
    bench = enrich_with_data(qa_sampled, kn_df)

    return bench


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="input path to benchmark",
        required=True,
    )
    args = parser.parse_args()
    source = Path(args.input)
    subset = create_subset(source)
    print(subset)
