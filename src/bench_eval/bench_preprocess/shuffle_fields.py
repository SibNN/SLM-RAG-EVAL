"""Mix datasets.

Reads jsonl source file with "dataset" field, shuffles groups and saves to the new jsonl
using the following scheme:
  - question  <- dataset[i]
  - context_ids <- dataset[(i+1) % k]
  - answer <- dataset[(i+2) % k]
  - golden_answer <- dataset[(i+3) % k]
"""

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42


def read_jsonl_to_df(path: Path) -> pd.DataFrame:
    """Read jsonl file.

    Args:
        path: input jsonl file

    Returns:
        pd.DataFrame

    """
    df = pd.read_json(path, lines=True, dtype=False)
    return df


def sample_with_replacement(df: pd.DataFrame, n: int, seed: int | None = None) -> pd.DataFrame:
    """Return dataframe of length n sampled with replacement.

    Args:
        df: pd.DataFrame
        n: number of samples
        seed: seed

    Returns:
        pd.DataFrame: shufled frame

    """
    return df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def sample_without_replacement_shuffled(df: pd.DataFrame, n: int, seed: int | None = None) -> pd.DataFrame:
    """Return dataframe of length n sampled without replacement.

    if n == len(df) just shuffle

    Args:
        df: dataframe
        n: number of samples
        seed: seed

    Returns:
        pd.DataFrame: shufled frame

    """
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def build_mixed_dataset(groups_ordered: list[tuple[str, pd.DataFrame]], target_per_group: int) -> list[dict]:
    """Mix given groups.

    Args:
        groups_ordered: list of (dataset_name, dataframe)
        target_per_group: int, number of examples to draw from each group (already chosen)

    Returns:
        list of dicts (rows) in the assembled order:
            for i in range(k):
                for j in range(target_per_group):
                produce one record combining rows with same j-index from groups i, (i+1),(i+2),(i+3)

    """
    k = len(groups_ordered)
    # ensure each group has length >= target_per_group (we assume dataframes sampled already)
    group_samples = []
    for name, g in groups_ordered:
        if len(g) < target_per_group:
            raise ValueError(f"group {name} length {len(g)} < target_per_group {target_per_group}")
        group_samples.append(g.reset_index(drop=True))

    rows = []
    next_id = 101000
    for i in range(k):
        for j in range(target_per_group):
            q_row = group_samples[i].iloc[j]
            ctx_row = group_samples[(i + 1) % k].iloc[j]
            ans_row = group_samples[(i + 2) % k].iloc[j]
            golden_row = group_samples[(i + 3) % k].iloc[j]

            new_item = OrderedDict()
            new_item["id"] = int(next_id)
            next_id += 1

            # fields per spec
            new_item["question"] = q_row.get("question") if "question" in q_row else None
            new_item["context_ids"] = (
                ctx_row.get("context_ids") if "context_ids" in ctx_row else ctx_row.get("context_ids", None)
            )
            new_item["answer"] = ans_row.get("answer") if "answer" in ans_row else None
            new_item["golden_answer"] = (
                golden_row.get("answer") if "answer" in golden_row else golden_row.get("golden_answer", None)
            )

            # preserve some useful auxiliary fields:
            # - question_type (from question source)
            if "question_type" in q_row:
                new_item["question_type"] = q_row.get("question_type")
            # - context text (from context source)
            if "context" in ctx_row:
                new_item["context"] = ctx_row.get("context")

            # meta: indicate sources used to compose the row (helpful for later analysis)
            new_item["meta"] = {
                "question_from_dataset": groups_ordered[i][0],
                "context_ids_from_dataset": groups_ordered[(i + 1) % k][0],
                "answer_from_dataset": groups_ordered[(i + 2) % k][0],
                "golden_answer_from_dataset": groups_ordered[(i + 3) % k][0],
            }

            # mark dataset as mixed
            new_item["dataset"] = "mixed_composed"

            rows.append(new_item)
    return rows


def main(input: Path, output: Path) -> None:
    """Shuffle the fields in the given dataset.

    With equal sized datasets inside of given jsonl file,
    each column needed for judge evaluation is taken from other dataset.

    Args:
        input: path to the source jsonl dataset file
        output: result path

    """
    df = read_jsonl_to_df(input)
    total_in = len(df)
    if "dataset" not in df.columns:
        raise ValueError(f"Bad input file: {input}")

    datasets = list(pd.Series(df["dataset"]).astype(str).drop_duplicates().tolist())
    k = len(datasets)
    if k == 0:
        raise ValueError(f"ERROR: no datasets found in {input}.")

    groups_ordered = []
    for d in datasets:
        g = df[df["dataset"] == d].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        groups_ordered.append((d, g))

    target_per_group = math.ceil(total_in / k)
    groups_sampled = []
    for name, g in groups_ordered:
        if len(g) >= target_per_group:
            sample_g = sample_without_replacement_shuffled(g, target_per_group, seed=RANDOM_SEED)
        else:
            sample_g = sample_with_replacement(g, target_per_group, seed=RANDOM_SEED)
        groups_sampled.append((name, sample_g))

    rows = build_mixed_dataset(groups_sampled, target_per_group)
    if len(rows) > total_in:
        rows = rows[:total_in]

    with open(output, "w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Input rows: {total_in}")
    print(f"Datasets (k): {k} -> {datasets}")
    print(f"Output rows: {len(rows)}")
    print(f"Output saved to: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mix RAG jsonl datasets as described.")
    p.add_argument("input", help="input jsonl file")
    p.add_argument("output", help="output jsonl file")
    args = p.parse_args()
    main(Path(args.input), Path(args.output))
