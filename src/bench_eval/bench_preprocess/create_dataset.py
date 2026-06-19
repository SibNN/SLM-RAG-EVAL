"""Save dataset with given number of instances."""

import argparse
from pathlib import Path

from src.bench_eval.bench_preprocess.create_subset import create_subset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="input path to benchmark",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of instances",
        required=True,
    )
    args = parser.parse_args()
    source = Path(args.input)
    subset = create_subset(source, args.number)

    save_path = source.parent / f"{source.stem}_{len(subset)}.jsonl"
    subset.to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(subset)
