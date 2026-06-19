"""Module for question classification.

Used to asign questions from the dataset some classes.
It helps to evaluate benchmark better and make results more comprehensible.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.progress import track

from src.model.vllm import VllmModel
from src.shared.prompts import CLASSIFY_QUESTION_PROMPT


def get_class_llm(llm_answer: str) -> str | None:
    """Parse llm's answer to get class.

    Args:
        llm_answer: llm with json guided decoding answer

    Returns:
        str | None: class name, if succeeded

    """
    q_classes = ["Factoid", "Comparison", "Instruction", "Reason", "Evidence-based", "Experience"]
    try:
        answer = json.loads(llm_answer)
        q_class = answer.get("question_type")

        if q_class in q_classes:
            return q_class
        else:
            return None
    except json.JSONDecodeError:
        return None


def assign_class(model: VllmModel, question: str) -> str | None:
    """Classify given question.

    Args:
        model (VllmModel): model to generate class
        question (str): question to classify

    Returns:
        str: question class name

    """
    tries_left = 10
    message = [
        {"role": "system", "content": CLASSIFY_QUESTION_PROMPT},
        {"role": "user", "content": f"question: {question}"},
    ]
    q_class = None
    while q_class is None and tries_left > 0:
        model_out = model.generate(
            messages=message,
            response_format={"type": "json_object"},
        )
        q_class = get_class_llm(model_out)
        tries_left -= 1
    return q_class


def classify_dataset(input_file: pd.DataFrame) -> pd.DataFrame:
    """Classify each dataset question.

    Args:
        input_file (pd.DataFrame): pandas dataframe with question field in it

    Returns:
        pd.DataFrame: same dataframe, but with question_type column

    """
    model = VllmModel()
    q_types = []
    for row in track(input_file.itertuples(), total=len(input_file)):
        question_type = assign_class(model, str(row.question))
        q_types.append(question_type)

    classified_df = input_file.copy()
    classified_df["question_type"] = q_types
    return classified_df


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
    subset = pd.read_json(source, lines=True)

    classified_set = classify_dataset(subset)
    cls_path = source.parent / f"{source.stem}_classified.jsonl"
    classified_set.to_json(cls_path, orient="records", lines=True, force_ascii=False)
