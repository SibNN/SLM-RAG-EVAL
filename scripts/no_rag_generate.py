"""Module for answer generation without using RAG."""

import argparse
from pathlib import Path

import pandas as pd
from rich.progress import track

from src.model.gemma_model import Gemma3TextGenerator
from src.model.vllm import VllmModel


def get_answer(model: VllmModel, question: str, tries_left: int = 10) -> str | None:
    """Get an answer to given question.

    Args:
        model (VllmModel): model to generate answer
        question (str): question to answer
        tries_left (int): Maximum number of generation attempts if the model
            fails to return a valid answer.

    Returns:
        str: answer

    """
    message = [
        {"role": "user", "content": question},
    ]
    answer = None
    while answer is None and tries_left > 0:
        model_out = model.generate(messages=message)
        answer = model_out.strip()
        tries_left -= 1
    return answer


def generate_answers(input_file: pd.DataFrame, tries_left: int = 10, use_gemma: bool = False) -> pd.DataFrame:
    """Generate answers to each question in the dataset without using RAG.

    Args:
        input_file (pd.DataFrame): pandas dataframe with question field in it.
        tries_left (int): Maximum number of generation attempts if the model
            fails to return a valid answer.
        use_gemma (bool): If True, generate answers using the Gemma model.
            If False, use the default text generation model served via vLLM.

    Returns:
        pd.DataFrame: same dataframe, but with no_rag_answer column

    """
    answers = []

    if use_gemma:
        generator = Gemma3TextGenerator()
        for row in track(input_file.itertuples(), total=len(input_file)):
            answer = generator.generate(row.question)
            answers.append(answer)
    else:
        model = VllmModel()
        for row in track(input_file.itertuples(), total=len(input_file)):
            answer = get_answer(model, row.question, tries_left)
            answers.append(answer)

    no_rag_df = input_file.copy()
    no_rag_df["no_rag_answer"] = answers
    return no_rag_df


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
        "-g",
        "--gemma",
        action="store_true",
        help="Use Gemma model",
    )
    parser.add_argument(
        "-a",
        "--attempts",
        type=int,
        default=10,
        help="Maximum number of generation attempts if the model fails to return a valid answer",
    )
    args = parser.parse_args()
    source = Path(args.input)
    dataset = pd.read_json(source, lines=True)

    no_rag_set = generate_answers(
        dataset,
        tries_left=args.attempts,
        use_gemma=args.gemma,
    )

    base = source.parent / f"{source.stem}_no_rag"
    i = 1
    output_path = Path(f"{base}_{i}.jsonl")

    while output_path.exists():
        i += 1
        output_path = Path(f"{base}_{i}.jsonl")
    no_rag_set.to_json(output_path, orient="records", lines=True, force_ascii=False)
