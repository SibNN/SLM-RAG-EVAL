"""Generation of the answers from the dataset."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
from dacite import from_dict
from rich.progress import track

from src.model.api_model import APIModel
from src.model.vllm import VllmModel
from src.shared.paths import RAG_ANSWERS_FOLDER
from src.shared.prompts import SYSTEM_PROMPT
from src.shared.schemas import RAGAnswerRow


def process_question(
    model: VllmModel | APIModel,
    question: str,
    contexts: list[str],
    tries_left: int = 10,
) -> str | None:
    """Process a user question using LLM.

    Args:
        model: Initialized VLLM or API model used for generation.
        question: Question to be answered.
        contexts: List of retrieved context strings.
        tries_left: Maximum number of retries if generation fails.

    Returns:
        str | None: answer from llm with given query

    """
    if contexts:
        joined_context = "\n\n".join(contexts)
        content = f"Контекст:\n{joined_context}\n\nВопрос: {question}"
    else:
        content = f"Вопрос: {question}"

    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content,
        },
    ]
    answer = None
    while answer is None and tries_left > 0:
        model_out = model.generate(messages=message)

        if model_out:
            answer = model_out.strip()

        tries_left -= 1

    return answer


def answer_over_dataset(  # noqa: PLR0913
    dataset_path: Path,
    out_path: Path,
    run_name: str,
    mode: str = "golden_context",
    generator_type: Literal["vllm", "api"] = "vllm",
    tries_left: int = 10,
) -> None:
    """Generate answers for a list of questions using RAG.

    Args:
        dataset_path: Path to the dataset, jsonl file
        out_path: path to the output file parent folder.
        run_name: name of the run.
        mode: Name of the generation mode (no_context, golden_context).
        generator_type: generation backend - 'api' to use OpenAI-compatible API,
            or 'vllm' to run generation through a local vLLM server.
        tries_left: Maximum number of retries per question.

    """
    model = VllmModel() if generator_type == "vllm" else APIModel()

    df = pd.read_json(dataset_path, lines=True)

    out_file = out_path / f"{run_name}.jsonl"

    total = len(df)

    print(f"Processing {total} questions...")

    with out_file.open("w", encoding="utf-8") as f:
        for _, row in track(df.iterrows(), total=total):
            question_id = row.get("id") or row.get("question_id")
            question = row["question"]
            golden_context_ids = row.get("context_ids") or row.get("golden_doc_ids")
            golden_answer = row["answer"]
            group = row["question_type"]
            dataset = row["dataset"]

            if mode == "golden_context":
                contexts = row.get("context") or row.get("doc_contents")
                doc_ids = golden_context_ids
            else:
                contexts = None
                doc_ids = None

            answer = process_question(
                model,
                question,
                contexts,
                tries_left,
            )

            result = {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "golden_answer": golden_answer,
                "doc_contents": contexts,
                "doc_ids": doc_ids,
                "golden_doc_ids": golden_context_ids,
                "question_type": group,
                "dataset": dataset,
            }

            answer_row = from_dict(data_class=RAGAnswerRow, data=result)

            f.write(json.dumps(asdict(answer_row), ensure_ascii=False) + "\n")

    print(f"Saved to: {out_file}")


def main(
    input_file: Path,
    run_name: str,
    mode: str = "golden_context",
    generator_type: Literal["vllm", "api"] = "vllm",
    tries_left: int = 10,
) -> None:
    """Generate model answers.

    Args:
        input_file: Path to the input dataset file.
        run_name: Name of the run (name should be without slashes and dots).
        mode: Name of the generation mode (no_context, golden_context).
        generator_type: generation backend - 'api' to use OpenAI-compatible API,
            or 'vllm' to run generation through a local vLLM server.
        tries_left: Maximum number of retries per question.

    """
    output_dir = RAG_ANSWERS_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_over_dataset(
        input_file,
        output_dir,
        run_name,
        mode,
        generator_type,
        tries_left,
    )
