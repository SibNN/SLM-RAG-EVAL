"""Module for evaluating RAG pipeline on non-LLM metrics using JSONL."""

import json
from collections import defaultdict
from pathlib import Path

from dacite import from_dict

from src.rag_eval.metrics.non_llm import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_recall_at_k,
    calculate_rouge,
)
from src.rag_eval.metrics_aggregation.aggregation import (
    compute_aggregates,
    print_table,
    save_tables,
)
from src.shared.paths import METRIC_RESULTS_FOLDER, RAG_ANSWERS_FOLDER
from src.shared.schemas import RAGAnswerRow


def gen_metrics(samples: list[RAGAnswerRow]) -> dict[str, float]:
    """Calculate metrics for a group of samples.

    Args:
        samples: given rag answers with evaluation entities

    Returns:
        dict: metric results

    """
    cands = [s.answer for s in samples]
    refs = [s.golden_answer for s in samples]

    doc_cands = [s.doc_ids for s in samples]
    doc_ref = [s.golden_doc_ids for s in samples]

    ref_folded = [[ref] for ref in refs]

    metrics = calculate_rouge(cands, ref_folded)

    metrics = {k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0}

    # retrieval metrics
    if doc_ref and doc_cands and doc_ref[0] is not None and doc_cands[0] is not None:
        if not isinstance(doc_ref[0], list):
            doc_ref = [[val] for val in doc_ref]

        metrics.update(calculate_hit_at_k(doc_cands, doc_ref))

        metrics["mrr"] = calculate_mrr(doc_cands, doc_ref)

        metrics.update(calculate_recall_at_k(doc_cands, doc_ref))

    return metrics


def bench(
    generated_answers: Path,
    save_path: Path | None = None,
) -> None:
    """Run non llm metric evaluation.

    Args:
        generated_answers: path to the given jsonl file
        save_path: output file path

    """
    rows = []
    all_metrics = defaultdict(list)

    with generated_answers.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            rag_answer = from_dict(data_class=RAGAnswerRow, data=sample)
            metrics = gen_metrics([rag_answer])

            # save metrics
            for k, v in metrics.items():
                all_metrics[k].append(v)

            row = {
                **sample,
                "_metrics": metrics,
            }

            rows.append(row)

    metric_names = sorted(all_metrics.keys())

    tables = {}

    for field in ["question_type", "dataset"]:
        table = compute_aggregates(
            rows=rows,
            metric_names=metric_names,
            group_field=field,
            all_metrics=all_metrics,
        )

        tables[field] = table

        print_table(
            table,
            f"Non-LLM Results by {field.upper()}",
        )

    if save_path:
        save_tables(
            save_path,
            tables,
        )


def main(name: str) -> None:
    """Measure RAG pipeline performance.

    Args:
        name: name of the run (and according jsonl file)

    """
    generated_answers = RAG_ANSWERS_FOLDER / f"{name}.jsonl"
    save_path = METRIC_RESULTS_FOLDER / f"{name}.jsonl"
    bench(generated_answers, save_path)
