"""Merge RAG evaluation results into one JSONL.

Inputs:
  1) answers_dir: JSONL files
     dataset_500_filtered_answers_<MODEL>[_with_empty].jsonl
     Fields used: question_id, question, answer, question_type, llm_time_taken_ms

  2) llm_metrics_dir: CSV files
     results_<MODEL>[_with_empty].csv
     Fields used:
       group, input, actual_output,
       Correctness_score, Faithfulness_score, AnswerRelevance_score, ContextRelevance_score

Join key:
  (model, mode, question, answer)

Output JSONL fields:
  sample_id, answer, model, mode, question_type,
  correctness, faithfulness, answer_relevance, context_relevance,
  llm_gen_time, cyrillic_ratio, latin_ratio, chinese_ratio, other_ratio

- If answer is empty/None or has 0 letters => ratios are None.
"""

import argparse
import csv
import json
from pathlib import Path

from src.shared.constants import PREFIX_ANSWERS, PREFIX_LLM_METRICS
from src.shared.schemas import JoinKey, MetricsRow
from src.stats.get_letter_stats import calc_letter_level_ratios, normalize_text
from src.stats.utils import (
    ms_to_seconds,
    parse_model_and_mode_from_name,
    read_jsonl,
    to_float_or_none,
)


def build_llm_metrics_index(
    llm_metrics_csv: Path,
    model: str,
    mode: str,
) -> tuple[dict[JoinKey, MetricsRow], dict[str, int]]:
    """Build mapping from JoinKey -> MetricsRow for one LLM-metrics CSV file.

    Also returns counters for duplicates, etc.
    """
    idx: dict[JoinKey, MetricsRow] = {}
    stats = {
        "rows_total": 0,
        "rows_skipped_empty_key": 0,
        "rows_duplicate_keys": 0,
    }

    with llm_metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "input",
            "actual_output",
            "Correctness_score",
            "Faithfulness_score",
            "AnswerRelevance_score",
            "ContextRelevance_score",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV {llm_metrics_csv} missing columns: {sorted(missing)}")

        for row in reader:
            stats["rows_total"] += 1

            q = normalize_text(row.get("input"))
            a = normalize_text(row.get("actual_output"))

            if q == "" and a == "":
                stats["rows_skipped_empty_key"] += 1
                continue

            key = JoinKey(model=model, mode=mode, question=q, answer=a)
            mr = MetricsRow(
                correctness=to_float_or_none(row.get("Correctness_score")),
                faithfulness=to_float_or_none(row.get("Faithfulness_score")),
                answer_relevance=to_float_or_none(row.get("AnswerRelevance_score")),
                context_relevance=to_float_or_none(row.get("ContextRelevance_score")),
            )

            if key in idx:
                # keep the first, count duplicates
                stats["rows_duplicate_keys"] += 1
                continue

            idx[key] = mr

    return idx, stats


def combine_results(
    output_file: Path,
    answer_files: list[Path],
    llm_metrics_by_pair: dict[tuple[str, str], dict[JoinKey, MetricsRow]],
    strict: bool,
) -> int:
    """Merge answer JSONL files with LLM-as-judge metrics and write a unified JSONL file.

    For each answer file:
    - Extract model and mode from filename.
    - Match each (question, answer) pair with corresponding evaluation metrics.
    - Compute letter-based ratios for the answer text.
    - Convert generation time from milliseconds to seconds.
    - Write a unified result row to the output file.

    If `strict` is True, raises ValueError when a metrics match is missing
    (except for predefined "NO RELEVANT DOCUMENTS FOUND" responses).
    """
    total_answer_rows = 0
    written_rows = 0
    missing_llm_metrics = 0
    missing_llm_metrics_pairs: dict[tuple[str, str], int] = {}
    answer_pairs_missing_csv: dict[tuple[str, str], int] = {}

    with output_file.open("w", encoding="utf-8") as out:
        for af in answer_files:
            model, mode = parse_model_and_mode_from_name(af.name, PREFIX_ANSWERS)

            llm_metrics_idx = llm_metrics_by_pair.get((model, mode))
            if llm_metrics_idx is None:
                # No CSV for this model/mode
                answer_pairs_missing_csv[(model, mode)] = answer_pairs_missing_csv.get((model, mode), 0) + 1
                # We'll still write rows but with None scores
                llm_metrics_idx = {}

            for row in read_jsonl(af):
                total_answer_rows += 1

                sample_id = row.get("question_id")
                question = normalize_text(row.get("question"))
                answer = normalize_text(row.get("answer"))
                question_type = row.get("question_type")

                key = JoinKey(model=model, mode=mode, question=question, answer=answer)
                metrics_row = llm_metrics_idx.get(key)

                if metrics_row is None:
                    if answer == "NO RELEVANT DOCUMENTS FOUND":
                        missing_llm_metrics_pairs[(model, mode)] = missing_llm_metrics_pairs.get((model, mode), 0) + 1
                    else:
                        missing_llm_metrics += 1
                        missing_llm_metrics_pairs[(model, mode)] = missing_llm_metrics_pairs.get((model, mode), 0) + 1

                        if strict:
                            raise ValueError(
                                f"No llm_metrics match for model={model}, mode={mode}, sample_id={sample_id}\n"
                                f"  question={question[:200]!r}\n"
                                f"  answer={answer[:200]!r}",
                            )

                # Ratios: letters only; if empty/no letters => None
                if answer == "":
                    cyr = lat = chi = oth = None
                else:
                    cyr, lat, chi, oth = calc_letter_level_ratios(answer)

                out_obj = {
                    "sample_id": sample_id,
                    "answer": answer,
                    "model": model,
                    "mode": mode,
                    "question_type": question_type,
                    "correctness": metrics_row.correctness if metrics_row else None,
                    "faithfulness": metrics_row.faithfulness if metrics_row else None,
                    "answer_relevance": metrics_row.answer_relevance if metrics_row else None,
                    "context_relevance": metrics_row.context_relevance if metrics_row else None,
                    "llm_gen_time": ms_to_seconds(row.get("llm_time_taken_ms")),
                    "cyrillic_ratio": cyr,
                    "latin_ratio": lat,
                    "chinese_ratio": chi,
                    "other_ratio": oth,
                }

                out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written_rows += 1

    return total_answer_rows, written_rows, missing_llm_metrics, missing_llm_metrics_pairs, answer_pairs_missing_csv


def build_unified_results(answers_dir: Path, llm_metrics_dir: Path, output_file: Path, strict: bool) -> None:
    """Aggregate and merge answer files from all models with their evaluation metrics into a unified file."""
    if not answers_dir.is_dir():
        raise NotADirectoryError(f"answers_dir is not a directory: {answers_dir}")
    if not llm_metrics_dir.is_dir():
        raise NotADirectoryError(f"llm_metrics_dir is not a directory: {llm_metrics_dir}")

    # Discover files
    answer_files = sorted(p for p in answers_dir.glob("*.jsonl") if p.name.startswith(PREFIX_ANSWERS))
    llm_metrics_files = sorted(p for p in llm_metrics_dir.glob("*.csv") if p.name.startswith(PREFIX_LLM_METRICS))

    if not answer_files:
        raise FileNotFoundError(f"No answer JSONL files found in {answers_dir}")
    if not llm_metrics_files:
        raise FileNotFoundError(f"No llm_metrics CSV files found in {llm_metrics_dir}")

    # Build llm_metrics indexes per (model, mode)
    llm_metrics_by_pair: dict[tuple[str, str], dict[JoinKey, MetricsRow]] = {}
    llm_metrics_file_stats: dict[str, dict[str, int]] = {}

    for f in llm_metrics_files:
        model, mode = parse_model_and_mode_from_name(f.name, PREFIX_LLM_METRICS)
        idx, stats = build_llm_metrics_index(f, model, mode)
        llm_metrics_by_pair[(model, mode)] = idx
        llm_metrics_file_stats[f.name] = stats

    # Process answers and write output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    (
        total_answer_rows,
        written_rows,
        missing_llm_metrics,
        missing_llm_metrics_pairs,
        answer_pairs_missing_csv,
    ) = combine_results(output_file, answer_files, llm_metrics_by_pair, strict)

    # Print stats
    print("\n=== INPUT FILES ===")
    print(f"Answers JSONL files: {len(answer_files)}")
    print(f"llm_metrics CSV files: {len(llm_metrics_files)}")

    print("\n=== LLM METRICS CSV STATS (per file) ===")
    for name, st in sorted(llm_metrics_file_stats.items()):
        print(
            f"{name}: rows_total={st['rows_total']}\n"
            f"skipped_empty_key={st['rows_skipped_empty_key']}\n"
            f"duplicate_keys={st['rows_duplicate_keys']}\n",
        )

    print("\n=== MERGE STATS ===")
    print(f"Answer rows read:    {total_answer_rows}")
    print(f"Rows written:        {written_rows}")
    print(f"Missing llm_metrics rows:  {missing_llm_metrics}")

    print("\n=== NO DOCUMENTS FOUND answers ===")
    if missing_llm_metrics_pairs:
        for (model, mode), count in sorted(missing_llm_metrics_pairs.items(), key=lambda x: (-x[1], x[0])):
            print(f"{model} | {mode}: {count}")

    if answer_pairs_missing_csv:
        print("\nNo llm_metrics CSV found for these (model, mode) pairs (still wrote rows with None scores):")
        for (model, mode), count in sorted(answer_pairs_missing_csv.items(), key=lambda x: (-x[1], x[0])):
            print(f"{model} | {mode}: {count} file(s)")

    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create combined JSONL with answers + llm metrics + letter ratios.",
    )
    ap.add_argument("--answers_dir", type=Path, required=True, help="Folder with answers JSONL files")
    ap.add_argument("--llm_metrics_dir", type=Path, required=True, help="Folder with llm_as_judge CSV files")
    ap.add_argument("--output_file", type=Path, required=True, help="Path to output combined JSONL")
    ap.add_argument("--strict", action="store_true", help="Fail if any answer row has no matching llm_metrics row")
    args = ap.parse_args()

    build_unified_results(args.answers_dir, args.llm_metrics_dir, args.output_file, args.strict)
