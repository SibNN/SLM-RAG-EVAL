"""Dataset collection script."""

import argparse
import json
from pathlib import Path

import get_data
import yaml

from src.shared.schemas import ContextEntry


def update_docs(docs_jsonl_path: Path, context_registry: dict[str, ContextEntry]) -> None:
    """Update source_dataset and source_filename in final_docs.jsonl based on global context_registry."""
    updated_records = []

    with open(docs_jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            context = record.get("context")
            entry = context_registry.get(context)
            if entry:
                record["source_dataset"] = list(entry.source_datasets)
                record["source_filename"] = list(entry.source_filenames)

            updated_records.append(record)

    with open(docs_jsonl_path, "w", encoding="utf-8") as f:
        for rec in updated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Docs file updated with merged source info: {len(updated_records)} records")


def main(args: argparse.Namespace) -> None:
    """Collect and serialize datasets into JSONL files."""
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "datasets_config.yaml"

    next_context_id = 0
    next_qa_id = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    qa_path = args.output_dir / "final_qa.jsonl"
    docs_path = args.output_dir / "final_docs.jsonl"

    for path in [qa_path, docs_path]:
        if path.exists():
            path.unlink()

    context_registry: dict[str, ContextEntry] = {}

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_handlers = {
        "grounded_rag": get_data.get_grounded_rag,
        "ru_rag_test": get_data.get_ru_rag_test,
        "sberquad": get_data.get_sberquad,
        "danetqa": get_data.get_danetqa,
        "croc": get_data.get_croc,
        "tape_multiq": get_data.get_tape_multiq,
        "ontico": get_data.get_ontico,
    }

    for name, ds in cfg["datasets"].items():
        handler = dataset_handlers[name]
        if name == "ontico":
            next_context_id, next_qa_id, context_registry = handler(
                args.dataset_dir / ds["input_path"],
                args.dataset_dir / ds["knowledge_base_dir"],
                qa_path,
                docs_path,
                context_registry,
                start_context_id=next_context_id,
                start_qa_id=next_qa_id,
            )
        else:
            next_context_id, next_qa_id, context_registry = handler(
                args.dataset_dir / ds["input_path"],
                qa_path,
                docs_path,
                context_registry,
                start_context_id=next_context_id,
                start_qa_id=next_qa_id,
            )
        print(f"{name} done: next_context_id={next_context_id}, next_qa_id={next_qa_id}\n")

    update_docs(docs_path, context_registry)

    print(f"Saved QA dataset to: {qa_path}")
    print(f"Saved documents dataset to: {docs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAG dataset.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to directory with datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output datasets.",
    )

    args = parser.parse_args()

    main(args)
