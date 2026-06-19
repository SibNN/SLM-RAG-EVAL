"""Dataset loaders and converters for dataset creation scripts."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils import clean_srt_text, clean_text, get_files

from src.shared.schemas import ContextEntry, DocRecord, QARecord


def get_or_create_context(  # noqa: PLR0913
    context: str,
    context_registry: dict[str, ContextEntry],
    doc_records: list[DocRecord],
    source_dataset: str,
    source_filename: str,
    context_id_counter: int,
    *,
    meta: dict[str, any],
) -> tuple[int, int]:
    """Get an existing context_id from the global context registry or create a new one."""
    entry = context_registry.get(context)
    if entry is None:
        context_id = context_id_counter
        context_id_counter += 1

        entry = ContextEntry(
            context_id=context_id,
            source_datasets={source_dataset},
            source_filenames={source_filename},
        )
        context_registry[context] = entry

        doc_records.append(
            DocRecord(
                context_id=context_id,
                context=context,
                meta=meta,
                source_dataset=[source_dataset],
                source_filename=[source_filename],
            ),
        )
    else:
        context_id = entry.context_id
        entry.source_datasets.add(source_dataset)
        entry.source_filenames.add(source_filename)

    return context_id, context_id_counter


def add_qa(  # noqa: PLR0913
    qa_records: list[QARecord],
    qa_id_counter: int,
    question: str,
    answer: str,
    context_ids: list[int],
    *,
    meta: dict[str, any],
) -> int:
    """Append a QA record to the QA list and increment the QA ID counter."""
    qa_records.append(
        QARecord(
            id=qa_id_counter,
            context_ids=context_ids,
            question=question,
            answer=answer,
            meta=meta,
        ),
    )
    return qa_id_counter + 1


def save_data(
    qa_records: list[QARecord],
    doc_records: list[DocRecord],
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    start_qa_id: int = 0,
) -> int:
    """Save QA and document records in JSONL format with deduplication and ID reindexing."""
    qa_rows: list[dict[str, any]] = [asdict(r) for r in qa_records]
    doc_rows: list[dict[str, any]] = [asdict(r) for r in doc_records]

    df = pd.DataFrame(qa_rows)

    df["qa_key"] = df.apply(
        lambda r: (r["question"].strip(), r["answer"].strip(), tuple(r["context_ids"])),
        axis=1,
    )
    dup_count = df["qa_key"].duplicated(keep=False).sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicates")
        df = df.drop_duplicates(subset="qa_key")
        print("Removed duplicates")

    df = df.drop(columns="qa_key").reset_index(drop=True)

    df["id"] = range(start_qa_id, start_qa_id + len(df))
    next_qa_id = start_qa_id + len(df)

    df.to_json(qa_jsonl_path, orient="records", mode="a", lines=True, force_ascii=False)
    pd.DataFrame(doc_rows).to_json(docs_jsonl_path, orient="records", mode="a", lines=True, force_ascii=False)

    return next_qa_id


def get_grounded_rag(  # noqa: PLR0913
    input_dir: Path,
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "Grounded-RAG-QA-RU",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from Grounded-RAG-QA-RU parquet files and append results to JSONL files."""
    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    parquet_files = get_files(input_dir)

    for parquet_path in tqdm(parquet_files, desc="Parquet files"):
        source_filename = parquet_path.name
        df = pd.read_parquet(parquet_path)

        for source_idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {source_filename}",
        ):
            conversation = row["conversation"]

            documents = None
            question = None
            answer = None

            # parse conversation
            for msg in conversation:
                role = msg.get("role")
                content = msg.get("content")

                if role == "documents":
                    try:
                        documents = json.loads(content)
                    except Exception:
                        documents = None

                elif role == "user":
                    question = content

                elif role == "assistant":
                    answer = content

            if not documents or not question or not answer:
                continue

            context_ids: list[int] = []

            # process documents
            for doc in documents:
                cleaned_context = clean_text(doc.get("content", ""))

                context_id, context_id_counter = get_or_create_context(
                    cleaned_context,
                    context_registry,
                    doc_records,
                    source_dataset,
                    source_filename,
                    context_id_counter,
                    meta={
                        "doc_id": doc.get("doc_id"),
                        "title": doc.get("title"),
                        "source_idx": source_idx,
                    },
                )

                context_ids.append(context_id)

            # QA record
            qa_id_counter = add_qa(
                qa_records,
                qa_id_counter,
                question,
                answer,
                context_ids,
                meta={
                    "source_dataset": source_dataset,
                    "source_filename": source_filename,
                },
            )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )

    return context_id_counter, qa_id_counter, context_registry


def get_ru_rag_test(  # noqa: PLR0913
    pkl_path: Path,
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "ru-rag-test",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from ru-rag-test pickle file and append results to JSONL files."""
    df = pd.read_pickle(pkl_path)  # noqa: S301

    source_filename = pkl_path.name

    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    for source_idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing {source_filename}",
    ):
        context = str(row["Контекст"]).strip()
        question = str(row["Вопрос"]).strip()
        answer = str(row["Правильный ответ"]).strip()
        source_file = str(row["Файл"]).strip()

        if not context or not question or not answer:
            continue

        context_id, context_id_counter = get_or_create_context(
            context,
            context_registry,
            doc_records,
            source_dataset,
            source_filename,
            context_id_counter,
            meta={
                "source_file": source_file,
                "source_idx": source_idx,
            },
        )

        # QA record
        qa_id_counter = add_qa(
            qa_records,
            qa_id_counter,
            question,
            answer,
            [context_id],
            meta={
                "source_dataset": source_dataset,
                "source_filename": source_filename,
            },
        )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )
    return context_id_counter, qa_id_counter, context_registry


def get_sberquad(  # noqa: PLR0913
    input_dir: Path,
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "SberQuAD",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from SberQuAD parquet files and append results to JSONL files."""
    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    parquet_files = get_files(input_dir)

    for parquet_path in tqdm(parquet_files, desc="Parquet files"):
        source_filename = parquet_path.name
        df = pd.read_parquet(parquet_path)

        for source_idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {source_filename}",
        ):
            context = str(row["context"]).strip()
            question = str(row["question"]).strip()
            title = str(row["title"]).strip()

            answer = str(row["answers"]["text"][0]).strip()
            answer_start = row["answers"]["answer_start"]

            if not context or not question or not answer:
                continue

            context_id, context_id_counter = get_or_create_context(
                context,
                context_registry,
                doc_records,
                source_dataset,
                source_filename,
                context_id_counter,
                meta={
                    "title": title,
                    "source_idx": source_idx,
                },
            )

            # QA record
            qa_id_counter = add_qa(
                qa_records,
                qa_id_counter,
                question,
                answer,
                [context_id],
                meta={
                    "answer_start": answer_start,
                    "source_dataset": source_dataset,
                    "source_filename": source_filename,
                },
            )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )
    return context_id_counter, qa_id_counter, context_registry


def get_danetqa(  # noqa: PLR0913
    input_dir: Path,
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "DaNetQA",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from DaNetQA jsonl files and append results to JSONL files."""
    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    jsonl_files = get_files(input_dir)

    for jsonl_path in tqdm(jsonl_files, desc="JSONL files"):
        source_filename = jsonl_path.name
        df = pd.read_json(jsonl_path, lines=True, encoding="utf-8")

        for source_idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {source_filename}",
        ):
            context = str(row["passage"]).strip()
            question = str(row["question"]).strip()
            answer = "" if source_filename == "test.jsonl" else str(row["label"]).strip()

            if not context or not question:
                continue

            context_id, context_id_counter = get_or_create_context(
                context,
                context_registry,
                doc_records,
                source_dataset,
                source_filename,
                context_id_counter,
                meta={
                    "source_idx": source_idx,
                },
            )

            # QA record
            qa_id_counter = add_qa(
                qa_records,
                qa_id_counter,
                question,
                answer,
                [context_id],
                meta={
                    "source_dataset": source_dataset,
                    "source_filename": source_filename,
                },
            )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )
    return context_id_counter, qa_id_counter, context_registry


def get_croc(  # noqa: PLR0913
    input_dir: Path,  # folder with croc.jsonl and Раздел_3.txt
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "croc",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from croc file and append results to JSONL files."""
    jsonl_path = input_dir / "croc.jsonl"
    doc_path = input_dir / "croc_docs/test_docs/Раздел_3.txt"

    df = pd.read_json(jsonl_path, lines=True, encoding="utf-8")

    source_filename = jsonl_path.name

    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    with open(doc_path, encoding="utf-8") as f:
        context = f.read()

    for source_idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing {source_filename}",
    ):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        context_id, context_id_counter = get_or_create_context(
            context,
            context_registry,
            doc_records,
            source_dataset,
            source_filename,
            context_id_counter,
            meta={
                "source_idx": source_idx,
            },
        )

        # QA record
        qa_id_counter = add_qa(
            qa_records,
            qa_id_counter,
            question,
            answer,
            [context_id],
            meta={
                "source_dataset": source_dataset,
                "source_filename": source_filename,
            },
        )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )
    return context_id_counter, qa_id_counter, context_registry


def get_tape_multiq(  # noqa: PLR0913
    input_dir: Path,
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "TAPE_MultiQ",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from TAPE MultiQ jsonl files and append results to JSONL files."""
    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    jsonl_files = get_files(input_dir)

    for jsonl_path in tqdm(jsonl_files, desc="JSONL files"):
        source_filename = jsonl_path.name
        df = pd.read_json(jsonl_path, lines=True, encoding="utf-8")

        for source_idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {source_filename}",
        ):
            question = str(row["question"]).strip()
            texts = {
                "support_text": str(row["support_text"]).strip(),
                "main_text": str(row["main_text"]).strip(),
            }

            if not question or not texts:
                continue

            context_ids: list[int] = []

            for text_type, context in texts.items():
                context_id, context_id_counter = get_or_create_context(
                    context,
                    context_registry,
                    doc_records,
                    source_dataset,
                    source_filename,
                    context_id_counter,
                    meta={
                        "text_type": text_type,
                        "source_idx": source_idx,
                    },
                )
                context_ids.append(context_id)

            main_answers = row["main_answers"]

            if not main_answers:
                qa_id_counter = add_qa(
                    qa_records,
                    qa_id_counter,
                    question,
                    "",
                    context_ids,
                    meta={
                        "source_dataset": source_dataset,
                        "source_filename": source_filename,
                    },
                )
                continue
            else:
                # one QA-record for one answer
                for ans in main_answers:
                    answer = str(ans["segment"]).strip()
                    if not answer:
                        continue

                    qa_id_counter = add_qa(
                        qa_records,
                        qa_id_counter,
                        question,
                        answer,
                        [context_id],
                        meta={
                            "source_dataset": source_dataset,
                            "source_filename": source_filename,
                        },
                    )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )

    return context_id_counter, qa_id_counter, context_registry


def get_ontico(  # noqa: PLR0913
    input_dir: Path,  # folder with bench.csv and updated_meta.csv
    knowledge_base_dir: Path,  # folder knowledge_base
    qa_jsonl_path: Path,
    docs_jsonl_path: Path,
    context_registry: dict[str, ContextEntry],
    source_dataset: str = "ontico",
    start_context_id: int = 0,
    start_qa_id: int = 0,
) -> tuple[int, int, dict[str, ContextEntry]]:
    """Build dataset from ontico benchmark with subtitle-based context and append results to JSONL files.."""
    qa_records: list[QARecord] = []
    doc_records: list[DocRecord] = []

    context_id_counter = start_context_id
    qa_id_counter = start_qa_id

    bench_path = input_dir / "bench.csv"
    meta_path = input_dir / "updated_meta.csv"

    source_filename = bench_path.name

    bench_df = pd.read_csv(bench_path)
    meta_df = pd.read_csv(meta_path)

    # lecture_title -> video_id
    lecture_to_video = dict(
        zip(meta_df["lecture_title"], meta_df["video_id"], strict=False),
    )

    for source_idx, row in tqdm(
        bench_df.iterrows(),
        total=len(bench_df),
        desc=f"Processing {source_filename}",
    ):
        question = str(row["Вопрос"]).strip()
        answer = str(row["Ответ"]).strip()
        lecture_title = str(row["Доклад"]).strip()

        if not question or not lecture_title:
            continue

        video_id = lecture_to_video.get(lecture_title)
        if not video_id:
            continue

        srt_path = knowledge_base_dir / video_id / "clean.srt"
        if not srt_path.exists():
            continue

        raw_srt = srt_path.read_text(encoding="utf-8")
        context = clean_srt_text(raw_srt)

        if not context:
            continue

        context_id, context_id_counter = get_or_create_context(
            context,
            context_registry,
            doc_records,
            source_dataset,
            source_filename,
            context_id_counter,
            meta={
                "video_id": video_id,
                "lecture_title": lecture_title,
                "source_idx": source_idx,
            },
        )

        # QA record
        qa_id_counter = add_qa(
            qa_records,
            qa_id_counter,
            question,
            answer,
            [context_id],
            meta={
                "source_dataset": source_dataset,
                "source_filename": source_filename,
            },
        )

    qa_id_counter = save_data(
        qa_records,
        doc_records,
        qa_jsonl_path,
        docs_jsonl_path,
        start_qa_id,
    )

    return context_id_counter, qa_id_counter, context_registry
