"""Dataclasses description."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTestConfig:
    """Config value for datasets evaluation."""

    func: Callable
    columns: Sequence[str | tuple[str, ...]]
    LLM: bool


@dataclass
class QARecord:
    """Question-answer record."""

    id: int
    context_ids: list[int]
    question: str
    answer: str
    meta: dict[str, any]


@dataclass
class DocRecord:
    """Document record."""

    context_id: int
    context: str
    meta: dict[str, any]
    source_dataset: list[str]
    source_filename: list[str]


@dataclass
class ContextEntry:
    """Registry entry for a unique context string."""

    context_id: int
    source_datasets: set[str]
    source_filenames: set[str]


@dataclass(frozen=True)
class JoinKey:
    """Unique key for merging results by model, mode, question, answer."""

    model: str
    mode: str
    question: str
    answer: str


@dataclass
class MetricsRow:
    """LLM-metrics for a single generated answer."""

    correctness: float | None
    faithfulness: float | None
    answer_relevance: float | None
    context_relevance: float | None


@dataclass
class RAGAnswerRow:
    """Input style for llm as judge and non llm evaluation approaches."""

    question_id: int
    question: str
    answer: str
    golden_answer: str
    doc_ids: list[int] | None
    golden_doc_ids: list[int] | None
    doc_contents: list[str] | None
    question_type: str | None
    dataset: str
