"""Utils for text processing."""

import json
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def vectorize(
    model: SentenceTransformer,
    sentences: list[str],
    batch_size: int = 4,
) -> np.ndarray:
    """Vectorize texts.

    Args:
        model: sentence-transformers model
        sentences: text corpora
        batch_size: number of consecutive batches

    Returns:
        np.ndarray: encoded sentences

    """
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return embeddings


def tokenize(text: str) -> list[str]:
    """Normalize text.

    Make text lower, remove non text symbols.

    Args:
        text (str): text to normalize

    Returns:
        list[str]: lowercased text splitted by whitespace without non letter symbols

    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def get_mark_llm(llm_answer: str) -> float | None:
    """Parse llm's answer to get mark.

    Args:
        llm_answer: llm with json guided decoding answer

    Returns:
        float | None: score, if succeeded

    """
    try:
        answer = json.loads(llm_answer)
        mark = answer.get("mark")

        try:
            mark = float(mark)
            return mark
        except TypeError:
            return None
        except ValueError:
            return None
    except json.JSONDecodeError:
        return None


def to_json_safe(obj: pd.DataFrame | pd.Series | dict) -> dict:
    """Process convertation to the dict.

    Args:
        obj: item to convert to dict

    Returns:
        dict: dict item

    Raises:
        ValueError: when not sucessfull

    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="index")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    raise ValueError("Bad input, try other format")
