"""Non llm metrics description."""

from collections import defaultdict

import spacy
from pymorphy3 import MorphAnalyzer
from rouge import Rouge
from spacy.language import Language


def lemmatize_text(text: str, morph: MorphAnalyzer, nlp: Language) -> str:
    """Normalize text.

    Args:
        text: text to normalize
        morph: morph analyzer
        nlp: spacy model

    Returns:
        str: normalized text

    """
    doc = nlp(text)
    lemmas = [morph.parse(token.text)[0].normal_form for token in doc]
    return " ".join(lemmas)


def calculate_rouge(
    cands: list[str],
    refs: list[list[str]],
    use_stemmer: bool = False,
    spacy_model: str | None = None,
) -> dict[str, list[float]]:
    """Calculate Rouge scores.

    Including Rouge-1, Rouge-2 and Rouge-L score.

    Args:
        cands: predicted values
        refs: list of ground thruth values
        use_stemmer: if there is a need to normalize texts
        spacy_model: model to normalize texts

    Returns:
        dict: evaluation results.

    """
    scorer = Rouge()
    morph, nlp = None, None
    if use_stemmer:
        morph = MorphAnalyzer()
        nlp = spacy.load(spacy_model)

    res = defaultdict(list)
    empty_pred_count = 0

    for pred, ref_list in zip(cands, refs, strict=False):
        if use_stemmer:
            predicted = lemmatize_text(pred, morph, nlp)
            reference = [lemmatize_text(ref, morph, nlp) for ref in ref_list]
        else:
            predicted, reference = pred, ref_list

        if not pred.strip():
            empty_pred_count += 1

        sample_scores = []
        for ref in reference:
            if not pred.strip():
                sample_scores.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            else:
                s = scorer.get_scores(predicted, ref)[0]  # get_scores expects single strings
                sample_scores.append(
                    {
                        "rouge1": s["rouge-1"]["f"],
                        "rouge2": s["rouge-2"]["f"],
                        "rougeL": s["rouge-l"]["f"],
                    },
                )

        for k in ["rouge1", "rouge2", "rougeL"]:
            res[k].append(max(s[k] for s in sample_scores))
    res_dict = dict(res)
    return res_dict


def calculate_hit_at_k(
    predicted: list[list],
    golden: list[list],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Calculate hit@k for multiple relevant documents.

    Args:
        predicted: list of predicted documetns
        golden: list of ground truth documents
        ks: size of k
            defaults to list of [1, 3, 5, 10]

    Returns:
        dict: calculated hit rate on the given documents lists

    """
    if len(predicted) != len(golden):
        raise ValueError(
            f"Length of predicted and golden should match. Got predicted:{len(predicted)} and golden:{len(golden)}",
        )
    if len(golden) == 0:
        raise ValueError("Length of passed arguments shouldn't be 0.")

    hit_rate: dict[str, float] = {}
    ks = ks if ks else [1, 3, 5, 10]
    for k in ks:
        hits = 0
        for preds, refs in zip(predicted, golden, strict=False):
            if not refs or not preds:
                continue
            top_k = preds[:k]
            if any(doc in refs for doc in top_k):
                hits += 1
        hit_rate[f"hit@{k}"] = hits / len(golden)

    return hit_rate


def calculate_recall_at_k(
    predicted: list[list],
    golden: list[list],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Calculate recall@k for multiple relevant documents.

    Args:
        predicted: list of predicted documetns
        golden: list of ground truth documents
        ks: size of k
            defaults to list of [1, 3, 5, 10]

    Returns:
        dict: calculated recall for given ks

    """
    if len(predicted) != len(golden):
        raise ValueError(
            f"Length of predicted and golden should match. Got predicted:{len(predicted)} and golden:{len(golden)}",
        )
    if len(golden) == 0:
        raise ValueError("Length of passed arguments shouldn't be 0.")

    ks = ks or [1, 3, 5, 10]
    recall: dict[str, float] = {}

    for k in ks:
        total_recall = 0.0
        for preds, refs in zip(predicted, golden, strict=False):
            if not refs or not preds:
                continue
            top_k = preds[:k]
            hit_count = sum(1 for doc in refs if doc in top_k)
            total_recall += hit_count / len(refs)
        recall[f"recall@{k}"] = total_recall / len(golden)

    return recall


def calculate_mrr(
    predicted: list[list],
    golden: list[list],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        predicted: list of predicted documetns
        golden: list of ground truth documents

    Returns:
        dict: calculated mrr for given documents

    """
    if len(predicted) != len(golden):
        raise ValueError(
            f"Length of predicted and golden should match. Got predicted:{len(predicted)} and golden:{len(golden)}",
        )
    if len(golden) == 0:
        raise ValueError("Length of passed arguments shouldn't be 0.")

    reciprocal_ranks = []

    for preds, refs in zip(predicted, golden, strict=False):
        rr = 0.0
        if not preds or not refs:
            continue
        for rank, doc in enumerate(preds, start=1):
            if doc in refs:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    if len(reciprocal_ranks) == 0:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
