"""Evaluation functions description."""

import gc
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.bench_eval.utils.stats import mean_confidence_interval
from src.bench_eval.utils.text_processors import get_mark_llm, to_json_safe, tokenize, vectorize
from src.model.vllm import VllmModel
from src.shared.load_config import config
from src.shared.prompts import EVAL_ANSWER_PROMPT, EVAL_QUESTION_PROMPT


def hs_evaluation(texts: list[str]) -> dict:
    """Calculate average cosine similarity distace between texts.

    Args:
        texts (list[str]): texts to calculate distance on

    Returns:
        dict: evaluation results with average cosine similarity and top pairs

    """
    model = SentenceTransformer(
        config["embedder"]["name"],
        trust_remote_code=True,
        device="cuda",
    )
    # embeddings and cosine similarity
    embs = vectorize(model, texts)
    cos_sim = np.matmul(embs, embs.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # the upper-triangle
    upper_idx = np.triu_indices(len(texts), k=1)
    pairwise_sims = cos_sim[upper_idx]
    mean_sim, ci_95_err = mean_confidence_interval(pairwise_sims)

    # sorting indexes
    sorted_idx = np.argsort(-pairwise_sims)[:10]
    top_pairs = []
    for idx in sorted_idx:
        i = int(upper_idx[0][idx])
        j = int(upper_idx[1][idx])
        similarity = float(pairwise_sims[idx])
        top_pairs.append(((i, j), similarity))
    del model
    del embs
    del cos_sim
    gc.collect()
    torch.cuda.empty_cache()
    return {"result": mean_sim, "ci95": ci_95_err, "top_pairs": str(top_pairs)}


def intra_inter_diversity(texts: list[str], datasets: list[str]) -> dict:
    """Compute intra-inter domain diversity score.

    Computes intra-domain and inter-domain semantic similarity statistics
    for a collection of texts using dense embeddings and cosine similarity.

    This function is designed to quantify domain diversity across multiple
    datasets (domains) in RAG-style corpora. It answers the following questions:

    1. How semantically homogeneous are texts within each dataset?
       (intra-domain similarity)
    2. How semantically similar are texts between different datasets?
       (inter-domain similarity)
    3. Are datasets meaningfully different from each other in embedding space,
       or do they effectively cover the same semantic domain?
       (diversity ratio)

    The function embeds all input texts into a shared vector space, computes
    a full cosine similarity matrix, and then aggregates similarities based on
    dataset membership.

    Interpretation guidelines:
    - Intra-domain similarity (higher is more homogeneous):
        * >0.8: tightly focused domain
        * ~0.4–0.8: moderate topical spread
        * <0.3: very heterogeneous domain

    - Inter-domain similarity (higher is more overlapping):
        * Close to intra-domain values → domains are not distinct
        * Much lower than intra-domain values → strong domain separation

    - Diversity ratio = mean(inter-domain) / mean(intra-domain):
        the less, the less diverse each domain and more diverse between domains

    Args:
        texts (list[str]):
            List of input texts to be embedded. All texts must be in the same
            language and embedding space. Typically this is one of:
            - context
            - question
            - answer
            - or their concatenation

        datasets (list[str]):
            List of dataset/domain identifiers. Must be the same length as
            `texts`. Each entry indicates which dataset (domain) the
            corresponding text belongs to.

    Returns:
        dict with the following structure:

        {
            "intra_domain_similarity": dict[str, float | None],
                Mean cosine similarity within each dataset.
                If a dataset contains fewer than 2 samples, the value is None.

            "inter_domain_similarity": dict[str, float],
                Mean cosine similarity between every unordered pair of datasets.
                Keys are formatted as "<dataset1>__vs__<dataset2>".

            "mean_intra_similarity": float | None,
                Mean of all valid intra-domain similarities.

            "mean_inter_similarity": float | None,
                Mean of all inter-domain similarities.

            "diversity_ratio": float | None,
                Ratio of mean inter-domain similarity to mean intra-domain
                similarity. Serves as a single scalar measure of domain diversity.

            "num_datasets": int,
                Number of unique datasets (domains) observed.
        }

    """
    model = SentenceTransformer(
        config["embedder"]["name"],
        trust_remote_code=True,
        device="cuda",
    )

    embs = vectorize(model, texts)
    cos_sim = np.matmul(embs, embs.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    dataset_to_idx = {}
    for i, d in enumerate(datasets):
        dataset_to_idx.setdefault(d, []).append(i)
    intra_domain = {}
    min_num_datasets_to_eval = 2
    for d, idxs in dataset_to_idx.items():
        if len(idxs) < min_num_datasets_to_eval:
            intra_domain[d] = None
            continue

        sub = cos_sim[np.ix_(idxs, idxs)]
        upper = sub[np.triu_indices(len(idxs), k=1)]
        intra_domain[d] = float(upper.mean())

    inter_domain = {}
    dataset_names = list(dataset_to_idx.keys())
    for i, j in combinations(range(len(dataset_names)), 2):
        d1, d2 = dataset_names[i], dataset_names[j]
        idx1, idx2 = dataset_to_idx[d1], dataset_to_idx[d2]

        sub = cos_sim[np.ix_(idx1, idx2)]
        inter_domain[f"{d1}__vs__{d2}"] = float(sub.mean())

    intra_vals = [v for v in intra_domain.values() if v is not None]
    mean_intra, err_intra = mean_confidence_interval(intra_vals)
    inter_vals = list(inter_domain.values())
    mean_inter, err_inter = mean_confidence_interval(inter_vals)
    diversity_ratio = float(np.mean(inter_vals) / np.mean(intra_vals)) if intra_vals and inter_vals else None
    del model
    del embs
    del cos_sim
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "intra_domain_similarity": intra_domain,
        "inter_domain_similarity": inter_domain,
        "mean_intra_similarity": mean_intra if intra_vals else None,
        "ci95_intra": err_intra if intra_vals else None,
        "mean_inter_similarity": mean_inter if inter_vals else None,
        "ci95_inter": err_inter if inter_vals else None,
        "diversity_ratio": diversity_ratio,
        "num_datasets": len(dataset_names),
    }


def ngram_diversity(texts: list[str], n_gram: int = 4) -> dict:
    """Calculate relation between unique n_grams in text and number of n_grams in text.

    Args:
        texts (list[str]): texts to calculate
        n_gram (int): number of consecutive charachters in the text

    Returns:
        dict: evaluation results

    """
    n_grams = defaultdict(list)
    for text in texts:
        tokens = tokenize(text)
        for n in range(1, n_gram + 1):
            for i in range(len(tokens) - n + 1):
                n_grams[n].append(tuple(tokens[i : i + n]))

    diversity = defaultdict(float)
    for key, values in n_grams.items():
        diversity[key] = len(set(values)) / len(values)

    diversity["mean"] = float(np.mean(list(diversity.values())))

    return dict(diversity)


def stats(texts: list[str]) -> dict:
    """Get low level statistics, based on the given texts.

    Args:
        texts: list of texts to calculate stats

    Retuns:
        dict:
            min: minimum length of texts
            max: maximum length of texts
            mean: average lenght of texts
            median: median lenght of texts

    """
    number_of_words = [len(tokenize(text)) for text in texts]
    statistics = {
        "min": int(np.min(number_of_words)),
        "max": int(np.max(number_of_words)),
        "mean": float(round(np.mean(number_of_words), 2)),
        "median": float(np.median(number_of_words)),
    }
    return statistics


def llm_rank_question(model: VllmModel, questions: list[str]) -> dict:
    """Evaluate questions from 1 to 10 using LLM.

    Args:
        model: vllm model
        questions: list of strings with the questiions

    Returns:
        dict: llm rankings on given questions

    """
    marks = []
    skipped = 0

    for question in questions:
        message = [
            {"role": "system", "content": EVAL_QUESTION_PROMPT},
            {"role": "user", "content": f"question: {question}"},
        ]
        model_out = model.generate(
            messages=message,
            response_format={"type": "json_object"},
        )

        mark = get_mark_llm(model_out)
        if mark is not None:
            marks.append(mark)
        else:
            skipped += 1
    mean_sim, ci_95_err = mean_confidence_interval(marks)
    return {"result": mean_sim, "ci95": ci_95_err, "skipped": skipped}


def llm_rank_answer(model: VllmModel, questions: list[str], answers: list[str]) -> dict:
    """Rank golden answers using llm as a judge.

    Args:
        model: vllm models
        questions: list of questions
        answers: list of answers

    Returns:
        dict: results of evaluation

    """
    marks = []
    skipped = 0
    for question, answer in zip(questions, answers, strict=False):
        message = [
            {"role": "system", "content": EVAL_ANSWER_PROMPT},
            {"role": "user", "content": f"question: {question}\nanswer: {answer}"},
        ]
        model_out = model.generate(
            messages=message,
            response_format={"type": "json_object"},
        )

        mark = get_mark_llm(model_out)
        if mark is not None:
            marks.append({"question": question, "mark": mark})
        else:
            skipped += 1

    worst_results = sorted(marks, key=lambda x: x["mark"])[:20]
    marks_vals = [val["mark"] for val in marks]
    mean_sim, ci_95_err = mean_confidence_interval(marks_vals)
    return {
        "result": mean_sim,
        "ci95": ci_95_err,
        "skipped": skipped,
        "worst": worst_results,
    }


def question_type_stats(datasets: list[str], qa_cls: list[str]) -> dict:
    """Get low level support for your dataset.

    Mainly this function returns number of each classes grouped by given dataset columns.

    Args:
        datasets (list[str]): first column to group by
        qa_cls (list[str]): second column to group by

    Returns:
        dict: evaluation results with absolute and percet for both given columns

    """
    results = {}
    data = {"dataset": datasets, "question_type": qa_cls}
    df = pd.DataFrame(data)

    pivot_counts = pd.crosstab(
        df["dataset"],
        df["question_type"],
    )
    results["pivot_counts"] = pivot_counts

    pivot_percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
    results["pivot_percent"] = pivot_percent

    global_counts = df["question_type"].value_counts()
    global_percent = df["question_type"].value_counts(normalize=True) * 100

    results["global_counts"] = global_counts
    results["global_percent"] = global_percent

    mean_per_dataset = pivot_counts.mean(axis=0)
    mean_percent_per_dataset = pivot_percent.mean(axis=0)

    results["mean_per_dataset"] = mean_per_dataset
    results["mean_percent_per_dataset"] = mean_percent_per_dataset

    extended_stats = pivot_counts.describe().loc[["mean", "std", "min", "max"]]
    results["extended_stats"] = extended_stats

    summary = pd.DataFrame(
        {
            "global_count": global_counts,
            "global_percent": global_percent,
            "mean_per_dataset": mean_per_dataset,
            "mean_percent_per_dataset": mean_percent_per_dataset,
        },
    ).fillna(0)

    results["summary"] = summary
    return to_json_safe(results)
