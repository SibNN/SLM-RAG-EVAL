# SLM-RAG-EVAL Benchmark

This repository contains code and supporting materials for the paper **“Little Brains, Big Feats: Exploring Compact Language Models”**, accepted to the **ECML PKDD 2026 Applied Data Science track**.

## Data

The public reproducibility package is available on Zenodo:

https://doi.org/10.5281/zenodo.20771850

Download `data.zip` from Zenodo and extract it into the repository root so that the `data/` directory has the following structure:

```text
data/
  bench/
  datasets/
  judge_eval/
```

The archive contains the public reproducibility package for the paper. It includes the 400-example public evaluation subset, the corresponding public knowledge base, model outputs, judge scores, and intermediate evaluation files.

The proprietary 100-example subset used in the full 500-example benchmark is not included due to licensing restrictions.

## Installation

We use `uv` for dependency management:

```bash
uv sync
source .venv/bin/activate
```

To set up pre-commit hooks, run:

```bash
pre-commit install
```

## Benchmark Evaluation

Example usage:

```bash
python src/bench_eval/benchmark_evaluation.py \
  -i data/datasets/base/wonan_400.jsonl \
  -o data/output/final.json \
  -n 50
```

## RAG Evaluation

Run RAG evaluation on the public 400-example subset:

```bash
python src/rag_eval/run_bench.py \
  data/datasets/base/wonan_400.jsonl \
  name_of_the_run
```

## Judge Evaluation

Generate judge ratings for the mixed evaluation dataset:

```bash
python src/judge_eval/generate_ratings.py \
  data/datasets/mixed/wonan_400.jsonl \
  judge_mixup/qwen8b
```

The evaluation results will be saved to:

```text
data/judge_eval/mixed/qwen8b.jsonl
```

## License

The source code in this repository is released under the MIT License.     
The released data package is distributed under the license specified in the Zenodo record. The archive contains curated samples and derived materials from third-party public datasets; please refer to the dataset documentation and third-party licenses when reusing the data.
