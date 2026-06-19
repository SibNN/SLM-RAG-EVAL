### SLM-RAG-EVAL BENCHMARK

## GET STARTED

```
uv sync
source .venv/bin/activate
```

setup pre-commit

```
pre-commit install
```

## Benchmark evaluation

here is usage example

```
python src/bench_eval/benchmark_evaluation.py -i data/datasets/common -o data/output/final.json -n 50
```

## RAG evaluation

```
python src/rag_eval/run_bench.py data/datasets/base/wonan_400.jsonl name_of_the_run
```

## Judge evaluation

```
python src/judge_eval/generate_ratings.py data/datasets/mixed/wonan_400.jsonl judge_mixup/qwen8b
```

to save evaluation results on mixed dataset in data/judge_eval/mixed/qwen8b.jsonl file
