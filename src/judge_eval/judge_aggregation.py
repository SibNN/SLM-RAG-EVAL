"""Aggregation judges metrics."""

import json
from pathlib import Path

import numpy as np

from src.shared.paths import JUDGE_EVAL_FOLDER

METRICS = [
    "Correctness",
    "AnswerRelevance",
    "ContextRelevance",
    "Faithfulness",
]


MetricScores = dict[str, float]
Samples = list[MetricScores]
Workers = dict[str, Samples]
Judges = dict[str, Workers]


class JudgeAggregator:
    """Aggregates judge results from folder.

    Structure like:
    JUDGE_EVAL_FOLDER/
        run_name/
            worker_0.jsonl
            worker_1.jsonl
            ...

    k = number of judges (run_name folders)
    x = concurrency (number of worker files)
    s = benchmark size (lines in file)
    """

    def __init__(self, judges_dir: Path, run_names: list[str] | None = None) -> None:
        """Initialize JudgeAggregator.

        Args:
            judges_dir (Path): the path to the directory where the judges scores are located (JUDGE_EVAL_FOLDER),
            as described in the class description.
            run_names (list[str] | None): names of the runs for aggregation.
            The names must match the names of the folders inside the judges_dir.

        """
        self.stats = {}
        self.run_names = run_names
        self.data = self._load_all(judges_dir)

        self.judges = list(self.data.keys())

        self.stats["judges"] = {f"{judge}_runs": len(workers) for judge, workers in self.data.items()}
        self.stats["judges"]["judges_number"] = len(self.judges)

        # calculating benchmark size
        first_run = next(iter(self.data.values()))
        first_worker = next(iter(first_run.values()))
        self.stats["benchmark_size"] = len(first_worker)

    def _load_metrics_jsonl(self, path: Path) -> Samples:
        """Load metrics from jsonl file.

        Args:
            path (Path): path for jsonl file

        Returns:
            List[Dict[str, float]]: list of dictionaries with metrics
            like [{'metic_1': score_1, 'metric_2': score_2}, ...]

        """
        rows: Samples = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)

                metrics = {m: sample["metrics"][m]["score"] for m in METRICS}

                rows.append(metrics)

        return rows

    def _load_all(self, judges_dir: Path) -> Judges:
        """Load all data from judges directory.

        data structure:

        data[judge][worker][sample][metric]

        Args:
            judges_dir (Path): the path to the directory where the judges scores are located

        """
        data: Judges = {}

        for judge_dir in judges_dir.iterdir():
            if not judge_dir.is_dir():
                continue

            # if run_names are specified, the data is loaded only from the specified runs
            if self.run_names is not None and judge_dir.name not in self.run_names:
                continue

            judge_name = judge_dir.name
            workers: Workers = {}

            for worker_file in judge_dir.glob("*.jsonl"):
                worker_id = worker_file.stem
                workers[worker_id] = self._load_metrics_jsonl(worker_file)

            data[judge_name] = workers

        return data

    def _empty_metrics(self) -> dict[str, list]:
        """Get the empty metrics sample.

        Returns:
            Dict[str, List]: empty metrics, for each metric empty list

        """
        return {m: [] for m in METRICS}

    def _mean(self, metric_values: dict[str, list[float]]) -> dict[str, float]:
        """Count mean value for metrics, where metrics.

        {'metric_1': [val_1, val_2, ...], ...}

        Args:
            metric_values (Dict[str, List[float]]):  dictionary with metrics

        Returns:
            Dict[str, float]: average values for each metric

        """
        return {m: float(np.mean(metric_values[m])) for m in METRICS}

    def aggregate_all(self) -> list[dict[int, MetricScores]]:
        """Averaging of all metrics for each of the benchmark questions.

        Output size = s

        Returns:
            List[Dict[int, MetricScores]]: list of average metrics for each question from the benchmark

        """
        result = []

        for sample_id in range(self.stats["benchmark_size"]):
            metric_values = self._empty_metrics()

            for judge in self.judges:
                for worker in self.data[judge]:
                    metrics = self.data[judge][worker][sample_id]

                    for m in METRICS:
                        metric_values[m].append(metrics[m])

            result.append({sample_id: self._mean(metric_values)})
        return result

    def aggregate_judges_only(self) -> dict[str, list[MetricScores]]:
        """Aggregate evaluation scores across judges.

        This method averages metric scores across all judges (k) for each worker (x) and benchmark sample (s).
        The concurrency dimension is preserved, meaning results remain separated by worker and question.
        Shape:
            k * x * s -> x * s

        Returns:
            dict[str, List[MetricScores]]:

        """
        result: dict[str, list[MetricScores]] = {}

        all_workers = {worker for judge in self.data.values() for worker in judge}

        for worker in all_workers:
            worker_results: list[MetricScores] = []
            for sample_id in range(self.stats["benchmark_size"]):
                metric_values = self._empty_metrics()

                for judge in self.judges:
                    metrics = self.data[judge][worker][sample_id]

                    for m in METRICS:
                        metric_values[m].append(metrics[m])

                worker_results.append(self._mean(metric_values))

            result[worker] = worker_results

        return result

    def save_result(
        self,
        result: dict[str, list[MetricScores]] | list[dict[int, MetricScores]],
        output_path: Path,
    ) -> None:
        """Save the aggregation result by specified path.

        This method saves the aggregation result.
        list result -> jsonl
        dict result -> json

        Raises:
            TypeError: result does not match the specified formats.

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(result, list):
            with open(output_path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
                for row in result:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif isinstance(result, dict):
            with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            raise TypeError("Unsupported type of result, result should be either dicr or list.")


if __name__ == "__main__":
    agg = JudgeAggregator(JUDGE_EVAL_FOLDER, run_names=["gpt"])

    print(f"stats:\n{agg.stats}\n")

    agg1 = agg.aggregate_all()
    agg2 = agg.aggregate_judges_only()

    print(f"aggregation over all for each question (k*x*s -> s):\n\n{agg1}\n\n")
    print(f"aggregation over judges, but keep concurency (k*x*s -> x*s):\n\n{agg2}")
