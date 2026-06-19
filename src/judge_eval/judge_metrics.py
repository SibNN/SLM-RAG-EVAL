"""Creating metrics for judges evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn.metrics import f1_score

from src.judge_eval.judge_aggregation import METRICS
from src.shared.paths import JUDGE_EVAL_FOLDER


class JudgeAnalysis:
    """Generates metrics for evaluating judges.

    Available metrics:
    - correlation_matrix
    - f1_over_golden
    - average_bad_score
    - icc
    - disagreement
    """

    def __init__(self, judge_files: list[Path]) -> None:
        """Initialize JudgeAnalysis with specified files.

        Args:
            judge_files(list[Path]): paths for judges evaluation.

        """
        self.judge_files = judge_files
        self.judge_names = [f.stem for f in judge_files]

        self.rows = self._load_all()
        self.data = pd.DataFrame(self.rows)

    def _load_all(self) -> list[dict[str, float | int]]:
        """Load files with the results of judges scores specified during initialization.

        Returns:
            list[dict[str, float | int]]: list of rows with judges scores.

        """
        rows = []
        for judge_file in self.judge_files:
            judge_name = judge_file.stem
            with open(judge_file, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    record = {
                        "judge": judge_name,
                        "item_id": i,
                        "dataset": data["dataset"],
                        "question_type": data["question_type"],
                        "golden": int(data["actual_output"] == data["expected_output"]),
                    }

                    for m in METRICS:
                        record[m] = data["metrics"][m]["score"]

                    rows.append(record)

        return rows

    def f1_over_golden(self, threshold: float = 0.5) -> dict[str, float]:
        """Compute F1 score for each judge on questions where a golden answer exists.

        Counts the f1 score for each of the judges.
        The metric is considered only for samples where the field is golden==1.
        For such samples, we know for sure that the answer is correct and look at the accuracy metric.

        Args:
            threshold (float): threshold to convert Correctness score into binary prediction.

        Returns:
            dict[str, float]: F1 score per judge.

        """
        results = {}

        for judge in self.judge_names:
            filtered = self.data[self.data["judge"] == judge]

            preds = (filtered["Correctness"] >= threshold).astype(int)
            results[judge] = f1_score(filtered["golden"], preds)

        return results

    def average_bad_score(self) -> dict[str, dict[str, float]]:
        """Generate average bad score for judges.

        Generate average metrics for golden==0 for each of the judges,
        ie the average metric scores based on bad examples.

        Returns:
            dict[str, dict[str, float]]: dictionary with metrics for each judge.

        """
        results = {}
        for judge in self.judge_names:
            df = self.data[(self.data["judge"] == judge) & (self.data["golden"] == 0)]
            results[judge] = df[METRICS].mean().to_dict()

        return results

    def correlation_matrix(self, metric: str = "Correctness") -> pd.DataFrame:
        """Create correlation matrix between judges by specified metric.

        Args:
            metric (str): any metric from the list (METRICS).

        Returns:
            pd.DataFrame: correlation matrix.

        Raises:
            ValueError: if specified an incorrect metric or its name.

        """
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric, specify one metric from the list:\n{METRICS}",
            )
        pivot = self.data.pivot(
            index="item_id",
            columns="judge",
            values=metric,
        )
        pivot["mean_judges"] = pivot.mean(axis=1)

        return pivot.corr()

    def plot_correlation_heatmap(self, metric: str = "Correctness") -> None:
        """Count correlation matrix and draws heatmap.

        Args:
            metric (str): any metric from the list (METRICS).

        Raises:
            ValueError: if specified an incorrect metric or its name.

        """
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric, specify one metric from the list:\n{METRICS}",
            )
        corr = self.correlation_matrix(metric)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        plt.title(f"Judge correlation ({metric})")
        plt.show()

    def list_judges(self) -> list[str]:
        """Return the list of unique judges available in the dataset.

        Returns:
            list[str]: list of judges in the dataset.

        """
        return self.data["judge"].unique().tolist()

    def icc(
        self,
        metric: str = "Correctness",
        judges: list[str] | None = None,
        include_golden: bool = False,
    ) -> float:
        """Compute the icc in different ways.

        - between judges (specify list of judges >= 2, and include_golden=False)
        - between judges and golden (specify list of judges >= 2, and include_golden=True)
        - single judge and golden (specify list of judges(len==1) and include_golden=True)

        Args:
            metric(str): any metric from the list (METRICS).
            judges(list[str] | None): list of judges, if not specified use all available judges.
            include_golden(bool): flag that allows to enable golden_answers while calculating.

        Returns:
            float: the icc score

        Raises:
            ValueError: if specified an incorrect metric or its name.

        """
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric, specify one metric from the list:\n{METRICS}",
            )

        data = self.data.copy()

        data = data.rename(columns={metric: "score"})

        # loading only selected judges
        if judges is not None:
            data = data[data["judge"].isin(judges)]

        if include_golden:
            # select only unique item_id
            data_golden = data.groupby("item_id")["golden"].first().reset_index()
            data_golden["judge"] = "golden"
            data_golden = data_golden.rename(columns={"golden": "score"})
            data = pd.concat(
                [data, data_golden[["item_id", "judge", "score"]]],
                ignore_index=True,
            )
        else:
            data = data[["item_id", "judge", "score"]]

        # check if judges more than 2
        if data["judge"].nunique() <= 1:
            raise ValueError("At least 2 judges are required to count the ICC.")

        icc = pg.intraclass_corr(
            data=data,
            targets="item_id",
            raters="judge",
            ratings="score",
            nan_policy="omit",
        )

        return icc.loc[icc["Type"] == "ICC2", "ICC"].values[0]

    def disagreement(self, metric: str = "Correctness") -> float:
        """Compute the average disagreement between judges by a specified metric.

        Args:
            metric (str): any metric from the list (METRICS).

        Returns:
            float: mean standard deviation of judges scores.

        Raises:
            ValueError: if specified an incorrect metric or its name.

        """
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric, specify one metric from the list:\n{METRICS}",
            )

        pivot = self.data.pivot(
            index="item_id",
            columns="judge",
            values=metric,
        )

        return pivot.std(axis=1).mean()


if __name__ == "__main__":
    test_path = JUDGE_EVAL_FOLDER / "judge_mixup"
    judge_files = [metrics_path for metrics_path in test_path.glob("*.jsonl")]

    analysis = JudgeAnalysis(judge_files)

    print(f"ICC:\n{analysis.icc()}")
    print(f"Correlation matrix:\n{analysis.correlation_matrix()}")
    print(f"F1:\n{analysis.f1_over_golden()}")
    print(f"Average bad score:\n{analysis.average_bad_score()}")
