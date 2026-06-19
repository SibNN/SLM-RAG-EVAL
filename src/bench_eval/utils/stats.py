"""Statisical utils needed for evaluation."""

import math

from scipy import stats


def mean_confidence_interval(
    data: list[float],
    confidence: float = 0.95,
) -> tuple[float, float] | tuple[None, None]:
    """Calculate CI on given list.

    Args:
        data: input data
        confidence: level of confidence interval
            - defaults to 0.95

    Returns:
        tuple | None: mean and error range for given confidence value, if exists

    """
    if not len(data) > 1:
        return (None, None)

    n = len(data)
    mean = sum(data) / n

    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(variance)

    sem = std / math.sqrt(n)

    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    error = float(t_value * sem)
    mean = float(mean)
    return mean, error
