"""Evaluation metrics for probabilistic forecasting.

Implements CRPS, Weighted Interval Score, Log Score, and other
metrics required for the Mosqlimate Sprint 2025 competition.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def crps_single(
    y_true: float,
    quantiles: np.ndarray,
    values: np.ndarray,
) -> float:
    """Compute CRPS for a single observation.

    Uses the quantile-based CRPS approximation.

    Args:
        y_true: True value
        quantiles: Array of quantile levels (0-1)
        values: Array of quantile predictions

    Returns:
        CRPS score
    """
    quantiles = np.asarray(quantiles)
    values = np.asarray(values)

    crps = 0.0
    for i, (q, v) in enumerate(zip(quantiles, values)):
        if i == 0:
            continue

        q_prev = quantiles[i - 1]
        v_prev = values[i - 1]

        crps += (v - v_prev) * (
            (y_true - v_prev) * (y_true >= v_prev) * (q_prev**2)
            + (y_true - v) * (y_true < v) * ((1 - q) ** 2)
            + (y_true - v_prev) * (y_true < v_prev) * ((q_prev - 1) ** 2)
            + (y_true - v) * (y_true >= v) * (q**2)
        )

    crps += np.sum(
        [
            2 * (y_true - v) * (y_true >= v) * q + 2 * (v - y_true) * (y_true < v) * (1 - q)
            for q, v in zip(quantiles, values)
        ]
    )

    return float(crps / len(quantiles))


def crps(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    quantile_cols: Optional[Dict[float, str]] = None,
) -> float:
    """Compute Continuous Ranked Probability Score.

    CRPS measures the integrated squared difference between the
    empirical CDF and the predicted CDF.

    Args:
        y_true: True values
        predictions: DataFrame with quantile predictions
        quantile_cols: Mapping of quantile levels to column names

    Returns:
        Mean CRPS
    """
    y_true = np.asarray(y_true).ravel()

    if quantile_cols is None:
        quantile_cols = {
            0.025: "lower_95",
            0.25: "lower_50",
            0.5: "median",
            0.75: "upper_50",
            0.975: "upper_95",
        }

    available_cols = {q: col for q, col in quantile_cols.items() if col in predictions.columns}

    if not available_cols:
        return float("nan")

    quantiles = sorted(available_cols.keys())
    values = np.column_stack([predictions[available_cols[q]].values for q in quantiles])

    crps_values = []
    for y, v in zip(y_true, values):
        crps_values.append(crps_single(y, quantiles, v))

    return float(np.mean(crps_values))


def weighted_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    median: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute Weighted Interval Score for a prediction interval.

    WIS combines interval sharpness with penalty for misses.

    Args:
        y_true: True values
        lower: Lower bound predictions
        upper: Upper bound predictions
        median: Median predictions
        alpha: 1 - confidence level (e.g., 0.05 for 95% interval)

    Returns:
        WIS value
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    median = np.asarray(median).ravel()

    interval_width = upper - lower

    below_lower = np.maximum(0, lower - y_true)
    above_upper = np.maximum(0, y_true - upper)

    penalty = (2 / alpha) * (below_lower + above_upper)

    wis = (alpha / 2) * interval_width + penalty

    wis += np.abs(y_true - median)

    return float(np.mean(wis))


def weighted_interval_score_total(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    levels: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
) -> float:
    """Compute total Weighted Interval Score across multiple levels.

    Args:
        y_true: True values
        predictions: DataFrame with quantile predictions
        levels: Confidence levels (e.g., [0.50, 0.80, 0.95])
        weights: Weights for each level

    Returns:
        Total WIS
    """
    y_true = np.asarray(y_true).ravel()
    levels = levels or [0.50, 0.80, 0.95]
    weights = weights or [1 / 3, 1 / 3, 1 / 3]

    total_wis = 0.0
    total_weight = 0.0
    for level, weight in zip(levels, weights):
        alpha = 1 - level
        col_name = int(level * 100)

        lower_col = f"lower_{col_name}"
        upper_col = f"upper_{col_name}"

        if lower_col not in predictions.columns or upper_col not in predictions.columns:
            continue

        lower = np.asarray(predictions[lower_col].values)
        upper = np.asarray(predictions[upper_col].values)
        median = np.asarray(predictions["median"].values)

        wis = weighted_interval_score(y_true, lower, upper, median, alpha)
        total_wis += weight * wis
        total_weight += weight

    if total_weight == 0:
        return float("nan")

    return total_wis / total_weight


def logarithmic_score(
    y_true: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Compute Logarithmic Score assuming normal distribution.

    Args:
        y_true: True values
        median: Median predictions
        scale: Scale (std) of predictions

    Returns:
        Mean log score
    """
    y_true = np.asarray(y_true).ravel()
    median = np.asarray(median).ravel()
    scale = np.asarray(scale).ravel()

    scale = np.maximum(scale, 1e-6)

    z = (y_true - median) / scale

    log_score = -stats.norm.logpdf(z) - np.log(scale)

    return float(np.mean(log_score))


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute prediction interval coverage.

    Args:
        y_true: True values
        lower: Lower bound predictions
        upper: Upper bound predictions

    Returns:
        Coverage rate (0-1)
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()

    in_interval = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(in_interval))


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute mean prediction interval width.

    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions

    Returns:
        Mean interval width
    """
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    return float(np.mean(upper - lower))


def sharpness(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    relative: bool = True,
) -> float:
    """Compute sharpness (precision) of prediction intervals.

    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions
        y_true: True values (for relative sharpness)
        relative: Whether to normalize by true values

    Returns:
        Sharpness score
    """
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()

    widths = upper - lower

    if relative and y_true is not None:
        y_true = np.asarray(y_true).ravel()
        y_true = np.maximum(y_true, 1)
        widths = widths / y_true

    return float(np.mean(widths))


def bias(
    y_true: np.ndarray,
    median: np.ndarray,
) -> float:
    """Compute bias (systematic error) of predictions.

    Args:
        y_true: True values
        median: Median predictions

    Returns:
        Bias value
    """
    y_true = np.asarray(y_true).ravel()
    median = np.asarray(median).ravel()
    return float(np.mean(median - y_true))


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    metric: str = "rmse",
) -> float:
    """Compute skill score relative to baseline.

    Args:
        y_true: True values
        y_pred: Model predictions
        y_baseline: Baseline predictions
        metric: Metric to use ('rmse', 'mae', 'mape')

    Returns:
        Skill score (positive = better than baseline)
    """
    if metric == "rmse":
        model_score = rmse(y_true, y_pred)
        baseline_score = rmse(y_true, y_baseline)
    elif metric == "mae":
        model_score = mae(y_true, y_pred)
        baseline_score = mae(y_true, y_baseline)
    elif metric == "mape":
        model_score = mape(y_true, y_pred)
        baseline_score = mape(y_true, y_baseline)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return 1 - (model_score / baseline_score)


def evaluate_forecast(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Comprehensive forecast evaluation.

    Args:
        y_true: True values
        predictions: DataFrame with predictions and intervals
        levels: Confidence levels to evaluate

    Returns:
        Dictionary with all evaluation metrics
    """
    y_true = np.asarray(y_true).ravel()
    levels = levels or [0.50, 0.80, 0.95]

    valid_mask = ~predictions["median"].isna()
    y_true = y_true[valid_mask]
    predictions = predictions[valid_mask].copy()

    results = {
        "rmse": rmse(y_true, predictions["median"]),
        "mae": mae(y_true, predictions["median"]),
        "mape": mape(y_true, predictions["median"]),
        "bias": bias(y_true, predictions["median"]),
    }

    for level in levels:
        col_name = int(level * 100)
        alpha = 1 - level

        lower_col = f"lower_{col_name}"
        upper_col = f"upper_{col_name}"

        if lower_col in predictions.columns and upper_col in predictions.columns:
            results[f"coverage_{col_name}"] = coverage(
                y_true, predictions[lower_col], predictions[upper_col]
            )
            results[f"wis_{col_name}"] = weighted_interval_score(
                y_true, predictions[lower_col], predictions[upper_col], predictions["median"], alpha
            )
            results[f"width_{col_name}"] = interval_width(
                predictions[lower_col], predictions[upper_col]
            )

    results["crps"] = crps(y_true, predictions)

    results["wis_total"] = weighted_interval_score_total(y_true, predictions, levels)

    return results


class ForecastEvaluator:
    """Evaluator class for forecast comparison and analysis.

    Args:
        levels: Confidence levels to evaluate
        baseline_method: Baseline method for skill scores
    """

    def __init__(
        self,
        levels: Optional[List[float]] = None,
        baseline_method: str = "naive",
    ):
        self.levels = levels or [0.50, 0.80, 0.95]
        self.baseline_method = baseline_method
        self.results: Dict[str, Dict[str, float]] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        predictions: pd.DataFrame,
        model_name: str,
    ) -> Dict[str, float]:
        """Evaluate a model's predictions.

        Args:
            y_true: True values
            predictions: DataFrame with predictions
            model_name: Name for storing results

        Returns:
            Evaluation metrics
        """
        metrics = evaluate_forecast(y_true, predictions, self.levels)
        self.results[model_name] = metrics
        return metrics

    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models.

        Returns:
            DataFrame with model comparison
        """
        return pd.DataFrame(self.results).T

    def get_best_model(self, metric: str = "crps") -> str:
        """Get best model for a given metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Name of best model
        """
        scores = {name: results[metric] for name, results in self.results.items()}
        return min(scores, key=scores.get)

    def summary(self) -> str:
        """Generate summary report.

        Returns:
            Summary string
        """
        df = self.compare_models()

        lines = ["=" * 60]
        lines.append("FORECAST EVALUATION SUMMARY")
        lines.append("=" * 60)

        for metric in ["rmse", "mae", "mape", "crps", "wis_total"]:
            if metric in df.columns:
                lines.append(f"\n{metric.upper()}:")
                for model, value in df[metric].sort_values().items():
                    lines.append(f"  {model}: {value:.4f}")

        for level in self.levels:
            col_name = int(level * 100)
            cov_col = f"coverage_{col_name}"
            if cov_col in df.columns:
                lines.append(f"\nCoverage {col_name}%:")
                for model, value in df[cov_col].sort_values(ascending=False).items():
                    lines.append(f"  {model}: {value:.2%}")

        return "\n".join(lines)
