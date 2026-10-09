"""Evaluation metrics for probabilistic forecasting.

Implements CRPS, Weighted Interval Score (Bracher et al. 2021), Log
Score, and other metrics used for the Mosqlimate IMDC competition
(the official scoring metric is the Weighted Interval Score, computed
weekly, over the 50/80/90/95% prediction intervals).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from mosqlimate_ai.evaluation.quantiles import (
    CONFIDENCE_LEVELS,
    crps_from_quantiles,
    interval_score,
    pinball_loss,
    quantile_col_name,
    wis_total_from_intervals,
)

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
    """Compute CRPS for a single observation from quantile predictions.

    Uses the standard quantile approximation
    ``CRPS = (2 / n_tau) * sum_tau pinball(y, z_tau, tau)``.

    Args:
        y_true: True value
        quantiles: Array of quantile levels (0-1)
        values: Array of quantile predictions

    Returns:
        CRPS score (non-negative)
    """
    quantiles = np.asarray(quantiles, dtype=float)
    values = np.asarray(values, dtype=float)

    if quantiles.size == 0:
        return float("nan")

    y_arr = np.full_like(values, y_true, dtype=float)
    losses = pinball_loss(y_arr, values, quantiles)
    return float(2.0 * np.mean(losses))


def _as_quantile_frame(
    predictions: pd.DataFrame,
    quantile_cols: Optional[dict[float, str]] = None,
) -> pd.DataFrame:
    """Normalize legacy interval columns or a custom mapping to quantile columns."""
    if quantile_cols is not None:
        available = {
            quantile_col_name(q): predictions[col]
            for q, col in quantile_cols.items()
            if col in predictions.columns
        }
        return pd.DataFrame(available, index=predictions.index)

    if any(col in predictions.columns for col in ("q025", "q050", "q500")):
        return predictions

    return intervals_to_quantiles_compat(predictions)


def intervals_to_quantiles_compat(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a quantile frame from submission-style interval columns."""
    mapping = {
        "lower_95": 0.025,
        "lower_90": 0.05,
        "lower_80": 0.10,
        "lower_50": 0.25,
        "median": 0.50,
        "upper_50": 0.75,
        "upper_80": 0.90,
        "upper_90": 0.95,
        "upper_95": 0.975,
    }
    available = {
        quantile_col_name(tau): predictions[col]
        for col, tau in mapping.items()
        if col in predictions.columns
    }
    return pd.DataFrame(available, index=predictions.index)


def crps(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    quantile_cols: Optional[dict[float, str]] = None,
) -> float:
    """Compute Continuous Ranked Probability Score.

    Accepts either canonical quantile columns (``q025``..``q975``),
    submission-style interval columns (``median``, ``lower_50``..), or an
    explicit ``quantile_cols`` mapping. Uses the quantile approximation
    ``CRPS ~ 2 x mean pinball loss``.

    Args:
        y_true: True values
        predictions: DataFrame with quantile predictions
        quantile_cols: Optional mapping of quantile levels to column names

    Returns:
        Mean CRPS (nan when no usable quantile columns exist)
    """
    y_true = np.asarray(y_true).ravel()

    qframe = _as_quantile_frame(predictions, quantile_cols)
    if qframe.shape[1] == 0:
        return float("nan")

    return crps_from_quantiles(y_true, qframe)


def weighted_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    median: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Interval score IS_alpha for a single prediction interval.

    ``IS_alpha = (u - l) + (2/alpha)(l - y) 1(y < l) + (2/alpha)(y - u) 1(y > u)``
    following Bracher et al. (2021).

    Args:
        y_true: True values
        lower: Lower bound predictions
        upper: Upper bound predictions
        median: Median predictions (unused; kept for API compatibility)
        alpha: 1 - confidence level (e.g., 0.05 for 95% interval)

    Returns:
        Mean interval score value
    """
    del median
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()

    return float(np.mean(interval_score(y_true, lower, upper, alpha)))


def weighted_interval_score_total(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    levels: Optional[list[float]] = None,
    weights: Optional[list[float]] = None,
) -> float:
    """Compute total Weighted Interval Score (Bracher et al. 2021).

    ``WIS = 1/(K + 1/2) ( w_0 |y - m| + sum_k w_k IS_{alpha_k} )`` with the
    standard weights ``w_0 = 1/2``, ``w_k = alpha_k / 2``. ``weights`` is
    accepted for API compatibility; when provided it must be a list of
    ``alpha_k / 2`` values matching ``levels`` (custom weighting).

    Args:
        y_true: True values
        predictions: DataFrame with median and lower_X/upper_X columns
        levels: Confidence levels (default 0.50/0.80/0.90/0.95)
        weights: Optional custom per-level weights (alpha_k/2 each)

    Returns:
        Total WIS (nan when no interval level is available)
    """
    if weights is not None:
        # Custom weighting path: WIS = sum_k w_k IS_{alpha_k} / sum_k w_k
        y_arr = np.asarray(y_true).ravel()
        total = 0.0
        total_w = 0.0
        for level, weight in zip(levels, weights):
            name = int(level * 100)
            lower_col, upper_col = f"lower_{name}", f"upper_{name}"
            if lower_col not in predictions.columns or upper_col not in predictions.columns:
                continue
            total = total + weight * interval_score(
                y_arr,
                predictions[lower_col].values,
                predictions[upper_col].values,
                1 - level,
            )
            total_w += weight
        if total_w == 0:
            return float("nan")
        return float(np.mean(total / total_w))

    return wis_total_from_intervals(np.asarray(y_true).ravel(), predictions, levels)


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
    levels: Optional[list[float]] = None,
) -> dict[str, float]:
    """Comprehensive forecast evaluation.

    Args:
        y_true: True values
        predictions: DataFrame with predictions and intervals
        levels: Confidence levels to evaluate

    Returns:
        Dictionary with all evaluation metrics
    """
    y_true = np.asarray(y_true).ravel()
    levels = levels or list(CONFIDENCE_LEVELS)

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


def evaluate_by_horizon(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    horizons: np.ndarray,
    levels: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Evaluate forecast metrics per forecast horizon.

    Long-horizon degradation is the main failure mode for dengue season
    forecasting; aggregate metrics hide it.

    Args:
        y_true: True values aligned with ``predictions`` rows
        predictions: DataFrame with median and interval columns
        horizons: Integer forecast horizon (1 = first step after training)
            for each prediction row
        levels: Confidence levels to evaluate

    Returns:
        DataFrame indexed by horizon with metric columns
    """
    horizons = np.asarray(horizons).ravel()
    records = []
    for h in np.unique(horizons):
        mask = horizons == h
        metrics = evaluate_forecast(y_true[mask], predictions[mask].reset_index(drop=True), levels)
        metrics["horizon"] = int(h)
        records.append(metrics)
    return pd.DataFrame(records).set_index("horizon")


class ForecastEvaluator:
    """Evaluator class for forecast comparison and analysis.

    Args:
        levels: Confidence levels to evaluate
        baseline_method: Baseline method for skill scores
    """

    def __init__(
        self,
        levels: Optional[list[float]] = None,
        baseline_method: str = "naive",
    ):
        self.levels = levels or [0.50, 0.80, 0.95]
        self.baseline_method = baseline_method
        self.results: dict[str, dict[str, float]] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        predictions: pd.DataFrame,
        model_name: str,
    ) -> dict[str, float]:
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
