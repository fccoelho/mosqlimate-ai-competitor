"""Validator Agent for Karl DBot."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.prompts import get_prompt
from mosqlimate_ai.evaluation.metrics import (
    ForecastEvaluator,
    coverage,
    evaluate_forecast,
)

logger = logging.getLogger(__name__)

PREDICTION_LEVELS = [0.50, 0.80, 0.90, 0.95]


class ValidatorAgent(BaseAgent):
    """Agent responsible for model validation and quality checks.

    This agent:
    - Performs cross-validation
    - Calculates performance metrics (CRPS, WIS, Log Score)
    - Checks for overfitting
    - Validates prediction intervals (coverage)
    - Compares models and selects best
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.evaluator = ForecastEvaluator(levels=PREDICTION_LEVELS)
        self.validation_results: Dict[str, Dict[str, float]] = {}
        self.system_prompt = get_prompt("validator")
        logger.info("ValidatorAgent ready")

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate forecasts.

        Args:
            task: Task description
            context: Context with keys:
                - forecasts: Dict of forecasts by model name
                - y_true: True values for validation
                - models: List of model names to validate (default: all)
                - levels: Prediction interval levels (default: [0.50, 0.80, 0.90, 0.95])
                - check_coverage: Whether to check interval coverage (default: True)
                - compare_models: Whether to compare models (default: True)

        Returns:
            Dictionary with validation results
        """
        logger.info(f"ValidatorAgent executing: {task}")
        context = context or {}

        try:
            forecasts = context.get("forecasts", {})
            y_true = context.get("y_true")
            models = context.get("models")
            levels = context.get("levels", PREDICTION_LEVELS)
            check_coverage = context.get("check_coverage", True)
            compare_models = context.get("compare_models", True)

            if not forecasts:
                raise ValueError("No forecasts provided in context")
            if y_true is None:
                raise ValueError("No y_true provided in context")

            y_true = np.asarray(y_true).ravel()

            if models:
                forecasts = {k: v for k, v in forecasts.items() if k in models}

            results = {}

            for model_name, forecast in forecasts.items():
                logger.info(f"Validating {model_name}...")

                metrics = self._validate_single_model(
                    forecast=forecast,
                    y_true=y_true,
                    levels=levels,
                    check_coverage=check_coverage,
                )
                results[model_name] = metrics
                self.validation_results[model_name] = metrics

            comparison = None
            best_model = None

            if compare_models and len(results) > 1:
                comparison = self._compare_models(results)
                best_model = self._select_best_model(results)

            validation_passed = self._check_validation_passed(results)

            return {
                "agent": self.config.name,
                "task": task,
                "status": "success",
                "output": {
                    "metrics": results,
                    "comparison": comparison,
                    "best_model": best_model,
                    "validation_passed": validation_passed,
                    "levels": levels,
                },
            }

        except Exception as e:
            logger.error(f"ValidatorAgent failed: {e}")
            return {
                "agent": self.config.name,
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _validate_single_model(
        self,
        forecast: pd.DataFrame,
        y_true: np.ndarray,
        levels: List[float],
        check_coverage: bool,
    ) -> Dict[str, float]:
        """Validate a single model's forecast."""
        metrics = evaluate_forecast(y_true, forecast, levels)

        if check_coverage:
            for level in levels:
                col_name = int(level * 100)
                lower_col = f"lower_{col_name}"
                upper_col = f"upper_{col_name}"

                if lower_col in forecast.columns and upper_col in forecast.columns:
                    valid_mask = ~forecast[lower_col].isna()
                    expected_coverage = level
                    actual_coverage = coverage(
                        y_true[valid_mask],
                        forecast.loc[valid_mask, lower_col],
                        forecast.loc[valid_mask, upper_col],
                    )
                    metrics[f"coverage_error_{col_name}"] = abs(actual_coverage - expected_coverage)

        return metrics

    def _compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare models across metrics."""
        df = pd.DataFrame(results).T
        return df

    def _select_best_model(
        self,
        results: Dict[str, Dict[str, float]],
        primary_metric: str = "crps",
    ) -> str:
        """Select best model based on primary metric."""
        scores = {
            name: metrics.get(primary_metric, float("inf")) for name, metrics in results.items()
        }
        return min(scores, key=scores.get)

    def _check_validation_passed(
        self,
        results: Dict[str, Dict[str, float]],
        max_crps: float = 100.0,
        min_coverage: float = 0.85,
    ) -> bool:
        """Check if validation passed basic criteria."""
        for model_name, metrics in results.items():
            if metrics.get("crps", float("inf")) > max_crps:
                logger.warning(f"{model_name} CRPS too high: {metrics['crps']}")
                return False

            for level in PREDICTION_LEVELS:
                col_name = int(level * 100)
                coverage_key = f"coverage_{col_name}"
                if coverage_key in metrics:
                    if metrics[coverage_key] < min_coverage * level:
                        logger.warning(
                            f"{model_name} coverage_{col_name} too low: {metrics[coverage_key]}"
                        )
                        return False

        return True

    def cross_validate_time_series(
        self,
        model_factory: Any,
        data: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 4,
        target_col: str = "casos",
    ) -> Dict[str, Any]:
        """Perform time-series cross-validation.

        Args:
            model_factory: Function that returns a new model instance
            data: DataFrame with features and target
            n_splits: Number of CV splits
            gap: Gap between train and test (in weeks)
            target_col: Target column name

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        cv_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            model = model_factory()
            model.fit(train_data)

            forecast = model.predict(test_data)

            y_true = test_data[target_col].values
            metrics = evaluate_forecast(y_true, forecast)
            metrics["fold"] = fold
            cv_metrics.append(metrics)

        cv_df = pd.DataFrame(cv_metrics)

        summary = {
            "crps_mean": cv_df["crps"].mean(),
            "crps_std": cv_df["crps"].std(),
            "rmse_mean": cv_df["rmse"].mean(),
            "mae_mean": cv_df["mae"].mean(),
            "coverage_95_mean": cv_df["coverage_95"].mean()
            if "coverage_95" in cv_df.columns
            else None,
            "n_splits": n_splits,
        }

        return {
            "fold_metrics": cv_df,
            "summary": summary,
        }

    def validate_prediction_intervals(
        self,
        forecasts: pd.DataFrame,
        y_true: np.ndarray,
        levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Validate prediction interval calibration.

        Args:
            forecasts: DataFrame with prediction intervals
            y_true: True values
            levels: Confidence levels to validate

        Returns:
            Interval validation results
        """
        levels = levels or PREDICTION_LEVELS
        y_true = np.asarray(y_true).ravel()

        results = {}

        for level in levels:
            col_name = int(level * 100)
            lower_col = f"lower_{col_name}"
            upper_col = f"upper_{col_name}"

            if lower_col in forecasts.columns and upper_col in forecasts.columns:
                valid_mask = ~forecasts[lower_col].isna()
                y_valid = y_true[valid_mask]
                lower = forecasts.loc[valid_mask, lower_col].values
                upper = forecasts.loc[valid_mask, upper_col].values

                actual_coverage = coverage(y_valid, lower, upper)
                expected_coverage = level

                results[f"level_{col_name}"] = {
                    "expected_coverage": expected_coverage,
                    "actual_coverage": actual_coverage,
                    "coverage_error": abs(actual_coverage - expected_coverage),
                    "n_below": int(np.sum(y_valid < lower)),
                    "n_above": int(np.sum(y_valid > upper)),
                    "mean_width": float(np.mean(upper - lower)),
                }

        return results

    def check_overfitting(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Check for overfitting by comparing train vs validation metrics.

        Args:
            train_metrics: Metrics on training data
            val_metrics: Metrics on validation data
            threshold: Maximum acceptable ratio of val/train

        Returns:
            Overfitting analysis
        """
        overfitting_detected = False
        analysis = {}

        for metric in ["rmse", "mae", "crps"]:
            if metric in train_metrics and metric in val_metrics:
                train_val = train_metrics[metric]
                val_val = val_metrics[metric]

                if train_val > 0:
                    ratio = val_val / train_val
                    is_overfit = ratio > (1 + threshold)

                    analysis[metric] = {
                        "train": train_val,
                        "validation": val_val,
                        "ratio": ratio,
                        "overfitting": is_overfit,
                    }

                    if is_overfit:
                        overfitting_detected = True

        return {
            "overfitting_detected": overfitting_detected,
            "analysis": analysis,
            "threshold": threshold,
        }

    def generate_report(
        self,
        results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Generate a validation report.

        Args:
            results: Validation results (default: stored results)

        Returns:
            Formatted report string
        """
        results = results or self.validation_results

        lines = ["=" * 60]
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)

        for model_name, metrics in results.items():
            lines.append(f"\n{model_name}:")
            lines.append(f"  CRPS:    {metrics.get('crps', 'N/A'):.4f}")
            lines.append(f"  RMSE:    {metrics.get('rmse', 'N/A'):.4f}")
            lines.append(f"  MAE:     {metrics.get('mae', 'N/A'):.4f}")
            lines.append(f"  WIS:     {metrics.get('wis_total', 'N/A'):.4f}")
            lines.append(f"  Bias:    {metrics.get('bias', 'N/A'):.4f}")
            lines.append("\n  Coverage:")
            for level in PREDICTION_LEVELS:
                col_name = int(level * 100)
                cov_key = f"coverage_{col_name}"
                if cov_key in metrics:
                    lines.append(f"    {col_name}%: {metrics[cov_key]:.2%}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def save_results(
        self,
        output_path: Union[str, Path],
        results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Save validation results to disk.

        Args:
            output_path: Path to save results
            results: Results to save (default: stored results)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        results = results or self.validation_results

        df = pd.DataFrame(results).T
        df.to_csv(output_path / "validation_results.csv")

        report = self.generate_report(results)
        with open(output_path / "validation_report.txt", "w") as f:
            f.write(report)

        logger.info(f"Saved validation results to {output_path}")
