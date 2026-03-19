"""Ensemble Agent for Karl DBot."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.prompts import get_prompt
from mosqlimate_ai.models.ensemble import EnsembleForecaster
from mosqlimate_ai.submission.formatter import SubmissionFormatter

logger = logging.getLogger(__name__)


class EnsembleAgent(BaseAgent):
    """Agent responsible for combining multiple models.

    This agent:
    - Weighs individual model predictions based on performance
    - Creates ensemble forecasts using CRPS-weighted averaging
    - Applies conformal calibration to prediction intervals
    - Formats output for Mosqlimate submission
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.ensemble: Optional[EnsembleForecaster] = None
        self.weights: Dict[str, float] = {}
        self.formatter = SubmissionFormatter()
        self.system_prompt = get_prompt("ensemble")
        logger.info("EnsembleAgent ready")

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create ensemble and format submission.

        Args:
            task: Task description
            context: Context with keys:
                - forecasts: Dict of forecasts by model name
                - y_true: True values for weight optimization (optional)
                - method: Ensemble method ('weighted_average', 'median')
                - weight_metric: Metric for weight optimization ('crps', 'rmse', 'mae')
                - calibrate: Whether to apply calibration (default: True)
                - format_submission: Whether to format for API (default: False)
                - model_id: Model ID for submission (required if format_submission=True)
                - dates: List of dates for submission (required if format_submission=True)

        Returns:
            Dictionary with ensemble predictions and metadata
        """
        logger.info(f"EnsembleAgent executing: {task}")
        context = context or {}

        try:
            forecasts = context.get("forecasts", {})
            _y_true = context.get("y_true")
            _method = context.get("method", "weighted_average")
            _weight_metric = context.get("weight_metric", "crps")
            _calibrate = context.get("calibrate", True)
            format_submission = context.get("format_submission", False)

            if not forecasts:
                raise ValueError("No forecasts provided in context")

            if isinstance(forecasts, dict) and any(isinstance(v, dict) for v in forecasts.values()):
                results = {}
                for state, state_forecasts in forecasts.items():
                    if isinstance(state_forecasts, dict):
                        state_context = {
                            **context,
                            "forecasts": state_forecasts,
                            "format_submission": False,
                        }
                        results[state] = self._create_ensemble(state_context)

                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": {
                        "ensemble_by_state": results,
                        "n_states": len(results),
                    },
                }
            else:
                result = self._create_ensemble(context)

                if format_submission:
                    submission_data = self._format_submission(result, context)
                    result["submission_data"] = submission_data

                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": result,
                }

        except Exception as e:
            logger.error(f"EnsembleAgent failed: {e}")
            return {
                "agent": self.config.name,
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _create_ensemble(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble from forecasts."""
        forecasts = context["forecasts"]
        y_true = context.get("y_true")
        method = context.get("method", "weighted_average")
        weight_metric = context.get("weight_metric", "crps")
        calibrate = context.get("calibrate", True)

        self.ensemble = EnsembleForecaster(
            method=method,
            weight_metric=weight_metric,
            calibrate_intervals=calibrate,
        )

        for name, forecast in forecasts.items():
            self.ensemble.add_model(name, forecast)

        if y_true is not None:
            y_true = np.asarray(y_true).ravel()
            self.ensemble.fit_weights(y_true, optimize=True)

            if calibrate:
                self.ensemble.calibrate(y_true)
        else:
            n_models = len(forecasts)
            self.weights = {name: 1.0 / n_models for name in forecasts}
            self.ensemble.weights = self.weights

        self.weights = self.ensemble.get_weights()

        predictions = self.ensemble.predict()

        if calibrate and self.ensemble.calibration_params:
            predictions = self.ensemble.apply_calibration(predictions)

        return {
            "predictions": predictions,
            "weights": self.weights,
            "method": method,
            "weight_metric": weight_metric,
            "calibrated": calibrate and bool(self.ensemble.calibration_params),
        }

    def _format_submission(
        self,
        ensemble_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format ensemble predictions for Mosqlimate submission."""
        predictions = ensemble_result["predictions"]
        model_id = context.get("model_id")
        dates = context.get("dates")
        uf = context.get("uf")

        if model_id is None:
            raise ValueError("model_id required for submission formatting")

        if dates is not None:
            predictions = predictions.copy()
            predictions["date"] = dates[: len(predictions)]

        if uf is not None:
            predictions = predictions.copy()
            predictions["uf"] = uf

        submission = self.formatter.format_predictions(
            predictions=predictions,
            model_id=model_id,
        )

        return submission

    def add_model(
        self,
        name: str,
        predictions: pd.DataFrame,
        weight: float = 1.0,
    ) -> "EnsembleAgent":
        """Add a model to the ensemble.

        Args:
            name: Model identifier
            predictions: DataFrame with predictions and intervals
            weight: Initial weight

        Returns:
            Self for chaining
        """
        if self.ensemble is None:
            self.ensemble = EnsembleForecaster()

        self.ensemble.add_model(name, predictions, weight)
        return self

    def fit_weights(
        self,
        y_true: np.ndarray,
        optimize: bool = True,
    ) -> "EnsembleAgent":
        """Optimize model weights.

        Args:
            y_true: True values
            optimize: Whether to optimize weights

        Returns:
            Self for chaining
        """
        if self.ensemble is None:
            raise ValueError("No models added to ensemble")

        self.ensemble.fit_weights(y_true, optimize=optimize)
        self.weights = self.ensemble.get_weights()
        return self

    def predict(
        self,
        apply_calibration: bool = True,
    ) -> pd.DataFrame:
        """Generate ensemble predictions.

        Args:
            apply_calibration: Whether to apply calibration

        Returns:
            DataFrame with ensemble predictions
        """
        if self.ensemble is None:
            raise ValueError("No ensemble created")

        predictions = self.ensemble.predict()

        if apply_calibration and self.ensemble.calibration_params:
            predictions = self.ensemble.apply_calibration(predictions)

        return predictions

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights.

        Returns:
            Dictionary of model weights
        """
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> "EnsembleAgent":
        """Set model weights manually.

        Args:
            weights: Dictionary of model weights

        Returns:
            Self for chaining
        """
        if self.ensemble is None:
            raise ValueError("No ensemble created")

        self.ensemble.set_weights(weights)
        self.weights = self.ensemble.get_weights()
        return self

    def calibrate(
        self,
        y_true: np.ndarray,
    ) -> "EnsembleAgent":
        """Apply conformal calibration to prediction intervals.

        Args:
            y_true: True values for calibration

        Returns:
            Self for chaining
        """
        if self.ensemble is None:
            raise ValueError("No ensemble created")

        self.ensemble.calibrate(y_true)
        return self

    def format_submission(
        self,
        predictions: pd.DataFrame,
        model_id: int,
        uf: Optional[str] = None,
        dates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format forecasts for Mosqlimate submission.

        Args:
            predictions: DataFrame with ensemble predictions
            model_id: Model ID from Mosqlimate
            uf: State abbreviation
            dates: List of dates for predictions

        Returns:
            Formatted submission data
        """
        predictions = predictions.copy()

        if dates is not None:
            predictions["date"] = dates[: len(predictions)]

        if uf is not None:
            predictions["uf"] = uf

        return self.formatter.format_predictions(
            predictions=predictions,
            model_id=model_id,
        )

    def save_ensemble(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """Save ensemble configuration and predictions.

        Args:
            output_path: Path to save ensemble
        """
        if self.ensemble is None:
            raise ValueError("No ensemble to save")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.ensemble.save(output_path)

        weights_df = pd.DataFrame([self.weights]).T
        weights_df.columns = ["weight"]
        weights_df.to_csv(output_path / "weights.csv")

        logger.info(f"Ensemble saved to {output_path}")

    def load_ensemble(
        self,
        path: Union[str, Path],
    ) -> "EnsembleAgent":
        """Load ensemble from disk.

        Args:
            path: Path to saved ensemble

        Returns:
            Self for chaining
        """
        path = Path(path)

        self.ensemble = EnsembleForecaster()
        self.ensemble.load(path)
        self.weights = self.ensemble.get_weights()

        logger.info(f"Ensemble loaded from {path}")
        return self

    def compare_methods(
        self,
        forecasts: Dict[str, pd.DataFrame],
        y_true: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare different ensemble methods.

        Args:
            forecasts: Dictionary of model forecasts
            y_true: True values
            methods: Methods to compare (default: all)

        Returns:
            DataFrame with comparison results
        """
        from mosqlimate_ai.evaluation.metrics import evaluate_forecast

        methods = methods or ["weighted_average", "median"]
        y_true = np.asarray(y_true).ravel()

        results = {}

        for method in methods:
            ensemble = EnsembleForecaster(method=method)

            for name, forecast in forecasts.items():
                ensemble.add_model(name, forecast)

            ensemble.fit_weights(y_true)
            predictions = ensemble.predict()

            metrics = evaluate_forecast(y_true, predictions)
            results[method] = metrics

        return pd.DataFrame(results).T

    def generate_report(self) -> str:
        """Generate ensemble report.

        Returns:
            Formatted report string
        """
        lines = ["=" * 60]
        lines.append("ENSEMBLE REPORT")
        lines.append("=" * 60)

        if self.weights:
            lines.append("\nModel Weights:")
            for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {weight:.4f}")

        if self.ensemble and self.ensemble.calibration_params:
            lines.append("\nCalibration Parameters:")
            for key, value in self.ensemble.calibration_params.items():
                lines.append(f"  {key}: {value:.4f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
