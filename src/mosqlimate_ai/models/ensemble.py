"""Ensemble model with CRPS-optimized weighting for dengue forecasting.

Combines multiple forecasting models using CRPS-based weighting
and conformal prediction for calibrated prediction intervals.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.metrics import crps

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """Ensemble forecaster combining multiple models.

    Uses CRPS-based weighting to optimally combine predictions
    from multiple forecasting models.

    Args:
        method: Ensemble method ('weighted_average', 'stacking', 'median')
        weight_metric: Metric for weight optimization ('crps', 'rmse', 'mae')
        calibrate_intervals: Whether to apply conformal calibration

    Example:
        >>> ensemble = EnsembleForecaster()
        >>> ensemble.add_model('xgboost', xgb_predictions, weight=0.5)
        >>> ensemble.add_model('lstm', lstm_predictions, weight=0.5)
        >>> ensemble.fit_weights(y_validation)
        >>> final_predictions = ensemble.predict()
    """

    def __init__(
        self,
        method: str = "weighted_average",
        weight_metric: str = "crps",
        calibrate_intervals: bool = True,
    ):
        self.method = method
        self.weight_metric = weight_metric
        self.calibrate_intervals = calibrate_intervals

        self.models: Dict[str, pd.DataFrame] = {}
        self.weights: Dict[str, float] = {}
        self.calibration_params: Dict[str, Any] = {}
        self.is_fitted = False

    def add_model(
        self,
        name: str,
        predictions: pd.DataFrame,
        weight: float = 1.0,
    ) -> "EnsembleForecaster":
        """Add a model's predictions to the ensemble.

        Args:
            name: Model identifier
            predictions: DataFrame with predictions and intervals
            weight: Initial weight (will be optimized later)

        Returns:
            Self for chaining
        """
        self.models[name] = predictions.copy()
        self.weights[name] = weight
        logger.info(f"Added model '{name}' with initial weight {weight}")
        return self

    def fit_weights(
        self,
        y_true: np.ndarray,
        optimize: bool = True,
    ) -> "EnsembleForecaster":
        """Optimize model weights based on validation performance.

        Args:
            y_true: True values for validation
            optimize: Whether to optimize weights (vs. just compute)

        Returns:
            Self for chaining
        """
        y_true = np.asarray(y_true).ravel()

        performance = {}
        for name, pred in self.models.items():
            valid_mask = ~pred["median"].isna()

            if self.weight_metric == "crps":
                score = crps(y_true[valid_mask], pred[valid_mask])
                performance[name] = score
            elif self.weight_metric == "rmse":
                from mosqlimate_ai.evaluation.metrics import rmse

                score = rmse(y_true[valid_mask], pred.loc[valid_mask, "median"])
                performance[name] = score
            elif self.weight_metric == "mae":
                from mosqlimate_ai.evaluation.metrics import mae

                score = mae(y_true[valid_mask], pred.loc[valid_mask, "median"])
                performance[name] = score

        if optimize:
            if self.weight_metric == "crps":
                total_inverse = sum(1 / (s + 1e-8) for s in performance.values())
                self.weights = {
                    name: (1 / (score + 1e-8)) / total_inverse
                    for name, score in performance.items()
                }
            else:
                total_inverse = sum(1 / (s + 1e-8) for s in performance.values())
                self.weights = {
                    name: (1 / (score + 1e-8)) / total_inverse
                    for name, score in performance.values()
                }

        logger.info(f"Optimized weights: {self.weights}")
        logger.info(f"Performance ({self.weight_metric}): {performance}")

        self.is_fitted = True
        return self

    def predict(
        self,
        return_individual: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """Generate ensemble predictions.

        Args:
            return_individual: Whether to return individual model predictions

        Returns:
            DataFrame with ensemble predictions (and optionally individual predictions)
        """
        if not self.models:
            raise ValueError("No models added to ensemble")

        if self.method == "weighted_average":
            predictions = self._weighted_average_predict()
        elif self.method == "median":
            predictions = self._median_predict()
        elif self.method == "stacking":
            predictions = self._stacking_predict()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        predictions["median"] = predictions["median"].clip(lower=0)
        for col in predictions.columns:
            if col.startswith("lower_"):
                predictions[col] = predictions[col].clip(lower=0)

        if return_individual:
            return predictions, self.models

        return predictions

    def _weighted_average_predict(self) -> pd.DataFrame:
        """Generate weighted average predictions."""
        first_model = list(self.models.values())[0]
        n_samples = len(first_model)

        total_weight = sum(self.weights.values())

        median_sum = np.zeros(n_samples)
        lower_50_sum = np.zeros(n_samples)
        upper_50_sum = np.zeros(n_samples)
        lower_80_sum = np.zeros(n_samples)
        upper_80_sum = np.zeros(n_samples)
        lower_95_sum = np.zeros(n_samples)
        upper_95_sum = np.zeros(n_samples)

        for name, pred in self.models.items():
            w = self.weights[name] / total_weight
            median_sum += pred["median"].values * w

            if "lower_50" in pred.columns:
                lower_50_sum += pred["lower_50"].values * w
                upper_50_sum += pred["upper_50"].values * w
            if "lower_80" in pred.columns:
                lower_80_sum += pred["lower_80"].values * w
                upper_80_sum += pred["upper_80"].values * w
            if "lower_95" in pred.columns:
                lower_95_sum += pred["lower_95"].values * w
                upper_95_sum += pred["upper_95"].values * w

        result = pd.DataFrame({"median": median_sum})

        if "lower_50" in first_model.columns:
            result["lower_50"] = lower_50_sum
            result["upper_50"] = upper_50_sum
        if "lower_80" in first_model.columns:
            result["lower_80"] = lower_80_sum
            result["upper_80"] = upper_80_sum
        if "lower_95" in first_model.columns:
            result["lower_95"] = lower_95_sum
            result["upper_95"] = upper_95_sum

        if "date" in first_model.columns:
            result["date"] = first_model["date"].values
        if "uf" in first_model.columns:
            result["uf"] = first_model["uf"].values

        return result

    def _median_predict(self) -> pd.DataFrame:
        """Generate median-of-medians predictions."""
        first_model = list(self.models.values())[0]

        medians = np.column_stack([pred["median"].values for pred in self.models.values()])
        ensemble_median = np.median(medians, axis=1)

        result = pd.DataFrame({"median": ensemble_median})

        for level in [50, 80, 95]:
            lower_col = f"lower_{level}"
            upper_col = f"upper_{level}"

            if lower_col in first_model.columns:
                lowers = np.column_stack([pred[lower_col].values for pred in self.models.values()])
                uppers = np.column_stack([pred[upper_col].values for pred in self.models.values()])

                result[lower_col] = np.percentile(lowers, 50, axis=1)
                result[upper_col] = np.percentile(uppers, 50, axis=1)

        if "date" in first_model.columns:
            result["date"] = first_model["date"].values
        if "uf" in first_model.columns:
            result["uf"] = first_model["uf"].values

        return result

    def _stacking_predict(self) -> pd.DataFrame:
        """Generate stacking predictions (placeholder for now)."""
        logger.warning("Stacking not fully implemented, falling back to weighted average")
        return self._weighted_average_predict()

    def calibrate(
        self,
        y_true: np.ndarray,
        level: float = 0.95,
    ) -> "EnsembleForecaster":
        """Apply conformal calibration to prediction intervals.

        Args:
            y_true: True values for calibration
            level: Confidence level for calibration

        Returns:
            Self for chaining
        """
        predictions = self.predict()
        y_true = np.asarray(y_true).ravel()

        valid_mask = ~predictions["median"].isna()
        y_valid = y_true[valid_mask]
        pred_valid = predictions.loc[valid_mask]

        for level in [50, 80, 95]:
            lower_col = f"lower_{level}"
            upper_col = f"upper_{level}"

            if lower_col in pred_valid.columns:
                residuals = np.concatenate(
                    [
                        pred_valid[lower_col].values - y_valid,
                        y_valid - pred_valid[upper_col].values,
                    ]
                )

                n = len(residuals)
                q = np.ceil((n + 1) * (level / 100)) / n
                correction = np.quantile(np.abs(residuals), min(q, 0.999))

                self.calibration_params[f"correction_{level}"] = correction

        logger.info(f"Calibration parameters: {self.calibration_params}")
        return self

    def apply_calibration(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply calibration corrections to predictions.

        Args:
            predictions: DataFrame with predictions

        Returns:
            Calibrated predictions
        """
        if not self.calibration_params:
            return predictions

        predictions = predictions.copy()

        for level in [50, 80, 95]:
            correction = self.calibration_params.get(f"correction_{level}", 0)
            lower_col = f"lower_{level}"
            upper_col = f"upper_{level}"

            if lower_col in predictions.columns:
                predictions[lower_col] = predictions[lower_col] - correction
                predictions[upper_col] = predictions[upper_col] + correction

        return predictions

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights.

        Returns:
            Dictionary of model weights
        """
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> "EnsembleForecaster":
        """Set model weights manually.

        Args:
            weights: Dictionary of model weights

        Returns:
            Self for chaining
        """
        for name, weight in weights.items():
            if name in self.models:
                self.weights[name] = weight
            else:
                logger.warning(f"Model '{name}' not found, skipping")

        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble configuration.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "method": self.method,
            "weight_metric": self.weight_metric,
            "weights": self.weights,
            "calibration_params": self.calibration_params,
        }
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        for name, pred in self.models.items():
            pred.to_csv(path / f"predictions_{name}.csv", index=False)

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: Union[str, Path]) -> "EnsembleForecaster":
        """Load ensemble configuration.

        Args:
            path: Path to saved configuration

        Returns:
            Loaded ensemble
        """
        path = Path(path)

        with open(path / "ensemble_config.json") as f:
            config = json.load(f)

        self.method = config["method"]
        self.weight_metric = config["weight_metric"]
        self.weights = config["weights"]
        self.calibration_params = config.get("calibration_params", {})

        for pred_file in path.glob("predictions_*.csv"):
            name = pred_file.stem.replace("predictions_", "")
            self.models[name] = pd.read_csv(pred_file)

        self.is_fitted = True
        logger.info(f"Ensemble loaded from {path}")
        return self


def create_ensemble(
    models: Dict[str, pd.DataFrame],
    y_true: Optional[np.ndarray] = None,
    method: str = "weighted_average",
    weight_metric: str = "crps",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Convenience function to create ensemble predictions.

    Args:
        models: Dictionary of model predictions
        y_true: True values for weight optimization
        method: Ensemble method
        weight_metric: Metric for weight optimization

    Returns:
        Tuple of (ensemble predictions, weights)
    """
    ensemble = EnsembleForecaster(method=method, weight_metric=weight_metric)

    for name, pred in models.items():
        ensemble.add_model(name, pred)

    if y_true is not None:
        ensemble.fit_weights(y_true)

    predictions = ensemble.predict()
    return predictions, ensemble.get_weights()
