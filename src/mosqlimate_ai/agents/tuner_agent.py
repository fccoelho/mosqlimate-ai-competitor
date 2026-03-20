"""Efficient hyperparameter tuning for validation pipeline.

Uses Bayesian Optimization with convergence-based early stopping
instead of fixed iteration limits.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available. Using fallback search strategy.")


class ConvergenceTracker:
    """Track convergence for early stopping.

    Monitors improvement in objective scores and signals when
    convergence is reached.

    Args:
        patience: Number of iterations without improvement before stopping
        min_improvement_rate: Minimum relative improvement to count as progress
        min_iterations: Minimum iterations before checking convergence
    """

    def __init__(
        self,
        patience: int = 5,
        min_improvement_rate: float = 0.001,
        min_iterations: int = 5,
    ):
        self.patience = patience
        self.min_improvement_rate = min_improvement_rate
        self.min_iterations = min_iterations
        self.best_score: Optional[float] = None
        self.iterations_without_improvement = 0
        self.score_history: List[float] = []
        self.iteration = 0

    def update(self, score: float) -> bool:
        """Update tracker with new score.

        Args:
            score: New objective score (lower is better)

        Returns:
            True if should continue, False if converged
        """
        self.iteration += 1
        self.score_history.append(score)

        if self.best_score is None:
            self.best_score = score
            return True

        if score < self.best_score:
            relative_improvement = (self.best_score - score) / (abs(self.best_score) + 1e-8)

            if relative_improvement > self.min_improvement_rate:
                self.best_score = score
                self.iterations_without_improvement = 0
                logger.debug(
                    f"Iteration {self.iteration}: Improved to {score:.6f} "
                    f"({relative_improvement:.2%} improvement)"
                )
            else:
                self.iterations_without_improvement += 1
                logger.debug(
                    f"Iteration {self.iteration}: Score {score:.6f} improved but below threshold"
                )
        else:
            self.iterations_without_improvement += 1
            logger.debug(
                f"Iteration {self.iteration}: Score {score:.6f} (no improvement, "
                f"patience: {self.iterations_without_improvement}/{self.patience})"
            )

        if self.iteration < self.min_iterations:
            return True

        if self.iterations_without_improvement >= self.patience:
            logger.info(
                f"Convergence reached after {self.iteration} iterations. "
                f"Best score: {self.best_score:.6f}"
            )
            return False

        return True

    def get_best_score(self) -> Optional[float]:
        """Get best score found."""
        return self.best_score

    def get_stats(self) -> Dict[str, Any]:
        """Get convergence statistics."""
        return {
            "total_iterations": self.iteration,
            "best_score": self.best_score,
            "iterations_without_improvement": self.iterations_without_improvement,
            "score_history": self.score_history,
            "converged": self.iterations_without_improvement >= self.patience,
        }


class EfficientHyperparameterTuner:
    """Moderate hyperparameter tuning using Bayesian Optimization.

    This tuner performs efficient hyperparameter search using Bayesian Optimization
    with convergence-based early stopping. Supports XGBoost, LSTM, Prophet, TFT,
    and N-BEATS models.
    """

    def __init__(
        self,
        max_iterations: int = 30,
        convergence_patience: int = 5,
        min_improvement_rate: float = 0.001,
        random_state: int = 42,
    ):
        """Initialize tuner.

        Args:
            max_iterations: Maximum number of optimization iterations (upper bound)
            convergence_patience: Iterations without improvement before stopping
            min_improvement_rate: Minimum relative improvement (0.001 = 0.1%)
            random_state: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.convergence_patience = convergence_patience
        self.min_improvement_rate = min_improvement_rate
        self.random_state = random_state
        self.search_spaces = self._define_search_spaces()

    def _define_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define search spaces for all model types."""
        if SKOPT_AVAILABLE:
            return {
                "xgboost": {
                    "learning_rate": Real(0.01, 0.15, prior="log-uniform"),
                    "max_depth": Integer(3, 10),
                    "n_estimators": Integer(200, 800),
                    "min_child_weight": Integer(1, 10),
                    "subsample": Real(0.6, 1.0),
                    "colsample_bytree": Real(0.6, 1.0),
                    "gamma": Real(0, 5),
                },
                "lstm": {
                    "hidden_size": Integer(64, 256),
                    "num_layers": Integer(1, 3),
                    "dropout": Real(0.1, 0.4),
                    "learning_rate": Real(0.0001, 0.01, prior="log-uniform"),
                },
                "prophet": {
                    "changepoint_prior_scale": Real(0.001, 0.5, prior="log-uniform"),
                    "seasonality_prior_scale": Real(0.1, 20.0),
                    "seasonality_mode": ["additive", "multiplicative"],
                },
                "tft": {
                    "hidden_size": Integer(16, 128),
                    "hidden_continuous_size": Integer(8, 64),
                    "attention_head_size": Integer(1, 8),
                    "dropout": Real(0.1, 0.4),
                    "learning_rate": Real(0.0001, 0.01, prior="log-uniform"),
                },
                "nbeats": {
                    "hidden_size": Integer(64, 512),
                    "learning_rate": Real(0.0001, 0.01, prior="log-uniform"),
                },
            }
        else:
            return {
                "xgboost": {
                    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
                    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                    "n_estimators": [200, 300, 400, 500, 600, 700, 800],
                    "min_child_weight": [1, 3, 5, 7, 10],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "gamma": [0, 0.5, 1, 2, 3, 5],
                },
                "lstm": {
                    "hidden_size": [64, 128, 256],
                    "num_layers": [1, 2, 3],
                    "dropout": [0.1, 0.2, 0.3, 0.4],
                    "learning_rate": [0.0001, 0.001, 0.005, 0.01],
                },
                "prophet": {
                    "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "seasonality_prior_scale": [0.1, 1.0, 5.0, 10.0, 20.0],
                    "seasonality_mode": ["additive", "multiplicative"],
                },
                "tft": {
                    "hidden_size": [16, 32, 64, 128],
                    "hidden_continuous_size": [8, 16, 32, 64],
                    "attention_head_size": [1, 2, 4, 8],
                    "dropout": [0.1, 0.2, 0.3, 0.4],
                    "learning_rate": [0.0001, 0.001, 0.005, 0.01],
                },
                "nbeats": {
                    "hidden_size": [64, 128, 256, 512],
                    "learning_rate": [0.0001, 0.001, 0.005, 0.01],
                },
            }

    def tune(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]] = None,
        model_type: str = "xgboost",
    ) -> Tuple[Dict[str, Any], float]:
        """Run hyperparameter tuning with convergence-based stopping.

        Args:
            objective_fn: Function that takes hyperparameters and returns score (lower is better)
            warm_start_params: Parameters from similar states to start from
            model_type: Type of model ('xgboost', 'lstm', 'prophet', 'tft', 'nbeats')

        Returns:
            Tuple of (best_params, best_score)
        """
        model_type = model_type.lower()
        if model_type not in self.search_spaces:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.search_spaces.keys())}"
            )

        if model_type == "xgboost":
            return self._tune_xgboost(objective_fn, warm_start_params)
        elif model_type == "lstm":
            return self._tune_lstm(objective_fn, warm_start_params)
        elif model_type == "prophet":
            return self._tune_prophet(objective_fn, warm_start_params)
        elif model_type == "tft":
            return self._tune_tft(objective_fn, warm_start_params)
        elif model_type == "nbeats":
            return self._tune_nbeats(objective_fn, warm_start_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _tune_xgboost(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune XGBoost hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize("xgboost", objective_fn, warm_start_params)
        else:
            return self._fallback_search("xgboost", objective_fn, warm_start_params)

    def _tune_lstm(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune LSTM hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize("lstm", objective_fn, warm_start_params)
        else:
            return self._fallback_search("lstm", objective_fn, warm_start_params)

    def _tune_prophet(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune Prophet hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize("prophet", objective_fn, warm_start_params)
        else:
            return self._fallback_search("prophet", objective_fn, warm_start_params)

    def _tune_tft(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune TFT hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize("tft", objective_fn, warm_start_params)
        else:
            return self._fallback_search("tft", objective_fn, warm_start_params)

    def _tune_nbeats(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune N-BEATS hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize("nbeats", objective_fn, warm_start_params)
        else:
            return self._fallback_search("nbeats", objective_fn, warm_start_params)

    def _bayesian_optimize(
        self,
        model_type: str,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Use Bayesian optimization with convergence-based stopping."""
        search_space = self.search_spaces[model_type]
        space = list(search_space.values())
        param_names = list(search_space.keys())

        tracker = ConvergenceTracker(
            patience=self.convergence_patience,
            min_improvement_rate=self.min_improvement_rate,
            min_iterations=5,
        )

        evaluated_scores: List[float] = []
        evaluated_params: List[List[Any]] = []

        def wrapped_objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            params = self._convert_params(params, model_type)
            score = objective_fn(params)
            evaluated_scores.append(score)
            evaluated_params.append(x)
            tracker.update(score)
            return score

        x0 = None
        y0 = None
        n_initial_points = 5

        if warm_start_params:
            x0 = []
            for name in param_names:
                x0.append(warm_start_params.get(name, self._get_default_param(model_type, name)))
            x0 = [x0]
            y0 = [objective_fn(self._convert_params(warm_start_params, model_type))]
            n_initial_points = 3
            tracker.update(y0[0])

        logger.info(
            f"Starting Bayesian optimization for {model_type} "
            f"(max {self.max_iterations} iterations, patience {self.convergence_patience})"
        )

        n_calls = 0
        result = None

        try:
            result = gp_minimize(
                wrapped_objective,
                space,
                x0=x0,
                y0=y0,
                n_initial_points=n_initial_points,
                n_calls=self.max_iterations,
                random_state=self.random_state,
                verbose=False,
                callback=lambda res: not tracker.get_stats()["converged"],
            )
            n_calls = len(result.func_vals)
        except Exception as e:
            logger.warning(f"Bayesian optimization failed: {e}. Using evaluated results.")

        if result is not None and hasattr(result, "x"):
            best_params = {name: val for name, val in zip(param_names, result.x)}
            best_score = float(result.fun)
        elif evaluated_scores:
            best_idx = int(np.argmin(evaluated_scores))
            best_params = {name: val for name, val in zip(param_names, evaluated_params[best_idx])}
            best_score = evaluated_scores[best_idx]
        else:
            best_params = self._get_default_params(model_type)
            best_score = objective_fn(best_params)

        best_params = self._convert_params(best_params, model_type)

        stats = tracker.get_stats()
        logger.info(
            f"Optimization completed for {model_type}: "
            f"{stats['total_iterations']} iterations, "
            f"best score: {best_score:.6f}, "
            f"converged: {stats['converged']}"
        )

        return best_params, best_score

    def _fallback_search(
        self,
        model_type: str,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Fallback search strategy when scikit-optimize is not available."""
        logger.info(f"Using fallback search strategy for {model_type}")

        tracker = ConvergenceTracker(
            patience=self.convergence_patience,
            min_improvement_rate=self.min_improvement_rate,
            min_iterations=3,
        )

        best_params = (
            warm_start_params.copy() if warm_start_params else self._get_default_params(model_type)
        )
        best_score = objective_fn(best_params)
        tracker.update(best_score)

        search_space = self.search_spaces[model_type]
        priority_params = self._get_priority_params(model_type)

        for param_name in priority_params:
            if not tracker.get_stats()["converged"]:
                break

            if param_name not in search_space:
                continue

            for value in search_space[param_name]:
                if tracker.iteration >= self.max_iterations:
                    break

                params = best_params.copy()
                params[param_name] = value

                try:
                    score = objective_fn(params)
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                except Exception as e:
                    logger.warning(f"Failed to evaluate {param_name}={value}: {e}")
                    score = 1e6

                if not tracker.update(score):
                    break

            if tracker.get_stats()["converged"]:
                break

        stats = tracker.get_stats()
        logger.info(
            f"Fallback search completed for {model_type}: "
            f"{stats['total_iterations']} iterations, "
            f"best score: {best_score:.6f}"
        )

        return best_params, best_score

    def _convert_params(self, params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Convert parameter types as needed."""
        int_params = {
            "xgboost": ["max_depth", "n_estimators", "min_child_weight"],
            "lstm": ["hidden_size", "num_layers"],
            "tft": ["hidden_size", "hidden_continuous_size", "attention_head_size"],
            "nbeats": ["hidden_size"],
        }

        if model_type in int_params:
            for param in int_params[model_type]:
                if param in params:
                    params[param] = int(params[param])

        return params

    def _get_default_param(self, model_type: str, param_name: str) -> Any:
        """Get default value for a single parameter."""
        defaults = self._get_default_params(model_type)
        return defaults.get(param_name)

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        defaults = {
            "xgboost": {
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_estimators": 500,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
            },
            "prophet": {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "seasonality_mode": "multiplicative",
            },
            "tft": {
                "hidden_size": 64,
                "hidden_continuous_size": 32,
                "attention_head_size": 4,
                "dropout": 0.1,
                "learning_rate": 0.001,
            },
            "nbeats": {
                "hidden_size": 256,
                "learning_rate": 0.001,
            },
        }
        return defaults.get(model_type, {})

    def _get_priority_params(self, model_type: str) -> List[str]:
        """Get parameters in priority order for fallback search."""
        priority = {
            "xgboost": [
                "learning_rate",
                "max_depth",
                "n_estimators",
                "subsample",
                "colsample_bytree",
            ],
            "lstm": ["learning_rate", "hidden_size", "num_layers", "dropout"],
            "prophet": ["changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode"],
            "tft": ["learning_rate", "hidden_size", "attention_head_size", "dropout"],
            "nbeats": ["learning_rate", "hidden_size"],
        }
        return priority.get(model_type, [])

    def suggest_focus_areas(
        self,
        validation_results: List[Dict[str, Any]],
    ) -> List[str]:
        """Suggest which hyperparameters to focus on based on validation results.

        Args:
            validation_results: List of validation result dictionaries

        Returns:
            List of parameter names to focus on
        """
        if not validation_results:
            return ["learning_rate", "max_depth"]

        latest = validation_results[-1]
        focus_areas = []

        if latest.get("coverage_95", 0.95) < 0.90:
            focus_areas.append("subsample")
            focus_areas.append("colsample_bytree")

        if abs(latest.get("bias", 0)) > 100:
            focus_areas.append("learning_rate")
            focus_areas.append("gamma")

        if latest.get("mape", 0) > 50:
            focus_areas.append("max_depth")
            focus_areas.append("min_child_weight")

        if not focus_areas:
            focus_areas = ["learning_rate", "n_estimators"]

        return list(set(focus_areas))

    def get_supported_models(self) -> List[str]:
        """Get list of supported model types."""
        return list(self.search_spaces.keys())
