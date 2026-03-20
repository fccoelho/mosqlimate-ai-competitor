"""Efficient hyperparameter tuning for validation pipeline.

Uses Bayesian Optimization for moderate, efficient search instead of full grid search.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import scikit-optimize, fall back to simple search if not available
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available. Using fallback search strategy.")


class EfficientHyperparameterTuner:
    """Moderate hyperparameter tuning using Bayesian Optimization.

    This tuner performs efficient hyperparameter search using Bayesian Optimization
    with 10 iterations instead of exhaustive grid search. It can warm-start from
    similar states' successful configurations.
    """

    def __init__(self, max_iterations: int = 10, random_state: int = 42):
        """Initialize tuner.

        Args:
            max_iterations: Maximum number of optimization iterations
            random_state: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.search_space = self._define_search_space()

    def _define_search_space(self) -> Dict[str, Any]:
        """Define search space for XGBoost hyperparameters."""
        if SKOPT_AVAILABLE:
            return {
                "learning_rate": Real(0.01, 0.15, prior="log-uniform"),
                "max_depth": Integer(3, 10),
                "n_estimators": Integer(200, 800),
                "min_child_weight": Integer(1, 10),
                "subsample": Real(0.6, 1.0),
                "colsample_bytree": Real(0.6, 1.0),
                "gamma": Real(0, 5),
            }
        else:
            # Fallback: discrete grid for manual search
            return {
                "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "n_estimators": [200, 300, 400, 500, 600, 700, 800],
                "min_child_weight": [1, 3, 5, 7, 10],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.5, 1, 2, 3, 5],
            }

    def tune(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]] = None,
        model_type: str = "xgboost",
    ) -> Tuple[Dict[str, Any], float]:
        """Run hyperparameter tuning.

        Args:
            objective_fn: Function that takes hyperparameters and returns score (lower is better)
            warm_start_params: Parameters from similar states to start from
            model_type: Type of model ('xgboost' or 'lstm')

        Returns:
            Tuple of (best_params, best_score)
        """
        if model_type == "xgboost":
            return self._tune_xgboost(objective_fn, warm_start_params)
        elif model_type == "lstm":
            return self._tune_lstm(objective_fn, warm_start_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _tune_xgboost(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune XGBoost hyperparameters."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize_xgboost(objective_fn, warm_start_params)
        else:
            return self._fallback_search_xgboost(objective_fn, warm_start_params)

    def _bayesian_optimize_xgboost(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Use Bayesian optimization for XGBoost."""
        space = list(self.search_space.values())
        param_names = list(self.search_space.keys())

        # Wrap objective function
        def wrapped_objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            # Convert integer parameters
            params["max_depth"] = int(params["max_depth"])
            params["n_estimators"] = int(params["n_estimators"])
            params["min_child_weight"] = int(params["min_child_weight"])
            return objective_fn(params)

        # If warm start provided, use it for initial points
        x0 = None
        y0 = None
        n_initial_points = 3

        if warm_start_params:
            x0 = [
                [
                    warm_start_params.get("learning_rate", 0.05),
                    warm_start_params.get("max_depth", 6),
                    warm_start_params.get("n_estimators", 500),
                    warm_start_params.get("min_child_weight", 1),
                    warm_start_params.get("subsample", 0.8),
                    warm_start_params.get("colsample_bytree", 0.8),
                    warm_start_params.get("gamma", 0),
                ]
            ]
            # Evaluate warm start
            y0 = [objective_fn(warm_start_params)]
            n_initial_points = 2  # Fewer random starts since we have warm start

        logger.info(f"Starting Bayesian optimization with {self.max_iterations} iterations")

        # Run optimization
        result = gp_minimize(
            wrapped_objective,
            space,
            x0=x0,
            y0=y0,
            n_initial_points=n_initial_points,
            n_calls=self.max_iterations,
            random_state=self.random_state,
            verbose=False,
        )

        best_params = {name: val for name, val in zip(param_names, result.x)}
        # Convert integer parameters
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["min_child_weight"] = int(best_params["min_child_weight"])

        logger.info(f"Best score: {result.fun:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params, result.fun

    def _fallback_search_xgboost(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Fallback search strategy when scikit-optimize is not available."""
        logger.info("Using fallback search strategy (grid search with limited iterations)")

        best_params = (
            warm_start_params.copy()
            if warm_start_params
            else {
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_estimators": 500,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
            }
        )
        best_score = objective_fn(best_params)

        # Iterative refinement: focus on most impactful parameters
        # Priority: learning_rate, max_depth, n_estimators

        iteration = 0
        for lr in self.search_space["learning_rate"]:
            if iteration >= self.max_iterations:
                break
            params = best_params.copy()
            params["learning_rate"] = lr
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params
            iteration += 1

        for depth in self.search_space["max_depth"]:
            if iteration >= self.max_iterations:
                break
            params = best_params.copy()
            params["max_depth"] = depth
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params
            iteration += 1

        for n_est in self.search_space["n_estimators"]:
            if iteration >= self.max_iterations:
                break
            params = best_params.copy()
            params["n_estimators"] = n_est
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params
            iteration += 1

        logger.info(f"Fallback search completed. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params, best_score

    def _tune_lstm(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Tune LSTM hyperparameters."""
        # For LSTM, use a simpler grid search due to training time
        search_space = {
            "hidden_size": [64, 128, 256],
            "num_layers": [1, 2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.005, 0.01],
        }

        best_params = (
            warm_start_params.copy()
            if warm_start_params
            else {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
            }
        )
        best_score = objective_fn(best_params)

        iteration = 0
        for hidden in search_space["hidden_size"]:
            if iteration >= self.max_iterations:
                break
            params = best_params.copy()
            params["hidden_size"] = hidden
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params
            iteration += 1

        for lr in search_space["learning_rate"]:
            if iteration >= self.max_iterations:
                break
            params = best_params.copy()
            params["learning_rate"] = lr
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params
            iteration += 1

        logger.info(f"LSTM tuning completed. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params, best_score

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

        # Check coverage
        if latest.get("coverage_95", 0.95) < 0.90:
            focus_areas.append("subsample")
            focus_areas.append("colsample_bytree")

        # Check bias
        if abs(latest.get("bias", 0)) > 100:
            focus_areas.append("learning_rate")
            focus_areas.append("gamma")

        # Check overfitting indicators
        if latest.get("mape", 0) > 50:
            focus_areas.append("max_depth")
            focus_areas.append("min_child_weight")

        # Default if no specific issues
        if not focus_areas:
            focus_areas = ["learning_rate", "n_estimators"]

        return list(set(focus_areas))  # Remove duplicates
