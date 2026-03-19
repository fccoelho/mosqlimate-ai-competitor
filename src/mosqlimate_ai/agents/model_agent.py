"""Model Architect Agent for Karl DBot."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.prompts import get_prompt
from mosqlimate_ai.models.lstm_model import LSTMForecaster
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)


class ModelArchitectAgent(BaseAgent):
    """Agent responsible for ML/DL model architecture and training.

    This agent:
    - Designs model architectures (XGBoost, LSTM)
    - Performs hyperparameter tuning
    - Trains multiple models in parallel
    - Selects best architectures
    - Saves trained models to disk
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.trained_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.system_prompt = get_prompt("model_architect")
        self._setup_default_configs()
        logger.info("ModelArchitectAgent ready")

    def _setup_default_configs(self) -> None:
        """Setup default model configurations."""
        self.model_configs = {
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "early_stopping_rounds": 50,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 200,
                "early_stopping_patience": 20,
                "mc_samples": 100,
            },
        }

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute model training task.

        Args:
            task: Task description
            context: Context with keys:
                - data: DataFrame with features and target
                - target_col: Target column name (default: 'casos')
                - models: List of models to train (default: ['xgboost', 'lstm'])
                - save_path: Path to save trained models (optional)
                - validation_size: Fraction for validation (default: 0.1)
                - hyperparameters: Dict of model-specific hyperparameters (optional)

        Returns:
            Dictionary with trained models and metadata
        """
        logger.info(f"ModelArchitectAgent executing: {task}")
        context = context or {}

        try:
            data = context.get("data")
            if data is None:
                raise ValueError("No data provided in context")

            if isinstance(data, dict):
                results = {}
                for state, state_df in data.items():
                    state_context = {**context, "data": state_df}
                    results[state] = self._train_models(task, state_context)

                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": {
                        "trained_states": list(results.keys()),
                        "models_by_state": results,
                    },
                }
            else:
                result = self._train_models(task, context)
                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": result,
                }

        except Exception as e:
            logger.error(f"ModelArchitectAgent failed: {e}")
            return {
                "agent": self.config.name,
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _train_models(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train models for a single dataset."""
        data = context["data"]
        target_col = context.get("target_col", "casos")
        models_to_train = context.get("models", ["xgboost", "lstm"])
        save_path = context.get("save_path")
        validation_size = context.get("validation_size", 0.1)
        hyperparameters = context.get("hyperparameters", {})

        trained = {}
        feature_cols = None

        if "xgboost" in models_to_train:
            logger.info("Training XGBoost model...")
            xgb_config = {**self.model_configs["xgboost"], **hyperparameters.get("xgboost", {})}

            xgb_forecaster = XGBoostForecaster(
                target_col=target_col,
                **xgb_config,
            )
            xgb_forecaster.fit(data, validation_size=validation_size, verbose=False)
            trained["xgboost"] = xgb_forecaster
            feature_cols = xgb_forecaster.feature_cols

            if save_path:
                model_path = Path(save_path) / "xgboost"
                xgb_forecaster.save(model_path)
                logger.info(f"XGBoost saved to {model_path}")

        if "lstm" in models_to_train:
            logger.info("Training LSTM model...")
            lstm_config = {**self.model_configs["lstm"], **hyperparameters.get("lstm", {})}

            lstm_forecaster = LSTMForecaster(
                target_col=target_col,
                sequence_length=52,
                **lstm_config,
            )
            lstm_forecaster.fit(data, validation_size=validation_size, verbose=False)
            trained["lstm"] = lstm_forecaster
            feature_cols = feature_cols or lstm_forecaster.feature_cols

            if save_path:
                model_path = Path(save_path) / "lstm"
                lstm_forecaster.save(model_path)
                logger.info(f"LSTM saved to {model_path}")

        self.trained_models.update(trained)

        return {
            "trained_models": list(trained.keys()),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "models": trained if not save_path else None,
            "save_path": str(save_path) if save_path else None,
        }

    def train_xgboost(
        self,
        data: pd.DataFrame,
        target_col: str = "casos",
        validation_size: float = 0.1,
        hyperparameters: Optional[Dict] = None,
    ) -> XGBoostForecaster:
        """Train XGBoost model.

        Args:
            data: DataFrame with features and target
            target_col: Target column name
            validation_size: Fraction for validation
            hyperparameters: Model hyperparameters

        Returns:
            Trained XGBoostForecaster
        """
        config = {**self.model_configs["xgboost"], **(hyperparameters or {})}
        forecaster = XGBoostForecaster(target_col=target_col, **config)
        forecaster.fit(data, validation_size=validation_size, verbose=False)
        self.trained_models["xgboost"] = forecaster
        return forecaster

    def train_lstm(
        self,
        data: pd.DataFrame,
        target_col: str = "casos",
        validation_size: float = 0.1,
        hyperparameters: Optional[Dict] = None,
    ) -> LSTMForecaster:
        """Train LSTM model.

        Args:
            data: DataFrame with features and target
            target_col: Target column name
            validation_size: Fraction for validation
            hyperparameters: Model hyperparameters

        Returns:
            Trained LSTMForecaster
        """
        config = {**self.model_configs["lstm"], **(hyperparameters or {})}
        forecaster = LSTMForecaster(target_col=target_col, **config)
        forecaster.fit(data, validation_size=validation_size, verbose=False)
        self.trained_models["lstm"] = forecaster
        return forecaster

    def load_models(
        self, path: Union[str, Path], models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load saved models from disk.

        Args:
            path: Base path to load models from
            models: List of model names to load (default: all)

        Returns:
            Dictionary of loaded models
        """
        path = Path(path)
        models = models or ["xgboost", "lstm"]
        loaded = {}

        if "xgboost" in models and (path / "xgboost").exists():
            forecaster = XGBoostForecaster()
            forecaster.load(path / "xgboost")
            loaded["xgboost"] = forecaster
            self.trained_models["xgboost"] = forecaster

        if "lstm" in models and (path / "lstm").exists():
            forecaster = LSTMForecaster()
            forecaster.load(path / "lstm")
            loaded["lstm"] = forecaster
            self.trained_models["lstm"] = forecaster

        logger.info(f"Loaded models: {list(loaded.keys())}")
        return loaded

    def get_model(self, name: str) -> Optional[Any]:
        """Get a trained model by name.

        Args:
            name: Model name ('xgboost' or 'lstm')

        Returns:
            Trained model or None
        """
        return self.trained_models.get(name)

    def cross_validate(
        self,
        data: pd.DataFrame,
        target_col: str = "casos",
        n_splits: int = 5,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation on models.

        Args:
            data: DataFrame with features and target
            target_col: Target column name
            n_splits: Number of CV splits
            models: Models to evaluate (default: all)

        Returns:
            Dictionary of CV results by model
        """
        models = models or ["xgboost"]
        results = {}

        for model_name in models:
            if model_name == "xgboost":
                forecaster = XGBoostForecaster(
                    target_col=target_col, **self.model_configs["xgboost"]
                )
                feature_cols = forecaster._infer_features(data)
                x_features = data[feature_cols].values
                y_target = data[target_col].values

                cv_results = forecaster.model.cross_validate(
                    x_features, y_target, n_splits=n_splits
                )
                results["xgboost"] = {
                    "rmse_mean": cv_results["rmse"]["mean"],
                    "rmse_std": cv_results["rmse"]["std"],
                    "mae_mean": cv_results["mae"]["mean"],
                    "coverage_95_mean": cv_results["coverage_95"]["mean"],
                }

        return results
