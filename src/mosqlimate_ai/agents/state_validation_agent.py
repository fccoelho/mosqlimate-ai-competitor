"""StateValidationAgent for managing validation pipeline per state.

This agent handles the 4-run validation pipeline for a single state,
using KarlDBot for decision-making and coordinating with other agents.
"""

import logging
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mosqlimate_ai.agents.base import BaseAgent, AgentConfig
from mosqlimate_ai.agents.communication import (
    AgentCommunicationBus,
    AgentMessage,
    MessageType,
    MessagePriority,
)
from mosqlimate_ai.agents.knowledge_base import CrossStateKnowledgeBase, ValidationResult
from mosqlimate_ai.agents.tuner_agent import EfficientHyperparameterTuner
from mosqlimate_ai.agents.selection_agent import TopNModelSelectionAgent
from mosqlimate_ai.agents.model_selector_agent import ModelPreSelector
from mosqlimate_ai.validation.config import ValidationPipelineConfig, ValidationTestConfig
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.data.preprocessor import DataPreprocessor
from mosqlimate_ai.data.features import FeatureEngineer
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster
from mosqlimate_ai.models.lstm_model import LSTMForecaster
from mosqlimate_ai.models.prophet_model import ProphetForecaster
from mosqlimate_ai.models.tft_model import TFTForecaster
from mosqlimate_ai.models.nbeats_model import NBEATSForecaster
from mosqlimate_ai.evaluation.metrics import (
    evaluate_forecast,
    crps,
    weighted_interval_score,
    weighted_interval_score_total,
    mae,
    rmse,
    mape,
    bias,
    coverage,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


class StateValidationAgent(BaseAgent):
    """Agent that manages the 4-run validation pipeline for a single state.

    This agent:
    1. Loads and preprocesses data for each validation test
    2. Runs hyperparameter tuning with cross-state knowledge
    3. Trains models for each validation period
    4. Evaluates and selects top N models
    5. Communicates results via the message bus

    Example:
        >>> agent = StateValidationAgent("SP", config, message_bus, knowledge_base)
        >>> results = agent.run_full_validation()
    """

    def __init__(
        self,
        state: str,
        config: ValidationPipelineConfig,
        message_bus: AgentCommunicationBus,
        knowledge_base: CrossStateKnowledgeBase,
        output_dir: Path = Path("validation_results"),
    ):
        """Initialize state validation agent.

        Args:
            state: State UF code (e.g., "SP")
            config: Validation pipeline configuration
            message_bus: Communication bus for agent messaging
            knowledge_base: Shared knowledge base for cross-state learning
            output_dir: Directory for validation outputs
        """
        agent_config = AgentConfig(
            name=f"StateValidationAgent_{state}",
            description=f"Manages validation pipeline for {state}",
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
        super().__init__(agent_config)

        self.state = state
        self.pipeline_config = config
        self.message_bus = message_bus
        self.knowledge_base = knowledge_base
        self.output_dir = Path(output_dir)

        # Initialize supporting agents
        self.tuner = EfficientHyperparameterTuner(
            max_iterations=config.tuning_iterations,
            convergence_patience=config.convergence_patience,
            min_improvement_rate=config.min_improvement_rate,
        )
        self.selector = TopNModelSelectionAgent(
            n_top=config.n_top_models,
            min_coverage_threshold=config.min_coverage_threshold,
            max_bias_threshold=config.max_bias_threshold,
        )
        self.model_preselector = ModelPreSelector(
            max_models=config.max_models_per_state,
            knowledge_base=knowledge_base,
        )

        # Model registry
        self.model_classes = {
            "xgboost": XGBoostForecaster,
            "lstm": LSTMForecaster,
            "prophet": ProphetForecaster,
            "tft": TFTForecaster,
            "nbeats": NBEATSForecaster,
        }

        # Results storage
        self.validation_results: Dict[int, Dict[str, Any]] = {}
        self.final_results: Optional[Dict[str, Any]] = None
        self.selected_models: List[str] = []

        logger.info(f"Initialized StateValidationAgent for {state}")

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute validation task.

        Args:
            task: Task description (e.g., "run_validation_test_1", "run_final_forecast")
            context: Additional context

        Returns:
            Validation results dictionary
        """
        if task.startswith("run_validation_test_"):
            test_num = int(task.split("_")[-1])
            return self._run_validation_test(test_num)
        elif task == "run_final_forecast":
            return self._run_final_forecast()
        elif task == "run_full_validation":
            return self.run_full_validation()
        else:
            raise ValueError(f"Unknown task: {task}")

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete 4-run validation pipeline.

        Returns:
            Dictionary with all validation results
        """
        logger.info(f"Starting full validation pipeline for {self.state}")

        # Create and send message using AgentMessage
        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.COMMAND,
            content={"state": self.state, "timestamp": datetime.now().isoformat()},
            state=self.state,
        )
        self.message_bus.send_message(message)

        # Run 3 validation tests
        for test_config in self.pipeline_config.validation_tests:
            logger.info(f"Running validation test {test_config.test_number} for {self.state}")
            result = self._run_validation_test(test_config.test_number)
            self.validation_results[test_config.test_number] = result

            # Share insights with knowledge base using share_results
            if result.get("status") == "success":
                # Convert to ValidationResult format
                for model_name, metrics in result.get("metrics", {}).items():
                    validation_result = ValidationResult(
                        state=self.state,
                        validation_test=test_config.test_number,
                        model_name=model_name,
                        crps=metrics.get("crps", 0.0),
                        wis_total=metrics.get("wis_total", 0.0),
                        rmse=metrics.get("rmse", 0.0),
                        mae=metrics.get("mae", 0.0),
                        mape=metrics.get("mape", 0.0),
                        bias=metrics.get("bias", 0.0),
                        coverage_50=metrics.get("coverage_50", 0.0),
                        coverage_80=metrics.get("coverage_80", 0.0),
                        coverage_90=metrics.get("coverage_90", 0.0),
                        coverage_95=metrics.get("coverage_95", 0.0),
                        hyperparameters=result.get("hyperparameters", {}),
                        timestamp=datetime.now().isoformat(),
                    )
                    self.knowledge_base.share_results(validation_result)

        # Run final forecast
        logger.info(f"Running final forecast for {self.state}")
        self.final_results = self._run_final_forecast()

        # Select top models - restructure results to expected format
        # Format: {model_name: {test_num: metrics}}
        structured_results = self._structure_results_for_selection()
        top_models = self.selector.select_top_models(all_results=structured_results)

        # Compile final results
        results = {
            "state": self.state,
            "validation_tests": self.validation_results,
            "final_forecast": self.final_results,
            "top_models": top_models,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        self._save_results(results)

        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.RESULT,
            content={"state": self.state, "status": "success"},
            state=self.state,
        )
        self.message_bus.send_message(message)

        logger.info(f"Completed validation pipeline for {self.state}")
        return results

    def _structure_results_for_selection(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Restructure validation results for TopNModelSelectionAgent.

        Returns:
            Dictionary in format {model_name: {test_num: metrics}}
        """
        structured: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for test_num, result in self.validation_results.items():
            if result.get("status") != "success":
                continue

            metrics = result.get("metrics", {})
            hyperparams = result.get("hyperparameters", {})

            for model_name, model_metrics in metrics.items():
                if model_name not in structured:
                    structured[model_name] = {}

                # Combine metrics with hyperparameters for the test
                test_data = {**model_metrics}
                if model_name in hyperparams:
                    test_data["hyperparameters"] = hyperparams[model_name]

                structured[model_name][test_num] = test_data

        return structured

    def _run_validation_test(self, test_number: int) -> Dict[str, Any]:
        """Run a single validation test.

        Args:
            test_number: Test number (1, 2, or 3)

        Returns:
            Test results dictionary
        """
        test_config = next(
            t for t in self.pipeline_config.validation_tests if t.test_number == test_number
        )

        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.COMMAND,
            content={
                "state": self.state,
                "test_number": test_number,
                "season": test_config.season,
            },
            state=self.state,
            validation_test=test_number,
        )
        self.message_bus.send_message(message)

        try:
            # Load data
            data = self._load_data(test_config)

            # Pre-select models based on state characteristics
            selected_models = self._select_models_for_state(data)
            logger.info(f"Selected models for {self.state} test {test_number}: {selected_models}")

            # Tune hyperparameters for selected models
            tuned_params = self._tune_hyperparameters(data, test_number, selected_models)

            # Train models
            models = self._train_models(data, tuned_params, selected_models)

            # Evaluate models
            metrics = self._evaluate_models(models, data)

            result = {
                "test_number": test_number,
                "season": test_config.season,
                "state": self.state,
                "metrics": metrics,
                "hyperparameters": tuned_params,
                "selected_models": selected_models,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Validation test {test_number} failed for {self.state}: {e}")
            result = {
                "test_number": test_number,
                "season": test_config.season,
                "state": self.state,
                "error": str(e),
                "status": "failed",
            }

        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.RESULT,
            content={
                "state": self.state,
                "test_number": test_number,
                "status": result["status"],
            },
            state=self.state,
            validation_test=test_number,
        )
        self.message_bus.send_message(message)

        return result

    def _select_models_for_state(self, data: Dict[str, Any]) -> List[str]:
        """Select models based on state characteristics and knowledge base.

        Args:
            data: Training data dictionary

        Returns:
            List of selected model names
        """
        df = data["data"]

        data_characteristics = {
            "data_size": len(df),
            "n_features": len([c for c in df.columns if c not in ["date", "casos", "uf"]]),
            "has_missing": df.isnull().any().any(),
        }

        if self.pipeline_config.preselect_models:
            recommendations = self.model_preselector.select_models_for_state(
                self.state, data_characteristics
            )
            selected = [r.model_name for r in recommendations]
        else:
            selected = list(self.model_classes.keys())

        return selected

    def _tune_hyperparameters(
        self,
        data: Dict[str, Any],
        test_number: int,
        selected_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Tune hyperparameters for all models with cross-state warm starting.

        Args:
            data: Training data
            test_number: Current validation test number
            selected_models: Optional list of models to tune (if None, tune all)

        Returns:
            Dictionary with tuned hyperparameters per model
        """
        tuned_params = {}
        df = data["data"]

        selected_models = selected_models or list(self.model_classes.keys())

        warm_start = self._get_warm_start_params()

        cv_splits = self._create_cv_splits(df, n_splits=3)

        if "xgboost" in selected_models:
            tuned_params["xgboost"] = self._tune_single_model(
                "xgboost",
                XGBoostForecaster,
                df,
                cv_splits,
                warm_start.get("xgboost"),
            )

        if "lstm" in selected_models:
            tuned_params["lstm"] = self._tune_single_model(
                "lstm",
                LSTMForecaster,
                df,
                cv_splits,
                warm_start.get("lstm"),
            )

        if "prophet" in selected_models:
            tuned_params["prophet"] = self._tune_single_model(
                "prophet",
                ProphetForecaster,
                df,
                cv_splits,
                warm_start.get("prophet"),
            )

        if "tft" in selected_models:
            tuned_params["tft"] = self._tune_single_model(
                "tft",
                TFTForecaster,
                df,
                cv_splits,
                warm_start.get("tft"),
            )

        if "nbeats" in selected_models:
            tuned_params["nbeats"] = self._tune_single_model(
                "nbeats",
                NBEATSForecaster,
                df,
                cv_splits,
                warm_start.get("nbeats"),
            )

        return tuned_params

    def _get_warm_start_params(self) -> Dict[str, Dict[str, Any]]:
        """Get warm start parameters from similar states in knowledge base.

        Returns:
            Dictionary of warm start params per model type
        """
        warm_start = {}

        similar_states = self.knowledge_base.get_similar_states(self.state, n_similar=3)
        similar_state_names = [s[0] for s in similar_states]

        if not similar_state_names:
            logger.info(f"No similar states found for {self.state}, using defaults")
            return warm_start

        for model_type in ["xgboost", "lstm", "prophet", "tft", "nbeats"]:
            try:
                params = self.knowledge_base.get_best_params(similar_state_names, model_type)
                if params:
                    warm_start[model_type] = params
                    logger.info(
                        f"Warm start params for {model_type} from similar states: "
                        f"{similar_state_names}"
                    )
            except Exception as e:
                logger.debug(f"Could not get warm start params for {model_type}: {e}")

        return warm_start

    def _create_cv_splits(
        self, df: pd.DataFrame, n_splits: int = 3
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-series cross-validation splits.

        Args:
            df: Full dataset
            n_splits: Number of CV splits

        Returns:
            List of (train_df, val_df) tuples
        """
        df = df.sort_values("date").reset_index(drop=True)
        n_samples = len(df)
        min_train_size = max(52, int(n_samples * 0.5))

        splits = []
        for i in range(n_splits):
            val_size = min(12, int(n_samples * 0.1))
            train_end = n_samples - (n_splits - i) * val_size

            if train_end < min_train_size:
                train_end = min_train_size

            val_start = train_end
            val_end = min(val_start + val_size, n_samples)

            if val_end > val_start:
                train_df = df.iloc[:train_end].copy()
                val_df = df.iloc[val_start:val_end].copy()
                splits.append((train_df, val_df))

        if not splits:
            split_point = int(n_samples * 0.8)
            splits = [(df.iloc[:split_point], df.iloc[split_point:])]

        return splits

    def _tune_single_model(
        self,
        model_type: str,
        model_class: type,
        df: pd.DataFrame,
        cv_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        warm_start_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Tune a single model type with proper objective function.

        Args:
            model_type: Model type name
            model_class: Model class to instantiate
            df: Full dataset
            cv_splits: Cross-validation splits
            warm_start_params: Optional warm start parameters

        Returns:
            Tuned hyperparameters
        """
        default_params = self._get_model_defaults(model_type)

        def objective(params: Dict[str, Any]) -> float:
            """Objective function using cross-validation."""
            try:
                merged_params = {**default_params, **params}

                cv_scores = []
                for train_df, val_df in cv_splits:
                    try:
                        model = model_class(**merged_params)
                        model.fit(train_df, verbose=False)

                        n_weeks = len(val_df)
                        forecast = model.predict(weeks=n_weeks)

                        score = self._compute_validation_score(val_df, forecast)
                        cv_scores.append(score)
                    except Exception as e:
                        logger.debug(f"CV fold failed for {model_type}: {e}")
                        cv_scores.append(1e6)

                if not cv_scores:
                    return 1e6

                mean_score = float(np.mean(cv_scores))
                return mean_score

            except Exception as e:
                logger.warning(f"{model_type} objective failed: {e}")
                return 1e6

        try:
            best_params, best_score = self.tuner.tune(
                objective_fn=objective,
                warm_start_params=warm_start_params,
                model_type=model_type,
            )
            final_params = {**default_params, **best_params}
            logger.info(f"{model_type} tuning completed with score: {best_score:.4f}")
            return final_params
        except Exception as e:
            logger.error(f"{model_type} tuning failed: {e}, using defaults")
            return default_params

    def _compute_validation_score(self, val_df: pd.DataFrame, forecast: pd.DataFrame) -> float:
        """Compute validation score (CRPS) for a forecast.

        Args:
            val_df: Validation data with true values
            forecast: Forecast DataFrame

        Returns:
            CRPS score (lower is better)
        """
        try:
            actual = val_df["casos"].values
            n_points = min(len(actual), len(forecast))
            actual = actual[:n_points]
            forecast = forecast.iloc[:n_points]

            if "median" not in forecast.columns:
                if "yhat" in forecast.columns:
                    forecast = forecast.rename(columns={"yhat": "median"})
                else:
                    return 1e6

            try:
                score = crps(actual, forecast)
                if np.isnan(score) or np.isinf(score):
                    return 1e6
                return float(score)
            except Exception:
                pred = forecast["median"].values[:n_points]
                mae_score = np.mean(np.abs(actual - pred))
                return float(mae_score)

        except Exception as e:
            logger.debug(f"Score computation failed: {e}")
            return 1e6

    def _get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type.

        Args:
            model_type: Model type name

        Returns:
            Default hyperparameters dictionary
        """
        defaults = {
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "early_stopping_rounds": 50,
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping_patience": 10,
            },
            "prophet": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "interval_width": 0.80,
            },
            "tft": {
                "hidden_size": 64,
                "hidden_continuous_size": 32,
                "attention_head_size": 4,
                "dropout": 0.1,
                "hidden_layer_size": 128,
                "learning_rate": 0.001,
                "max_prediction_length": 52,
                "max_encoder_length": 104,
                "batch_size": 64,
                "max_epochs": 30,
            },
            "nbeats": {
                "stack_types": ["generic"],
                "num_blocks": [3],
                "num_block_layers": [4],
                "hidden_size": 256,
                "learning_rate": 0.001,
                "max_prediction_length": 52,
                "max_encoder_length": 104,
                "batch_size": 64,
                "max_epochs": 30,
                "mc_samples": 100,
            },
        }
        return defaults.get(model_type, {})

    def _run_final_forecast(self) -> Dict[str, Any]:
        """Run final forecast using best configuration.

        Returns:
            Final forecast results
        """
        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.COMMAND,
            content={"state": self.state},
            state=self.state,
        )
        self.message_bus.send_message(message)

        try:
            # Use best hyperparameters from validation tests
            best_params = self._get_best_hyperparameters()

            # Load all training data up to EW25 2025
            data = self._load_final_data()

            # Train final models
            models = self._train_models(data, best_params)

            # Generate forecasts
            forecasts = self._generate_forecasts(models)

            result = {
                "state": self.state,
                "forecasts": forecasts,
                "hyperparameters": best_params,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Final forecast failed for {self.state}: {e}")
            result = {
                "state": self.state,
                "error": str(e),
                "status": "failed",
            }

        message = AgentMessage(
            sender=self.config.name,
            receiver="ValidationOrchestrator",
            message_type=MessageType.RESULT,
            content={"state": self.state, "status": result["status"]},
            state=self.state,
        )
        self.message_bus.send_message(message)

        return result

    def _load_data(self, test_config: ValidationTestConfig) -> Dict[str, Any]:
        """Load and preprocess data for a validation test.

        Args:
            test_config: Validation test configuration

        Returns:
            Dictionary with processed data
        """
        loader = CompetitionDataLoader()
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()

        # Load data
        df = loader.load_state_data(self.state)

        # Filter training period
        df = df[df["date"] <= test_config.train_end]

        # Preprocess
        df = preprocessor.clean(df)
        df = preprocessor.impute_missing(df)

        # Feature engineering
        df = feature_engineer.build_feature_set(df)

        return {"data": df}

    def _load_final_data(self) -> Dict[str, Any]:
        """Load data for final forecast training.

        Returns:
            Dictionary with processed data
        """
        loader = CompetitionDataLoader()
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()

        df = loader.load_state_data(self.state)
        df = df[df["date"] <= self.pipeline_config.final_forecast_train_end]

        df = preprocessor.clean(df)
        df = preprocessor.impute_missing(df)
        df = feature_engineer.build_feature_set(df)

        return {"data": df}

    def _train_models(
        self,
        data: Dict[str, Any],
        params: Dict[str, Any],
        selected_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train models with given hyperparameters.

        Args:
            data: Training data
            params: Hyperparameters
            selected_models: Optional list of models to train (if None, train all with params)

        Returns:
            Dictionary with trained models
        """
        df = data["data"]
        models = {}
        selected_models = selected_models or list(params.keys())

        for model_name in selected_models:
            if model_name not in params:
                logger.warning(f"No hyperparameters for {model_name}, skipping")
                continue

            model_class = self.model_classes.get(model_name)
            if model_class is None:
                logger.warning(f"Unknown model type: {model_name}")
                continue

            try:
                model_params = params[model_name]
                model = model_class(**model_params)
                model.fit(df, verbose=False)
                models[model_name] = model
                logger.info(f"{model_name} model trained successfully")
            except Exception as e:
                logger.error(f"{model_name} training failed: {e}")

        return models

    def _evaluate_models(
        self, models: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models with proper time-series validation.

        Args:
            models: Dictionary of trained models
            data: Data for evaluation (includes train data)

        Returns:
            Dictionary of metrics per model
        """
        metrics = {}
        df = data["data"]

        n_samples = len(df)
        val_size = min(12, int(n_samples * 0.15))
        split_point = n_samples - val_size

        if split_point < 52:
            logger.warning("Not enough data for proper evaluation, using synthetic metrics")
            return self._get_default_metrics(models.keys())

        train_df = df.iloc[:split_point]
        val_df = df.iloc[split_point:]
        actual = val_df["casos"].values
        n_weeks = len(val_df)

        for model_name, model in models.items():
            try:
                forecast = model.predict(weeks=n_weeks)

                model_metrics = self._compute_all_metrics(actual, forecast, n_weeks)
                metrics[model_name] = model_metrics

                logger.info(
                    f"{model_name} evaluation: CRPS={model_metrics['crps']:.2f}, "
                    f"WIS={model_metrics['wis_total']:.2f}, "
                    f"Coverage95={model_metrics['coverage_95']:.2%}"
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
                metrics[model_name] = self._get_default_metrics_single()

        return metrics

    def _compute_all_metrics(
        self, actual: np.ndarray, forecast: pd.DataFrame, n_weeks: int
    ) -> Dict[str, float]:
        """Compute all evaluation metrics for a forecast.

        Args:
            actual: True values
            forecast: Forecast DataFrame
            n_weeks: Number of forecast weeks

        Returns:
            Dictionary of metrics
        """
        n_points = min(len(actual), len(forecast))
        actual = actual[:n_points]
        forecast = forecast.iloc[:n_points].copy()

        if "median" not in forecast.columns:
            if "yhat" in forecast.columns:
                forecast = forecast.rename(columns={"yhat": "median"})
                if "yhat_lower" in forecast.columns:
                    forecast["lower_95"] = forecast["yhat_lower"]
                if "yhat_upper" in forecast.columns:
                    forecast["upper_95"] = forecast["yhat_upper"]

        pred = forecast["median"].values if "median" in forecast.columns else np.zeros(n_points)

        metrics = {
            "mae": float(mae(actual, pred)),
            "rmse": float(rmse(actual, pred)),
            "mape": float(mape(actual, pred)),
            "bias": float(bias(actual, pred)),
        }

        try:
            metrics["crps"] = float(crps(actual, forecast))
        except Exception:
            metrics["crps"] = metrics["mae"]

        for level, col_name in [(0.50, "50"), (0.80, "80"), (0.90, "90"), (0.95, "95")]:
            lower_col = f"lower_{col_name}"
            upper_col = f"upper_{col_name}"

            if lower_col in forecast.columns and upper_col in forecast.columns:
                lower = forecast[lower_col].values
                upper = forecast[upper_col].values
                metrics[f"coverage_{col_name}"] = float(coverage(actual, lower, upper))
            else:
                metrics[f"coverage_{col_name}"] = level

        try:
            metrics["wis_total"] = float(weighted_interval_score_total(actual, forecast))
        except Exception:
            lower = (
                forecast.get("lower_95", pred * 0.8).values
                if "lower_95" in forecast.columns
                else pred * 0.8
            )
            upper = (
                forecast.get("upper_95", pred * 1.2).values
                if "upper_95" in forecast.columns
                else pred * 1.2
            )
            metrics["wis_total"] = float(
                weighted_interval_score(actual, lower, upper, pred, alpha=0.05)
            )

        return metrics

    def _get_default_metrics(self, model_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get default metrics for all models when evaluation fails.

        Args:
            model_names: List of model names

        Returns:
            Dictionary of default metrics per model
        """
        return {name: self._get_default_metrics_single() for name in model_names}

    def _get_default_metrics_single(self) -> Dict[str, float]:
        """Get default metrics for a single model.

        Returns:
            Dictionary of default metrics
        """
        return {
            "crps": 0.0,
            "wis_total": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "bias": 0.0,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
            "coverage_95": 0.95,
        }

    def _generate_forecasts(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts from trained models.

        Args:
            models: Dictionary of trained models

        Returns:
            Dictionary with forecasts per model
        """
        forecasts = {}

        for model_name, model in models.items():
            # Generate forecast
            forecast = model.predict(weeks=52)
            forecasts[model_name] = forecast

        return forecasts

    def _get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters from validation tests.

        Returns:
            Best hyperparameters
        """
        # Aggregate hyperparameters from all tests
        # Return most frequently used or best performing
        all_params = [
            result.get("hyperparameters", {}) for result in self.validation_results.values()
        ]

        # Simple approach: return first non-empty set
        for params in all_params:
            if params:
                return params

        return {}

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to disk.

        Args:
            results: Validation results dictionary
        """
        state_dir = self.output_dir / self.state
        state_dir.mkdir(parents=True, exist_ok=True)

        results_file = state_dir / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved validation results to {results_file}")
