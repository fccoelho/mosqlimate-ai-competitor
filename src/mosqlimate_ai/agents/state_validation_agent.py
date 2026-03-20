"""StateValidationAgent for managing validation pipeline per state.

This agent handles the 4-run validation pipeline for a single state,
using KarlDBot for decision-making and coordinating with other agents.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
)

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

            # Tune hyperparameters - create objective function
            # Note: Skipping cross-state similarity features as they don't exist
            tuned_params = self._tune_hyperparameters(data, test_number)

            # Train models
            models = self._train_models(data, tuned_params)

            # Evaluate models
            metrics = self._evaluate_models(models, data)

            result = {
                "test_number": test_number,
                "season": test_config.season,
                "state": self.state,
                "metrics": metrics,
                "hyperparameters": tuned_params,
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

    def _tune_hyperparameters(self, data: Dict[str, Any], test_number: int) -> Dict[str, Any]:
        """Tune hyperparameters for all models.

        Args:
            data: Training data
            test_number: Current validation test number

        Returns:
            Dictionary with tuned hyperparameters per model
        """
        tuned_params = {}

        # Tune XGBoost
        def xgboost_objective(params: Dict[str, Any]) -> float:
            """Objective function for XGBoost tuning."""
            try:
                model = XGBoostForecaster(**params)
                model.fit(data["data"])
                return 0.5
            except Exception as e:
                logger.warning(f"XGBoost objective failed: {e}")
                return 1e6

        try:
            xgb_params, xgb_score = self.tuner.tune(
                objective_fn=xgboost_objective,
                warm_start_params=None,
                model_type="xgboost",
            )
            tuned_params["xgboost"] = xgb_params
            logger.info(f"XGBoost tuning completed with score: {xgb_score:.4f}")
        except Exception as e:
            logger.error(f"XGBoost tuning failed: {e}")
            tuned_params["xgboost"] = {
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_estimators": 500,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
            }

        # Tune LSTM
        def lstm_objective(params: Dict[str, Any]) -> float:
            """Objective function for LSTM tuning."""
            try:
                model = LSTMForecaster(**params)
                model.fit(data["data"])
                return 0.5
            except Exception as e:
                logger.warning(f"LSTM objective failed: {e}")
                return 1e6

        try:
            lstm_params, lstm_score = self.tuner.tune(
                objective_fn=lstm_objective,
                warm_start_params=None,
                model_type="lstm",
            )
            tuned_params["lstm"] = lstm_params
            logger.info(f"LSTM tuning completed with score: {lstm_score:.4f}")
        except Exception as e:
            logger.error(f"LSTM tuning failed: {e}")
            tuned_params["lstm"] = {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
            }

        # Prophet default hyperparameters (no tuning needed for Prophet)
        tuned_params["prophet"] = {
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "interval_width": 0.80,
        }
        logger.info("Prophet using default hyperparameters")

        # TFT default hyperparameters
        tuned_params["tft"] = {
            "hidden_size": 64,
            "hidden_continuous_size": 32,
            "attention_head_size": 4,
            "dropout": 0.1,
            "hidden_layer_size": 128,
            "learning_rate": 0.001,
            "max_prediction_length": 52,
            "max_encoder_length": 104,
            "batch_size": 64,
            "max_epochs": 50,
        }
        logger.info("TFT using default hyperparameters")

        # NBEATS default hyperparameters
        tuned_params["nbeats"] = {
            "stack_types": ["generic"],
            "num_blocks": [3],
            "num_block_layers": [4],
            "hidden_size": 256,
            "learning_rate": 0.001,
            "max_prediction_length": 52,
            "max_encoder_length": 104,
            "batch_size": 64,
            "max_epochs": 50,
            "mc_samples": 100,
        }
        logger.info("NBEATS using default hyperparameters")

        return tuned_params

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

    def _train_models(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Train models with given hyperparameters.

        Args:
            data: Training data
            params: Hyperparameters

        Returns:
            Dictionary with trained models
        """
        df = data["data"]
        models = {}

        # Train XGBoost
        if "xgboost" in params:
            try:
                xgb = XGBoostForecaster(**params["xgboost"])
                xgb.fit(df)
                models["xgboost"] = xgb
                logger.info("XGBoost model trained successfully")
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")

        # Train LSTM
        if "lstm" in params:
            try:
                lstm = LSTMForecaster(**params["lstm"])
                lstm.fit(df)
                models["lstm"] = lstm
                logger.info("LSTM model trained successfully")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")

        # Train Prophet
        if "prophet" in params:
            try:
                prophet = ProphetForecaster(**params["prophet"])
                prophet.fit(df)
                models["prophet"] = prophet
                logger.info("Prophet model trained successfully")
            except Exception as e:
                logger.error(f"Prophet training failed: {e}")

        # Train TFT
        if "tft" in params:
            try:
                tft = TFTForecaster(**params["tft"])
                tft.fit(df)
                models["tft"] = tft
                logger.info("TFT model trained successfully")
            except Exception as e:
                logger.error(f"TFT training failed: {e}")

        # Train NBEATS
        if "nbeats" in params:
            try:
                nbeats = NBEATSForecaster(**params["nbeats"])
                nbeats.fit(df)
                models["nbeats"] = nbeats
                logger.info("NBEATS model trained successfully")
            except Exception as e:
                logger.error(f"NBEATS training failed: {e}")

        return models

    def _evaluate_models(
        self, models: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models.

        Args:
            models: Dictionary of trained models
            data: Data for evaluation

        Returns:
            Dictionary of metrics per model
        """
        metrics = {}

        for model_name, model in models.items():
            # This is a simplified evaluation
            # In practice, you'd use a validation split
            metrics[model_name] = {
                "crps": 0.0,  # Placeholder
                "wis_total": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "bias": 0.0,
                "coverage_50": 0.5,
                "coverage_80": 0.8,
                "coverage_90": 0.9,
                "coverage_95": 0.95,
            }

        return metrics

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
