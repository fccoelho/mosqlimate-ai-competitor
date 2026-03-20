"""Agent orchestrator for Karl DBot multi-agent system."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mosqlimate_ai.agents.base import BaseAgent
from mosqlimate_ai.evaluation.metrics import crps, mae, rmse

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task in the workflow."""

    id: str
    name: str
    agent: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """Represents a complete forecasting workflow."""

    id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FineTuningConfig:
    """Configuration for performance-based fine-tuning."""

    max_rounds: int = 5
    convergence_patience: int = 3
    min_improvement_rate: float = 0.01
    target_metric: str = "crps"
    optimization_direction: str = "minimize"
    model_types: List[str] = field(default_factory=lambda: ["xgboost", "lstm"])
    validation_size: float = 0.15
    n_cv_splits: int = 3


@dataclass
class FineTuningResult:
    """Results from a fine-tuning round."""

    round_num: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    improvement: float
    is_best: bool
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """Tracks performance metrics across fine-tuning rounds."""

    def __init__(
        self, target_metric: str = "crps", patience: int = 3, min_improvement: float = 0.01
    ):
        self.target_metric = target_metric
        self.patience = patience
        self.min_improvement = min_improvement
        self.history: List[FineTuningResult] = []
        self.best_result: Optional[FineTuningResult] = None
        self.rounds_without_improvement = 0

    def update(
        self, round_num: int, hyperparams: Dict[str, Any], metrics: Dict[str, float]
    ) -> bool:
        current_score = metrics.get(self.target_metric, float("inf"))

        improvement = 0.0
        is_best = False

        if self.best_result is None:
            is_best = True
            improvement = 1.0
        else:
            best_score = self.best_result.metrics.get(self.target_metric, float("inf"))
            improvement = (best_score - current_score) / (abs(best_score) + 1e-8)

            if improvement > self.min_improvement:
                is_best = True
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1

        result = FineTuningResult(
            round_num=round_num,
            hyperparameters=hyperparams,
            metrics=metrics,
            improvement=improvement,
            is_best=is_best,
        )
        self.history.append(result)

        if is_best:
            self.best_result = result

        return self.should_continue()

    def should_continue(self) -> bool:
        return self.rounds_without_improvement < self.patience

    def get_best_hyperparameters(self) -> Dict[str, Any]:
        if self.best_result:
            return self.best_result.hyperparameters
        return {}

    def get_convergence_summary(self) -> Dict[str, Any]:
        return {
            "total_rounds": len(self.history),
            "best_round": self.best_result.round_num if self.best_result else None,
            "best_metrics": self.best_result.metrics if self.best_result else {},
            "converged": not self.should_continue(),
            "improvement_history": [r.improvement for r in self.history],
        }


class AgentOrchestrator:
    """Orchestrates multi-agent workflow for dengue forecasting.

    This is the central coordinator for Karl DBot agents,
    managing task execution, inter-agent communication,
    and workflow optimization.

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> orchestrator.register_agent(DataEngineerAgent(config))
        >>> result = orchestrator.run_forecast_workflow(uf="SP")
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.message_queue: List[Dict[str, Any]] = []
        logger.info("AgentOrchestrator initialized")

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.config.name] = agent
        logger.info(f"Registered agent: {agent.config.name}")

    def create_workflow(self, name: str, tasks: List[Task]) -> Workflow:
        """Create a new workflow.

        Args:
            name: Workflow name
            tasks: List of tasks

        Returns:
            Created workflow
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow = Workflow(id=workflow_id, name=name, tasks={t.id: t for t in tasks})
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id}")
        return workflow

    def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow results
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow.status = "running"
        logger.info(f"Starting workflow: {workflow_id}")

        completed = set()
        failed = []
        max_iterations = len(workflow.tasks) * 2
        iteration = 0

        while len(completed) + len(failed) < len(workflow.tasks):
            iteration += 1
            if iteration > max_iterations:
                logger.error("Max iterations reached, stopping workflow")
                break

            ready = [
                t
                for t in workflow.tasks.values()
                if t.status == "pending" and all(dep in completed for dep in t.dependencies)
            ]

            if not ready:
                if len(completed) + len(failed) < len(workflow.tasks):
                    logger.error("Dependency deadlock - pending tasks cannot run")
                break

            for task in ready:
                try:
                    result = self._execute_task(task)
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    completed.add(task.id)
                    logger.info(f"Task completed: {task.name}")
                except Exception as e:
                    task.status = "failed"
                    failed.append(task.id)
                    logger.error(f"Task failed: {task.name} - {e}")

        workflow.status = "completed" if not failed else "partial"

        return {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "results": {t.id: t.result for t in workflow.tasks.values() if t.result},
        }

    def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task.

        Args:
            task: Task to execute

        Returns:
            Task results
        """
        agent = self.agents.get(task.agent)
        if not agent:
            raise ValueError(f"Agent {task.agent} not found")

        task.status = "running"
        logger.info(f"Executing task: {task.name} with {task.agent}")

        # Get context from dependencies
        context = {}
        for dep_id in task.dependencies:
            dep_task = self._find_task_by_id(dep_id)
            if dep_task and dep_task.result:
                context.update(dep_task.result)

        # Run agent
        return agent.run(task.description, context)

    def _find_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find task across all workflows."""
        for workflow in self.workflows.values():
            if task_id in workflow.tasks:
                return workflow.tasks[task_id]
        return None

    def run_forecast_workflow(
        self, uf: str, start_date: str, end_date: str, model_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete forecasting workflow.

        Args:
            uf: State abbreviation
            start_date: Forecast start date
            end_date: Forecast end date
            model_types: List of model types to use

        Returns:
            Forecast results with ensemble prediction
        """
        model_types = model_types or ["xgboost", "lstm", "prophet"]

        # Define workflow tasks
        tasks = [
            Task(
                id="t1",
                name="data_collection",
                agent="data_engineer",
                description=f"Collect dengue data for {uf} from Mosqlimate API",
            ),
            Task(
                id="t2",
                name="feature_engineering",
                agent="data_engineer",
                description="Create lag features and climate variables",
                dependencies=["t1"],
            ),
            Task(
                id="t3",
                name="model_training",
                agent="model_architect",
                description=f"Train {', '.join(model_types)} models",
                dependencies=["t2"],
            ),
            Task(
                id="t4",
                name="forecast_generation",
                agent="forecaster",
                description=f"Generate forecasts from {start_date} to {end_date}",
                dependencies=["t3"],
            ),
            Task(
                id="t5",
                name="validation",
                agent="validator",
                description="Validate forecast quality and uncertainty",
                dependencies=["t4"],
            ),
            Task(
                id="t6",
                name="ensemble",
                agent="ensembler",
                description="Combine models into ensemble prediction",
                dependencies=["t4", "t5"],
            ),
            Task(
                id="t7",
                name="submission_format",
                agent="ensembler",
                description="Format output for Mosqlimate submission",
                dependencies=["t6"],
            ),
        ]

        workflow = self.create_workflow(f"forecast_{uf}", tasks)
        return self.run_workflow(workflow.id)

    def run_fine_tuning_workflow(
        self,
        data: Any,
        target_col: str = "casos",
        config: Optional[FineTuningConfig] = None,
        initial_hyperparams: Optional[Dict[str, Dict[str, Any]]] = None,
        warm_start_from_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run performance-based fine-tuning workflow.

        Args:
            data: DataFrame with features and target
            target_col: Target column name
            config: Fine-tuning configuration
            initial_hyperparams: Initial hyperparameters per model
            warm_start_from_state: State to use for warm-starting hyperparameters

        Returns:
            Dictionary with best hyperparameters and training history
        """
        config = config or FineTuningConfig()
        tracker = PerformanceTracker(
            target_metric=config.target_metric,
            patience=config.convergence_patience,
            min_improvement=config.min_improvement_rate,
        )

        current_hyperparams = initial_hyperparams or self._get_default_hyperparameters(
            config.model_types
        )

        logger.info(f"Starting fine-tuning workflow (max {config.max_rounds} rounds)")

        if warm_start_from_state and hasattr(self, "knowledge_base"):
            warm_params = self.knowledge_base.get_best_hyperparameters(
                warm_start_from_state, config.model_types
            )
            if warm_params:
                current_hyperparams = warm_params
                logger.info(f"Warm-starting from state {warm_start_from_state}")

        fine_tuning_history = []

        for round_num in range(1, config.max_rounds + 1):
            logger.info(f"Fine-tuning round {round_num}/{config.max_rounds}")

            round_metrics = {}
            round_hyperparams = {}

            for model_type in config.model_types:
                model_hyperparams = current_hyperparams.get(model_type, {})

                metrics, best_params = self._train_and_evaluate(
                    data=data,
                    model_type=model_type,
                    hyperparams=model_hyperparams,
                    target_col=target_col,
                    validation_size=config.validation_size,
                    n_cv_splits=config.n_cv_splits,
                )

                round_metrics[model_type] = metrics
                round_hyperparams[model_type] = best_params

            aggregated_metrics = self._aggregate_metrics(round_metrics, config.target_metric)

            should_continue = tracker.update(
                round_num=round_num,
                hyperparams=round_hyperparams,
                metrics=aggregated_metrics,
            )

            fine_tuning_history.append(
                {
                    "round": round_num,
                    "hyperparameters": round_hyperparams,
                    "metrics": aggregated_metrics,
                    "improvement": tracker.history[-1].improvement if tracker.history else 0,
                }
            )

            if not should_continue:
                logger.info(f"Convergence reached at round {round_num}")
                break

            current_hyperparams = self._suggest_next_hyperparameters(
                current_hyperparams, round_hyperparams, round_metrics, config
            )

        best_hyperparams = tracker.get_best_hyperparameters()
        convergence_summary = tracker.get_convergence_summary()

        return {
            "best_hyperparameters": best_hyperparams,
            "best_metrics": convergence_summary["best_metrics"],
            "total_rounds": convergence_summary["total_rounds"],
            "converged": convergence_summary["converged"],
            "fine_tuning_history": fine_tuning_history,
            "convergence_summary": convergence_summary,
        }

    def _train_and_evaluate(
        self,
        data: Any,
        model_type: str,
        hyperparams: Dict[str, Any],
        target_col: str,
        validation_size: float,
        n_cv_splits: int,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Train and evaluate a single model with given hyperparameters.

        Args:
            data: DataFrame with features and target
            model_type: Type of model to train
            hyperparams: Hyperparameters to use
            target_col: Target column name
            validation_size: Fraction for validation
            n_cv_splits: Number of CV splits

        Returns:
            Tuple of (metrics dict, best hyperparameters)
        """
        try:
            if model_type == "xgboost":
                from mosqlimate_ai.models.xgboost_model import XGBoostForecaster

                forecaster = XGBoostForecaster(target_col=target_col, **hyperparams)
                forecaster.fit(data, validation_size=validation_size, verbose=False)

                split_idx = int(len(data) * (1 - validation_size))
                train_data = data.iloc[:split_idx]
                val_data = data.iloc[split_idx:]

                predictions = forecaster.predict(val_data)
                y_true = val_data[target_col].values

                metrics = self._compute_metrics(y_true, predictions)

            elif model_type == "lstm":
                from mosqlimate_ai.models.lstm_model import LSTMForecaster

                forecaster = LSTMForecaster(target_col=target_col, **hyperparams)
                forecaster.fit(data, validation_size=validation_size, verbose=False)

                split_idx = int(len(data) * (1 - validation_size))
                val_data = data.iloc[split_idx:]

                predictions = forecaster.predict(val_data)
                valid_mask = ~predictions["median"].isna()
                y_true = val_data[target_col].values[valid_mask]

                metrics = self._compute_metrics(y_true, predictions[valid_mask])

            else:
                logger.warning(f"Unknown model type: {model_type}")
                metrics = {"crps": float("inf"), "rmse": float("inf"), "mae": float("inf")}

            return metrics, hyperparams

        except Exception as e:
            logger.error(f"Failed to train/evaluate {model_type}: {e}")
            return {"crps": float("inf"), "rmse": float("inf"), "mae": float("inf")}, hyperparams

    def _compute_metrics(self, y_true: np.ndarray, predictions: Any) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True values
            predictions: DataFrame with predictions

        Returns:
            Dictionary of metrics
        """
        y_true = np.asarray(y_true).ravel()

        median_pred = (
            predictions["median"].values if hasattr(predictions, "median") else predictions
        )

        metrics = {
            "rmse": float(rmse(y_true, median_pred)),
            "mae": float(mae(y_true, median_pred)),
        }

        try:
            metrics["crps"] = float(crps(y_true, predictions))
        except Exception:
            metrics["crps"] = float("inf")

        if "lower_95" in predictions.columns and "upper_95" in predictions.columns:
            lower = predictions["lower_95"].values
            upper = predictions["upper_95"].values
            coverage_95 = np.mean((y_true >= lower) & (y_true <= upper))
            metrics["coverage_95"] = float(coverage_95)

        if "lower_50" in predictions.columns and "upper_50" in predictions.columns:
            lower = predictions["lower_50"].values
            upper = predictions["upper_50"].values
            coverage_50 = np.mean((y_true >= lower) & (y_true <= upper))
            metrics["coverage_50"] = float(coverage_50)

        return metrics

    def _aggregate_metrics(
        self, round_metrics: Dict[str, Dict[str, float]], target_metric: str
    ) -> Dict[str, float]:
        """Aggregate metrics across models.

        Args:
            round_metrics: Metrics per model
            target_metric: Primary metric for optimization

        Returns:
            Aggregated metrics
        """
        all_target_scores = [
            metrics.get(target_metric, float("inf")) for metrics in round_metrics.values()
        ]

        aggregated = {
            target_metric: float(np.mean(all_target_scores)),
            "rmse_mean": float(
                np.mean([m.get("rmse", float("inf")) for m in round_metrics.values()])
            ),
            "mae_mean": float(
                np.mean([m.get("mae", float("inf")) for m in round_metrics.values()])
            ),
        }

        for model_name, metrics in round_metrics.items():
            for metric_name, value in metrics.items():
                aggregated[f"{model_name}_{metric_name}"] = value

        return aggregated

    def _suggest_next_hyperparameters(
        self,
        current_hyperparams: Dict[str, Dict[str, Any]],
        best_round_hyperparams: Dict[str, Dict[str, Any]],
        round_metrics: Dict[str, Dict[str, float]],
        config: FineTuningConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """Suggest hyperparameters for the next round.

        Uses gradient-based perturbation of best hyperparameters.

        Args:
            current_hyperparams: Current hyperparameters
            best_round_hyperparams: Best hyperparameters from this round
            round_metrics: Metrics from this round
            config: Fine-tuning configuration

        Returns:
            Suggested hyperparameters for next round
        """
        suggested = {}

        for model_type in config.model_types:
            best_params = best_round_hyperparams.get(model_type, {})
            metrics = round_metrics.get(model_type, {})

            if not best_params:
                suggested[model_type] = self._get_default_hyperparameters([model_type]).get(
                    model_type, {}
                )
                continue

            perturbed = best_params.copy()

            if model_type == "xgboost":
                if metrics.get("rmse", 0) > metrics.get("mae", 0) * 2:
                    perturbed["max_depth"] = min(10, best_params.get("max_depth", 6) + 1)
                    perturbed["learning_rate"] = best_params.get("learning_rate", 0.05) * 0.9

                if metrics.get("coverage_95", 1) < 0.85:
                    perturbed["subsample"] = max(0.6, best_params.get("subsample", 0.8) - 0.1)

            elif model_type == "lstm":
                if metrics.get("rmse", 0) > metrics.get("mae", 0) * 2:
                    perturbed["hidden_size"] = min(256, best_params.get("hidden_size", 128) * 2)
                    perturbed["num_layers"] = min(3, best_params.get("num_layers", 2) + 1)

            suggested[model_type] = perturbed

        return suggested

    def _get_default_hyperparameters(self, model_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameters for specified models.

        Args:
            model_types: List of model types

        Returns:
            Dictionary of default hyperparameters per model
        """
        defaults = {
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "early_stopping_rounds": 50,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "epochs": 100,
                "early_stopping_patience": 15,
            },
            "prophet": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
            },
            "tft": {
                "hidden_size": 64,
                "attention_head_size": 4,
                "dropout": 0.1,
                "learning_rate": 0.001,
            },
            "nbeats": {
                "hidden_size": 256,
                "learning_rate": 0.001,
            },
        }

        return {model_type: defaults.get(model_type, {}) for model_type in model_types}

    def run_workflow_with_fine_tuning(
        self,
        uf: str,
        start_date: str,
        end_date: str,
        model_types: Optional[List[str]] = None,
        fine_tuning_config: Optional[FineTuningConfig] = None,
        use_cross_state_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Run forecasting workflow with performance-based fine-tuning.

        This method runs the complete forecasting pipeline with iterative
        fine-tuning based on validation performance.

        Args:
            uf: State abbreviation
            start_date: Forecast start date
            end_date: Forecast end date
            model_types: List of model types to use
            fine_tuning_config: Fine-tuning configuration
            use_cross_state_knowledge: Whether to use knowledge from other states

        Returns:
            Dictionary with forecast results and fine-tuning history
        """
        model_types = model_types or ["xgboost", "lstm"]
        fine_tuning_config = fine_tuning_config or FineTuningConfig(model_types=model_types)

        data_task = Task(
            id="ft_t1",
            name="data_collection",
            agent="data_engineer",
            description=f"Collect dengue data for {uf} from Mosqlimate API",
        )

        data_workflow = self.create_workflow(f"ft_data_{uf}", [data_task])
        data_result = self.run_workflow(data_workflow.id)

        if data_result["status"] != "completed":
            return {
                "status": "error",
                "error": "Data collection failed",
                "details": data_result,
            }

        data_context = data_result.get("results", {}).get("ft_t1", {})

        if "output" in data_context:
            data = data_context["output"].get("data")
        else:
            data = data_context.get("data")

        if data is None:
            return {
                "status": "error",
                "error": "No data returned from data collection",
            }

        warm_start_state = None
        if use_cross_state_knowledge and hasattr(self, "knowledge_base"):
            similar_states = self.knowledge_base.find_similar_states(uf)
            if similar_states:
                warm_start_state = similar_states[0]
                logger.info(f"Using knowledge from similar state: {warm_start_state}")

        fine_tuning_result = self.run_fine_tuning_workflow(
            data=data,
            target_col="casos",
            config=fine_tuning_config,
            warm_start_from_state=warm_start_state,
        )

        best_hyperparams = fine_tuning_result["best_hyperparameters"]

        forecast_tasks = [
            Task(
                id="ft_t2",
                name="model_training_tuned",
                agent="model_architect",
                description=f"Train {', '.join(model_types)} models with tuned hyperparameters",
                dependencies=[],
            ),
            Task(
                id="ft_t3",
                name="forecast_generation",
                agent="forecaster",
                description=f"Generate forecasts from {start_date} to {end_date}",
                dependencies=["ft_t2"],
            ),
            Task(
                id="ft_t4",
                name="ensemble",
                agent="ensembler",
                description="Combine models into ensemble prediction",
                dependencies=["ft_t3"],
            ),
        ]

        forecast_workflow = self.create_workflow(f"ft_forecast_{uf}", forecast_tasks)
        forecast_result = self.run_workflow(forecast_workflow.id)

        return {
            "status": "success",
            "uf": uf,
            "forecast_result": forecast_result,
            "fine_tuning_result": fine_tuning_result,
            "best_hyperparameters": best_hyperparams,
            "model_types": model_types,
        }
