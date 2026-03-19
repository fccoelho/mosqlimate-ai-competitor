"""Data Engineer Agent for Karl DBot."""

import logging
from typing import Any, Dict, Optional

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.prompts import get_prompt
from mosqlimate_ai.data.features import FeatureEngineer
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class DataEngineerAgent(BaseAgent):
    """Agent responsible for data collection and preprocessing.

    This agent handles:
    - Data ingestion from cached competition data
    - Data cleaning and validation
    - Feature engineering (lag features, rolling averages)
    - Climate data integration

    Example:
        >>> config = AgentConfig(name="data_engineer", description="Data pipeline")
        >>> agent = DataEngineerAgent(config)
        >>> result = agent.run("Prepare SP data for modeling")
    """

    def __init__(self, config: AgentConfig):
        """Initialize data engineer agent."""
        super().__init__(config)
        self.loader = CompetitionDataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.system_prompt = get_prompt("data_engineer")

        self.register_tool("load_state_data", self.loader.load_state_data)
        self.register_tool("load_all_states", self.loader.load_all_states)
        self.register_tool("load_ocean_data", self.loader.load_ocean_data)
        logger.info("DataEngineerAgent ready")

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute data engineering task.

        Args:
            task: Task description
            context: Additional context with keys:
                - uf: State abbreviation (optional, default: all states)
                - start_date: Start date filter (optional)
                - end_date: End date filter (optional)
                - aggregate: Whether to aggregate to state level (default: True)
                - build_features: Whether to build feature set (default: True)

        Returns:
            Dictionary with processed dataset and metadata
        """
        logger.info(f"DataEngineerAgent executing: {task}")
        context = context or {}

        try:
            uf = context.get("uf")
            start_date = context.get("start_date")
            end_date = context.get("end_date")
            aggregate = context.get("aggregate", True)
            build_features = context.get("build_features", True)

            if uf:
                df = self.loader.load_state_data(
                    uf=uf, start_date=start_date, end_date=end_date, aggregate=aggregate
                )
                states_data = {uf: df}
            else:
                states_data = self.loader.load_all_states(
                    start_date=start_date, end_date=end_date, aggregate=aggregate
                )

            ocean_df = self.loader.load_ocean_data()

            processed_data = {}
            feature_names = None

            for state, state_df in states_data.items():
                state_df = self.preprocessor.clean(state_df)
                state_df = self.preprocessor.impute_missing(state_df)
                state_df = self.preprocessor.add_epidemiological_features(state_df)

                if build_features:
                    state_df = self.feature_engineer.build_feature_set(
                        state_df,
                        target_col="casos",
                        ocean_df=ocean_df if not ocean_df.empty else None,
                    )
                    if feature_names is None:
                        feature_names = self.feature_engineer.select_features(state_df, "casos")

                processed_data[state] = state_df

            result_df = processed_data[uf] if uf else processed_data

            result = {
                "agent": self.config.name,
                "task": task,
                "status": "success",
                "output": {
                    "data": result_df,
                    "states_data": processed_data if not uf else None,
                    "feature_names": feature_names,
                    "target": "casos",
                    "n_states": len(processed_data),
                    "total_rows": sum(len(df) for df in processed_data.values()),
                    "date_range": {
                        "min": self.loader.get_date_range()["min_date"],
                        "max": self.loader.get_date_range()["max_date"],
                    },
                },
            }

            self.add_to_memory("last_result", result)
            return result

        except Exception as e:
            logger.error(f"DataEngineerAgent failed: {e}")
            return {
                "agent": self.config.name,
                "task": task,
                "status": "error",
                "error": str(e),
            }
