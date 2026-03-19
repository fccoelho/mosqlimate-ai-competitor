"""Forecast Agent for Karl DBot."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.prompts import get_prompt
from mosqlimate_ai.models.lstm_model import LSTMForecaster
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)

FORECAST_QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
PREDICTION_LEVELS = [0.50, 0.80, 0.90, 0.95]


class ForecastAgent(BaseAgent):
    """Agent responsible for generating predictions.

    This agent:
    - Generates point forecasts using trained models
    - Calculates prediction intervals
    - Produces uncertainty estimates
    - Creates multi-horizon forecasts
    - Handles recursive forecasting for future dates
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.system_prompt = get_prompt("forecast")
        logger.info("ForecastAgent ready")

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate forecasts.

        Args:
            task: Task description
            context: Context with keys:
                - models: Dict of trained models by name
                - data: DataFrame with features for forecasting
                - forecast_start: Start date for forecast (default: last date + 1)
                - forecast_end: End date for forecast (optional)
                - n_weeks: Number of weeks to forecast (default: 52)
                - levels: Prediction interval levels (default: [0.50, 0.80, 0.90, 0.95])
                - uf: State abbreviation (for multi-state)

        Returns:
            Dictionary with forecasts and metadata
        """
        logger.info(f"ForecastAgent executing: {task}")
        context = context or {}

        try:
            models = context.get("models", {})
            data = context.get("data")
            _forecast_start = context.get("forecast_start")
            _forecast_end = context.get("forecast_end")
            _n_weeks = context.get("n_weeks", 52)
            _levels = context.get("levels", PREDICTION_LEVELS)
            _uf = context.get("uf")

            if not models:
                raise ValueError("No models provided in context")
            if data is None:
                raise ValueError("No data provided in context")

            if isinstance(data, dict):
                results = {}
                for state, state_df in data.items():
                    state_context = {**context, "data": state_df, "uf": state}
                    results[state] = self._generate_forecasts(state_context)

                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": {
                        "forecasts_by_state": results,
                        "n_states": len(results),
                    },
                }
            else:
                result = self._generate_forecasts(context)
                return {
                    "agent": self.config.name,
                    "task": task,
                    "status": "success",
                    "output": result,
                }

        except Exception as e:
            logger.error(f"ForecastAgent failed: {e}")
            return {
                "agent": self.config.name,
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _generate_forecasts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts from all models."""
        models = context["models"]
        data = context["data"]
        n_weeks = context.get("n_weeks", 52)
        levels = context.get("levels", PREDICTION_LEVELS)
        uf = context.get("uf")

        all_forecasts = {}

        for model_name, model in models.items():
            logger.info(f"Generating forecasts with {model_name}...")

            try:
                if isinstance(model, XGBoostForecaster):
                    forecast = self._forecast_xgboost(model, data, n_weeks, levels)
                elif isinstance(model, LSTMForecaster):
                    forecast = self._forecast_lstm(model, data, n_weeks, levels)
                else:
                    logger.warning(f"Unknown model type: {type(model)}")
                    continue

                if uf:
                    forecast["uf"] = uf

                all_forecasts[model_name] = forecast
                self.forecasts[model_name] = forecast

            except Exception as e:
                logger.error(f"Forecast failed for {model_name}: {e}")
                continue

        return {
            "forecasts": all_forecasts,
            "n_weeks": n_weeks,
            "levels": levels,
            "uf": uf,
        }

    def _forecast_xgboost(
        self,
        model: XGBoostForecaster,
        data: pd.DataFrame,
        n_weeks: int,
        levels: List[float],
    ) -> pd.DataFrame:
        """Generate XGBoost forecasts with prediction intervals."""
        forecast = model.predict(data, levels=levels)

        if "date" in data.columns:
            forecast["date"] = data["date"].values

        return forecast

    def _forecast_lstm(
        self,
        model: LSTMForecaster,
        data: pd.DataFrame,
        n_weeks: int,
        levels: List[float],
        n_mc_samples: int = 100,
    ) -> pd.DataFrame:
        """Generate LSTM forecasts with MC Dropout uncertainty."""
        forecast = model.predict(data, n_mc_samples=n_mc_samples)

        if "date" in data.columns:
            valid_idx = ~forecast["median"].isna()
            forecast.loc[valid_idx, "date"] = data.loc[valid_idx, "date"].values

        return forecast

    def generate_future_dates(
        self,
        last_date: Union[str, datetime],
        n_weeks: int,
    ) -> List[datetime]:
        """Generate future epidemiological weeks.

        Args:
            last_date: Last historical date
            n_weeks: Number of weeks to generate

        Returns:
            List of future dates
        """
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)

        dates = []
        current = last_date + timedelta(days=7)

        for _ in range(n_weeks):
            dates.append(current)
            current += timedelta(days=7)

        return dates

    def recursive_forecast(
        self,
        model: Any,
        data: pd.DataFrame,
        n_weeks: int,
        levels: List[float],
        target_col: str = "casos",
    ) -> pd.DataFrame:
        """Generate recursive multi-step forecasts.

        This method uses predicted values as lag features for
        subsequent predictions.

        Args:
            model: Trained forecaster
            data: Historical data with features
            n_weeks: Number of weeks to forecast
            levels: Prediction interval levels
            target_col: Target column name

        Returns:
            DataFrame with recursive forecasts
        """
        forecast_data = data.copy()
        forecasts = []

        for _week in range(n_weeks):
            pred = model.predict(forecast_data, levels=levels)

            next_pred = pred.iloc[[-1]].copy()
            forecasts.append(next_pred)

            new_row = forecast_data.iloc[[-1]].copy()
            for col in forecast_data.columns:
                if col.startswith(f"{target_col}_lag_"):
                    lag_num = int(col.split("_")[-1])
                    if lag_num == 1:
                        new_row[col] = pred["median"].iloc[-1]
                    elif f"{target_col}_lag_{lag_num-1}" in forecast_data.columns:
                        new_row[col] = forecast_data[f"{target_col}_lag_{lag_num-1}"].iloc[-1]

            forecast_data = pd.concat([forecast_data, new_row], ignore_index=True)

        return pd.concat(forecasts, ignore_index=True)

    def get_forecast(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get stored forecast by model name.

        Args:
            model_name: Model identifier

        Returns:
            Forecast DataFrame or None
        """
        return self.forecasts.get(model_name)

    def save_forecasts(
        self,
        output_path: Union[str, Path],
        model_name: Optional[str] = None,
    ) -> None:
        """Save forecasts to disk.

        Args:
            output_path: Path to save forecasts
            model_name: Specific model to save (default: all)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if model_name:
            if model_name in self.forecasts:
                self.forecasts[model_name].to_csv(
                    output_path / f"forecast_{model_name}.csv", index=False
                )
                logger.info(f"Saved {model_name} forecast to {output_path}")
        else:
            for name, forecast in self.forecasts.items():
                forecast.to_csv(output_path / f"forecast_{name}.csv", index=False)
            logger.info(f"Saved all forecasts to {output_path}")

    def load_forecasts(
        self,
        path: Union[str, Path],
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load forecasts from disk.

        Args:
            path: Path to saved forecasts
            model_names: Specific models to load (default: all found)

        Returns:
            Dictionary of loaded forecasts
        """
        path = Path(path)
        loaded = {}

        for forecast_file in path.glob("forecast_*.csv"):
            name = forecast_file.stem.replace("forecast_", "")
            if model_names is None or name in model_names:
                loaded[name] = pd.read_csv(forecast_file)
                self.forecasts[name] = loaded[name]

        logger.info(f"Loaded forecasts: {list(loaded.keys())}")
        return loaded

    def combine_forecasts(
        self,
        forecasts: Optional[Dict[str, pd.DataFrame]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Combine forecasts from multiple models.

        Simple weighted average combination. For more sophisticated
        ensemble methods, use EnsembleAgent.

        Args:
            forecasts: Dictionary of forecasts (default: stored forecasts)
            weights: Model weights (default: equal weights)

        Returns:
            Combined forecast DataFrame
        """
        forecasts = forecasts or self.forecasts

        if not forecasts:
            raise ValueError("No forecasts available to combine")

        if weights is None:
            weights = {name: 1.0 / len(forecasts) for name in forecasts}

        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        first_forecast = list(forecasts.values())[0]
        combined = pd.DataFrame()

        for col in [
            "median",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "lower_90",
            "upper_90",
            "lower_95",
            "upper_95",
        ]:
            if col in first_forecast.columns:
                weighted_sum = np.zeros(len(first_forecast))
                for name, forecast in forecasts.items():
                    if col in forecast.columns:
                        weighted_sum += forecast[col].values * weights.get(name, 0)
                combined[col] = weighted_sum

        for col in ["date", "uf"]:
            if col in first_forecast.columns:
                combined[col] = first_forecast[col].values

        return combined
