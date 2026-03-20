"""Prophet model for dengue forecasting with uncertainty intervals.

Implements Facebook Prophet for time series forecasting with
built-in uncertainty quantification via posterior predictive samples.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
QUANTILE_MAP = {
    0.025: "lower_95",
    0.05: "lower_90",
    0.10: "lower_80",
    0.25: "lower_50",
    0.50: "median",
    0.75: "upper_50",
    0.90: "upper_80",
    0.95: "upper_90",
    0.975: "upper_95",
}


class ProphetModel:
    """Prophet model for time-series forecasting with uncertainty.

    Uses Prophet's built-in uncertainty intervals from posterior samples.
    Supports custom seasonality and regressors for dengue forecasting.

    Args:
        yearly_seasonality: Yearly seasonality mode (True, False, or 'auto')
        weekly_seasonality: Weekly seasonality mode
        daily_seasonality: Daily seasonality mode (usually False for dengue)
        seasonality_mode: 'additive' or 'multiplicative'
        changepoint_prior_scale: Flexibility of trend changes
        seasonality_prior_scale: Strength of seasonality
        holidays_prior_scale: Strength of holiday effects
        mcmc_samples: Number of MCMC samples for uncertainty (0 for MAP)
        interval_width: Width of uncertainty intervals
        quantiles: Quantiles to predict
        extra_regressors: List of additional regressor column names

    Example:
        >>> model = ProphetModel(yearly_seasonality=True)
        >>> model.fit(df)
        >>> predictions = model.predict_quantiles(future_df)
    """

    def __init__(
        self,
        yearly_seasonality: Union[bool, str] = True,
        weekly_seasonality: Union[bool, str] = False,
        daily_seasonality: Union[bool, str] = False,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        quantiles: Optional[List[float]] = None,
        extra_regressors: Optional[List[str]] = None,
        n_forecast_samples: int = 1000,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.quantiles = quantiles or QUANTILES
        self.extra_regressors = extra_regressors or []
        self.n_forecast_samples = n_forecast_samples

        self.model_ = None
        self.is_fitted_ = False
        self.feature_names_: Optional[List[str]] = None

        self._check_prophet()

    def _check_prophet(self) -> None:
        """Check if Prophet is available."""
        try:
            from prophet import Prophet

            self._Prophet = Prophet
        except ImportError as e:
            raise ImportError("Prophet is required. Install with: pip install prophet") from e

    def _create_model(self) -> Any:
        """Create Prophet model instance."""
        model = self._Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
        )

        model.add_seasonality(
            name="quarterly",
            period=91.25,
            fourier_order=4,
        )

        return model

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "casos",
        verbose: bool = False,
    ) -> "ProphetModel":
        """Fit Prophet model.

        Args:
            df: DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column
            verbose: Whether to print training progress

        Returns:
            Fitted model
        """
        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(df[date_col]),
                "y": df[target_col].values,
            }
        )

        self.model_ = self._create_model()

        for regressor in self.extra_regressors:
            if regressor in df.columns:
                self.model_.add_regressor(regressor)
                prophet_df[regressor] = df[regressor].values
                if self.feature_names_ is None:
                    self.feature_names_ = []
                self.feature_names_.append(regressor)

        logger.info("Training Prophet model...")

        if verbose:
            self.model_.fit(prophet_df)
        else:
            import logging as _logging

            prophet_logger = _logging.getLogger("prophet")
            prophet_logger.setLevel(_logging.WARNING)
            self.model_.fit(prophet_df)

        self.is_fitted_ = True
        logger.info("Prophet training completed")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> np.ndarray:
        """Predict point estimates (median).

        Args:
            df: DataFrame with dates
            date_col: Name of date column

        Returns:
            Median predictions
        """
        self._check_fitted()

        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df[date_col])})

        for regressor in self.extra_regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor].values

        forecast = self.model_.predict(prophet_df)
        return forecast["yhat"].values

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict quantiles for prediction intervals.

        Uses Prophet's predictive samples to estimate quantiles.

        Args:
            df: DataFrame with dates
            date_col: Name of date column
            n_samples: Number of samples to draw (default: self.n_forecast_samples)

        Returns:
            DataFrame with quantile predictions
        """
        self._check_fitted()
        n_samples = n_samples or self.n_forecast_samples

        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df[date_col])})

        for regressor in self.extra_regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor].values

        forecast = self.model_.predict(prophet_df)

        if hasattr(self.model_, "predictive_samples"):
            samples = self.model_.predictive_samples(prophet_df)
            yhat_samples = samples["yhat"]
        else:
            yhat = forecast["yhat"].values
            lower = forecast["yhat_lower"].values
            upper = forecast["yhat_upper"].values
            std_estimate = (upper - lower) / (2 * 1.28)
            yhat_samples = np.random.normal(
                loc=yhat[:, np.newaxis],
                scale=std_estimate[:, np.newaxis],
                size=(len(yhat), n_samples),
            )

        result = pd.DataFrame()

        for q in self.quantiles:
            col_name = QUANTILE_MAP.get(q, f"q{q}")
            result[col_name] = np.quantile(yhat_samples, q, axis=1)

        return result

    def predict_with_intervals(
        self,
        df: pd.DataFrame,
        levels: Optional[List[float]] = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Predict with prediction intervals.

        Args:
            df: DataFrame with dates
            levels: Confidence levels (e.g., [0.50, 0.80, 0.95])
            date_col: Name of date column

        Returns:
            DataFrame with predictions and intervals
        """
        levels = levels or [0.50, 0.80, 0.95]
        quantile_pred = self.predict_quantiles(df, date_col)

        result = pd.DataFrame()
        result["median"] = quantile_pred["median"]

        for level in levels:
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            lower_col = QUANTILE_MAP.get(lower_q, f"q{lower_q}")
            upper_col = QUANTILE_MAP.get(upper_q, f"q{upper_q}")

            if lower_col in quantile_pred.columns and upper_col in quantile_pred.columns:
                result[f"lower_{int(level*100)}"] = quantile_pred[lower_col]
                result[f"upper_{int(level*100)}"] = quantile_pred[upper_col]

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import json
        from prophet.serialize import model_to_json

        with open(path / "model.json", "w") as f:
            json.dump(model_to_json(self.model_), f)

        config = {
            "quantiles": self.quantiles,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "extra_regressors": self.extra_regressors,
            "feature_names": self.feature_names_,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "ProphetModel":
        """Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        self.quantiles = config["quantiles"]
        self.yearly_seasonality = config["yearly_seasonality"]
        self.weekly_seasonality = config["weekly_seasonality"]
        self.daily_seasonality = config["daily_seasonality"]
        self.seasonality_mode = config["seasonality_mode"]
        self.changepoint_prior_scale = config["changepoint_prior_scale"]
        self.seasonality_prior_scale = config["seasonality_prior_scale"]
        self.extra_regressors = config.get("extra_regressors", [])
        self.feature_names_ = config.get("feature_names")

        from prophet.serialize import model_from_json

        with open(path / "model.json") as f:
            self.model_ = model_from_json(json.load(f))

        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")


class ProphetForecaster:
    """High-level forecaster using Prophet models.

    Args:
        target_col: Target column name
        date_col: Date column name
        extra_regressors: Additional regressor columns
        **model_kwargs: Arguments passed to ProphetModel
    """

    def __init__(
        self,
        target_col: str = "casos",
        date_col: str = "date",
        extra_regressors: Optional[List[str]] = None,
        **model_kwargs,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.extra_regressors = extra_regressors or []
        self.model = ProphetModel(extra_regressors=extra_regressors, **model_kwargs)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> "ProphetForecaster":
        """Fit model on DataFrame.

        Args:
            df: DataFrame with date and target
            verbose: Training verbosity

        Returns:
            Fitted forecaster
        """
        self.model.fit(
            df,
            date_col=self.date_col,
            target_col=self.target_col,
            verbose=verbose,
        )
        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with prediction intervals.

        Args:
            df: DataFrame with dates
            levels: Confidence levels for intervals

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        predictions = self.model.predict_with_intervals(
            df,
            levels=levels,
            date_col=self.date_col,
        )

        if self.date_col in df.columns:
            predictions[self.date_col] = df[self.date_col].values
        if "uf" in df.columns:
            predictions["uf"] = df["uf"].values

        return predictions

    def save(self, path: Union[str, Path]) -> None:
        """Save forecaster."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        import pickle

        with open(path / "forecaster_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "target_col": self.target_col,
                    "date_col": self.date_col,
                    "extra_regressors": self.extra_regressors,
                },
                f,
            )

    def load(self, path: Union[str, Path]) -> "ProphetForecaster":
        """Load forecaster."""
        import pickle

        path = Path(path)

        self.model.load(path)
        with open(path / "forecaster_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.target_col = config["target_col"]
        self.date_col = config["date_col"]
        self.extra_regressors = config["extra_regressors"]
        self.is_fitted = True

        return self
