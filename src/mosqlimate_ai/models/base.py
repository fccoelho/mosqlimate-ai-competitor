"""Unified forecasting model interface.

Every forecasting model in the pipeline implements the same contract:

    model.fit(train_df)                 # train_df: weekly rows, 'date' + target column
    forecast = model.predict(horizon)   # quantile DataFrame indexed by target dates

``predict(horizon)`` returns a DataFrame with the nine canonical quantile
columns (``q025``..``q975``, see
:mod:`mosqlimate_ai.evaluation.quantiles`) indexed by the weekly target
dates ``last_train_date + 7*h`` for ``h = 1..horizon``.

Competition context: the IMDC target window (EW41..EW40+) starts 16 weeks
after the training cutoff (EW25), so the production forecast requires
``horizon=67`` and evaluation slices horizons 16..67. All models must
therefore be genuine multi-horizon forecasters, not one-step predictors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mosqlimate_ai.evaluation.quantiles import (
    QUANTILE_LEVELS,
    sort_quantiles,
)

logger = logging.getLogger(__name__)

WEEK_OFFSET = pd.Timedelta(days=7)


def make_forecast_dates(last_train_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Weekly target dates for horizons 1..horizon after the training cutoff."""
    last_train_date = pd.Timestamp(last_train_date)
    return pd.DatetimeIndex([last_train_date + WEEK_OFFSET * h for h in range(1, horizon + 1)])


class BaseForecaster(ABC):
    """Abstract base class for probabilistic weekly forecasters.

    Subclasses must implement :meth:`_fit` and :meth:`_predict`. The
    public :meth:`fit` / :meth:`predict` wrap them with date bookkeeping,
    quantile-column validation, and quantile-crossing correction.
    """

    def __init__(self, quantiles: list | None = None):
        self.quantiles = sorted(quantiles) if quantiles else list(QUANTILE_LEVELS)
        self.target_col: str = "casos"
        self.last_train_date_: pd.Timestamp | None = None
        self.n_train_weeks_: int = 0
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "casos") -> BaseForecaster:
        """Fit the model on a weekly training frame.

        Args:
            df: DataFrame with a ``date`` column (weekly) and the target.
                Exogenous feature columns may be present; models that use
                them document which ones.
            target_col: Name of the target column (default ``casos``).

        Returns:
            self
        """
        if "date" not in df.columns:
            raise ValueError("training frame must contain a 'date' column")
        if target_col not in df.columns:
            raise ValueError(f"training frame must contain target column '{target_col}'")

        self.target_col = target_col
        df = df.sort_values("date").reset_index(drop=True)
        self.last_train_date_ = pd.Timestamp(df["date"].max())
        self.n_train_weeks_ = len(df)
        self._fit(df)
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Probabilistic forecast for horizons 1..horizon.

        Args:
            horizon: Number of weekly steps ahead (including any gap
                between the training cutoff and the target season).

        Returns:
            DataFrame indexed by weekly target dates with quantile
            columns ``q025``..``q975``; crossing-free.
        """
        if not self.is_fitted_:
            raise RuntimeError("model is not fitted; call fit() first")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        qf = self._predict(horizon)
        qf = qf.sort_index()
        expected = make_forecast_dates(self.last_train_date_, horizon)
        qf.index = expected[: len(qf)]
        qf = sort_quantiles(qf)
        return qf

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def predict_intervals(self, horizon: int) -> pd.DataFrame:
        """Forecast as submission-style median/interval columns."""
        from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals

        return quantiles_to_intervals(self.predict(horizon).reset_index(names="date"))

    @property
    def feature_importances_(self) -> pd.Series | None:
        """Per-feature importance when the underlying model exposes it."""
        return getattr(self, "_feature_importances", None)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the fitted model (subclass hook)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    @classmethod
    def load(cls, path: str | Path) -> BaseForecaster:
        """Restore a fitted model (subclass hook)."""
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> pd.DataFrame: ...


def slice_training_window(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Rows strictly up to and including ``train_end`` (no future data)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(train_end)].sort_values("date").reset_index(drop=True)


def target_window_slice(
    forecast: pd.DataFrame,
    target_start: str,
    target_end: str,
) -> pd.DataFrame:
    """Restrict a forecast to the competition target window."""
    idx = pd.to_datetime(forecast.index)
    mask = (idx >= pd.Timestamp(target_start)) & (idx <= pd.Timestamp(target_end))
    return forecast[mask]
