"""Baseline forecasters.

Every competitive model must beat these cheap reference forecasts on the
validation seasons; they also serve as ensemble members and as the
"skill floor" in reports.

- ``SeasonalNaiveForecaster``: repeats the value observed 52 weeks before
  each target date (the canonical seasonal baseline for weekly
  epidemiological series).
- ``FlatForecaster``: flat median of the last ``window`` training weeks
  with quantiles from the empirical residual distribution.
- ``LogLinearTrendForecaster``: log-linear trend + yearly harmonics fit
  by least squares (a smoothed epidemic-season reference).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS
from mosqlimate_ai.models.base import WEEK_OFFSET, BaseForecaster, make_forecast_dates

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the observation from 52 weeks before each target date.

    Quantiles come from the empirical distribution of *additive*
    seasonal errors (``value_t - value_{t-52}``) observed in training;
    additive errors remain well-defined for zero-inflated series
    (e.g. chikungunya before its introduction), where multiplicative
    ratios degenerate to 0/0.
    """

    def __init__(self, season_lag: int = 52):
        super().__init__()
        self.season_lag = season_lag

    def _fit(self, df: pd.DataFrame) -> None:
        s = df.set_index("date")[self.target_col].astype(float)
        self.history_ = s
        errors = s - s.shift(self.season_lag)
        self.error_quantiles_ = errors.dropna().quantile(list(QUANTILE_COLS.keys()))
        if self.error_quantiles_.isna().any():
            # degenerate series (shorter than the seasonal lag)
            self.error_quantiles_ = self.error_quantiles_.fillna(0.0)

    def _predict(self, horizon: int) -> pd.DataFrame:
        dates = make_forecast_dates(self.last_train_date_, horizon)
        base = []
        for d in dates:
            seasonal_date = d - pd.Timedelta(weeks=self.season_lag)
            v = np.nan
            # walk back from the seasonal date to the most recent
            # non-missing observation (series may contain NaN weeks)
            lookup = seasonal_date
            while lookup >= self.history_.index.min():
                if lookup in self.history_.index and pd.notna(self.history_.loc[lookup]):
                    v = float(self.history_.loc[lookup])
                    break
                lookup -= WEEK_OFFSET
            if pd.isna(v):
                valid = self.history_.last_valid_index()
                v = float(self.history_.loc[valid]) if valid is not None else 0.0
            base.append(max(v, 0.0))
        base = np.array(base, dtype=float)

        qf = pd.DataFrame(index=dates)
        for tau, col in QUANTILE_COLS.items():
            if tau == 0.5:
                qf[col] = base
            else:
                qf[col] = np.maximum(base + float(self.error_quantiles_.loc[tau]), 0.0)
        return qf.clip(lower=0.0)


class FlatForecaster(BaseForecaster):
    """Flat forecast at the recent median with empirical residual quantiles."""

    def __init__(self, window: int = 8):
        super().__init__()
        self.window = window

    def _fit(self, df: pd.DataFrame) -> None:
        s = df.set_index("date")[self.target_col].astype(float)
        self.history_ = s
        recent = s.iloc[-self.window :]
        self.level_ = float(recent.median())
        residuals = recent - recent.mean()
        self.residual_quantiles_ = residuals.quantile(list(QUANTILE_COLS.keys()))

    def _predict(self, horizon: int) -> pd.DataFrame:
        dates = make_forecast_dates(self.last_train_date_, horizon)
        qf = pd.DataFrame(index=dates)
        for tau, col in QUANTILE_COLS.items():
            qf[col] = max(self.level_ + float(self.residual_quantiles_.loc[tau]), 0.0)
        qf[QUANTILE_COLS[0.5]] = max(self.level_, 0.0)
        return qf.clip(lower=0.0)


class LogLinearTrendForecaster(BaseForecaster):
    """Log-linear trend with yearly and half-yearly harmonics.

    Fitted by ridge-regularized least squares on log1p(cases); quantiles
    from Gaussian residual quantiles, transformed back to counts.
    """

    def __init__(self, n_harmonics: int = 2):
        super().__init__()
        self.n_harmonics = n_harmonics

    @staticmethod
    def _design(t: np.ndarray, n_harmonics: int) -> np.ndarray:
        cols = [np.ones_like(t), t / 52.0]
        for k in range(1, n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * t / 52.0))
            cols.append(np.cos(2 * np.pi * k * t / 52.0))
        return np.column_stack(cols)

    def _fit(self, df: pd.DataFrame) -> None:
        s = df.set_index("date")[self.target_col].astype(float)
        t = np.arange(len(s), dtype=float)
        y = np.log1p(np.maximum(s.values, 0.0))
        X = self._design(t, self.n_harmonics)
        # exclude missing weeks from the least-squares fit (NaN targets
        # would poison the normal equations); the time axis stays
        # calendar-aligned so remaining weeks keep their positions
        mask = np.isfinite(y)
        if mask.sum() < 10:
            raise ValueError("too few observations to fit the log-linear trend")
        X, y_t = X[mask], y[mask]
        lam = 1.0
        A = X.T @ X + lam * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, X.T @ y_t)
        residuals = y_t - X @ self.coef_
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.5
        self.sigma_ = max(sigma, 1e-3)
        self.t_last_ = t[-1]
        self.z_quantiles_ = {tau: float(_norm_ppf(tau)) for tau in QUANTILE_COLS}

    def _predict(self, horizon: int) -> pd.DataFrame:
        from scipy import stats as _stats

        dates = make_forecast_dates(self.last_train_date_, horizon)
        t = self.t_last_ + np.arange(1, horizon + 1, dtype=float)
        X = self._design(t, self.n_harmonics)
        mu = X @ self.coef_
        qf = pd.DataFrame(index=dates)
        for tau, col in QUANTILE_COLS.items():
            z = _stats.norm.ppf(tau)
            qf[col] = np.expm1(mu + z * self.sigma_)
        return qf.clip(lower=0.0)


def _norm_ppf(tau: float) -> float:
    from scipy import stats as _stats

    return float(_stats.norm.ppf(tau))
