"""Direct multi-horizon quantile regression with gradient boosting.

The workhorse model of the pipeline. One gradient-boosted tree ensemble
per quantile level is trained on (forecast origin, horizon) pairs:

- **Origin features** (known at forecast time): target lags
  (1..8, 12, 26, 52), rolling means/std (4, 8, 12 weeks), log-gap since
  last observation, plus origin-time exogenous values.
- **Target-time features** (known or projected for the target week):
  calendar/seasonality harmonics of the target date, forecast horizon,
  and future exogenous covariates (ECMWF climate forecasts, ocean
  indices) when supplied via ``future_exog``.

Modeling is done on ``log1p(cases)`` so quantile errors are
multiplicative; predictions are mapped back with ``expm1``.

This design is leak-free by construction: lag/rolling features are
computed *inside* the model from the training window only, and future
covariates must be supplied explicitly (no silent reuse of history).
"""

from __future__ import annotations

from typing import Dict, List, Optional  # noqa: F401 - used in annotations

import logging

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS, QUANTILE_LEVELS
from mosqlimate_ai.models.base import BaseForecaster, make_forecast_dates

logger = logging.getLogger(__name__)

DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 26, 52)
DEFAULT_ROLLING = (4, 8, 12)

# Columns never used as features, whatever the input frame contains.
_NON_FEATURE_COLS = {
    "date",
    "uf",
    "geocode",
    "epiweek",
    "macroregional_geocode",
    "regional_geocode",
    "year",
}


def _target_derived(col: str, target_col: str) -> bool:
    prefixes = ("casos_", "target_", "train_")
    return col.startswith(prefixes) or col == target_col


def _calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Seasonality features from target dates (always known in advance)."""
    doy = dates.dayofyear
    week = dates.isocalendar().week.astype(float).values
    out = pd.DataFrame(index=dates)
    out["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin_half"] = np.sin(4 * np.pi * doy / 365.25)
    out["cos_half"] = np.cos(4 * np.pi * doy / 365.25)
    out["epiweek_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["epiweek_cos"] = np.cos(2 * np.pi * week / 52.0)
    # Southern-hemisphere dengue season flag (Nov-Apr)
    out["is_summer"] = ((dates.month >= 11) | (dates.month <= 4)).astype(float)
    return out


class DirectMultiHorizonGBM(BaseForecaster):
    """Direct multi-horizon quantile GBM (XGBoost or LightGBM).

    Args:
        model_type: ``"xgboost"`` or ``"lightgbm"``.
        quantiles: Quantile levels to predict (default: the nine IMDC
            levels).
        max_horizon: Largest horizon to train on. Must cover the 16-week
            gap + 52 target weeks for the competition (default 67).
        lags: Target lag features (in weeks).
        rolling: Rolling mean/std window sizes.
        exog_cols: Exogenous columns to use from the training frame
            (origin-time reanalysis values). ``None`` = auto-detect all
            numeric non-target columns.
        exog_lookup: Optional :class:`~mosqlimate_ai.data.future_exog.ExogLookup`
            supplying *target-time* known covariates (ECMWF climate
            forecasts by issue/lead, ocean indices) as a function of
            (origin_date, target_date). Used identically at fit and
            predict time, keeping train/predict distributions aligned.
        recency_halflife_weeks: Sample-weight half-life for training
            origins (``None`` disables weighting).
        params: Booster hyperparameters (tuned upstream).
        quantile_jobs: Number of parallel workers for per-quantile model
            fits (0 = fit serially with all cores per booster).
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        quantiles: Optional[List[float]] = None,
        max_horizon: int = 80,
        lags: List[int] = None,
        rolling: List[int] = None,
        exog_cols: Optional[List[str]] = None,
        exog_lookup=None,
        recency_halflife_weeks: Optional[int] = 104,
        params: Optional[Dict] = None,
        quantile_jobs: int = 4,
    ):
        super().__init__(quantiles=quantiles or list(QUANTILE_LEVELS))
        if model_type not in ("xgboost", "lightgbm"):
            raise ValueError(f"unsupported model_type: {model_type}")
        self.model_type = model_type
        self.max_horizon = max_horizon
        self.lags = list(lags) if lags else list(DEFAULT_LAGS)
        self.rolling = list(rolling) if rolling else list(DEFAULT_ROLLING)
        self.exog_cols = exog_cols
        self.exog_lookup = exog_lookup
        self.recency_halflife_weeks = recency_halflife_weeks
        self.params = params or {}
        self.quantile_jobs = quantile_jobs

    # ------------------------------------------------------------------
    # Feature construction (train-window only; leak-free)
    # ------------------------------------------------------------------
    def _origin_features(self, history: pd.Series, t_idx: int) -> dict[str, float]:
        """Features computed from data available at forecast origin t_idx."""
        vals = history.values[: t_idx + 1]
        log_vals = np.log1p(np.maximum(vals, 0.0))
        feats: dict[str, float] = {}
        for lag in self.lags:
            feats[f"lag_{lag}"] = log_vals[-lag] if len(log_vals) >= lag else np.nan
        for w in self.rolling:
            feats[f"roll_mean_{w}"] = log_vals[-w:].mean() if len(log_vals) >= 1 else np.nan
            feats[f"roll_std_{w}"] = log_vals[-w:].std() if len(log_vals) >= 2 else np.nan
        feats["last_obs_age"] = 0.0 if vals[-1] > 0 else 1.0
        feats["n_zero_last4"] = float((vals[-4:] == 0).sum())
        return feats

    def _resolve_exog_cols(self, df: pd.DataFrame) -> list[str]:
        if self.exog_cols is not None:
            return [c for c in self.exog_cols if c in df.columns]
        return [
            c
            for c in df.columns
            if c not in _NON_FEATURE_COLS and not _target_derived(c, self.target_col)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _exog_at(self, df: pd.DataFrame, exog_cols: list[str], date: pd.Timestamp) -> dict[str, float]:
        """Exogenous values at a given date; NaN when the date is absent."""
        if not exog_cols:
            return {}
        sub = df.loc[df["date"] == date]
        if sub.empty:
            return {c: np.nan for c in exog_cols}
        row = sub.iloc[-1]
        return {c: float(row[c]) if pd.notna(row[c]) else np.nan for c in exog_cols}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _fit(self, df: pd.DataFrame) -> None:
        df = df.sort_values("date").reset_index(drop=True)
        history = pd.Series(df[self.target_col].astype(float).values, index=df["date"])
        exog_cols = self._resolve_exog_cols(df)
        self.exog_cols_ = exog_cols
        self._last_exog_ = {
            c: (float(df[c].iloc[-1]) if pd.notna(df[c].iloc[-1]) else np.nan)
            for c in exog_cols
        }

        min_lag = max(max(self.lags), max(self.rolling))
        n = len(df)
        if n <= min_lag + 2:
            raise ValueError(f"not enough training rows ({n}) for lags {self.lags}")

        origins = range(min_lag, n)
        rows: List[Dict[str, float]] = []
        targets: List[float] = []
        horizons: List[int] = []
        origin_times: List[int] = []

        y_all = history.values
        for t in origins:
            origin_feats = self._origin_features(history, t)
            origin_exog = self._exog_at(df, exog_cols, df["date"].iloc[t])
            t_date = df["date"].iloc[t]
            for h in range(1, self.max_horizon + 1):
                target_pos = t + h
                if target_pos >= n:
                    break
                y_t = y_all[target_pos]
                if np.isnan(y_t):
                    # missing week (incomplete data): excluded from
                    # training rather than silently treated as zero
                    continue
                target_date = df["date"].iloc[target_pos]
                target_exog = (
                    self.exog_lookup.get(t_date, target_date) if self.exog_lookup else {}
                )
                rows.append({**origin_feats, **origin_exog, **target_exog})
                targets.append(float(np.log1p(max(y_t, 0.0))))
                horizons.append(h)
                origin_times.append(t_date)

        X = pd.DataFrame(rows)
        X["horizon"] = np.array(horizons, dtype=float)
        cal = _calendar_features(
            pd.DatetimeIndex(origin_times)
            + pd.to_timedelta(7 * np.array(horizons), unit="D")
        )
        cal.index = X.index
        X = pd.concat([X, cal], axis=1)
        y = np.array(targets)

        self.feature_names_ = list(X.columns)

        sample_weight = None
        if self.recency_halflife_weeks:
            ages = np.array([(df["date"].max() - t).days / 7.0 for t in origin_times])
            sample_weight = 0.5 ** (ages / self.recency_halflife_weeks)

        self.models_ = {}
        if self.quantile_jobs and self.quantile_jobs > 1:
            import os

            from joblib import Parallel, delayed

            # Respect the caller's thread budget to avoid oversubscription
            # when several workers run in parallel (booster_n_jobs wins if
            # explicitly provided).
            n_par = min(self.quantile_jobs, len(self.quantiles))
            budget = int(self.params.get("thread_budget") or 0) or (os.cpu_count() or 8)
            per_model_jobs = int(self.params.get("booster_n_jobs") or 0) or max(
                1, budget // n_par
            )
            fitted = Parallel(n_jobs=n_par)(
                delayed(self._fit_booster)(X, y, tau, sample_weight, per_model_jobs)
                for tau in self.quantiles
            )
            self.models_ = dict(zip(self.quantiles, fitted))
        else:
            booster_n_jobs = int(self.params.get("booster_n_jobs") or 0) or -1
            for tau in self.quantiles:
                self.models_[tau] = self._fit_booster(X, y, tau, sample_weight, booster_n_jobs)

        importances = {}
        for tau, model in self.models_.items():
            for f, v in self._importances(model).items():
                importances[f] = importances.get(f, 0.0) + v / len(self.models_)
        self._feature_importances = pd.Series(importances).sort_values(ascending=False)

    def _fit_booster(self, X: pd.DataFrame, y: np.ndarray, tau: float, sample_weight, n_jobs: int = -1):
        common = {
            "n_estimators": int(self.params.get("n_estimators", 500)),
            "max_depth": int(self.params.get("max_depth", 5)),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "subsample": float(self.params.get("subsample", 0.8)),
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.8)),
            "min_child_weight": float(self.params.get("min_child_weight", 5.0)),
            "reg_alpha": float(self.params.get("reg_alpha", 0.1)),
            "reg_lambda": float(self.params.get("reg_lambda", 1.0)),
            "random_state": int(self.params.get("random_state", 42)),
            "n_jobs": n_jobs,
        }
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor

            model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=tau,
                tree_method=self.params.get("tree_method", "hist"),
                **common,
            )
        else:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="quantile",
                alpha=tau,
                verbosity=-1,
                **common,
            )
        model.fit(X, y, sample_weight=sample_weight)
        return model

    def _importances(self, model) -> dict[str, float]:
        try:
            return dict(zip(self.feature_names_, model.feature_importances_))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict(self, horizon: int) -> pd.DataFrame:
        history = self._history_
        dates = make_forecast_dates(self.last_train_date_, horizon)
        n = len(history)

        # Origin features from the most recent observed state
        origin_feats = self._origin_features(history, n - 1)
        origin_exog = self._last_exog_
        horizon_arr = np.arange(1, horizon + 1, dtype=float)

        rows = []
        for _h, d in zip(horizon_arr, dates):
            row = {**origin_feats, **origin_exog}
            if self.exog_lookup is not None:
                row.update(self.exog_lookup.get(self.last_train_date_, d))
            rows.append(row)
        X = pd.DataFrame(rows, index=dates)
        X["horizon"] = horizon_arr
        X = pd.concat([X, _calendar_features(dates)], axis=1)
        X = X[self.feature_names_]

        qf = pd.DataFrame(index=dates)
        for tau in self.quantiles:
            pred_log = self.models_[tau].predict(X)
            qf[QUANTILE_COLS[tau]] = np.expm1(pred_log)
        return qf.clip(lower=0.0)

    # fit() stores history for prediction-time feature building
    def fit(self, df: pd.DataFrame, target_col: str = "casos", **kwargs):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        self._history_ = (
            pd.Series(df[target_col].astype(float).values, index=df["date"]).sort_index()
        )
        return super().fit(df, target_col=target_col)


class XGBoostDirectForecaster(DirectMultiHorizonGBM):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_type", "xgboost")
        super().__init__(**kwargs)


class LightGBMDirectForecaster(DirectMultiHorizonGBM):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_type", "lightgbm")
        super().__init__(**kwargs)
