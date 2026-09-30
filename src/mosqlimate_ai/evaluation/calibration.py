"""Conformal quantile recalibration for probabilistic forecasts.

Quantile-regression models are systematically overconfident for long
horizons (empirical coverage of the 50% interval was ~0.05 in early
backtests). This module implements per-quantile additive recalibration
(a conformal prediction variant, cf. Romano et al. 2019, "Conformalized
Quantile Regression", in the per-quantile form):

    1. Fit the model on data up to ``train_end - calib_weeks``.
    2. Forecast the following ``calib_weeks`` and collect residuals
       ``r_i(tau) = y_i - q_tau(x_i)`` at each quantile level.
    3. Offset ``c_tau = Quantile_tau(r_i)``; applied forecasts are
       ``q'_tau = q_tau + c_tau``.

Under residual exchangeability the corrected quantile has coverage
approximately ``tau``. Offsets are computed for the same horizon
structure as deployment (gap + season), and quantiles are re-sorted
after shifting to guarantee monotonicity.

The calibration fit costs one extra model fit per state/test — negligible
for the GBM zoo (~1s each) and affordable for the TFT.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS, sort_quantiles
from mosqlimate_ai.models.base import BaseForecaster

logger = logging.getLogger(__name__)


def conformal_quantile_offsets(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    calib_weeks: int = 67,
    min_calib_points: int = 8,
) -> dict[float, float]:
    """Per-quantile additive offsets from a held-out calibration window.

    Args:
        model: *Unfitted* forecaster (a fresh clone is fitted internally).
        train_df: Full training frame (up to the deploy cutoff).
        calib_weeks: Length of the held-out calibration window; should
            match the deployment horizon structure (16-week gap + 52
            target weeks = 67).
        min_calib_points: Minimum residuals required per quantile to
            compute an offset (fewer -> no offset).

    Returns:
        Mapping quantile level -> additive offset (log-count scale is
        *not* used; offsets are in count space, applied after expm1).
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    if len(train_df) <= calib_weeks + 10:
        logger.debug("not enough data for calibration window; skipping")
        return {}

    calib_end = train_df["date"].max()
    fit_end = calib_end - pd.Timedelta(weeks=calib_weeks)
    fit_df = train_df[train_df["date"] <= fit_end]
    if len(fit_df) < 52:
        return {}

    model.fit(fit_df)
    forecast = model.predict(calib_weeks)

    actual = train_df.set_index("date")[model.target_col].astype(float)
    actual = actual[(actual.index > fit_end) & (actual.index <= calib_end)]

    common = forecast.index.intersection(actual.index)
    if len(common) < min_calib_points:
        logger.debug("only %d calibration points; skipping", len(common))
        return {}

    y = actual.loc[common].values
    offsets = {}
    for tau, col in QUANTILE_COLS.items():
        if col not in forecast.columns:
            continue
        residuals = y - forecast.loc[common, col].values
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < min_calib_points:
            continue
        offsets[tau] = float(np.quantile(residuals, tau))
    return offsets


def apply_offsets(
    forecast: pd.DataFrame,
    offsets: dict[float, float],
) -> pd.DataFrame:
    """Apply per-quantile additive offsets and re-sort (crossing fix)."""
    out = forecast.copy()
    for tau, offset in offsets.items():
        col = QUANTILE_COLS.get(tau)
        if col in out.columns and np.isfinite(offset):
            out[col] = out[col] + offset
    return sort_quantiles(out.clip(lower=0.0))


def calibrate_forecast(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    horizon: int,
    calib_weeks: int = 67,
) -> pd.DataFrame:
    """Fit on the full window, predict, and conformally recalibrate.

    One-shot helper for backtests: assumes ``model`` is an unfitted clone
    that may be fitted twice (calibration + production).
    """
    offsets = conformal_quantile_offsets(model, train_df, calib_weeks=calib_weeks)
    model.fit(train_df)
    forecast = model.predict(horizon)
    if offsets:
        forecast = apply_offsets(forecast, offsets)
    return forecast
