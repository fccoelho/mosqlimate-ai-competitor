"""Covariate lag estimation and stepwise inclusion for the model zoo.

Climate covariates affect arbovirus cases through vector dynamics with
a multi-week delay: the useful predictor for cases at week ``T`` is the
covariate at ``T - k`` for some optimal lag ``k``, not the contemporary
value. This module estimates those lags from training data only
(leak-safe) and gates covariates through a forward (stepwise) inclusion
protocol before they reach the forecasters.

Components:

- :func:`estimate_covariate_lags` — per covariate, the lag in
  ``0..max_lag`` maximizing the absolute Spearman correlation between
  the *seasonally adjusted* covariate and *seasonally adjusted*
  ``log1p(cases)`` (annual harmonics + linear trend removed from both,
  so shared seasonality cannot dominate the lag choice).

- :func:`stepwise_select` — forward inclusion under rolling-origin
  cross-validation: starting from purely autoregressive base features,
  a covariate (at its optimal lag) is kept only when it reduces the
  mean out-of-sample MAE on ``log1p(cases)`` by at least ``min_gain``
  (relative). Ridge regression is the cheap evaluation proxy; the real
  forecasters still learn from the selected features themselves.

- :class:`LaggedExogLookup` — wraps any lookup so ``get(origin, target)``
  returns the inner lookup's value for ``target - lag`` per feature
  (same as-of origin). Exposes only the selected features via
  ``features_``; with none selected the wrapper reports no features and
  models fall back to their covariate-free behavior.

- :func:`optimize_exog_lookup` — one-call entry point used by the
  backtest workflow: build the covariate history from the lookup,
  estimate lags, run stepwise selection, and return the wrapped lookup
  (or None when nothing is useful) plus a reportable info dict.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MAX_LAG_WEEKS = 16
MIN_CORR_POINTS = 40
CV_FOLDS = 12
MIN_GAIN = 0.02  # relative MAE improvement needed to keep a covariate
AR_LAGS = (1, 2, 3, 4, 8, 12, 26, 52)


def _deseasonalize(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Remove annual harmonics (1st + 2nd) and a linear trend."""
    values = np.asarray(values, dtype=float)
    t = np.arange(len(values), dtype=float)
    woy = pd.DatetimeIndex(dates).isocalendar().week.to_numpy(dtype=float)
    design = np.column_stack(
        [
            np.ones_like(t),
            t,
            np.sin(2 * np.pi * woy / 52.0),
            np.cos(2 * np.pi * woy / 52.0),
            np.sin(4 * np.pi * woy / 52.0),
            np.cos(4 * np.pi * woy / 52.0),
        ]
    )
    mask = np.isfinite(values)
    if mask.sum() < design.shape[1] + 10:
        return values - np.nanmean(values)
    beta, *_ = np.linalg.lstsq(design[mask], values[mask], rcond=None)
    out = values - design @ beta
    return out


def covariate_history(lookup, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Best-known weekly values of every lookup feature over ``dates``."""
    feats = list(getattr(lookup, "features_", ()))
    if not feats:
        return pd.DataFrame(index=pd.DatetimeIndex(dates))
    rows = [lookup.get(d, d) for d in pd.DatetimeIndex(dates)]
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=feats)
    return out.loc[:, out.notna().any(axis=0)]


def estimate_covariate_lags(
    cases: pd.Series,
    cov_history: pd.DataFrame,
    max_lag: int = MAX_LAG_WEEKS,
    min_corr_points: int = MIN_CORR_POINTS,
) -> Dict[str, Dict[str, float]]:
    """Optimal lag per covariate via |Spearman corr| on adjusted series.

    Returns ``{feature: {"lag": k, "corr": r}}`` for covariates with
    enough overlapping weeks; ties resolve to the smaller lag.
    """
    y = _deseasonalize(np.log1p(cases.to_numpy(dtype=float)), cases.index)
    out: Dict[str, Dict[str, float]] = {}
    for feat in cov_history.columns:
        x = _deseasonalize(cov_history[feat].to_numpy(dtype=float), cov_history.index)
        best: Optional[tuple[int, float]] = None
        for k in range(max_lag + 1):
            shifted = np.full_like(x, np.nan)
            shifted[k:] = x[: len(x) - k] if k else x
            m = np.isfinite(shifted) & np.isfinite(y)
            if m.sum() < min_corr_points:
                continue
            r = float(spearmanr(shifted[m], y[m]).statistic)
            if not np.isfinite(r):
                continue
            if best is None or abs(r) > abs(best[1]) + 1e-12:
                best = (k, r)
        if best is not None:
            out[feat] = {"lag": float(best[0]), "corr": best[1]}
    return out


class LaggedExogLookup:
    """Lookup wrapper that shifts each feature by its estimated lag.

    ``get(origin, target)`` returns, per kept feature, the inner
    lookup's value for ``target - lag`` weeks (as known at ``origin``),
    preserving the as-of (leak-safe) semantics of the inner lookup.
    """

    def __init__(self, inner, lags: Dict[str, int]):
        order = [f for f in getattr(inner, "features_", ()) if f in lags]
        self.inner = inner
        self.lags = {f: int(lags[f]) for f in order}
        self.features_ = tuple(order)

    def get(self, origin_date, target_date) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for feat, lag in self.lags.items():
            shifted = pd.Timestamp(target_date) - pd.Timedelta(weeks=lag)
            out[feat] = self.inner.get(origin_date, shifted).get(feat, np.nan)
        return out


def _ridge_fit_predict(train_X, train_y, test_X, l2: float = 1.0) -> np.ndarray:
    """Closed-form standardized ridge; handles NaN-free numpy arrays."""
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (train_X - mu) / sd
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    A = Xs.T @ Xs + l2 * np.eye(Xs.shape[1])
    A[0, 0] -= l2  # don't penalize the intercept
    beta = np.linalg.solve(A, Xs.T @ train_y)
    Zs = np.column_stack([np.ones(len(test_X)), (test_X - mu) / sd])
    return Zs @ beta


def stepwise_select(
    train_df: pd.DataFrame,
    lookup,
    lags_info: Dict[str, Dict[str, float]],
    folds: int = CV_FOLDS,
    min_gain: float = MIN_GAIN,
    horizons: tuple[int, ...] = (4, 8, 12, 16),
) -> list[str]:
    """Forward inclusion of lagged covariates under rolling-origin CV.

    Base features are autoregressive lags of ``log1p(cases)``; each
    candidate covariate enters at its estimated lag, evaluated as the
    value for the target week as known at the fold origin (the same
    as-of convention the GBMs use). Evaluation mirrors the deployment
    task: direct multi-horizon prediction at several lead weeks, where
    climate information is actually informative. A covariate is kept
    when it lowers mean out-of-sample MAE by at least ``min_gain``
    (relative) and improves the majority of folds.
    """
    df = train_df.sort_values("date").reset_index(drop=True)
    y_log = np.log1p(df["casos"].astype(float).to_numpy())
    dates = pd.DatetimeIndex(df["date"])
    n = len(df)

    max_ar = max(AR_LAGS)
    rows, targets, origin_pos, row_h = [], [], [], []
    for t in range(max_ar, n):
        base = [y_log[t - l] for l in AR_LAGS]
        base.append(float(np.nanmean(y_log[t - 3 : t + 1])))
        if not np.isfinite(base).all():
            continue
        for h in horizons:
            if t + h >= n or not np.isfinite(y_log[t + h]):
                continue
            rows.append(base)
            targets.append(y_log[t + h])
            origin_pos.append(t)
            row_h.append(h)
    if len(rows) < folds + max_ar:
        return []

    base_X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    origins = np.asarray(origin_pos)
    row_h_arr = np.asarray(row_h)

    cand_vals: Dict[str, np.ndarray] = {}
    for feat, info in sorted(lags_info.items(), key=lambda kv: -abs(kv[1]["corr"])):
        lag = int(info["lag"])
        col = np.full(len(rows), np.nan)
        for i, t in enumerate(origin_pos):
            target_week = dates[t] + pd.Timedelta(weeks=int(row_h_arr[i]) - lag)
            col[i] = lookup.get(dates[t], target_week).get(feat, np.nan)
        if np.isfinite(col).mean() > 0.8:  # usable in most folds
            col[~np.isfinite(col)] = np.nanmean(col)
            cand_vals[feat] = col

    # folds = last `folds` origin weeks; each fold tests that origin's
    # rows (one per horizon), trained on everything strictly earlier
    fold_origins = np.unique(origins)[-folds:]

    def fold_errors(columns: Dict[str, np.ndarray]) -> np.ndarray:
        """Mean absolute error per fold (one value per fold origin)."""
        X = np.column_stack([base_X] + [columns[c] for c in sorted(columns)]) if columns else base_X
        # scale covariates to the AR features' magnitude for the shared L2
        scale = np.std(base_X, axis=0).mean()
        X = X.copy()
        if columns:
            for j in range(base_X.shape[1], X.shape[1]):
                X[:, j] *= max(scale, 1.0) / max(np.std(X[:, j]), 1e-9)
        errs = []
        for fo in fold_origins:
            train_mask = origins < fo
            test_mask = origins == fo
            pred = _ridge_fit_predict(X[train_mask], y[train_mask], X[test_mask])
            errs.append(float(np.mean(np.abs(pred - y[test_mask]))))
        return np.asarray(errs)

    def _paired_t(trial_errs: np.ndarray, base_errs: np.ndarray) -> float:
        diff = base_errs - trial_errs  # >0 means trial is better
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        if sd < 1e-12:
            return np.inf if diff.mean() > 0 else 0.0
        return float(diff.mean() / (sd / np.sqrt(len(diff))))

    kept: Dict[str, np.ndarray] = {}
    best_errs = fold_errors(kept)
    for feat, col in cand_vals.items():
        trial = {**kept, feat: col}
        errs = fold_errors(trial)
        # a covariate must (a) lower mean CV MAE by min_gain, (b) improve
        # the majority of folds, and (c) clear a paired t-test whose bar
        # rises with every kept feature (a light multiple-comparison
        # guard against the greedy search riding on noise)
        if (
            errs.mean() <= best_errs.mean() * (1 - min_gain)
            and (errs < best_errs).mean() >= 0.6
            and _paired_t(errs, best_errs) >= 1.0 + 0.5 * len(kept)
        ):
            kept = trial
            best_errs = errs
    return list(kept)


def optimize_exog_lookup(lookup, train_df: pd.DataFrame, max_lag: int = MAX_LAG_WEEKS):
    """Estimate lags + stepwise-select covariates for one training window.

    Returns ``(wrapped_lookup_or_None, info)``; ``None`` means no
    covariate proved useful and models should stay covariate-free.

    Lags are estimated on a prefix that excludes the stepwise CV
    evaluation tail, so the best-of-``max_lag`` correlation search
    cannot leak into the fold it is judged on.
    """
    if lookup is None:
        return None, {"lags": {}, "selected": []}

    df = train_df.sort_values("date").reset_index(drop=True)
    cv_tail = CV_FOLDS + 24  # fold weeks + horizon span buffer
    est_df = df.iloc[:-cv_tail] if len(df) > 2 * cv_tail else df

    dates = pd.DatetimeIndex(pd.to_datetime(est_df["date"]))
    cases = pd.Series(est_df["casos"].astype(float).to_numpy(), index=dates)
    hist = covariate_history(lookup, dates)
    if hist.empty or cases.notna().sum() < MIN_CORR_POINTS + max_lag:
        return None, {"lags": {}, "selected": []}

    lags_info = estimate_covariate_lags(cases, hist, max_lag=max_lag)
    if not lags_info:
        return None, {"lags": {}, "selected": []}

    try:
        selected = stepwise_select(df, lookup, lags_info)
    except Exception as exc:  # defensive: selection must never break a backtest
        logger.warning("stepwise selection failed, using all lagged covariates: %s", exc)
        selected = list(lags_info)

    info = {
        "lags": {f: int(v["lag"]) for f, v in lags_info.items()},
        "corr": {f: round(float(v["corr"]), 3) for f, v in lags_info.items()},
        "selected": selected,
    }
    if not selected:
        return None, info
    return LaggedExogLookup(lookup, {f: lags_info[f]["lag"] for f in selected}), info
