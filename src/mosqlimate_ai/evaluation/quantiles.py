"""Canonical quantile forecast format and operations.

The IMDC competition requires forecasts expressed as a median plus
50%, 80%, 90% and 95% prediction intervals, i.e. the quantile levels

    {0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975}.

Throughout the pipeline, probabilistic forecasts are represented as a
DataFrame with one row per target date and one column per quantile level,
named ``q025``, ``q050``, ``q100``, ``q250``, ``q500``, ``q750``,
``q900``, ``q950``, ``q975``.

This module also implements the competition scoring quantities
(WIS following Bracher et al. 2021, and the quantile approximation of
CRPS) directly from that representation.
"""

from __future__ import annotations

from typing import Dict, List, Optional  # noqa: F401 - used in annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The nine quantile levels required by the IMDC submission format.
QUANTILE_LEVELS: tuple = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)

# Interval confidence levels implied by QUANTILE_LEVELS.
CONFIDENCE_LEVELS: tuple = (0.50, 0.80, 0.90, 0.95)


def quantile_col_name(tau: float) -> str:
    """Return the canonical column name for a quantile level.

    >>> quantile_col_name(0.025)
    'q025'
    >>> quantile_col_name(0.5)
    'q500'
    """
    return f"q{int(round(tau * 1000)):03d}"


QUANTILE_COLS: dict[float, str] = {tau: quantile_col_name(tau) for tau in QUANTILE_LEVELS}

# Mapping from quantile column names to the submission interval columns.
_INTERVAL_MAP = {
    "lower_95": "q025",
    "lower_90": "q050",
    "lower_80": "q100",
    "lower_50": "q250",
    "median": "q500",
    "upper_50": "q750",
    "upper_80": "q900",
    "upper_90": "q950",
    "upper_95": "q975",
}


def empty_quantile_frame(index: pd.Index) -> pd.DataFrame:
    """Create an all-NaN quantile frame for the given index."""
    return pd.DataFrame(
        {col: np.nan for col in QUANTILE_COLS.values()},
        index=index,
    )


def ensure_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a quantile frame indexed by date."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def quantiles_to_intervals(
    quantile_df: pd.DataFrame,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert canonical quantile columns to submission interval columns.

    Produces ``median``, ``lower_50``, ``upper_50``, ``lower_80``,
    ``upper_80``, ``lower_90``, ``upper_90``, ``lower_95``, ``upper_95``.
    A ``date`` column (or explicit ``date_col``) is preserved and the
    result is sorted by it; otherwise the index is kept.
    """
    out = quantile_df.copy()
    if date_col is None and "date" in out.columns:
        date_col = "date"

    renamed = {}
    for interval_col, quantile_col in _INTERVAL_MAP.items():
        if quantile_col in out.columns:
            renamed[interval_col] = out[quantile_col]
    result = pd.DataFrame(renamed, index=out.index)

    if date_col and date_col in out.columns:
        result[date_col] = out[date_col]
        result = result.sort_values(date_col).reset_index(drop=True)
    return result


def intervals_to_quantiles(
    interval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert submission interval columns back to canonical quantile columns."""
    renamed = {}
    for interval_col, quantile_col in _INTERVAL_MAP.items():
        if interval_col in interval_df.columns:
            renamed[quantile_col] = interval_df[interval_col]
    return pd.DataFrame(renamed, index=interval_df.index)


def sort_quantiles(quantile_df: pd.DataFrame) -> pd.DataFrame:
    """Fix quantile crossing by isotonic (sorted) rearrangement per row.

    Independent per-quantile models can produce non-monotonic quantile
    curves; rearrangement is the standard post-hoc fix and is optimal
    under a pinball-loss criterion. Columns are returned in canonical
    order (extra non-quantile columns are kept at the end).
    """
    cols = [c for c in QUANTILE_COLS.values() if c in quantile_df.columns]
    out = quantile_df.copy()
    if cols:
        out[cols] = np.sort(out[cols].values, axis=1)
        ordered = cols + [c for c in out.columns if c not in cols]
        out = out[ordered]
    return out


def widen_quantiles(quantile_df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Scale the quantile spread around the median by ``k``.

    ``q'_tau = median + k * (q_tau - median)`` — a multiplicative
    interval-widening (``k > 1``) or tightening (``k < 1``) used for
    post-hoc recalibration of over/under-confident forecasts. Values
    are clipped at zero; monotonicity is preserved because the spread
    is scaled uniformly per row. Non-quantile columns pass through.
    """
    cols = [c for c in QUANTILE_COLS.values() if c in quantile_df.columns]
    out = quantile_df.copy()
    if cols and "q500" in cols and k != 1.0:
        med = quantile_df["q500"]
        for col in cols:
            out[col] = (med + k * (quantile_df[col] - med)).clip(lower=0.0)
    return out


def pinball_loss(
    y_true: np.ndarray,
    z: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Pinball (quantile) loss per observation.

    ``rho_tau(y, z) = tau * (y - z)`` if ``y >= z`` else
    ``(1 - tau) * (z - y)``.

    Args:
        y_true: Observed values.
        z: Predicted tau-quantile values.
        tau: Quantile level in (0, 1).

    Returns:
        Per-observation pinball losses (non-negative).
    """
    y_true = np.asarray(y_true, dtype=float)
    z = np.asarray(z, dtype=float)
    diff = y_true - z
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_df: pd.DataFrame,
    levels: Sequence[float] | None = None,
) -> float:
    """CRPS approximation from quantile forecasts.

    Uses ``CRPS ~ (2 / n_tau) * sum_tau rho_tau`` (twice the mean pinball
    loss), the standard approximation for quantile-based forecasts. With
    the nine IMDC quantile levels this equals the weighted interval score
    with the usual Bracher et al. (2021) weights.

    Args:
        y_true: Observed values aligned with the quantile frame rows.
        quantile_df: Quantile frame (columns ``q025``..``q975``).
        levels: Restrict to these quantile levels (default: all nine).

    Returns:
        Mean CRPS across observations.
    """
    levels = levels or QUANTILE_LEVELS
    y_true = np.asarray(y_true, dtype=float)
    losses = []
    for tau in levels:
        col = quantile_col_name(tau)
        if col not in quantile_df.columns:
            continue
        z = quantile_df[col].values
        mask = ~(np.isnan(y_true) | np.isnan(z))
        if not mask.any():
            continue
        losses.append(pinball_loss(y_true[mask], z[mask], tau))
    if not losses:
        return float("nan")
    stacked = np.vstack(losses)
    return float(2.0 * np.mean(stacked))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted interval score component IS_alpha (per observation).

    ``IS_alpha = (u - l) + (2/alpha)(l - y) 1(y < l) + (2/alpha)(y - u) 1(y > u)``
    following Bracher et al. (2021).
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = np.maximum(0.0, lower - y_true)
    above = np.maximum(0.0, y_true - upper)
    return width + (2.0 / alpha) * (below + above)


def wis_from_quantiles(
    y_true: np.ndarray,
    quantile_df: pd.DataFrame,
    levels: Sequence[float] | None = None,
) -> float:
    """Weighted Interval Score (Bracher et al. 2021) from quantile forecasts.

    With the standard weights ``w_0 = 1/2`` and ``w_k = alpha_k / 2``, the
    WIS is exactly twice the mean pinball loss across the 2K+1 quantile
    levels, i.e. identical to :func:`crps_from_quantiles` on the same set
    of quantiles. Kept as a separate, explicitly named function for
    clarity in reports.
    """
    return crps_from_quantiles(y_true, quantile_df, levels)


def wis_total_from_intervals(
    y_true: np.ndarray,
    interval_df: pd.DataFrame,
    levels: Sequence[float] | None = None,
    median_col: str = "median",
) -> float:
    """WIS following Bracher et al. (2021) from interval-form forecasts.

    ``WIS = 1/(K + 1/2) * ( w_0 |y - m| + sum_k w_k IS_{alpha_k} )``
    with ``w_0 = 1/2`` and ``w_k = alpha_k / 2``.

    Args:
        y_true: Observed values.
        interval_df: DataFrame with ``median`` and ``lower_X``/``upper_X``
            columns for the requested confidence levels.
        levels: Confidence levels (default 0.50/0.80/0.90/0.95).
        median_col: Name of the median column.

    Returns:
        Mean WIS across observations (nan when no level is available).
    """
    y_true = np.asarray(y_true, dtype=float)
    levels = CONFIDENCE_LEVELS if levels is None else levels
    levels = list(levels)

    available = []
    for level in levels:
        name = int(level * 100)
        lower_col, upper_col = f"lower_{name}", f"upper_{name}"
        if lower_col in interval_df.columns and upper_col in interval_df.columns:
            available.append((level, lower_col, upper_col))

    if not available:
        return float("nan")

    median = interval_df[median_col].values if median_col in interval_df.columns else None

    weighted = 0.0
    for level, lower_col, upper_col in available:
        alpha = 1.0 - level
        weighted += (alpha / 2.0) * interval_score(
            y_true, interval_df[lower_col].values, interval_df[upper_col].values, alpha
        )

    total = weighted
    if median is not None:
        total = total + 0.5 * np.abs(y_true - np.asarray(median, dtype=float))
        denom = len(available) + 0.5
    else:
        denom = len(available)

    return float(np.mean(total / denom))
