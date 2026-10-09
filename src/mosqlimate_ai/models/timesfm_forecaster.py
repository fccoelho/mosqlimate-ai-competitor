"""Google TimesFM (3.0) zero-shot foundation model on the unified interface.

Role in the model zoo: a pretrained time-series foundation model that
forecasts the weekly case counts *without local training*. ``fit()``
only buffers the target history (and, when an ``exog_lookup`` is
supplied, the aligned covariate history); ``predict()`` runs decoder
inference and maps the model's native quantile head onto the nine IMDC
quantile levels (levels inside the head's range are interpolated; the
outer tails are extended with Gaussian tails scaled from the inner 80%
of the predicted distribution).

Covariates: when constructed with an
:class:`~mosqlimate_ai.data.future_exog.ExogLookup`, the six
target-time-known covariates (ECMWF state temperature/humidity/
precipitation forecasts and the ENSO/IOD/PDO ocean indices) are passed
to TimesFM as ``past_future_covariates`` — the same leakage-safe values
the GBM models consume: history weeks use each week's own as-of values,
future weeks use the values known at the training cutoff. Without a
lookup the model stays purely univariate.

Requires the ``timesfm`` package (>= 3.0, already a project
dependency). The pretrained checkpoint (default
``google/timesfm-3.0-pytorch``) is downloaded from the HuggingFace Hub
on first use (~500 MB) and cached; one engine instance is shared per
process so backtest clones never reload it.
"""

from __future__ import annotations

import logging
from statistics import NormalDist

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS
from mosqlimate_ai.models.base import BaseForecaster, make_forecast_dates

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "google/timesfm-3.0-pytorch"

_INV_CDF = NormalDist().inv_cdf

# One loaded checkpoint per process, shared by every forecaster clone
# (the backtest harness copies models per calibration fit).
_ENGINE_CACHE: dict[tuple, object] = {}


def load_timesfm_engine(
    repo_id: str = DEFAULT_REPO_ID,
    device: str | None = None,
    per_core_batch_size: int = 4,
) -> object:
    """Load (and cache) a TimesFM 3.0 forecaster engine.

    Args:
        repo_id: HuggingFace repo or local checkpoint directory.
        device: ``"cuda"``/``"cpu"`` (default: auto-detect).
        per_core_batch_size: Inference batch size.

    Raises:
        ImportError: When ``timesfm`` is not installed.
    """
    key = (repo_id, device, per_core_batch_size)
    if key not in _ENGINE_CACHE:
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "TimesFM requires the 'timesfm' package (>= 3.0). "
                "Install it with: uv sync  (or pip install 'timesfm>=3.0.2')"
            ) from exc
        logger.info("Loading TimesFM checkpoint %s (first use downloads it)...", repo_id)
        _ENGINE_CACHE[key] = timesfm.TimesFM3Forecaster.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            device=device,
            per_core_batch_size=per_core_batch_size,
        )
        logger.info(
            "TimesFM engine ready (quantile head: %s)",
            getattr(_ENGINE_CACHE[key].config, "quantiles", "?"),
        )
    return _ENGINE_CACHE[key]


def map_quantile_curve(
    avail_levels: np.ndarray,
    avail_values: np.ndarray,
    target_level: float,
) -> float:
    """Quantile value at ``target_level`` from a predicted quantile curve.

    Levels inside the available range are linearly interpolated. Levels
    outside (e.g. the IMDC 0.025/0.975 tails when the head only emits
    0.1..0.9) are extended with a Gaussian tail whose scale is
    estimated from the inner 80% of the curve — flat clamping would
    systematically under-cover and be crushed by the WIS score.
    """
    order = np.argsort(avail_levels)
    levels = np.asarray(avail_levels, dtype=float)[order]
    values = np.asarray(avail_values, dtype=float)[order]

    if target_level >= levels[0] and target_level <= levels[-1]:
        return float(np.interp(target_level, levels, values))

    lo, hi = 0.1, 0.9
    if lo < levels[0] or hi > levels[-1]:
        lo, hi = levels[0], levels[-1]
    q_lo = float(np.interp(lo, levels, values))
    q_hi = float(np.interp(hi, levels, values))
    sigma = (q_hi - q_lo) / (_INV_CDF(hi) - _INV_CDF(lo))
    median = float(np.interp(0.5, levels, values))
    return median + _INV_CDF(target_level) * sigma


class TimesFMForecaster(BaseForecaster):
    """Zero-shot TimesFM 3.0 foundation model.

    Args:
        repo_id: HuggingFace checkpoint repo (default
            ``google/timesfm-3.0-pytorch``).
        device: ``"cuda"``/``"cpu"``; ``None`` auto-detects.
        per_core_batch_size: Inference batch size.
        exog_lookup: Optional
            :class:`~mosqlimate_ai.data.future_exog.ExogLookup` whose six
            target-time-known covariates (climate forecasts + ocean
            indices) are fed to the model as ``past_future_covariates``.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: str | None = None,
        per_core_batch_size: int = 4,
        exog_lookup=None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.device = device
        self.per_core_batch_size = per_core_batch_size
        self.exog_lookup = exog_lookup
        self.cov_history_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def _covariate_frame(
        self, dates: pd.DatetimeIndex, origin: pd.Timestamp | None = None
    ) -> pd.DataFrame | None:
        """Covariate rows for ``dates`` as known at each date (or at one origin).

        Returns None when the lookup carries no usable values at all;
        remaining interior gaps are forward/backward-filled (the engine
        would interpolate them anyway, but edge NaNs must not survive).
        """
        feats = list(self.exog_lookup.features_)
        rows = []
        for d in dates:
            vals = self.exog_lookup.get(origin if origin is not None else d, d)
            rows.append([vals.get(f, np.nan) for f in feats])
        out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=feats)
        if not out.notna().any().any():
            return None
        return out.ffill().bfill().fillna(0.0)

    # ------------------------------------------------------------------
    def _fit(self, df: pd.DataFrame) -> None:
        """Buffer the target history (the checkpoint is not fine-tuned)."""
        s = df.set_index("date")[self.target_col].astype(float).sort_index()
        # The foundation model needs a dense, finite context: explicit
        # gap weeks (NaN) are forward-filled, never zero-filled.
        self.history_ = s.ffill().dropna()
        if len(self.history_) < 2:
            raise ValueError("TimesFM needs at least two finite training weeks")
        self.cov_history_ = self._covariate_frame(self.history_.index) if self.exog_lookup else None

    # ------------------------------------------------------------------
    def _predict(self, horizon: int) -> pd.DataFrame:
        engine = load_timesfm_engine(self.repo_id, self.device, self.per_core_batch_size)
        context = self.history_.to_numpy(dtype=np.float64)

        # covariates aligned with the context window plus the horizon:
        # (n_covariates, context_len + horizon), the TimesFM 3.x layout
        past_future_covariates = None
        if self.cov_history_ is not None:
            fut_dates = make_forecast_dates(self.last_train_date_, horizon)
            cov_fut = self._covariate_frame(fut_dates, origin=self.last_train_date_)
            frames = [self.cov_history_] + ([cov_fut] if cov_fut is not None else [])
            full = pd.concat(frames)
            past_future_covariates = full.to_numpy(dtype=np.float32).T

        out = engine.predict(
            context,
            horizon,
            past_future_covariates=past_future_covariates,
            return_quantiles=True,
            make_positive=True,
            sort_quantiles=True,
        )
        point = np.asarray(out.forecast, dtype=float).ravel()[:horizon]

        levels = list(getattr(engine.config, "quantiles", None) or [])
        quants = getattr(out, "quantiles", None)
        if quants is not None:
            quants = np.asarray(quants, dtype=float)

        dates = make_forecast_dates(self.last_train_date_, horizon)
        qf = pd.DataFrame(index=dates)

        usable = (
            quants is not None
            and quants.ndim == 2
            and len(levels) > 1
            and quants.shape[-1] in (len(levels), horizon)
        )
        if usable:
            # engine returns (horizon, n_quantiles); tolerate transposed
            if quants.shape[0] == len(levels) and quants.shape[1] != len(levels):
                quants = quants.T
            curves = quants[:horizon]
            for tau, col in QUANTILE_COLS.items():
                qf[col] = [map_quantile_curve(levels, row, tau) for row in curves]
        else:
            logger.warning(
                "TimesFM returned no usable quantile head; issuing flat "
                "quantiles around the point forecast (conformal "
                "calibration will widen them)"
            )
            for col in QUANTILE_COLS.values():
                qf[col] = point

        # the median column should reflect the model's own median
        if 0.5 in QUANTILE_COLS:
            qf[QUANTILE_COLS[0.5]] = point

        return qf.clip(lower=0.0)
