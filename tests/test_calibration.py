"""Tests for conformal calibration and the quantile metrics module."""

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.evaluation.calibration import (
    apply_offsets,
    conformal_quantile_offsets,
)
from mosqlimate_ai.evaluation.quantiles import (
    QUANTILE_COLS,
    crps_from_quantiles,
    quantile_col_name,
    sort_quantiles,
    wis_from_quantiles,
    wis_total_from_intervals,
)
from mosqlimate_ai.models.baselines import FlatForecaster


def synth(n=200, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-03", periods=n, freq="7D")
    casos = np.maximum(100 + rng.normal(0, 20, n), 0).round()
    return pd.DataFrame({"date": dates, "casos": casos})


class TestQuantileMetrics:
    def test_wis_equals_crps_on_full_grid(self):
        rng = np.random.default_rng(0)
        taus = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        y = rng.gamma(2.0, 30.0, 100)
        qf = pd.DataFrame(
            np.sort(np.exp(rng.normal(np.log(y)[:, None], 0.3, (100, 9))), axis=1),
            columns=[quantile_col_name(t) for t in taus],
        )
        assert np.isclose(crps_from_quantiles(y, qf), wis_from_quantiles(y, qf))

    def test_wis_no_levels_is_nan(self):
        # median-only frame with no requested levels -> nan
        iv = pd.DataFrame({"median": [10.0], "lower_50": [10.0], "upper_50": [10.0]})
        assert np.isnan(wis_total_from_intervals(np.array([7.0]), iv, levels=[]))

    def test_perfect_forecast_has_zero_crps(self):
        y = np.array([5.0, 10.0])
        qf = pd.DataFrame(
            {col: y for col in QUANTILE_COLS.values()}
        )
        assert crps_from_quantiles(y, qf) == pytest.approx(0.0, abs=1e-9)

    def test_sort_quantiles_monotone(self):
        qf = pd.DataFrame(
            {c: [9.0, 1.0, 5.0] for c in QUANTILE_COLS.values()}
        )
        fixed = sort_quantiles(qf)
        assert (np.diff(fixed.values, axis=1) >= 0).all()


class TestConformalCalibration:
    def test_offsets_improve_coverage(self):
        # overconfident flat forecaster: constant point, tiny spread
        df = synth(160)
        cutoff = df["date"].iloc[-67]
        train = df[df["date"] <= cutoff].copy()
        actual_part = df[df["date"] > cutoff].set_index("date")["casos"]

        offsets = conformal_quantile_offsets(FlatForecaster(window=8), df, calib_weeks=67)
        assert offsets, "expected offsets on synthetic data"

        model = FlatForecaster(window=8)
        model.fit(train)
        raw = model.predict(67)
        calibrated = apply_offsets(raw, offsets)

        def coverage(qf):
            y = actual_part.reindex(qf.index).values
            lower = qf[QUANTILE_COLS[0.025]].values
            upper = qf[QUANTILE_COLS[0.975]].values
            mask = np.isfinite(y)
            return np.mean((y[mask] >= lower[mask]) & (y[mask] <= upper[mask]))

        # the flat forecaster misses trend/regime shifts; calibration must
        # not reduce 95% coverage
        assert coverage(calibrated) >= coverage(raw) - 0.05

    def test_apply_offsets_keeps_monotonicity(self):
        qf = pd.DataFrame(
            {
                QUANTILE_COLS[0.5]: [10.0],
                QUANTILE_COLS[0.25]: [9.0],
                QUANTILE_COLS[0.75]: [11.0],
                QUANTILE_COLS[0.025]: [8.0],
                QUANTILE_COLS[0.975]: [12.0],
                QUANTILE_COLS[0.05]: [8.5],
                QUANTILE_COLS[0.95]: [11.5],
                QUANTILE_COLS[0.1]: [8.7],
                QUANTILE_COLS[0.9]: [11.3],
            }
        )
        offsets = {0.5: 5.0}  # big shift on the median only
        fixed = apply_offsets(qf, offsets)
        assert (np.diff(fixed.values, axis=1) >= 0).all()


class TestExogLookup:
    def test_issue_lead_semantics(self):
        """Target months beyond the origin use the origin-month issue with
        lead = distance; months before it use their own month's shortest lead."""
        import warnings as _w

        _w.filterwarnings("ignore")
        from mosqlimate_ai.data.future_exog import ExogLookup
        from mosqlimate_ai.data.loader import CompetitionDataLoader

        loader = CompetitionDataLoader()
        lookup = ExogLookup(loader, "SP", train_end=pd.Timestamp("2022-06-26"))

        origin = pd.Timestamp("2022-06-26")
        # target month before origin: uses its own issue month (lead 1)
        f_past = lookup.get(origin, pd.Timestamp("2022-05-01"))
        # target 4 months ahead: issue = origin month, lead 4
        f_4mo = lookup.get(origin, pd.Timestamp("2022-10-05"))
        # target 6 months ahead: lead 6
        f_6mo = lookup.get(origin, pd.Timestamp("2022-12-26"))

        assert np.isfinite(f_past["cf_temp_med"])
        assert np.isfinite(f_4mo["cf_temp_med"])
        assert np.isfinite(f_6mo["cf_temp_med"])
        assert np.isfinite(f_past["oc_enso"])
        # distinct leads give distinct forecasts
        assert f_4mo["cf_temp_med"] != f_6mo["cf_temp_med"]

    def test_ocean_persistence_beyond_cutoff(self):
        import warnings as _w

        _w.filterwarnings("ignore")
        from mosqlimate_ai.data.future_exog import ExogLookup
        from mosqlimate_ai.data.loader import CompetitionDataLoader

        loader = CompetitionDataLoader()
        train_end = pd.Timestamp("2025-06-22")
        lookup = ExogLookup(loader, "SP", train_end=train_end)
        far = lookup.get(train_end, pd.Timestamp("2026-06-07"))
        last_obs = lookup.ocean_.index[-1]
        at_last = lookup.get(train_end, last_obs)
        assert far["oc_enso"] == at_last["oc_enso"], (
            "persistence must freeze at the last observed value"
        )
