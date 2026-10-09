"""Tests for the unified forecaster interface, baselines, and GBM model."""

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS
from mosqlimate_ai.models.base import BaseForecaster, make_forecast_dates
from mosqlimate_ai.models.baselines import (
    FlatForecaster,
    LogLinearTrendForecaster,
    SeasonalNaiveForecaster,
)


def make_synth_series(n_weeks: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-04", periods=n_weeks, freq="7D")
    t = np.arange(n_weeks)
    seasonal = 50 + 40 * np.sin(2 * np.pi * t / 52.0)
    noise = rng.gamma(1.5, 10, n_weeks)
    casos = np.maximum(seasonal + noise + t * 0.2, 0).round()
    return pd.DataFrame({"date": dates, "casos": casos})


class TestMakeForecastDates:
    def test_weekly_steps(self):
        dates = make_forecast_dates(pd.Timestamp("2022-06-26"), 5)
        assert len(dates) == 5
        assert dates[0] == pd.Timestamp("2022-07-03")
        assert (dates.diff().dropna() == pd.Timedelta(days=7)).all()

    def test_gap_alignment(self):
        # competition: train_end EW25 (2022-06-26) -> target_start EW41
        # (2022-10-09) = 15-week gap; 52 target weeks = horizons 15..66
        dates = make_forecast_dates(pd.Timestamp("2022-06-26"), 67)
        assert dates[14] == pd.Timestamp("2022-10-09"), "horizon 15 must hit EW41"
        assert dates[65] == pd.Timestamp("2023-10-01"), "horizon 66 is the 52nd target week"
        assert len(dates) == 67


class TestBaselines:
    @pytest.mark.parametrize(
        "factory",
        [
            SeasonalNaiveForecaster,
            FlatForecaster,
            LogLinearTrendForecaster,
        ],
    )
    def test_output_contract(self, factory):
        df = make_synth_series()
        model = factory()
        model.fit(df)
        horizon = 67
        qf = model.predict(horizon)

        assert len(qf) == horizon
        assert list(qf.columns) == list(QUANTILE_COLS.values())
        # quantiles are monotone (crossing-free)
        values = qf.values
        assert (np.diff(values, axis=1) >= -1e-9).all()
        # non-negative
        assert (values >= 0).all()

    def test_seasonal_naive_uses_lag52(self):
        df = make_synth_series(160)
        model = SeasonalNaiveForecaster()
        model.fit(df)
        qf = model.predict(4)
        expected = df.set_index("date")["casos"]
        for i, date in enumerate(qf.index):
            lag_date = date - pd.Timedelta(weeks=52)
            assert qf[QUANTILE_COLS[0.5]].iloc[i] == pytest.approx(expected.loc[lag_date])

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            SeasonalNaiveForecaster().predict(4)


class TestBaseContract:
    def test_fit_requires_date_column(self):
        df = pd.DataFrame({"casos": [1, 2, 3]})
        with pytest.raises(ValueError):
            SeasonalNaiveForecaster().fit(df)

    def test_predict_horizon_validation(self):
        model = SeasonalNaiveForecaster()
        model.fit(make_synth_series(120))
        with pytest.raises(ValueError):
            model.predict(0)
