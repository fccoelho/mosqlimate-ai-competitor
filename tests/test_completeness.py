"""Tests for weekly data-completeness checks and NaN-aware training."""

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.data.completeness import find_missing_weeks, warn_missing_weeks
from mosqlimate_ai.data.loader import reindex_weekly
from mosqlimate_ai.models.baselines import LogLinearTrendForecaster, SeasonalNaiveForecaster


def weekly_df(n=200, start="2016-01-03", seed=0, drop_weeks=(), nan_weeks=()):
    dates = pd.date_range(start, periods=n, freq="7D")
    rng = np.random.default_rng(seed)
    casos = np.maximum(100 + rng.normal(0, 15, n), 0).round()
    df = pd.DataFrame({"date": dates, "casos": casos})
    drop = {pd.Timestamp(w) for w in drop_weeks}
    df = df[~df.date.isin(drop)].reset_index(drop=True)
    if nan_weeks:
        idx = df.date.isin({pd.Timestamp(w) for w in nan_weeks})
        df.loc[idx, "casos"] = np.nan
    return df


class TestFindMissingWeeks:
    def test_missing_rows_detected(self):
        df = weekly_df(52, drop_weeks=["2016-03-06"])
        missing = find_missing_weeks(df)
        assert missing == [pd.Timestamp("2016-03-06")]

    def test_nan_values_detected(self):
        df = weekly_df(52, nan_weeks=["2016-03-06"])
        missing = find_missing_weeks(df)
        assert missing == [pd.Timestamp("2016-03-06")]

    def test_complete_series_has_no_gaps(self):
        assert find_missing_weeks(weekly_df(52)) == []

    def test_explicit_bounds(self):
        df = weekly_df(20)
        missing = find_missing_weeks(df, start="2016-08-07", end="2016-09-04")
        assert len(missing) == 5  # Sundays: Aug 7 .. Sep 4


class TestWarnMissingWeeks:
    def test_warning_emitted(self):
        df = weekly_df(52, drop_weeks=["2016-03-06"])
        with pytest.warns(UserWarning, match="INCOMPLETE"):
            missing = warn_missing_weeks(df, "XX/dengue")
        assert len(missing) == 1

    def test_no_warning_when_complete(self):
        import warnings as _w

        df = weekly_df(52)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            missing = warn_missing_weeks(df, "XX/dengue")
        assert missing == []
        assert not [w for w in caught if "INCOMPLETE" in str(w.message)]


class TestReindexWeekly:
    def test_missing_weeks_become_nan_rows(self):
        df = weekly_df(52, drop_weeks=["2016-03-06"])
        out = reindex_weekly(df)
        assert len(out) == 52
        assert out[out.casos.isna()].date.iloc[0] == pd.Timestamp("2016-03-06")

    def test_complete_frame_unchanged(self):
        df = weekly_df(52)
        out = reindex_weekly(df)
        assert len(out) == 52
        assert out.casos.notna().all()


class TestNaNAwareModels:
    def test_seasonal_naive_survives_nan_weeks(self):
        df = weekly_df(160, nan_weeks=["2016-06-05", "2016-06-12"])
        model = SeasonalNaiveForecaster()
        model.fit(df)
        qf = model.predict(8)
        assert qf.notna().all().all()

    def test_seasonal_naive_walks_back_from_missing_lag(self):
        df = weekly_df(160, drop_weeks=["2018-11-04"])  # creates NaN after reindex
        df = reindex_weekly(df)
        model = SeasonalNaiveForecaster()
        model.fit(df)
        # a horizon date whose lag-52 week is the NaN week
        horizon_date = pd.Timestamp("2018-11-04") + pd.Timedelta(weeks=52)
        qf = model.predict(52)
        assert horizon_date in qf.index
        assert qf.loc[horizon_date].notna().all()

    def test_loglinear_fits_with_nan_weeks(self):
        df = weekly_df(200, nan_weeks=[d for d in pd.date_range("2016-06-05", periods=4, freq="7D")])
        model = LogLinearTrendForecaster()
        model.fit(df)
        qf = model.predict(4)
        assert qf.notna().all().all()
        assert (qf >= 0).all().all()

    def test_gbm_skips_nan_targets(self):
        xgboost = pytest.importorskip("xgboost")
        from mosqlimate_ai.models.gbm_direct import XGBoostDirectForecaster

        df = weekly_df(160, nan_weeks=["2016-08-01"])
        model = XGBoostDirectForecaster(
            max_horizon=8, quantiles=[0.25, 0.5, 0.75], recency_halflife_weeks=None,
            params={"n_estimators": 50, "booster_n_jobs": 4},
        )
        model.fit(df)
        qf = model.predict(8)
        assert qf.notna().all().all()


class TestVerificationManifest:
    def test_mark_and_skip(self, tmp_path):
        from mosqlimate_ai.data.completeness import (
            file_is_verified,
            mark_files_verified,
        )

        f = tmp_path / "dengue.csv.gz"
        f.write_bytes(b"12345")
        assert not file_is_verified(tmp_path, "dengue.csv.gz")
        mark_files_verified(tmp_path, ["dengue.csv.gz"])
        assert file_is_verified(tmp_path, "dengue.csv.gz")

    def test_invalidated_when_file_changes(self, tmp_path):
        import time

        from mosqlimate_ai.data.completeness import (
            file_is_verified,
            mark_files_verified,
        )

        f = tmp_path / "dengue.csv.gz"
        f.write_bytes(b"12345")
        mark_files_verified(tmp_path, ["dengue.csv.gz"])
        time.sleep(0.02)
        f.write_bytes(b"12345678")  # content changed -> fingerprint differs
        assert not file_is_verified(tmp_path, "dengue.csv.gz")

    def test_missing_file_not_verified(self, tmp_path):
        from mosqlimate_ai.data.completeness import file_is_verified

        assert not file_is_verified(tmp_path, "absent.csv.gz")
