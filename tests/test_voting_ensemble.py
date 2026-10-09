"""Tests for the inverse-WIS voting ensemble (``ens_vote``)."""

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.calibration import calibrate_forecast_scored
from mosqlimate_ai.models.baselines import (
    FlatForecaster,
    LogLinearTrendForecaster,
    SeasonalNaiveForecaster,
)
from mosqlimate_ai.validation.backtest import run_single_backtest, voting_weights
from mosqlimate_ai.validation.config import ValidationTestConfig


def synth(n=370, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="7D")
    t = np.arange(n)
    seasonal = 100 * (1 + np.sin(2 * np.pi * t / 52.0))
    casos = np.maximum(seasonal + rng.normal(0, 15, n), 0).round()
    return pd.DataFrame({"date": dates, "casos": casos})


TEST_CFG = ValidationTestConfig(
    test_number=1,
    season="synthetic",
    train_end="2022-06-26",
    target_start="2022-10-09",
    target_end="2023-10-08",
    description="synthetic test",
)


def split(df):
    train = df[df["date"] <= pd.Timestamp(TEST_CFG.train_end)]
    actual = df[
        (df["date"] > pd.Timestamp(TEST_CFG.train_end))
        & (df["date"] <= pd.Timestamp(TEST_CFG.target_end))
    ]
    return train, actual


class TestVotingWeights:
    def test_weights_sum_to_one_and_order_by_skill(self):
        w = voting_weights({"a": 10.0, "b": 20.0, "c": 40.0})
        assert set(w) == {"a", "b", "c"}
        assert w["a"] > w["b"] > w["c"]
        assert np.isclose(sum(w.values()), 1.0)

    def test_nan_and_inf_scores_dropped(self):
        w = voting_weights({"a": 10.0, "bad": float("nan"), "worse": float("inf"), "b": 30.0})
        assert set(w) == {"a", "b"}
        assert np.isclose(sum(w.values()), 1.0)

    def test_fewer_than_two_valid_scores_returns_empty(self):
        assert voting_weights({"a": 5.0}) == {}
        assert voting_weights({"a": float("nan"), "b": float("nan")}) == {}

    def test_equal_scores_give_equal_weights(self):
        w = voting_weights({"a": 7.0, "b": 7.0, "c": 7.0})
        assert np.allclose(list(w.values()), 1 / 3)

    def test_regularization_bounds_a_freak_perfect_score(self):
        w = voting_weights({"perfect": 0.001, "avg1": 100.0, "avg2": 100.0})
        # a raw inverse score would hand this model ~100% of the vote
        assert w["perfect"] < 0.9
        assert w["perfect"] > w["avg1"]


class TestCalibrateForecastScored:
    def test_returns_forecast_and_positive_skill(self):
        df = synth()
        model = FlatForecaster(window=8)
        forecast, skill = calibrate_forecast_scored(model, df, horizon=67)
        assert not forecast.empty
        assert "q500" in forecast.columns
        assert len(forecast) == 67
        assert skill is not None
        assert np.isfinite(skill)
        assert skill > 0

    def test_unscorable_window_returns_none_skill(self):
        # too short for a calibration window -> no offsets, no skill
        model = FlatForecaster(window=8)
        forecast, skill = calibrate_forecast_scored(model, synth(60), horizon=10)
        assert not forecast.empty
        assert skill is None


class TestBacktestVotingEnsemble:
    registry = {
        "flat": FlatForecaster(window=8),
        "loglin_trend": LogLinearTrendForecaster(),
        "seas_naive": SeasonalNaiveForecaster(),
    }

    def test_ens_vote_present_with_all_members(self):
        train, actual = split(synth())
        result = run_single_backtest(
            "SY",
            TEST_CFG,
            disease="dengue",
            model_registry=self.registry,
            calibrate=True,
            train_df=train,
            actual_df=actual,
            future_exog=None,
        )
        assert "ens_vote" in result["models"]
        metrics = result["models"]["ens_vote"]["metrics"]
        assert np.isfinite(metrics["wis_total"])
        assert result["models"]["ens_vote"]["n_eval_weeks"] == 52

        votes = result["ensemble_votes"]
        # all trained models vote, including the seasonal-naive baseline
        assert set(votes["weights"]) == set(self.registry)
        assert set(votes["calib_wis"]) == set(self.registry)
        assert np.isclose(sum(votes["weights"].values()), 1.0)

        # the equal-weight ensembles remain alongside the voting one
        assert "ens_qavg" in result["models"]
        assert "ens_median" in result["models"]

    def test_ens_vote_skipped_without_calibration(self):
        train, actual = split(synth())
        result = run_single_backtest(
            "SY",
            TEST_CFG,
            disease="dengue",
            model_registry=self.registry,
            calibrate=False,
            train_df=train,
            actual_df=actual,
            future_exog=None,
        )
        assert "ens_vote" not in result["models"]
        assert "ensemble_votes" not in result
        assert "ens_qavg" in result["models"]
