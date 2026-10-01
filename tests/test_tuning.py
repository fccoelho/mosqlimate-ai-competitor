"""Tests for per-state hyperparameter tuning."""


import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.models.baselines import FlatForecaster
from mosqlimate_ai.validation.tuning import (
    effective_params,
    load_cached_params,
    search_space,
    tune_state_model,
)


def synth(n=220, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-04", periods=n, freq="7D")
    t = np.arange(n)
    casos = np.maximum(80 + 30 * np.sin(2 * np.pi * t / 52.0) + rng.normal(0, 8, n), 0).round()
    return pd.DataFrame({"date": dates, "casos": casos})


def flat_factory(params):
    return FlatForecaster(window=8)


class TestSearchSpace:
    def test_effective_params_none_for_zero(self):
        p = effective_params({**search_space(np.random.default_rng(0)), "recency_halflife_weeks": 0})
        assert p["recency_halflife_weeks"] is None

    def test_effective_params_passes_through(self):
        p = effective_params({**search_space(np.random.default_rng(0)), "recency_halflife_weeks": 52})
        assert p["recency_halflife_weeks"] == 52


class TestTuneStateModel:
    @pytest.mark.slow
    def test_tuning_runs_and_caches(self, tmp_path):
        df = synth(220)
        result = tune_state_model(
            "XX",
            "dengue",
            "flat_test",
            flat_factory,
            df,
            n_trials=2,
            calib_weeks=40,
        )
        assert result["state"] == "XX"
        assert np.isfinite(result["baseline_wis"])
        assert len(result["trials"]) == 3  # baseline + 2 trials

    def test_cache_roundtrip(self, tmp_path):
        result = {
            "state": "XX",
            "disease": "dengue",
            "model": "m1",
            "best_params": {"n_estimators": 300, "recency_halflife_weeks": None},
            "wis": 1.0,
            "n_trials": 2,
            "trials": [{"params": {}, "wis": 1.0}],
        }
        from mosqlimate_ai.validation.tuning import save_tuning_result

        save_tuning_result(tmp_path, result)
        loaded = load_cached_params(tmp_path, "XX", "dengue", "m1")
        assert loaded == {"n_estimators": 300, "recency_halflife_weeks": None}
        assert load_cached_params(tmp_path, "YY", "dengue", "m1") is None
