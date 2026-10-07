"""Tests for the TimesFM zero-shot forecaster wrapper.

The pretrained checkpoint is never touched: ``load_timesfm_engine`` is
monkeypatched with a fake engine that mimics the ``timesfm`` 3.x
``TimesFM3Forecaster`` API (``predict`` -> ``ForecastOutput`` with a
``forecast`` point path and a ``(horizon, n_quantiles)`` quantile head).
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mosqlimate_ai.models.timesfm_forecaster as tsm
from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS
from mosqlimate_ai.models.base import make_forecast_dates
from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster, map_quantile_curve

HEAD_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def make_synth_series(n_weeks: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-04", periods=n_weeks, freq="7D")
    t = np.arange(n_weeks)
    casos = np.maximum(50 + 40 * np.sin(2 * np.pi * t / 52.0) + rng.gamma(1.5, 10, n_weeks), 0)
    return pd.DataFrame({"date": dates, "casos": casos})


class FakeEngine:
    """Mimics TimesFM3Forecaster: median + widening quantile head."""

    def __init__(self, levels=HEAD_LEVELS):
        self.config = SimpleNamespace(quantiles=list(levels))
        self.calls = []
        self.last = None

    def predict(self, context, horizon, return_quantiles=True, **kwargs):
        self.calls.append((np.asarray(context), horizon, kwargs))
        rng = np.random.default_rng(len(self.calls))
        med = float(np.mean(context[-52:])) * np.ones(horizon)
        spread = np.linspace(0.5, 3.0, horizon)
        z = np.sort([rng.normal(0, 1) for _ in self.config.quantiles])
        quants = np.stack(
            [
                np.maximum(med + spread * z[i] + 10.0 * (tau - 0.5), 0.0)
                for i, tau in enumerate(sorted(self.config.quantiles))
            ],
            axis=1,
        )
        quants = np.sort(quants, axis=1)  # monotone head, as real models emit
        med_idx = min(range(len(self.config.quantiles)),
                      key=lambda i: abs(self.config.quantiles[i] - 0.5))
        self.last = SimpleNamespace(forecast=quants[:, med_idx].copy(), quantiles=quants)
        return self.last


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(tsm, "load_timesfm_engine", lambda *a, **k: engine)
    return engine


class TestMapQuantileCurve:
    def test_interpolation_inside_head(self):
        levels = np.array(HEAD_LEVELS)  # q(0.1)=0, q(0.2)=1, ..., q(0.9)=8
        values = np.arange(9, dtype=float)
        assert map_quantile_curve(levels, values, 0.1) == pytest.approx(0.0)
        assert map_quantile_curve(levels, values, 0.5) == pytest.approx(4.0)
        assert map_quantile_curve(levels, values, 0.25) == pytest.approx(1.5)

    def test_gaussian_tails_outside_head(self):
        levels = np.array(HEAD_LEVELS)
        values = np.arange(9, dtype=float)
        from statistics import NormalDist

        inv = NormalDist().inv_cdf
        sigma = (8.0 - 0.0) / (inv(0.9) - inv(0.1))
        assert map_quantile_curve(levels, values, 0.025) == pytest.approx(4.0 + inv(0.025) * sigma)
        assert map_quantile_curve(levels, values, 0.975) == pytest.approx(4.0 + inv(0.975) * sigma)

    def test_monotone_across_all_imdc_levels(self):
        rng = np.random.default_rng(1)
        values = np.sort(rng.uniform(10, 100, len(HEAD_LEVELS)))
        mapped = [map_quantile_curve(HEAD_LEVELS, values, tau) for tau in QUANTILE_COLS]
        assert np.all(np.diff(mapped) >= -1e-9)


class TestTimesFMForecaster:
    def test_output_contract(self, fake_engine):
        df = make_synth_series()
        model = TimesFMForecaster()
        model.fit(df)
        horizon = 67
        qf = model.predict(horizon)

        assert len(qf) == horizon
        assert list(qf.columns) == list(QUANTILE_COLS.values())
        expected = make_forecast_dates(df["date"].max(), horizon)
        assert (qf.index == expected).all()
        values = qf.values
        assert (np.diff(values, axis=1) >= -1e-9).all()  # crossing-free
        assert (values >= 0).all()

    def test_median_is_the_model_point_forecast(self, fake_engine):
        df = make_synth_series()
        model = TimesFMForecaster().fit(df)
        qf = model.predict(10)
        assert len(fake_engine.calls) == 1
        assert np.allclose(qf[QUANTILE_COLS[0.5]], fake_engine.last.forecast)

    def test_zero_shot_fit_does_not_touch_engine(self, fake_engine):
        df = make_synth_series()
        model = TimesFMForecaster().fit(df)
        assert fake_engine.calls == []
        assert len(model.history_) == len(df)

    def test_fit_forward_fills_gap_weeks(self, fake_engine):
        df = make_synth_series(120)
        df.loc[10:15, "casos"] = np.nan  # explicit gap weeks
        model = TimesFMForecaster().fit(df)
        assert not model.history_.isna().any()
        assert len(model.history_) == 120  # ffilled, never dropped

    def test_engine_shared_across_predicts(self, fake_engine):
        df = make_synth_series(120)
        m1 = TimesFMForecaster().fit(df)
        m2 = TimesFMForecaster().fit(df)
        m1.predict(4)
        m2.predict(4)
        assert len(fake_engine.calls) == 2  # same engine object

    def test_transposed_head_tolerated(self, monkeypatch):
        engine = FakeEngine()
        orig = engine.predict

        def transposed_predict(context, horizon, **kwargs):
            out = orig(context, horizon, **kwargs)
            out.quantiles = np.asarray(out.quantiles).T
            return out

        engine.predict = transposed_predict
        monkeypatch.setattr(tsm, "load_timesfm_engine", lambda *a, **k: engine)
        model = TimesFMForecaster().fit(make_synth_series(120))
        qf = model.predict(8)
        assert len(qf) == 8
        assert (np.diff(qf.values, axis=1) >= -1e-9).all()

    def test_flat_fallback_without_quantile_head(self, monkeypatch):
        engine = FakeEngine()
        engine.predict = lambda context, horizon, **kw: SimpleNamespace(
            forecast=np.full(horizon, 7.0), quantiles=None
        )
        monkeypatch.setattr(tsm, "load_timesfm_engine", lambda *a, **k: engine)
        qf = TimesFMForecaster().fit(make_synth_series(120)).predict(5)
        assert (qf == 7.0).all().all()

    def test_missing_package_message(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "timesfm", None)  # forces ImportError
        monkeypatch.setattr(tsm, "_ENGINE_CACHE", {})
        model = TimesFMForecaster().fit(make_synth_series(120))
        with pytest.raises(ImportError, match="timesfm"):
            model.predict(4)


class TestEngineCache:
    def test_cached_per_key(self, monkeypatch):
        loads = []

        def fake_from_pretrained(**kwargs):
            loads.append(kwargs)
            return FakeEngine()

        fake_pkg = types.ModuleType("timesfm")
        fake_pkg.TimesFM3Forecaster = SimpleNamespace(
            from_pretrained=fake_from_pretrained
        )
        monkeypatch.setitem(sys.modules, "timesfm", fake_pkg)
        monkeypatch.setattr(tsm, "_ENGINE_CACHE", {})

        e1 = tsm.load_timesfm_engine("r", None, 4)
        e2 = tsm.load_timesfm_engine("r", None, 4)
        e3 = tsm.load_timesfm_engine("r", "cpu", 4)
        assert e1 is e2  # same key -> cached
        assert e1 is not e3  # different key -> new engine
        assert len(loads) == 2


class FakeLookup:
    """Mimics ExogLookup: six features, as-of semantics."""

    features_ = (
        "cf_temp_med",
        "cf_umid_med",
        "cf_precip_tot",
        "oc_enso",
        "oc_iod",
        "oc_pdo",
    )

    def __init__(self):
        self.calls = []

    def get(self, origin_date, target_date):
        self.calls.append((origin_date, target_date))
        return {f: float(pd.Timestamp(target_date).dayofyear) for f in self.features_}


class TestCovariates:
    def test_predict_passes_past_future_covariates(self, fake_engine):
        df = make_synth_series(120)
        lookup = FakeLookup()
        model = TimesFMForecaster(exog_lookup=lookup).fit(df)
        horizon = 10
        qf = model.predict(horizon)

        assert len(qf) == horizon
        _, _, kwargs = fake_engine.calls[-1]
        pf = kwargs["past_future_covariates"]
        assert pf.shape == (len(FakeLookup.features_), len(df) + horizon)
        assert np.isfinite(pf).all()
        # future block queried at the training cutoff (leak-safe origin)
        origins = {o for o, _ in lookup.calls}
        assert pd.Timestamp(df["date"].max()) in origins

    def test_without_lookup_stays_univariate(self, fake_engine):
        df = make_synth_series(120)
        model = TimesFMForecaster().fit(df)
        model.predict(6)
        _, _, kwargs = fake_engine.calls[-1]
        assert kwargs["past_future_covariates"] is None

    def test_empty_lookup_degrades_to_univariate(self, fake_engine):
        df = make_synth_series(120)
        empty = SimpleNamespace(features_=FakeLookup.features_, get=lambda o, t: {})
        model = TimesFMForecaster(exog_lookup=empty).fit(df)
        assert model.cov_history_ is None
        model.predict(6)
        _, _, kwargs = fake_engine.calls[-1]
        assert kwargs["past_future_covariates"] is None


class TestRegistry:
    def test_default_includes_timesfm_with_exog(self):
        from mosqlimate_ai.validation.backtest import default_model_registry

        class Dummy:
            pass

        lookup = Dummy()
        registry = default_model_registry(exog_lookup=lookup)
        assert isinstance(registry["timesfm"], TimesFMForecaster)
        assert registry["timesfm"].exog_lookup is lookup

    def test_default_includes_timesfm(self):
        from mosqlimate_ai.validation.backtest import default_model_registry

        assert isinstance(default_model_registry()["timesfm"], TimesFMForecaster)

    def test_flag_excludes_timesfm(self):
        from mosqlimate_ai.validation.backtest import default_model_registry

        assert "timesfm" not in default_model_registry(include_timesfm=False)
