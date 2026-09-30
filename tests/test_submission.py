"""Tests for model selection and IMDC submission building."""

import json

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS, quantiles_to_intervals
from mosqlimate_ai.submission.imdc import IMDCSubmissionBuilder
from mosqlimate_ai.validation.selection import BASELINE_MODEL, select_model


def make_state_result(wis_by_model: dict) -> dict:
    """Fabricate a backtest result with the given mean WIS per model."""
    tests = {}
    for test_number in ("1", "2", "3"):
        models = {}
        for model, wis in wis_by_model.items():
            models[model] = {"metrics": {"wis_total": wis * (1 + 0.05 * int(test_number))}}
        tests[test_number] = {
            "state": "XX",
            "disease": "dengue",
            "season": f"20{test_number}",
            "models": models,
        }
    return {"state": "XX", "disease": "dengue", "tests": tests}


class TestSelection:
    def test_best_model_selected(self):
        result = make_state_result(
            {"xgb_direct": 500.0, "lgbm_direct": 700.0, BASELINE_MODEL: 2000.0}
        )
        sel = select_model(result)
        assert sel["selected"] == "xgb_direct"
        assert not sel["fallback"]

    def test_skill_gate_deploys_baseline(self):
        # every competitive model loses to the naive baseline -> baseline
        result = make_state_result(
            {
                "xgb_direct": 5000.0,
                "lgbm_direct": 6000.0,
                "ens_qavg": 3000.0,
                BASELINE_MODEL: 2000.0,
            }
        )
        sel = select_model(result)
        assert sel["selected"] == BASELINE_MODEL
        assert sel["fallback"]

    def test_baseline_when_nothing_beats_it(self):
        result = make_state_result(
            {"xgb_direct": 5000.0, "ens_qavg": 3000.0, BASELINE_MODEL: 1000.0}
        )
        sel = select_model(result)
        assert sel["selected"] == BASELINE_MODEL

    def test_empty_result_falls_back(self):
        result = {"state": "XX", "disease": "dengue", "tests": {}}
        sel = select_model(result)
        assert sel["selected"] == "ens_qavg"
        assert sel["fallback"]


def synth_quantile_forecast(start: str, weeks: int = 52) -> pd.DataFrame:
    dates = pd.date_range(start, periods=weeks, freq="7D")
    idx = np.arange(weeks)
    qf = pd.DataFrame(index=dates)
    base = 100.0 + idx
    for tau, col in QUANTILE_COLS.items():
        qf[col] = base * (0.5 + tau)
    return qf


class TestIMDCSubmission:
    def test_payload_structure(self):
        builder = IMDCSubmissionBuilder(model_id=1, description="test dengue final")
        fc = quantiles_to_intervals(synth_quantile_forecast("2026-10-04").reset_index(names="date"))
        payload = builder.add_forecast("SP", fc, "final", disease="dengue")
        assert payload["adm_1"] == "SP"
        pred = payload["prediction"]
        assert len(pred["dates"]) == 52
        for key in ("lower_95", "lower_90", "lower_80", "lower_50", "preds", "upper_95"):
            assert key in pred

    def test_median_containment_enforced(self):
        builder = IMDCSubmissionBuilder(model_id=1)
        fc = quantiles_to_intervals(
            synth_quantile_forecast("2026-10-04").reset_index(names="date")
        )
        # push the median above the upper 95 bound; builder must clip it
        fc["median"] = fc["upper_95"] + 10
        payload = builder.add_forecast("SP", fc, "final")
        pred = payload["prediction"]
        assert all(m <= u for m, u in zip(pred["preds"], pred["upper_95"]))
        issues = builder.validate_completeness(expected_splits=["final"])
        assert not any("median outside" in i for i in issues)

    def test_missing_intervals_raise(self):
        builder = IMDCSubmissionBuilder(model_id=1)
        fc = synth_quantile_forecast("2026-10-04").reset_index(names="date")
        fc = fc[["date", "q500"]]
        with pytest.raises(ValueError):
            builder.add_forecast("SP", fc, "final")

    def test_completeness_missing_units(self):
        builder = IMDCSubmissionBuilder(model_id=1)
        fc = quantiles_to_intervals(
            synth_quantile_forecast("2026-10-04").reset_index(names="date")
        )
        builder.add_forecast("SP", fc, "final")
        issues = builder.validate_completeness(
            expected_states=["SP", "RJ", "BA"], expected_splits=["final"]
        )
        assert any("missing: dengue/final/RJ" in i for i in issues)
        assert any("missing: dengue/final/BA" in i for i in issues)
        assert not any("SP" in i and "missing" in i for i in issues)

    def test_negative_clipping(self):
        builder = IMDCSubmissionBuilder(model_id=1)
        qf = synth_quantile_forecast("2026-10-04")
        qf[QUANTILE_COLS[0.025]] = -5.0
        fc = quantiles_to_intervals(qf.reset_index(names="date"))
        payload = builder.add_forecast("SP", fc, "final")
        assert all(v >= 0 for v in payload["prediction"]["lower_95"])
