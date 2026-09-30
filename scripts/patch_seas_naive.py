"""Recompute seas_naive forecasts/metrics in existing backtest results.

One-off patch after switching SeasonalNaiveForecaster to additive
seasonal errors (the multiplicative variant degenerated to NaN on
zero-inflated chikungunya series). Fits are near-instant.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.evaluation.calibration import calibrate_forecast
from mosqlimate_ai.evaluation.metrics import evaluate_by_horizon, evaluate_forecast
from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.models.baselines import SeasonalNaiveForecaster
from mosqlimate_ai.validation.backtest import FULL_HORIZON, _clone_model

loader = CompetitionDataLoader()
states_cache = {}
backtest_dir = Path("validation_results/backtest")
fc_dir = backtest_dir / "forecasts"
patched = 0

for path in sorted(backtest_dir.glob("*_backtest.json")):
    result = json.loads(path.read_text())
    uf, disease = result["state"], result["disease"]

    buckets = dict(result.get("tests", {}))
    if result.get("final"):
        buckets["final"] = result["final"]

    # strip-forecast copy created BEFORE mutation
    slim = json.loads(json.dumps(result))
    slim_buckets = list(slim.get("tests", {}).values())
    if slim.get("final"):
        slim_buckets.append(slim["final"])
    for t in slim_buckets:
        for m in t.get("models", {}).values():
            m.pop("forecast", None)
            m.pop("by_horizon", None)

    for split_key, test_res in buckets.items():
        models = test_res.get("models", {})
        train_end = pd.Timestamp(test_res["train_end"])
        target_start = pd.Timestamp(test_res["target_start"])

        key = (uf, disease)
        if key not in states_cache:
            states_cache[key] = loader.load_all_states(aggregate=True, disease=disease)[uf]
        state_df = states_cache[key]
        train = state_df[state_df["date"] <= train_end]
        actual = state_df[
            (state_df["date"] > train_end)
            & (state_df["date"] <= target_start + pd.Timedelta(weeks=52))
        ]

        target_dates = pd.date_range(target_start, periods=52, freq="7D")
        y_true = actual.set_index("date")["casos"].astype(float).reindex(target_dates)
        y_obs_idx = y_true[y_true.notna()].index
        if not len(y_obs_idx):
            continue

        model = SeasonalNaiveForecaster()
        forecast = calibrate_forecast(_clone_model(model, None), train, FULL_HORIZON)
        f_win = forecast[forecast.index.isin(y_obs_idx)]
        f_obs = f_win.loc[y_obs_idx]
        iv = quantiles_to_intervals(f_obs.reset_index(names="date"))
        metrics = evaluate_forecast(y_true.loc[y_obs_idx].values, iv)
        horizons = np.array([(d - train_end).days // 7 for d in f_obs.index])
        by_h = evaluate_by_horizon(y_true.loc[y_obs_idx].values, iv, horizons)

        models["seas_naive"] = {
            "metrics": metrics,
            "n_eval_weeks": int(len(y_obs_idx)),
            "runtime_s": 0.1,
            "by_horizon": by_h.reset_index().to_dict(orient="records"),
            "forecast": f_win.reset_index(names="date"),
        }
        # mirror the metrics into the slim copy
        slim_models = slim.setdefault("tests", {}).setdefault(split_key, {}).setdefault(
            "models", {}
        )
        if split_key == "final" and "final" in slim and "tests" not in slim:
            slim_models = slim["final"].setdefault("models", {})
        slim_models["seas_naive"] = {
            k: v for k, v in models["seas_naive"].items() if k not in ("forecast", "by_horizon")
        }
        patched += 1

        fname = fc_dir / f"{uf}_{disease}_{split_key}_seas_naive.csv.gz"
        f_win.reset_index(names="date").to_csv(fname, index=False)

    # write the slim JSON
    path.write_text(json.dumps(slim, indent=2, default=str))
    print(f"patched {path.name}")

print(f"total patched model entries: {patched}")
