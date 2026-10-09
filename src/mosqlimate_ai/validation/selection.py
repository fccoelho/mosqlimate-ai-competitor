"""Model selection from backtest results.

Selects, per state and disease, the forecasting strategy to deploy for
the final forecast:

- Candidates: every individual model plus the two ensembles.
- Score: mean WIS across validation tests that have observations
  (currently tests 1-3; test 4 actuals arrive with future data updates).
- Skill gate: the selected candidate must beat the seasonal-naive
  baseline on the same tests; otherwise fall back to the calibrated
  quantile ensemble (``ens_qavg``), the most robust strategy in
  backtests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "ens_qavg"
BASELINE_MODEL = "seas_naive"


def selection_table(state_result: dict) -> pd.DataFrame:
    """Mean WIS per model across the evaluable tests of one state result."""
    rows = []
    buckets = state_result.get("tests", {})
    if not buckets and "models" in state_result:
        buckets = {str(state_result.get("test_number", "?")): state_result}
    for test_key, test_res in buckets.items():
        for model_name, model_res in test_res.get("models", {}).items():
            metrics = model_res.get("metrics") or {}
            wis = metrics.get("wis_total")
            if wis is None or np.isnan(wis):
                continue
            rows.append({"model": model_name, "test": test_key, "wis": wis})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("model")["wis"].agg(["mean", "count"]).sort_values("mean")


def select_model(state_result: dict, skill_gate: bool = True) -> dict:
    """Select the best model for a state from its backtest result.

    Returns a dict with the chosen model, its score, the full score
    table, and whether the fallback was applied.
    """
    table = selection_table(state_result)
    if table.empty:
        return {
            "state": state_result.get("state"),
            "disease": state_result.get("disease"),
            "selected": FALLBACK_MODEL,
            "mean_wis": None,
            "fallback": True,
            "reason": "no evaluable test results",
            "scores": {},
        }

    best = table.index[0]
    selected = best
    reason = "best mean WIS across validation tests"
    fallback = False

    if BASELINE_MODEL in table.index and skill_gate:
        baseline_wis = table.loc[BASELINE_MODEL, "mean"]
        competitive = table.drop(index=BASELINE_MODEL)
        if table.loc[best, "mean"] > baseline_wis:
            selected = BASELINE_MODEL
            fallback = True
            reason = "skill gate: no model beats the seasonal-naive baseline"
        elif best == BASELINE_MODEL and not competitive.empty:
            # the baseline itself is the best model: the zoo has no skill
            fallback = True
            reason = "seasonal-naive is the best model; model zoo shows no skill"

    return {
        "state": state_result.get("state"),
        "disease": state_result.get("disease"),
        "selected": selected,
        "mean_wis": float(table.loc[selected, "mean"]),
        "fallback": fallback,
        "reason": reason,
        "scores": {m: float(v) for m, v in table["mean"].items()},
    }


def select_from_backtest_dir(backtest_dir: Path) -> pd.DataFrame:
    """Run selection for every state result JSON in a backtest output dir."""
    rows = []
    for path in sorted(Path(backtest_dir).glob("*_backtest.json")):
        result = json.loads(path.read_text())
        try:
            rows.append(select_model(result))
        except Exception as exc:
            logger.exception("selection failed for %s: %s", path.name, exc)
    return pd.DataFrame(rows)
