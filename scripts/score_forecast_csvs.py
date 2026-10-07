"""Score forecast CSVs that are not covered by the backtest JSONs.

Reads ``<backtest_dir>/forecasts/{uf}_{disease}_{test}_{model}.csv.gz``
for one or more models (e.g. the downloaded ``imdc_bb`` baseline or
``timesfm`` forecasts from an earlier run), evaluates them against the
same observed state-level case counts and with the same WIS/coverage
functions used by the backtests, and writes
``<backtest_dir>/{model}_scores.json`` per model with one row per
(state, disease, test) in the schema consumed by
``make_validation_report.py``:

    state, disease, test, season, model, wis, mae,
    coverage_50, coverage_95, n_eval_weeks

The final 2026-2027 season has no observations yet and is not scored.
For the Mosqlimate baseline, local scores may differ slightly from the
platform's own scores (the registry uses its own case counts).

Usage::

    python scripts/score_forecast_csvs.py                      # imdc_bb
    python scripts/score_forecast_csvs.py --model timesfm
    python scripts/score_forecast_csvs.py --model imdc_bb --model timesfm
    python scripts/score_forecast_csvs.py --backtest-dir validation_results/backtest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.evaluation.metrics import evaluate_forecast
from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.validation.config import get_validation_config

ROOT = Path(__file__).resolve().parents[1]


def score_series(fc_path: Path, actual: pd.DataFrame, test) -> dict:
    """Mirror run_single_backtest._evaluate for one forecast CSV."""
    target_dates = pd.date_range(test.target_start, periods=52, freq="7D")
    y_true = (
        actual.assign(date=pd.to_datetime(actual["date"]))
        .set_index("date")["casos"]
        .astype(float)
        .reindex(target_dates)
    )

    fc = pd.read_csv(fc_path)
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc.set_index("date").sort_index()
    f_win = fc[fc.index.isin(target_dates)]
    y_obs_idx = y_true[y_true.notna()].index
    overlap = f_win.index.intersection(y_obs_idx)
    if f_win.empty or not len(overlap):
        return {}
    iv = quantiles_to_intervals(f_win.loc[overlap].reset_index(names="date"))
    return {
        "metrics": evaluate_forecast(y_true.loc[overlap].values, iv),
        "n_eval_weeks": len(overlap),
    }


def score_model(model: str, backtest_dir: Path, loader=None) -> pd.DataFrame:
    """Score every forecast CSV of one model; writes {model}_scores.json."""
    if loader is None:
        loader = CompetitionDataLoader()
    fc_dir = backtest_dir / "forecasts"
    tests = get_validation_config().validation_tests

    rows = []
    actual, loaded_key = None, None
    for path in sorted(fc_dir.glob(f"*_{model}.csv.gz")):
        # stem = {uf}_{disease}_{test}_{model}; the model name may
        # itself contain underscores, so strip it as a whole suffix
        stem = path.name[: -len(".csv.gz")]
        core = stem[: -len(f"_{model}")]
        uf_disease, test_key = core.rsplit("_", 1)
        if test_key not in {str(t.test_number) for t in tests}:
            continue  # final season: no observations yet
        uf, disease = uf_disease.split("_", 1)
        test = next(t for t in tests if str(t.test_number) == test_key)

        # one full-history load per (uf, disease); sliced per test below
        if (uf, disease) != loaded_key:
            actual = loader.load_state_data(uf, disease=disease)
            loaded_key = (uf, disease)
        scored = score_series(path, actual, test)
        metrics = scored.get("metrics", {})
        rows.append(
            {
                "state": uf,
                "disease": disease,
                "test": test_key,
                "season": test.season,
                "model": model,
                "wis": metrics.get("wis_total"),
                "mae": metrics.get("mae"),
                "coverage_50": metrics.get("coverage_50"),
                "coverage_95": metrics.get("coverage_95"),
                "n_eval_weeks": scored.get("n_eval_weeks", 0),
            }
        )

    out_path = backtest_dir / f"{model}_scores.json"
    out_path.write_text(json.dumps(rows, indent=2))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score forecast CSVs of extra models.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model name to score (repeatable; default: imdc_bb)",
    )
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=ROOT / "validation_results" / "backtest",
    )
    args = parser.parse_args()

    models = args.models or ["imdc_bb"]
    loader = CompetitionDataLoader()
    for model in models:
        df = score_model(model, args.backtest_dir, loader=loader)
        out_path = args.backtest_dir / f"{model}_scores.json"
        print(f"[{model}] scored {len(df)} forecasts -> {out_path}")
        if not df.empty:
            summary = df.groupby(["disease", "test"]).agg(
                n=("wis", "size"), eval_weeks=("n_eval_weeks", "sum"), mean_wis=("wis", "mean")
            )
            print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
