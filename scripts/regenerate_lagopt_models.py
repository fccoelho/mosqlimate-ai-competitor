"""Regenerate covariate-model forecasts with lag-optimized covariates.

Runs backtest jobs (validation tests 1-4 + final) for the three
covariate-consuming forecasters under ``*_lagopt`` names, feeding them
covariates that (a) enter at their estimated optimal climate->cases
lag and (b) survived the stepwise inclusion protocol (see
``mosqlimate_ai.data.lag_selection``). Uses the same
``run_single_backtest`` pipeline (windows, conformal calibration,
evaluation) as the standard backtests, so results are directly
comparable with the contemporaneous-covariate models.

Outputs:

- ``<backtest_dir>/forecasts/{uf}_{disease}_{test}_{model}_lagopt.csv.gz``
  for xgb_lagopt / lgbm_lagopt / timesfm_lagopt
- ``<backtest_dir>/lag_selection_summary.json`` — per (state, disease,
  test): estimated lags, lag correlations, and selected covariates

Then refresh the dependent artifacts::

    python scripts/score_forecast_csvs.py --model xgb_lagopt \
        --model lgbm_lagopt --model timesfm_lagopt
    python scripts/make_validation_report.py

Usage::

    python scripts/regenerate_lagopt_models.py            # all states
    python scripts/regenerate_lagopt_models.py --states SP,RJ --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regenerate_lagopt")

from mosqlimate_ai.validation.backtest import (  # noqa: E402
    _final_test_config,
    run_single_backtest,
)
from mosqlimate_ai.validation.config import get_validation_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("xgb_lagopt", "lgbm_lagopt", "timesfm_lagopt")


def _job(args: tuple) -> dict:
    """Worker: one (uf, disease, test) lagopt backtest job."""
    uf, disease, test_key, train_df, actual_df, tmp_dir, models, timesfm_device = args
    from mosqlimate_ai.data.future_exog import ExogLookup
    from mosqlimate_ai.data.lag_selection import optimize_exog_lookup
    from mosqlimate_ai.data.loader import CompetitionDataLoader
    from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster

    cfg = get_validation_config()
    test = (
        _final_test_config()
        if test_key == "final"
        else next(t for t in cfg.validation_tests if str(t.test_number) == test_key)
    )

    loader = CompetitionDataLoader()
    base_lookup = ExogLookup(loader, uf, train_end=pd.Timestamp(test.train_end))
    lookup, lag_info = optimize_exog_lookup(base_lookup, train_df)

    registry = {}
    if "xgb_lagopt" in models:
        from mosqlimate_ai.models.gbm_direct import XGBoostDirectForecaster

        registry["xgb_lagopt"] = XGBoostDirectForecaster(
            exog_lookup=lookup, params={"thread_budget": 1}
        )
    if "lgbm_lagopt" in models:
        from mosqlimate_ai.models.gbm_direct import LightGBMDirectForecaster

        registry["lgbm_lagopt"] = LightGBMDirectForecaster(
            exog_lookup=lookup, params={"thread_budget": 1}
        )
    if "timesfm_lagopt" in models:
        registry["timesfm_lagopt"] = TimesFMForecaster(exog_lookup=lookup, device=timesfm_device)

    start = time.time()
    result = run_single_backtest(
        uf,
        test,
        disease=disease,
        model_registry=registry,
        calibrate=True,
        train_df=train_df,
        actual_df=actual_df,
    )

    written, errors = [], {}
    for name in models:
        model_res = result["models"].get(name, {})
        fc = model_res.get("forecast")
        if fc is None or not len(fc):
            errors[name] = model_res.get("error", "no forecast")
            continue
        fc.to_csv(
            Path(tmp_dir) / f"{uf}_{disease}_{test_key}_{name}.csv.gz",
            index=False,
            date_format="%Y-%m-%d",
        )
        written.append(name)

    return {
        "key": [uf, disease, test_key],
        "lag_info": lag_info,
        "written": written,
        "errors": errors,
        "wis": {
            n: (result["models"].get(n, {}).get("metrics") or {}).get("wis_total") for n in models
        },
        "runtime_s": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate covariate models with lag-optimized covariates."
    )
    parser.add_argument(
        "--backtest-dir", type=Path, default=ROOT / "validation_results" / "backtest"
    )
    parser.add_argument("--states", help="Comma-separated UF list (default: all)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--models",
        help="Comma-separated subset of "
        + ",".join(MODELS)
        + " (default: all; split phases to cap GPU/RAM load, "
        "e.g. run the GBMs first, then timesfm alone)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run combinations whose CSVs already exist"
    )
    parser.add_argument(
        "--timesfm-device",
        default="cpu",
        help="Device for the TimesFM engine (cpu/cuda; default cpu to cap GPU memory)",
    )
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(",")) if args.models else MODELS
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown models: {unknown}")

    cfg = get_validation_config()
    states = [s.strip() for s in args.states.split(",")] if args.states else list(cfg.states)

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    out_dir = (args.backtest_dir / "forecasts").resolve()
    tmp = ROOT / "forecasts_tmp"
    tmp.mkdir(exist_ok=True)

    # recover finished forecasts from an interrupted run first, so the
    # default skip-existing logic can see them
    import shutil

    for path in tmp.glob("*_lagopt.csv.gz"):
        shutil.move(str(path), out_dir / path.name)

    def _done(uf: str, disease: str, test_key: str) -> bool:
        return all((out_dir / f"{uf}_{disease}_{test_key}_{m}.csv.gz").exists() for m in models)

    jobs = []
    loader = CompetitionDataLoader()
    for disease in ("dengue", "chikungunya"):
        all_states = loader.load_all_states(aggregate=True, disease=disease)
        for uf in states:
            if uf not in all_states:
                logger.warning("no %s data for %s; skipping", disease, uf)
                continue
            df = all_states[uf].copy()
            df["date"] = pd.to_datetime(df["date"])
            tests = [(str(t.test_number), t) for t in cfg.validation_tests]
            tests.append(("final", _final_test_config()))
            for test_key, test in tests:
                if not args.force and _done(uf, disease, test_key):
                    continue
                train_end = pd.Timestamp(test.train_end)
                train_df = df[df["date"] <= train_end]
                actual_df = df[
                    (df["date"] > train_end) & (df["date"] <= pd.Timestamp(test.target_end))
                ]
                jobs.append(
                    (
                        uf,
                        disease,
                        test_key,
                        train_df,
                        actual_df,
                        str(tmp),
                        models,
                        args.timesfm_device,
                    )
                )
        loader._merged_cache.clear()

    logger.info("prepared %d lagopt jobs (%s)", len(jobs), ",".join(models))
    summary_path = args.backtest_dir / "lag_selection_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    failures = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_job, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    res = fut.result()
                except Exception as exc:
                    failures.append(str(exc))
                    logger.exception("job crashed")
                    continue
                uf, disease, test_key = res["key"]
                summary[f"{uf}/{disease}/{test_key}"] = res["lag_info"]
                if res["errors"]:
                    failures.append(f"{res['key']}: {res['errors']}")
                if i % 20 == 0:
                    logger.info("done %d/%d", i, len(jobs))
                if i % 20 == 0 or i == len(jobs):
                    summary_path.write_text(json.dumps(summary, indent=2))

        for name in models:
            for path in tmp.glob(f"*_{name}.csv.gz"):
                shutil.move(str(path), out_dir / path.name)
        if tmp.exists() and not list(tmp.glob("*.csv.gz")):
            tmp.rmdir()

    args.backtest_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"lagopt forecasts written to {out_dir}; summary -> {summary_path.name}")
    if failures:
        print(f"{len(failures)} failures:")
        for f in failures[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
