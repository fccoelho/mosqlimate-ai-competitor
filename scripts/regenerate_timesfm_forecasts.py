"""Regenerate the TimesFM forecasts with covariates.

Re-runs the TimesFM backtest jobs (validation tests 1-4 + the final
season) for every competition state and disease through the standard
``run_single_backtest`` pipeline — same training windows, same conformal
calibration, same evaluation — but with a registry containing ONLY the
(now covariate-aware) TimesFM model, so the other models' saved results
are untouched. The calibrated forecasts overwrite
``<backtest_dir>/forecasts/{uf}_{disease}_{test}_timesfm.csv.gz``.

Afterwards refresh the dependent artifacts::

    python scripts/score_forecast_csvs.py --model timesfm --model timesfm_base
    python scripts/rebuild_ensembles_with_timesfm.py
    python scripts/score_forecast_csvs.py --model ens_qavg_tf --model ens_median_tf
    python scripts/make_validation_report.py

Usage::

    python scripts/regenerate_timesfm_forecasts.py            # all states
    python scripts/regenerate_timesfm_forecasts.py --states SP,RJ
    python scripts/regenerate_timesfm_forecasts.py --workers 3
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regenerate_timesfm")

from mosqlimate_ai.validation.backtest import (  # noqa: E402
    _final_test_config,
    run_single_backtest,
)
from mosqlimate_ai.validation.config import get_validation_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _job(args: tuple) -> dict:
    """Worker: one (uf, disease, test) TimesFM-only backtest job."""
    uf, disease, test_key, train_df, actual_df, tmp_dir = args
    from mosqlimate_ai.data.future_exog import ExogLookup
    from mosqlimate_ai.data.loader import CompetitionDataLoader
    from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster

    cfg = get_validation_config()
    test = (
        _final_test_config()
        if test_key == "final"
        else next(t for t in cfg.validation_tests if str(t.test_number) == test_key)
    )

    loader = CompetitionDataLoader()
    lookup = ExogLookup(loader, uf, train_end=pd.Timestamp(test.train_end))
    registry = {"timesfm": TimesFMForecaster(exog_lookup=lookup)}

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
    model_res = result["models"].get("timesfm", {})
    fc = model_res.get("forecast")
    if fc is None or not len(fc):
        return {"key": (uf, disease, test_key), "error": model_res.get("error", "no forecast")}
    fc.to_csv(
        Path(tmp_dir) / f"{uf}_{disease}_{test_key}_timesfm.csv.gz",
        index=False,
        date_format="%Y-%m-%d",
    )
    return {
        "key": (uf, disease, test_key),
        "wis": (model_res.get("metrics") or {}).get("wis_total"),
        "n_eval_weeks": model_res.get("n_eval_weeks", 0),
        "runtime_s": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate TimesFM forecasts with covariates.")
    parser.add_argument(
        "--backtest-dir", type=Path, default=ROOT / "validation_results" / "backtest"
    )
    parser.add_argument("--states", help="Comma-separated UF list (default: all)")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    cfg = get_validation_config()
    states = [s.strip() for s in args.states.split(",")] if args.states else list(cfg.states)

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    out_dir = (args.backtest_dir / "forecasts").resolve()
    tmp = ROOT / "forecasts_tmp"
    tmp.mkdir(exist_ok=True)

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
                train_end = pd.Timestamp(test.train_end)
                train_df = df[df["date"] <= train_end]
                actual_df = df[
                    (df["date"] > train_end) & (df["date"] <= pd.Timestamp(test.target_end))
                ]
                jobs.append((uf, disease, test_key, train_df, actual_df, str(tmp)))
        loader._merged_cache.clear()

    logger.info("prepared %d timesfm jobs", len(jobs))
    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_job, j): j[:3] for j in jobs}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if "error" in res:
                    failed.append((res["key"], res["error"]))
                    logger.error("failed %s: %s", res["key"], res["error"])
                else:
                    done += 1
                    if done % 20 == 0:
                        logger.info(
                            "done %d/%d (last %s wis=%s)", done, len(jobs), res["key"], res["wis"]
                        )
            except Exception as exc:
                failed.append((futures[fut], str(exc)))
                logger.exception("job %s crashed", futures[fut])

    # move the freshly written CSVs into the forecasts directory
    import shutil

    for path in tmp.glob("*_timesfm.csv.gz"):
        shutil.move(str(path), out_dir / path.name)
    tmp.rmdir()

    print(f"regenerated {done} timesfm forecasts in {out_dir}")
    if failed:
        print(f"{len(failed)} failures:")
        for key, err in failed:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
