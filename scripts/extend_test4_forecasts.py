"""Regenerate test-4 forecast CSVs with the full 53-week platform window.

The platform's validation test 4 covers 2025-10-05 .. 2026-10-04
inclusive (53 weekly dates — one more than the 52-week grid the
backtests store), so submissions for test 4 need the extra week. This
script re-fits the submission model (univariate ``timesfm_base``) on
the test-4 training window with the same conformal calibration as the
backtests, and overwrites
``forecasts/{uf}_{disease}_4_timesfm_base.csv.gz`` with the 53-week
window.

Local scoring is unaffected: ``score_forecast_csvs`` reindexes to the
52-date evaluation grid, so the extra week never enters the metrics.

Usage::

    python scripts/extend_test4_forecasts.py [--states SP,RJ] [--workers 2]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extend_test4")

from mosqlimate_ai.evaluation.calibration import calibrate_forecast_scored  # noqa: E402
from mosqlimate_ai.validation.backtest import FULL_HORIZON  # noqa: E402
from mosqlimate_ai.validation.config import get_validation_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ("2025-10-05", "2026-10-04")  # 53 weekly dates, platform test 4


def _job(args: tuple) -> dict:
    uf, disease, train_df, tmp_dir, timesfm_device = args
    from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster

    dates = pd.date_range(WINDOW[0], WINDOW[1], freq="7D")
    assert len(dates) == 53

    train_df = train_df.sort_values("date").reset_index(drop=True)
    last_train = pd.Timestamp(train_df["date"].max())
    horizon = max(FULL_HORIZON, int(np.ceil((dates[-1] - last_train).days / 7)))

    model = TimesFMForecaster(exog_lookup=None, device=timesfm_device)
    fc, _ = calibrate_forecast_scored(model, train_df, horizon)
    fc.index = pd.to_datetime(fc.index)
    fc = fc.sort_index()

    win = fc.reindex(dates)
    if win["q500"].isna().any():
        return {"key": (uf, disease), "error": "forecast does not cover the 53-week window"}
    win.index.name = "date"
    model_name = "timesfm_base"
    win.to_csv(Path(tmp_dir) / f"{uf}_{disease}_4_{model_name}.csv.gz", date_format="%Y-%m-%d")
    return {"key": (uf, disease), "model": model_name, "rows": len(win)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend test-4 forecasts to 53 weeks.")
    parser.add_argument("--states", help="Comma-separated UF list (default: all)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timesfm-device", default="cuda")
    args = parser.parse_args()

    cfg = get_validation_config()
    states = [s.strip() for s in args.states.split(",")] if args.states else list(cfg.states)
    test4 = next(t for t in cfg.validation_tests if t.test_number == 4)

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    tmp = ROOT / "forecasts_tmp"
    tmp.mkdir(exist_ok=True)
    out_dir = (ROOT / "validation_results" / "backtest" / "forecasts").resolve()

    jobs = []
    loader = CompetitionDataLoader()
    for disease in ("dengue", "chikungunya"):
        all_states = loader.load_all_states(aggregate=True, disease=disease)
        for uf in states:
            if uf not in all_states:
                continue
            df = all_states[uf].copy()
            df["date"] = pd.to_datetime(df["date"])
            train_df = df[df["date"] <= pd.Timestamp(test4.train_end)]
            jobs.append((uf, disease, train_df, str(tmp), args.timesfm_device))
        loader._merged_cache.clear()

    logger.info("prepared %d test4-extension jobs", len(jobs))
    done, failures = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            if "error" in res:
                failures.append(res)
                logger.error("failed %s: %s", res["key"], res["error"])
            else:
                done += 1

    import shutil

    for path in tmp.glob("*_4_*.csv.gz"):
        shutil.move(str(path), out_dir / path.name)
    if tmp.exists() and not list(tmp.glob("*.csv.gz")):
        tmp.rmdir()

    print(f"extended {done} test-4 forecasts to 53 weeks in {out_dir}")
    if failures:
        for f in failures[:10]:
            print(f"  FAIL {f['key']}: {f['error']}")


if __name__ == "__main__":
    main()
