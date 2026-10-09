"""Rebuild the deterministic quantile ensembles with TimesFM included.

The backtest JSONs currently on disk came from runs without TimesFM in
the registry (it used to be opt-in), so their ``ens_qavg`` /
``ens_median`` / ``ens_vote`` rows exclude it. This script rebuilds the
two deterministic ensembles from the calibrated member forecast CSVs
saved under ``<backtest_dir>/forecasts/``, exactly mirroring the
construction in ``run_single_backtest``:

- ``ens_qavg_tf``: equal-weight quantile average of all members except
  ``seas_naive`` (xgb_direct, lgbm_direct, loglin_trend, timesfm)
- ``ens_median_tf``: per-date median of the same members

Members missing on disk are skipped, as unfit members are at runtime;
fewer than two available members skips the combination.

The voting ensemble (``ens_vote``) needs per-member calibration WIS that
was never recorded for TimesFM and cannot be rebuilt; re-run the
backtests to refresh all ensembles natively now that TimesFM is in the
default registry (``scripts/run_backtests.py``).

After rebuilding, score the new ensembles with::

    python scripts/score_forecast_csvs.py --model ens_qavg_tf --model ens_median_tf

Usage::

    python scripts/rebuild_ensembles_with_timesfm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mosqlimate_ai.validation.config import get_validation_config

MEMBERS = ["xgb_direct", "lgbm_direct", "loglin_trend", "timesfm"]
ENSEMBLES = ["ens_qavg_tf", "ens_median_tf"]
ROOT = Path(__file__).resolve().parents[1]


def rebuild_combination(fc_dir: Path, uf: str, disease: str, test_key: str) -> int:
    """Build and write the _tf ensembles for one (uf, disease, test)."""
    frames = {}
    for member in MEMBERS:
        path = fc_dir / f"{uf}_{disease}_{test_key}_{member}.csv.gz"
        if not path.exists():
            continue
        f = pd.read_csv(path)
        f["date"] = pd.to_datetime(f["date"])
        frames[member] = f.set_index("date").sort_index()
    if len(frames) < 2:
        return 0

    # union grid: identical to runtime behavior when member indexes match
    grid = frames[next(iter(frames))].index
    for f in frames.values():
        grid = grid.union(f.index)
    aligned = [f.reindex(grid) for f in frames.values()]

    qavg = sum(aligned) / len(aligned)
    qavg.index.name = "date"
    median = pd.concat(aligned).groupby(level=0).median()
    median.index.name = "date"

    qavg.to_csv(fc_dir / f"{uf}_{disease}_{test_key}_ens_qavg_tf.csv.gz", date_format="%Y-%m-%d")
    median.to_csv(
        fc_dir / f"{uf}_{disease}_{test_key}_ens_median_tf.csv.gz", date_format="%Y-%m-%d"
    )
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ensembles including TimesFM.")
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=ROOT / "validation_results" / "backtest",
    )
    args = parser.parse_args()

    fc_dir = args.backtest_dir / "forecasts"
    known_tests = {str(t.test_number) for t in get_validation_config().validation_tests}
    known_tests.add("final")

    built, incomplete = 0, []
    for path in sorted(fc_dir.glob("*_timesfm.csv.gz")):
        # stem = {uf}_{disease}_{test}_timesfm
        stem = path.name[: -len(".csv.gz")]
        uf_disease, test_key = stem[: -len("_timesfm")].rsplit("_", 1)
        if test_key not in known_tests:
            continue
        uf, disease = uf_disease.split("_", 1)
        n = rebuild_combination(fc_dir, uf, disease, test_key)
        if n:
            built += 1
            if n < len(MEMBERS):
                incomplete.append(f"{uf}/{disease}/t{test_key}: {n}/{len(MEMBERS)} members")

    print(f"built {built} x {len(ENSEMBLES)} ensemble files in {fc_dir}")
    if incomplete:
        print("combinations with missing members:")
        for note in incomplete:
            print(f"  {note}")


if __name__ == "__main__":
    main()
