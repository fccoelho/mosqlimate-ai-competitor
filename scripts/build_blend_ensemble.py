"""Build the cross-family blend ensemble (``ens_blend``).

Quantile-averages the strongest univariate foundation model
(``timesfm_base``) with the strongest GBM-side ensemble per disease —
``ens_median`` for chikungunya, ``ens_qavg`` for dengue (the best
non-TimesFM ensembles in the validation report) — from the calibrated
member forecast CSVs saved under ``<backtest_dir>/forecasts/``,
mirroring the runtime ensemble construction (union-date alignment,
mean of quantile frames).

Score it with::

    python scripts/score_forecast_csvs.py --model ens_blend
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# per-disease GBM-side ensemble member (best non-TimesFM ensemble)
BLEND_MEMBERS = {
    "dengue": ("timesfm_base", "ens_qavg"),
    "chikungunya": ("timesfm_base", "ens_median"),
}
TEST_KEYS = ("1", "2", "3", "4", "final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ens_blend forecast CSVs.")
    parser.add_argument(
        "--backtest-dir", type=Path, default=ROOT / "validation_results" / "backtest"
    )
    parser.add_argument("--states", help="Comma-separated UF list (default: all)")
    args = parser.parse_args()

    fc_dir = args.backtest_dir / "forecasts"
    states = [s.strip() for s in args.states.split(",")] if args.states else None

    built, missing = 0, []
    for path in sorted(fc_dir.glob("*_timesfm_base.csv.gz")):
        stem = path.name[: -len(".csv.gz")]
        uf_disease, test_key = stem[: -len("_timesfm_base")].rsplit("_", 1)
        if test_key not in TEST_KEYS:
            continue
        uf, disease = uf_disease.split("_", 1)
        if states and uf not in states:
            continue
        members = BLEND_MEMBERS[disease]
        frames = []
        ok = True
        for member in members:
            mpath = fc_dir / f"{uf}_{disease}_{test_key}_{member}.csv.gz"
            if not mpath.exists():
                ok = False
                break
            f = pd.read_csv(mpath)
            f["date"] = pd.to_datetime(f["date"])
            frames.append(f.set_index("date").sort_index())
        if not ok:
            missing.append(f"{uf}/{disease}/t{test_key}")
            continue

        grid = frames[0].index.union(frames[1].index)
        blend = sum(f.reindex(grid) for f in frames) / len(frames)
        blend.index.name = "date"
        blend.to_csv(fc_dir / f"{uf}_{disease}_{test_key}_ens_blend.csv.gz", date_format="%Y-%m-%d")
        built += 1

    print(f"built {built} ens_blend files in {fc_dir} (members: {BLEND_MEMBERS})")
    if missing:
        print(f"{len(missing)} combinations with missing members:")
        for m in missing[:10]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
