"""Post-hoc interval widening screen (``*_recal`` variants).

Our submission candidates are systematically overconfident: 50%
intervals cover only ~34-39% of actuals (nominal 50%). This script
screens a multiplicative widening of the quantile spread around the
median,

    q'_tau = median + k * (q_tau - median),   k in a grid,

in a leakage-controlled way: the factor ``k*`` is chosen per (model,
disease) on the EARLY validation tests (1-2) only, and evaluated on
the held-back later tests (3-4). ``*_recal`` CSVs are written ONLY
where the held-out evaluation confirms the gain (eval WIS at ``k*``
beats ``k=1``); a ``k*`` that fails to transfer — e.g. dengue, where
the tune-set optimum over-widens explosively for later seasons — is
recorded in ``recal_factors.json`` as evidence but not deployed.

Outputs:
- ``<backtest_dir>/forecasts/*_{model}_recal.csv.gz``
- ``<backtest_dir>/recal_factors.json`` — chosen k and the tune/eval
  WIS evidence per (model, disease)

Score the recal variants with::

    python scripts/score_forecast_csvs.py --model timesfm_base_recal ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.evaluation.metrics import evaluate_forecast
from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals, widen_quantiles
from mosqlimate_ai.validation.config import get_validation_config

ROOT = Path(__file__).resolve().parents[1]
K_GRID = (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5)
TUNE_TESTS = ("1", "2")  # k* chosen here
EVAL_TESTS = ("3", "4")  # honest out-of-sample check
MODELS = ("timesfm_base", "ens_median", "ens_qavg", "ens_blend")


def wis_of(
    fc: pd.DataFrame, y_true: pd.Series, target_dates: pd.DatetimeIndex
) -> tuple[float, int]:
    f = fc.set_index("date")
    f = f[f.index.isin(target_dates)]
    obs = y_true[y_true.notna()]
    overlap = f.index.intersection(obs.index)
    if not len(overlap):
        return np.nan, 0
    iv = quantiles_to_intervals(f.loc[overlap].reset_index(names="date"))
    return float(evaluate_forecast(y_true.loc[overlap].values, iv)["wis_total"]), len(overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interval-widening screen for submission models.")
    parser.add_argument(
        "--backtest-dir", type=Path, default=ROOT / "validation_results" / "backtest"
    )
    parser.add_argument("--models", help=f"Comma-separated subset of {','.join(MODELS)}")
    args = parser.parse_args()
    models = tuple(m.strip() for m in args.models.split(",")) if args.models else MODELS

    fc_dir = args.backtest_dir / "forecasts"
    tests = {str(t.test_number): t for t in get_validation_config().validation_tests}

    loader = CompetitionDataLoader()
    actual_cache: dict[tuple, pd.DataFrame] = {}

    def actuals_for(uf: str, disease: str) -> pd.DataFrame:
        if (uf, disease) not in actual_cache:
            actual_cache[(uf, disease)] = loader.load_state_data(uf, disease=disease)
        return actual_cache[(uf, disease)]

    factors: dict[str, dict] = {}
    for model in models:
        per_disease: dict[str, dict] = {}
        for path in sorted(fc_dir.glob(f"*_{model}.csv.gz")):
            stem = path.name[: -len(".csv.gz")]
            uf_disease, test_key = stem[: -len(f"_{model}")].rsplit("_", 1)
            if test_key not in tests:
                continue  # final season: no observations
            uf, disease = uf_disease.split("_", 1)
            test = tests[test_key]
            target_dates = pd.date_range(test.target_start, periods=52, freq="7D")
            actual = actuals_for(uf, disease)
            y = (
                actual.assign(date=pd.to_datetime(actual["date"]))
                .set_index("date")["casos"]
                .astype(float)
                .reindex(target_dates)
            )
            fc0 = pd.read_csv(path)
            fc0["date"] = pd.to_datetime(fc0["date"])

            if disease not in per_disease:
                per_disease[disease] = {
                    "tune": {k: [] for k in K_GRID},
                    "eval": {k: [] for k in K_GRID},
                }
            bucket = per_disease[disease]
            for k in K_GRID:
                w, _ = wis_of(widen_quantiles(fc0, k), y, target_dates)
                (bucket["tune"] if test_key in TUNE_TESTS else bucket["eval"])[k].append(w)

        for disease, bucket in per_disease.items():
            tune_mean = {k: float(np.nanmean(v)) for k, v in bucket["tune"].items() if v}
            k_star = min(tune_mean, key=tune_mean.get)
            eval_mean = {k: float(np.nanmean(v)) for k, v in bucket["eval"].items() if v}
            factors[f"{model}/{disease}"] = {
                "k": k_star,
                "deployed": bool(eval_mean.get(k_star, np.inf) < eval_mean.get(1.0, np.inf)),
                "tune_wis": {str(k): round(v, 2) for k, v in tune_mean.items()},
                "eval_wis": {str(k): round(v, 2) for k, v in eval_mean.items()},
                "eval_wis_at_k": round(eval_mean.get(k_star, np.nan), 2),
                "eval_wis_at_1": round(eval_mean.get(1.0, np.nan), 2),
            }

        # write recalibrated CSVs only where the held-out eval confirmed
        # the gain (all tests + final use the early-tuned k*)
        for path in sorted(fc_dir.glob(f"*_{model}.csv.gz")):
            stem = path.name[: -len(".csv.gz")]
            uf_disease, test_key = stem[: -len(f"_{model}")].rsplit("_", 1)
            disease = uf_disease.split("_", 1)[1]
            entry = factors.get(f"{model}/{disease}")
            if entry is None or not entry["deployed"]:
                continue
            fc0 = pd.read_csv(path)
            fc0["date"] = pd.to_datetime(fc0["date"])
            widen_quantiles(fc0, entry["k"]).to_csv(
                fc_dir / f"{uf_disease}_{test_key}_{model}_recal.csv.gz",
                index=False,
                date_format="%Y-%m-%d",
            )

    out = args.backtest_dir / "recal_factors.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(factors)
    out.write_text(json.dumps(existing, indent=2))

    print(f"k factors -> {out}")
    for key, entry in sorted(factors.items()):
        gain = (
            100 * (1 - entry["eval_wis_at_k"] / entry["eval_wis_at_1"])
            if entry["eval_wis_at_1"]
            else 0
        )
        print(
            f"  {key}: k*={entry['k']}  "
            f"tune WIS {entry['tune_wis'][str(entry['k'])]} (k=1: {entry['tune_wis']['1.0']})  "
            f"eval WIS {entry['eval_wis_at_k']} vs {entry['eval_wis_at_1']} at k=1 ({gain:+.1f}%)"
        )


if __name__ == "__main__":
    main()
