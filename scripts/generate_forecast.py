"""Generate competition forecasts for a target season.

Trains the selected model per state/disease on all data up to the
cutoff (default: EW25 2026 for the final 2026-27 season), produces the
calibrated 67-step probabilistic forecast (15-week gap + 52 target
weeks), and saves the 52 target-window weeks as quantile CSVs.

Model choice per state defaults to the skill-gated selection from the
latest backtest run (``validation_results/backtest``); override with
``--model``.

Usage:
    python scripts/generate_forecast.py \
        [--states SP,RJ] [--diseases dengue,chikungunya] \
        [--model ens_qavg] [--cutoff 2026-06-21] \
        [--out forecasts/final] [--no-calibrate] [--workers 4] \
        [--include-tft] [--timesfm]
"""

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_forecast")

from mosqlimate_ai.data.completeness import warn_missing_weeks
from mosqlimate_ai.data.future_exog import ExogLookup
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.validation.backtest import (
    _final_test_config,
    default_model_registry,
    run_single_backtest,
)
from mosqlimate_ai.validation.selection import select_from_backtest_dir
from mosqlimate_ai.validation.tuning import load_cached_params

TARGET_WEEKS = 52


def slice_target(forecast: pd.DataFrame, target_start: str) -> pd.DataFrame:
    idx = pd.to_datetime(forecast["date"])
    mask = (idx >= pd.Timestamp(target_start)) & (
        idx < pd.Timestamp(target_start) + pd.Timedelta(weeks=TARGET_WEEKS)
    )
    return forecast[mask].reset_index(drop=True)


def tuned_params_for(cache_dir: Path, uf: str, disease: str, family: str):
    cached = load_cached_params(cache_dir, uf, disease, family)
    if not cached:
        return None
    out = dict(cached)
    hl = out.pop("recency_halflife_weeks", 104)
    out["recency_halflife_weeks"] = None if hl in (0, "none") else hl
    return out


def main() -> None:

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", default=None, help="Comma-separated UFs (default: all)")
    ap.add_argument("--diseases", default="dengue,chikungunya")
    ap.add_argument("--model", default=None,
                    help="Model name (default: skill-gated selection from the backtest)")
    ap.add_argument("--cutoff", default=None,
                    help="Training cutoff YYYY-MM-DD (default: EW25 2026)")
    ap.add_argument("--target-start", default=None,
                    help="Target season start (default: EW41 2026)")
    ap.add_argument("--out", default="forecasts/final")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip conformal recalibration")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include-tft", action="store_true")
    ap.add_argument("--timesfm", action="store_true",
                    help="Add the zero-shot TimesFM foundation model to the "
                         "registry (checkpoint downloaded on first use)")
    args = ap.parse_args()

    cfg_test = _final_test_config()
    cutoff = args.cutoff or cfg_test.train_end
    target_start = args.target_start or cfg_test.target_start
    states = args.states.split(",") if args.states else None
    diseases = tuple(d.strip() for d in args.diseases.split(",") if d.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backtest_dir = Path("validation_results/backtest")
    selections = None
    if backtest_dir.exists():
        try:
            selections = select_from_backtest_dir(backtest_dir)
        except Exception as exc:
            logger.warning("selection unavailable: %s", exc)

    loader = CompetitionDataLoader()
    for disease in diseases:
        try:
            states_data = loader.load_all_states(aggregate=True, disease=disease)
        except FileNotFoundError:
            logger.warning("skipping %s: data unavailable", disease)
            continue

        for uf in (states or sorted(states_data)):
            if uf not in states_data:
                logger.warning("no %s data for %s", disease, uf)
                continue
            state_df = states_data[uf]
            state_df["date"] = pd.to_datetime(state_df["date"])
            train = state_df[state_df["date"] <= pd.Timestamp(cutoff)]
            if len(train) < 60:
                logger.warning("%s/%s: insufficient training rows", uf, disease)
                continue

            warn_missing_weeks(train, f"{uf}/{disease}", context="forecast training window",
                               end=cutoff)

            chosen = args.model
            if chosen is None and selections is not None:
                sel = selections[
                    (selections.state == uf) & (selections.disease == disease)
                ]
                chosen = sel.iloc[0].selected if len(sel) else "ens_qavg"
            chosen = chosen or "ens_qavg"

            lookup = ExogLookup(loader, uf, train_end=pd.Timestamp(cutoff))
            params_by_model = {}
            for family in ("xgb_direct", "lgbm_direct"):
                tp = tuned_params_for(backtest_dir / "hyperparams", uf, disease, family)
                if tp:
                    params_by_model[family] = tp

            registry = default_model_registry(
                exog_lookup=lookup,
                include_tft=args.include_tft,
                include_timesfm=args.timesfm,
                params_by_model=params_by_model,
            )
            registry.pop("seas_naive", None)  # ensembles exclude it anyway

            result = run_single_backtest(
                uf,
                cfg_test,
                disease=disease,
                model_registry=registry,
                calibrate=not args.no_calibrate,
                train_df=train,
                actual_df=train.iloc[0:0],  # no actuals beyond the cutoff
                future_exog=None,
            )

            # choose the forecast: requested/selected model, else best ensemble
            models = result.get("models", {})
            fc_entry = models.get(chosen, {}).get("forecast")
            if fc_entry is None:
                for fallback in ("ens_qavg", "ens_median"):
                    fc_entry = models.get(fallback, {}).get("forecast")
                    if fc_entry is not None:
                        logger.info("%s/%s: %s unavailable, using %s",
                                    uf, disease, chosen, fallback)
                        chosen = fallback
                        break
            if fc_entry is None:
                logger.error("%s/%s: no forecast produced", uf, disease)
                continue

            fc = fc_entry
            win = slice_target(fc, target_start)
            dest = out_dir / disease / f"{uf}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            win.to_csv(dest, index=False)
            logger.info("%s/%s: %s -> %s (%d weeks)",
                        uf, disease, chosen, dest, len(win))

    print(f"Forecasts written under {out_dir}/")


if __name__ == "__main__":
    main()
