"""Generate competition forecasts for a target season.

Trains the selected model per state/disease on all data up to the
cutoff (default: EW25 2026 for the final 2026-27 season), produces the
calibrated 67-step probabilistic forecast (15-week gap + 52 target
weeks), and saves the 52 target-window weeks as quantile CSVs.

Model choice per state resolves in order: ``--model``, the per-disease
deployment recipe (``validation_results/backtest/deployment_recipe.json``,
written from the validation screens), the skill-gated selection from
the latest backtest run, and finally ``ens_qavg``. Recipe models know
their members: ``timesfm_base`` is the univariate TimesFM, ``ens_*``
ensembles are generated from the exact registry they were validated
with, and ``ens_blend`` averages the univariate TimesFM with the
per-disease GBM ensemble. A recipe ``recal_k`` > 1 widens the final
quantile spread around the median.

Usage:
    python scripts/generate_forecast.py \
        [--states SP,RJ] [--diseases dengue,chikungunya] \
        [--model ens_qavg] [--cutoff 2026-06-21] \
        [--out forecasts/final] [--no-calibrate] [--workers 4] \
        [--include-tft] [--timesfm/--no-timesfm]
"""

import argparse
import json
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
from mosqlimate_ai.evaluation.quantiles import widen_quantiles
from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster
from mosqlimate_ai.validation.backtest import (
    _final_test_config,
    default_model_registry,
    run_single_backtest,
)
from mosqlimate_ai.validation.selection import select_from_backtest_dir
from mosqlimate_ai.validation.tuning import load_cached_params

TARGET_WEEKS = 52
RECIPE_PATH = Path("validation_results/backtest/deployment_recipe.json")

# ens_blend members per disease (mirrors scripts/build_blend_ensemble.py)
BLEND_MEMBERS = {"dengue": "ens_qavg", "chikungunya": "ens_median"}


def load_recipe() -> dict:
    try:
        return json.loads(RECIPE_PATH.read_text())
    except FileNotFoundError:
        return {}


def registry_for(chosen: str, lookup, params_by_model: dict, include_tft: bool) -> dict:
    """Registry whose ensembles match how ``chosen`` was validated."""
    if chosen == "timesfm_base":
        # the univariate foundation model: no covariate lookup
        return {"timesfm_base": TimesFMForecaster(exog_lookup=None)}
    if chosen in ("ens_qavg", "ens_median", "ens_blend"):
        # validated ens_* were built from the registry WITHOUT timesfm;
        # ens_blend members are generated separately below
        return default_model_registry(
            exog_lookup=lookup,
            include_tft=include_tft,
            include_timesfm=False,
            params_by_model=params_by_model,
        )
    return default_model_registry(
        exog_lookup=lookup,
        include_tft=include_tft,
        include_timesfm=True,
        params_by_model=params_by_model,
    )


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
    ap.add_argument(
        "--model",
        default=None,
        help="Model name (default: skill-gated selection from the backtest)",
    )
    ap.add_argument(
        "--cutoff", default=None, help="Training cutoff YYYY-MM-DD (default: EW25 2026)"
    )
    ap.add_argument("--target-start", default=None, help="Target season start (default: EW41 2026)")
    ap.add_argument("--out", default="forecasts/final")
    ap.add_argument("--no-calibrate", action="store_true", help="Skip conformal recalibration")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include-tft", action="store_true")
    ap.add_argument(
        "--timesfm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include TimesFM in the default registry (default on; "
        "recipe ensembles always use their validated membership)",
    )
    args = ap.parse_args()

    cfg_test = _final_test_config()
    cutoff = args.cutoff or cfg_test.train_end
    target_start = args.target_start or cfg_test.target_start
    states = args.states.split(",") if args.states else None
    diseases = tuple(d.strip() for d in args.diseases.split(",") if d.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backtest_dir = Path("validation_results/backtest")
    recipe = load_recipe()
    if recipe:
        logger.info(
            "deployment recipe: %s",
            {d: recipe[d]["model"] for d in ("dengue", "chikungunya") if d in recipe},
        )
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

        for uf in states or sorted(states_data):
            if uf not in states_data:
                logger.warning("no %s data for %s", disease, uf)
                continue
            state_df = states_data[uf]
            state_df["date"] = pd.to_datetime(state_df["date"])
            train = state_df[state_df["date"] <= pd.Timestamp(cutoff)]
            if len(train) < 60:
                logger.warning("%s/%s: insufficient training rows", uf, disease)
                continue

            warn_missing_weeks(
                train, f"{uf}/{disease}", context="forecast training window", end=cutoff
            )

            chosen = args.model
            if chosen is None:
                recipe_entry = recipe.get(disease) or {}
                chosen = recipe_entry.get("model")
            if chosen is None and selections is not None:
                sel = selections[(selections.state == uf) & (selections.disease == disease)]
                chosen = sel.iloc[0].selected if len(sel) else None
            chosen = chosen or recipe.get("fallback") or "ens_qavg"

            lookup = ExogLookup(loader, uf, train_end=pd.Timestamp(cutoff))
            params_by_model = {}
            for family in ("xgb_direct", "lgbm_direct"):
                tp = tuned_params_for(backtest_dir / "hyperparams", uf, disease, family)
                if tp:
                    params_by_model[family] = tp

            registry = registry_for(chosen, lookup, params_by_model, args.include_tft)

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
            if chosen == "ens_blend":
                # blend = univariate TimesFM + the per-disease GBM ensemble
                tfm_reg = {"timesfm_base": TimesFMForecaster(exog_lookup=None)}
                tfm_res = run_single_backtest(
                    uf,
                    cfg_test,
                    disease=disease,
                    model_registry=tfm_reg,
                    calibrate=not args.no_calibrate,
                    train_df=train,
                    actual_df=train.iloc[0:0],
                    future_exog=None,
                )
                member = BLEND_MEMBERS.get(disease, "ens_qavg")
                f_ens = models.get(member, {}).get("forecast")
                f_tfm = tfm_res.get("models", {}).get("timesfm_base", {}).get("forecast")
                if f_ens is None or f_tfm is None:
                    logger.error("%s/%s: ens_blend member missing", uf, disease)
                    continue
                f_ens = f_ens.set_index("date")
                f_tfm = f_tfm.set_index("date")
                fc_entry = (f_ens + f_tfm.reindex(f_ens.index)) / 2.0
                fc_entry["date"] = fc_entry.index
                fc_entry = fc_entry.reset_index(drop=True)
            else:
                fc_entry = models.get(chosen, {}).get("forecast")
            if fc_entry is None:
                for fallback in ("ens_vote", "ens_qavg", "ens_median"):
                    fc_entry = models.get(fallback, {}).get("forecast")
                    if fc_entry is not None:
                        logger.info(
                            "%s/%s: %s unavailable, using %s", uf, disease, chosen, fallback
                        )
                        chosen = fallback
                        break
            if fc_entry is None:
                logger.error("%s/%s: no forecast produced", uf, disease)
                continue

            fc = fc_entry
            win = slice_target(fc, target_start)
            k = float((recipe.get(disease) or {}).get("recal_k", 1.0)) if not args.model else 1.0
            if k != 1.0:
                win = widen_quantiles(win, k)
            dest = out_dir / disease / f"{uf}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            win.to_csv(dest, index=False)
            logger.info(
                "%s/%s: %s (k=%.2f) -> %s (%d weeks)", uf, disease, chosen, k, dest, len(win)
            )

    print(f"Forecasts written under {out_dir}/")


if __name__ == "__main__":
    main()
