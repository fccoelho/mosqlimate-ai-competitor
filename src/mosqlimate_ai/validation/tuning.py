"""Per-state hyperparameter tuning for the GBM forecasters.

Strategy
--------
Hyperparameters are a property of a state's series (scale, seasonality,
noise regime), not of a specific season. Each (state, disease, model
family) is therefore tuned **once** against the earliest validation
setup and the result is cached to disk and reused across all splits
and future runs.

Criterion: WIS on a held-out calibration window with the exact
deployment structure — train on data up to ``train_end - calib_weeks``,
forecast the following ``calib_weeks`` (15-week gap + 52 target weeks),
score against the observations. This mirrors the conformal calibration
window, so the tuning signal reflects long-horizon performance rather
than one-step skill.

Search: seeded random search over compact GBM ranges (robust for
quantile boosting, no gradient-modelling overhead), plus a recency
half-life choice. Trials fit reduced trees (``search_n_estimators``)
for speed; the winning configuration is rescaled to the production
tree count proportionally to the learning rate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.metrics import evaluate_forecast
from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals

logger = logging.getLogger(__name__)

CALIB_WEEKS = 67
DEFAULT_TRIALS = 12
SEARCH_N_ESTIMATORS = 200

# Fixed production defaults (used when tuning is disabled)
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "recency_halflife_weeks": 104,
}


def search_space(rng: np.random.Generator) -> Dict:
    """Draw one hyperparameter configuration for a GBM forecaster.

    ``recency_halflife_weeks`` uses sentinel ``0`` for "no weighting".
    """
    return {
        "n_estimators": SEARCH_N_ESTIMATORS,
        "max_depth": int(rng.choice([3, 4, 5, 6, 7])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
        "min_child_weight": float(rng.choice([1.0, 2.0, 5.0, 10.0, 20.0])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "recency_halflife_weeks": int(rng.choice([0, 52, 104, 208])),
    }


def effective_params(params: Dict) -> Dict:
    """Translate search-space sentinels into forecaster kwargs."""
    out = dict(params)
    if out.get("recency_halflife_weeks") in (0, "none", None):
        out["recency_halflife_weeks"] = None
    else:
        out["recency_halflife_weeks"] = int(out["recency_halflife_weeks"])
    return out


def _tuning_split(
    train_df: pd.DataFrame, calib_weeks: int = CALIB_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit frame, observed frame) for the tuning window."""
    df = train_df.sort_values("date").reset_index(drop=True)
    calib_end = df["date"].max()
    fit_end = calib_end - pd.Timedelta(weeks=calib_weeks)
    fit = df[df["date"] <= fit_end]
    observed = df[(df["date"] > fit_end) & (df["date"] <= calib_end)]
    return fit, observed


def _score_params(
    model_factory: Callable[[Dict], object],
    params: Dict,
    train_df: pd.DataFrame,
    calib_weeks: int,
    booster_n_jobs: int = 4,
) -> float:
    """WIS of one configuration on the held-out calibration window.

    Ranks configurations by raw WIS (the additive conformal correction
    is skipped here — it is monotone across configurations and would
    double the cost of every trial). Returns ``np.inf`` on failure.
    """
    fit, observed = _tuning_split(train_df, calib_weeks)
    if len(fit) < 60 or observed.empty:
        return np.inf

    model = model_factory(effective_params(params))
    try:
        model.fit(fit)
        forecast = model.predict(calib_weeks)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("trial failed: %s", exc)
        return np.inf

    y = observed.set_index("date")[model.target_col].astype(float)
    common = forecast.index.intersection(y.dropna().index)
    if len(common) < 8:
        return np.inf

    f_obs = forecast.loc[common]
    iv = quantiles_to_intervals(f_obs.reset_index(names="date"))
    metrics = evaluate_forecast(y.loc[common].values, iv)
    wis = metrics.get("wis_total")
    return float(wis) if wis is not None and np.isfinite(wis) else np.inf


def tune_state_model(
    uf: str,
    disease: str,
    model_name: str,
    model_factory: Callable[[Dict], object],
    train_df: pd.DataFrame,
    n_trials: int = DEFAULT_TRIALS,
    calib_weeks: int = CALIB_WEEKS,
    booster_n_jobs: int = 4,
    seed: int = 42,
) -> Dict:
    """Random-search tuning for one (state, disease, model family).

    Args:
        model_factory: callable ``params -> unfitted forecaster``.
        train_df: full training frame for the earliest test (train data
            only; the function holds out the last ``calib_weeks``).
        n_trials: number of random configurations to evaluate.

    Returns:
        Dict with ``best_params`` (production tree count), ``search``,
        ``wis``, and the trial log.
    """
    rng = np.random.default_rng(seed)
    trials: List[Dict] = []

    # baseline first, so tuning can only improve on it
    base = dict(DEFAULT_PARAMS)
    base_wis = _score_params(
        model_factory, base, train_df, calib_weeks, booster_n_jobs
    )
    trials.append({"params": base, "wis": base_wis})
    best_params, best_wis = base, base_wis
    logger.info("[%s/%s/%s] baseline WIS %.1f", uf, disease, model_name, base_wis)

    for i in range(n_trials):
        params = search_space(rng)
        wis = _score_params(
            model_factory, params, train_df, calib_weeks, booster_n_jobs
        )
        eff = effective_params(params)
        trials.append({"params": eff, "wis": wis})
        if wis < best_wis:
            best_wis, best_params = wis, eff
            logger.info(
                "[%s/%s/%s] trial %d WIS %.1f *", uf, disease, model_name, i + 1, wis
            )

    # rescale trees to production budget, inversely proportional to lr
    production = dict(best_params)
    if production.get("n_estimators") == SEARCH_N_ESTIMATORS:
        lr = max(production.get("learning_rate", 0.05), 1e-3)
        scale = min(2.0, max(0.75, 0.05 / lr))
        production["n_estimators"] = int(min(900, SEARCH_N_ESTIMATORS * scale))

    return {
        "state": uf,
        "disease": disease,
        "model": model_name,
        "best_params": production,
        "wis": best_wis,
        "baseline_wis": base_wis,
        "n_trials": n_trials,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: Path, uf: str, disease: str, model_name: str) -> Path:
    d = Path(cache_dir) / "hyperparams"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{uf}_{disease}_{model_name}.json"


def load_cached_params(
    cache_dir: Path, uf: str, disease: str, model_name: str
) -> Optional[Dict]:
    path = cache_path(cache_dir, uf, disease, model_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("best_params")
    except Exception:  # pragma: no cover - defensive
        return None


def save_tuning_result(cache_dir: Path, result: Dict) -> Path:
    path = cache_path(cache_dir, result["state"], result["disease"], result["model"])
    slim = {k: v for k, v in result.items() if k != "trials"}
    slim["n_trials"] = result.get("n_trials")
    slim["trials_wis"] = [t["wis"] for t in result.get("trials", [])]
    path.write_text(json.dumps(slim, indent=2, default=str))
    return path


def tune_all_cached(
    cache_dir: Path,
    uf: str,
    disease: str,
    factories: Dict[str, Callable[[Dict], object]],
    train_df: pd.DataFrame,
    n_trials: int,
    booster_n_jobs: int = 4,
) -> Dict[str, Dict]:
    """Tune every model family that lacks a cache entry; return all params."""
    out: Dict[str, Dict] = {}
    for model_name, factory in factories.items():
        cached = load_cached_params(cache_dir, uf, disease, model_name)
        if cached is not None:
            out[model_name] = cached
            continue
        result = tune_state_model(
            uf,
            disease,
            model_name,
            factory,
            train_df,
            n_trials=n_trials,
            booster_n_jobs=booster_n_jobs,
        )
        save_tuning_result(cache_dir, result)
        out[model_name] = result["best_params"]
    return out
