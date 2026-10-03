"""Deterministic out-of-sample backtest harness.

Implements the IMDC validation protocol without the legacy LLM-agent
layer: for each state, disease and validation test, models are trained
strictly on data up to EW25 of the training year and asked for a 67-week
probabilistic forecast (15-week gap + 52 target weeks). Metrics are
computed on the full target window (EW41..EW40) with the official
Weighted Interval Score.

Results are plain dictionaries / parquet files; no agent chatter.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional  # noqa: F401 - used in annotations

import numpy as np
import pandas as pd

from mosqlimate_ai.data.completeness import warn_missing_weeks
from mosqlimate_ai.data.future_exog import build_future_exog
from mosqlimate_ai.evaluation.metrics import evaluate_by_horizon, evaluate_forecast
from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.models.base import BaseForecaster
from mosqlimate_ai.validation.config import get_validation_config

logger = logging.getLogger(__name__)

GAP_WEEKS = 15  # EW26..EW40 between train cutoff (EW25) and target start (EW41)
FULL_HORIZON = GAP_WEEKS + 52  # 67; target window = horizons 15..66


# ---------------------------------------------------------------------------
# Voting ensemble weights
# ---------------------------------------------------------------------------
def voting_weights(
    scores: dict[str, float],
    regularization: float = 0.1,
) -> dict[str, float]:
    """Inverse-skill voting weights for ensemble members.

    Each model's vote is proportional to ``1 / (score + reg)`` where
    ``reg = regularization * median(score)``. The shrinkage term keeps a
    single freak-perfect calibration window from monopolizing the vote
    (a raw inverse score would let one near-zero WIS take all the
    weight).

    Args:
        scores: Per-model skill scores (lower = better, e.g. WIS on the
            calibration window). NaN/inf entries are dropped.
        regularization: Shrinkage as a fraction of the median score.

    Returns:
        Normalized weights (sum to 1); empty when fewer than two valid
        scores remain.
    """
    valid = {m: float(s) for m, s in scores.items() if np.isfinite(s)}
    if len(valid) < 2:
        return {}
    reg = regularization * float(np.median(list(valid.values()))) + 1e-12
    inv = {m: 1.0 / (s + reg) for m, s in valid.items()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def default_model_registry(
    future_exog: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    include_tft: bool = False,
    include_timesfm: bool = False,
    exog_lookup=None,
    params_by_model: Optional[Dict[str, Dict]] = None,
) -> Dict[str, BaseForecaster]:
    """Default model zoo for backtests (each entry is an unfitted forecaster).

    Args:
        future_exog: Unused legacy hook (kept for API compatibility).
        params: Base GBM hyperparameters shared by both GBM families.
        include_tft: Add the (slower, GPU) TFT for extra diversity.
        include_timesfm: Add the zero-shot TimesFM foundation model
            (downloads a ~500 MB checkpoint on first use).
        exog_lookup: :class:`ExogLookup` for target-time known covariates.
        params_by_model: Optional per-model overrides (merged over
            ``params``). A ``recency_halflife_weeks`` key is routed to
            the forecaster constructor instead of the booster config.
    """
    from mosqlimate_ai.models.baselines import (
        LogLinearTrendForecaster,
        SeasonalNaiveForecaster,
    )
    from mosqlimate_ai.models.gbm_direct import (
        LightGBMDirectForecaster,
        XGBoostDirectForecaster,
    )

    base = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05}
    base.update(params or {})
    overrides = params_by_model or {}

    def _build(family: str, ctor) -> BaseForecaster:
        cfg = dict(base)
        cfg.update(overrides.get(family, {}))
        halflife = cfg.pop("recency_halflife_weeks", 104)
        return ctor(
            exog_lookup=exog_lookup,
            recency_halflife_weeks=None if halflife in (0, "none") else halflife,
            params=cfg,
        )

    registry = {
        "xgb_direct": _build("xgb_direct", XGBoostDirectForecaster),
        "lgbm_direct": _build("lgbm_direct", LightGBMDirectForecaster),
        "seas_naive": SeasonalNaiveForecaster(),
        "loglin_trend": LogLinearTrendForecaster(),
    }
    if include_tft:
        from mosqlimate_ai.models.tft_direct import TFTDirectForecaster

        registry["tft_direct"] = TFTDirectForecaster(epochs=15)
    if include_timesfm:
        from mosqlimate_ai.models.timesfm_forecaster import TimesFMForecaster

        registry["timesfm"] = TimesFMForecaster()
    return registry


# ---------------------------------------------------------------------------
# Single test backtest
# ---------------------------------------------------------------------------
def run_single_backtest(
    uf: str,
    test_config,
    disease: str = "dengue",
    loader=None,
    model_registry: dict[str, BaseForecaster] | None = None,
    use_future_exog: bool = True,
    calibrate: bool = True,
    calib_weeks: int = FULL_HORIZON,
    train_df: pd.DataFrame | None = None,
    actual_df: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
) -> dict:
    """Run one validation test for one state.

    Args:
        uf: State abbreviation.
        test_config: ``ValidationTestConfig`` (train_end, target_start, target_end).
        disease: ``"dengue"`` or ``"chikungunya"``.
        loader: Shared :class:`CompetitionDataLoader` (created when omitted;
            not needed when ``train_df``/``actual_df`` are supplied).
        model_registry: Unitted forecasters by name (default registry when omitted).
        use_future_exog: Attach climate-forecast + ocean covariates.
        calibrate: Apply conformal quantile recalibration (one extra fit
            per model on a held-out calibration window).
        calib_weeks: Calibration window length (default 67 = gap + season).
        train_df: Pre-loaded aggregated training frame (skips data loading).
        actual_df: Pre-loaded aggregated frame covering the post-cutoff
            actuals (skips data loading).
        future_exog: Pre-built future covariates (skips building).

    Returns:
        Result dictionary with per-model metrics, forecasts, and
        ensembles of the calibrated members: ``ens_qavg`` (equal-weight
        quantile average, baselines excluded), ``ens_median`` (per-date
        median), and — when ``calibrate`` is on — ``ens_vote``, a
        weighted average of *all* trained models with inverse-WIS votes
        from the leakage-free calibration window (weights recorded under
        ``ensemble_votes``).
    """
    if train_df is None or actual_df is None:
        if loader is None:
            from mosqlimate_ai.data.loader import CompetitionDataLoader

            loader = CompetitionDataLoader()
        train_df = loader.load_state_data(uf, end_date=test_config.train_end, disease=disease)
        actual_df = loader.load_state_data(
            uf,
            start_date=(pd.Timestamp(test_config.train_end) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            end_date=test_config.target_end,
            disease=disease,
        )

    train = train_df.copy()
    train["date"] = pd.to_datetime(train["date"])
    actual = actual_df.copy()
    actual["date"] = pd.to_datetime(actual["date"])

    # 52 weekly target dates from target_start (EW41..EW40); the config's
    # target_end is one week beyond the season in the Sunday-start convention
    target_dates = pd.date_range(test_config.target_start, periods=52, freq="7D")
    y_true = actual.set_index("date")["casos"].astype(float).reindex(target_dates)

    if future_exog is None and use_future_exog and loader is not None:
        future_exog = build_future_exog(
            loader,
            uf,
            pd.date_range(test_config.train_end, periods=FULL_HORIZON + 1, freq="7D")[1:],
            train_end=pd.Timestamp(test_config.train_end),
        )

    if model_registry is None:
        model_registry = default_model_registry(future_exog=future_exog)

    from mosqlimate_ai.evaluation.calibration import calibrate_forecast_scored

    # Dynamic horizon: cover the full target window even when the local
    # data cache is older than the canonical cutoff (67 weeks from
    # EW25). With fresher data the final season needs more steps.
    last_train = pd.Timestamp(train["date"].max())
    horizon_needed = int(np.ceil((pd.Timestamp(test_config.target_end) - last_train).days / 7))
    horizon = max(FULL_HORIZON, horizon_needed)

    models = {}
    calib_wis: dict[str, float] = {}
    for name, model in model_registry.items():
        try:
            start = time.time()
            cloned = _clone_model(model, future_exog)
            if calibrate:
                forecast, skill = calibrate_forecast_scored(
                    cloned, train, horizon, calib_weeks=calib_weeks
                )
                if skill is not None:
                    calib_wis[name] = skill
            else:
                cloned.fit(train)
                forecast = cloned.predict(horizon)
            models[name] = {
                "model": cloned,
                "forecast": forecast,
                "runtime": time.time() - start,
            }
        except Exception as exc:
            logger.exception("model %s failed for %s/%s: %s", name, uf, disease, exc)

    results = {
        "state": uf,
        "disease": disease,
        "test_number": test_config.test_number,
        "season": test_config.season,
        "train_end": test_config.train_end,
        "target_start": test_config.target_start,
        "target_end": test_config.target_end,
        "n_train_weeks": len(train),
        "n_target_weeks_observed": int(y_true.notna().sum()),
        "calibrated": calibrate,
        "models": {},
    }

    y_obs_idx = y_true[y_true.notna()].index

    def _evaluate(name: str, forecast: pd.DataFrame, runtime: float) -> None:
        # store the full 52-week target-window forecast (submission
        # completeness); evaluate on the overlapping observed subset
        f_win = forecast[forecast.index.isin(target_dates)]
        overlap = f_win.index.intersection(y_obs_idx)
        if f_win.empty:
            results["models"][name] = {"error": "no overlapping evaluation dates"}
            return
        f_obs = f_win.loc[overlap]
        iv = quantiles_to_intervals(f_obs.reset_index(names="date"))
        metrics = (
            evaluate_forecast(y_true.loc[overlap].values, iv)
            if len(overlap)
            else {}
        )
        horizons = np.array(
            [(d - pd.Timestamp(test_config.train_end)).days // 7 for d in f_obs.index]
        )
        by_h = (
            evaluate_by_horizon(y_true.loc[overlap].values, iv, horizons)
            if len(overlap)
            else pd.DataFrame()
        )
        results["models"][name] = {
            "metrics": metrics,
            "n_eval_weeks": int(len(overlap)),
            "runtime_s": round(runtime, 1),
            "by_horizon": by_h.reset_index().to_dict(orient="records"),
            "forecast": f_win.reset_index(names="date"),
        }

    for name, bundle in models.items():
        _evaluate(name, bundle["forecast"], bundle.get("runtime", 0.0))

    # Ensembles over the (calibrated) individual forecasts
    good = {
        name: b["forecast"]
        for name, b in models.items()
        if name not in ("seas_naive",) and isinstance(b.get("forecast"), pd.DataFrame) and len(b["forecast"])
    }
    if len(good) >= 2:
        import time as _time

        aligned = [f.reindex(good[next(iter(good))].index) for f in good.values()]
        start = _time.time()
        qavg = sum(aligned) / len(aligned)
        _evaluate("ens_qavg", qavg, _time.time() - start)
        start = _time.time()
        stacked = pd.concat(aligned)
        ens_median = stacked.groupby(level=0).median()
        _evaluate("ens_median", ens_median, _time.time() - start)

    # Voting ensemble: every trained model votes with weight proportional
    # to its inverse WIS on the leakage-free calibration window (which
    # only exists when conformal calibration ran). Unlike ens_qavg /
    # ens_median this includes the baselines — a weak member is
    # automatically down-weighted by its vote.
    votes = voting_weights(calib_wis) if calibrate else {}
    members = {
        name: b["forecast"]
        for name, b in models.items()
        if name in votes and isinstance(b.get("forecast"), pd.DataFrame) and len(b["forecast"])
    }
    if len(members) >= 2:
        import time as _time

        ref_index = members[next(iter(members))].index
        start = _time.time()
        ens_vote = None
        for name, fc in members.items():
            term = fc.reindex(ref_index) * votes[name]
            ens_vote = term if ens_vote is None else ens_vote + term
        _evaluate("ens_vote", ens_vote, _time.time() - start)
        results["ensemble_votes"] = {
            "weights": {m: round(w, 6) for m, w in votes.items() if m in members},
            "calib_wis": {m: round(s, 4) for m, s in calib_wis.items() if m in members},
        }

    return results


_WORKER_LOOKUP_CACHE: Dict[tuple, object] = {}
_WORKER_PARAMS_CACHE: Dict[tuple, Optional[Dict]] = {}
_WORKER_LOADER: Optional[object] = None


def _worker_lookup(uf: str, train_end) -> object:
    """Worker-local ExogLookup cache (loads the light exogenous sources)."""
    global _WORKER_LOADER
    from mosqlimate_ai.data.future_exog import ExogLookup

    key = (uf, str(pd.Timestamp(train_end)))
    if key not in _WORKER_LOOKUP_CACHE:
        # One shared light loader per worker process: ocean/population are
        # global tables (cached on the loader) and climate forecasts are
        # cached per state, so reusing it avoids re-reading the CSVs for
        # every (state, window) job.
        if _WORKER_LOADER is None:
            from mosqlimate_ai.data.loader import CompetitionDataLoader

            _WORKER_LOADER = CompetitionDataLoader()
        _WORKER_LOOKUP_CACHE[key] = ExogLookup(
            _WORKER_LOADER, uf, train_end=pd.Timestamp(train_end)
        )
    return _WORKER_LOOKUP_CACHE[key]


def _tune_job(args: tuple) -> Dict:
    """Worker entry point: tune one (state, disease) GBM pair.

    Results are written to the on-disk cache; returns a small summary.
    """
    uf, disease, train_df, n_trials, booster_n_jobs, cache_dir = args

    from mosqlimate_ai.models.gbm_direct import (
        LightGBMDirectForecaster,
        XGBoostDirectForecaster,
    )
    from mosqlimate_ai.validation.tuning import tune_all_cached

    lookup = _worker_lookup(uf, train_df["date"].max())

    def factory(family_ctor):
        def build(params: Dict):
            cfg = dict(params)
            # keep every worker inside its thread budget: the search-space
            # configs do not carry thread settings of their own
            cfg.setdefault("thread_budget", max(1, int(booster_n_jobs)))
            halflife = cfg.pop("recency_halflife_weeks", 104)
            return family_ctor(
                exog_lookup=lookup,
                recency_halflife_weeks=None if halflife in (0, "none") else halflife,
                params=cfg,
            )

        return build

    params_map = tune_all_cached(
        Path(cache_dir),
        uf,
        disease,
        {
            "xgb_direct": factory(XGBoostDirectForecaster),
            "lgbm_direct": factory(LightGBMDirectForecaster),
        },
        train_df,
        n_trials=n_trials,
        booster_n_jobs=max(1, int(booster_n_jobs)),
    )
    return {"state": uf, "disease": disease, "params": params_map}


def _test_job(args: tuple) -> dict:
    """Worker entry point: one (state, disease, split) backtest.

    Receives small per-split frames prepared by the main process so that
    workers never hold the full municipality tables (memory-bounded).
    Applies cached per-state tuned hyperparameters when available.
    """
    (
        uf,
        test_config,
        disease,
        train_df,
        actual_df,
        calibrate,
        include_tft,
        include_timesfm,
        booster_n_jobs,
        cache_dir,
    ) = args

    lookup = _worker_lookup(uf, test_config.train_end)

    cache_key = (uf, disease)
    if cache_key not in _WORKER_PARAMS_CACHE:
        from mosqlimate_ai.validation.tuning import DEFAULT_PARAMS, load_cached_params

        per_model: Dict[str, Dict] = {}
        for family in ("xgb_direct", "lgbm_direct"):
            cached = load_cached_params(Path(cache_dir), uf, disease, family)
            if cached:
                tuned = dict(cached)
                hl = tuned.pop(
                    "recency_halflife_weeks", DEFAULT_PARAMS["recency_halflife_weeks"]
                )
                tuned.setdefault("n_estimators", DEFAULT_PARAMS["n_estimators"])
                tuned["recency_halflife_weeks"] = hl
                per_model[family] = tuned
        _WORKER_PARAMS_CACHE[cache_key] = per_model or None
    params_by_model = _WORKER_PARAMS_CACHE[cache_key]

    # Per-worker thread budget computed at job-prep time
    # (total cores / number of workers), so all workers together exactly
    # fill the machine without oversubscription.
    thread_budget = max(1, int(booster_n_jobs))
    params = {"thread_budget": thread_budget}
    registry = default_model_registry(
        exog_lookup=lookup,
        include_tft=include_tft,
        include_timesfm=include_timesfm,
        params=params,
        params_by_model=params_by_model,
    )
    return run_single_backtest(
        uf,
        test_config,
        disease=disease,
        model_registry=registry,
        calibrate=calibrate,
        train_df=train_df,
        actual_df=actual_df,
        future_exog=None,
    )


def _prepare_test_job_args(
    states: list[str],
    diseases: tuple,
    include_final: bool,
    include_tft: bool,
    calibrate: bool,
    test_numbers: list[int] | None = None,
    max_workers: int = 4,
    cache_dir: Path = Path("validation_results/backtest"),
    loader=None,
    include_timesfm: bool = False,
) -> list[tuple]:
    """Prepare all worker job payloads in the main process.

    Keeps only the small per-state aggregated frames; the heavy merged
    municipality caches are released before workers are dispatched.
    """
    import os

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    booster_n_jobs = max(1, (os.cpu_count() or 8) // max(1, max_workers))

    if loader is None:
        loader = CompetitionDataLoader()
    cfg = get_validation_config()

    jobs: List[tuple] = []
    for disease in diseases:
        try:
            states_data = loader.load_all_states(aggregate=True, disease=disease)
        except FileNotFoundError:
            logger.warning("skipping %s: data unavailable", disease)
            continue

        test_cfgs = list(cfg.validation_tests)
        if include_final:
            test_cfgs.append(_final_test_config())
        if test_numbers is not None:
            test_cfgs = [t for t in test_cfgs if t.test_number in test_numbers]
            if not test_cfgs:
                continue

        for uf in states:
            if uf not in states_data:
                logger.warning("no %s data for %s; skipping", disease, uf)
                continue
            state_df = states_data[uf]
            state_df["date"] = pd.to_datetime(state_df["date"])
            for test_config in test_cfgs:
                if uf not in cfg.states and test_config.test_number != 5:
                    continue
                train_df = state_df[state_df["date"] <= pd.Timestamp(test_config.train_end)]
                actual_df = state_df[
                    (state_df["date"] > pd.Timestamp(test_config.train_end))
                    & (state_df["date"] <= pd.Timestamp(test_config.target_end))
                ]
                if len(train_df) < 60:
                    continue
                # completeness gate: warn (never fail) before training so
                # users know exactly which weeks are missing
                warn_missing_weeks(
                    train_df,
                    f"{uf}/{disease}",
                    context=f"{test_config.season} TRAINING window (cutoff {test_config.train_end})",
                    end=test_config.train_end,
                )
                warn_missing_weeks(
                    actual_df,
                    f"{uf}/{disease}",
                    context=f"{test_config.season} TARGET window",
                    start=test_config.target_start,
                )
                jobs.append(
                    (
                        uf,
                        test_config,
                        disease,
                        train_df,
                        actual_df,
                        calibrate,
                        include_tft,
                        include_timesfm,
                        booster_n_jobs,
                        str(Path(cache_dir)),
                    )
                )

    return jobs


def _prepare_tune_job_args(
    states: list[str],
    diseases: tuple,
    tune_trials: int,
    max_workers: int,
    cache_dir: Path = Path("validation_results/backtest"),
    loader=None,
) -> list[tuple]:
    """One tuning job per (state, disease): uses the earliest test window.

    Tuned configurations are cached on disk under
    ``<cache_dir>/hyperparams/`` and reused by the backtest jobs.
    """
    import os

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    booster_n_jobs = max(1, (os.cpu_count() or 8) // max(1, max_workers))

    if loader is None:
        loader = CompetitionDataLoader()
    cfg = get_validation_config()

    tune_jobs: List[tuple] = []
    for disease in diseases:
        try:
            states_data = loader.load_all_states(aggregate=True, disease=disease)
        except FileNotFoundError:
            logger.warning("skipping %s tuning: data unavailable", disease)
            continue

        first_train_end = cfg.validation_tests[0].train_end
        for uf in states:
            if uf not in states_data or uf not in cfg.states:
                continue
            state_df = states_data[uf]
            state_df["date"] = pd.to_datetime(state_df["date"])
            train_df = state_df[state_df["date"] <= pd.Timestamp(first_train_end)]
            if len(train_df) < 60:
                continue
            tune_jobs.append(
                (uf, disease, train_df, tune_trials, booster_n_jobs, str(Path(cache_dir)))
            )

    return tune_jobs


def run_full_pipeline(
    states: list[str] | None = None,
    diseases: tuple = ("dengue", "chikungunya"),
    include_final: bool = False,
    include_tft: bool = False,
    include_timesfm: bool = False,
    calibrate: bool = True,
    max_workers: int = 5,
    out_dir: Path = Path("validation_results/backtest"),
    test_numbers: list[int] | None = None,
    tune_trials: int = 0,
) -> pd.DataFrame:
    """Backtest all states and diseases with process-level parallelism.

    The main process prepares small per-split frames; worker processes
    only fit/predict/evaluate (bounded memory). Saves per-state JSON +
    forecast CSVs and a combined summary CSV.

    Args:
        test_numbers: Restrict to these test numbers (1..4 validation,
            5 = final forecast).
        tune_trials: Per-state random-search trials for the GBM
            hyperparameters (0 = use fixed defaults). Results are cached
            under ``<out_dir>/hyperparams/`` and reused on re-runs.
        include_timesfm: Add the zero-shot TimesFM foundation model to
            the registry (checkpoint downloaded on first use).
"""
    if states is None:
        states = get_validation_config().states

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One shared loader: the heavy merged municipality tables are built
    # once and reused by both prep phases (never rebuilt in between).
    import gc

    from mosqlimate_ai.data.loader import CompetitionDataLoader

    loader = CompetitionDataLoader()

    if tune_trials > 0:
        tune_jobs = _prepare_tune_job_args(
            states, diseases, tune_trials, max_workers, out_dir, loader=loader
        )
        logger.info("prepared %d tuning jobs (%d trials each)", len(tune_jobs), tune_trials)
        def _tune_failure(job, exc):
            logger.error("tuning job failed: %s/%s -> %s: %s",
                         job[0], job[1], type(exc).__name__, exc)

        _run_jobs(
            _tune_job,
            tune_jobs,
            max_workers,
            lambda job, result: logger.info("tuned %s/%s", job[0], job[1]),
            _tune_failure,
        )

    jobs = _prepare_test_job_args(
        states, diseases, include_final, include_tft, calibrate, test_numbers, max_workers,
        out_dir, loader=loader, include_timesfm=include_timesfm,
    )
    logger.info("prepared %d test jobs", len(jobs))

    # release the heavy caches before dispatching workers
    loader._merged_cache.clear()
    gc.collect()

    by_state: dict[tuple, dict] = {}
    summaries: list[pd.DataFrame] = []
    expected_counts: dict[tuple, int] = {}
    for job in jobs:
        key = (job[0], job[2])
        expected_counts[key] = expected_counts.get(key, 0) + 1

    def on_result(job, result):
        _collect_result(job, result, by_state, summaries, out_dir, expected_counts)

    def on_test_failure(job, result):
        logger.error("test job failed: %s/%s/%s", job[0], job[2], job[1].season)

    _run_jobs(_test_job, jobs, max_workers, on_result, on_test_failure)

    # persist any states whose jobs did not all complete
    for key, entry in list(by_state.items()):
        if entry["tests"] or entry["final"] is not None:
            save_backtest_results(entry, out_dir)
            summaries.append(summarize_backtest(entry))
            logger.warning("incomplete state saved: %s/%s", key[0], key[1])
            by_state.pop(key)

    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined.to_csv(out_dir / "summary.csv", index=False)
    return combined


def _run_jobs(fn, jobs, max_workers, on_success, on_failure) -> None:
    """Dispatch jobs through a spawn-context process pool (or serially)."""
    if max_workers <= 1:
        for job in jobs:
            try:
                result = fn(job)
            except Exception as exc:
                logger.exception("job failed: %s", job[:2])
                on_failure(job, exc)
                continue
            on_success(job, result)
        return

    # "spawn" is essential: forked workers inherit OpenMP/XGBoost
    # thread state and deadlock on the first booster fit.
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {pool.submit(fn, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                on_failure(job, exc)
                continue
            on_success(job, result)


def _collect_result(job, result, by_state, summaries, out_dir, expected_counts) -> None:
    uf, test_config, disease = job[0], job[1], job[2]
    key = (uf, disease)
    entry = by_state.setdefault(
        key, {"state": uf, "disease": disease, "tests": {}, "final": None}
    )
    if test_config.test_number == 5:
        entry["final"] = result
    else:
        entry["tests"][str(test_config.test_number)] = result

    have = len(entry["tests"]) + (1 if entry["final"] is not None else 0)
    if have >= expected_counts.get(key, have):
        save_backtest_results(entry, out_dir)
        summaries.append(summarize_backtest(entry))
        by_state.pop(key)
        logger.info("done %s/%s", uf, disease)


def _clone_model(model: BaseForecaster, future_exog: pd.DataFrame | None) -> BaseForecaster:
    """Re-instantiate a registry model bound to this test's future covariates."""
    import copy

    clone = copy.copy(model)
    if hasattr(clone, "future_exog"):
        clone.future_exog = future_exog
    clone.is_fitted_ = False
    return clone


# ---------------------------------------------------------------------------
# Per-state, multi-test pipelines
# ---------------------------------------------------------------------------
def run_state_pipeline(
    uf: str,
    disease: str = "dengue",
    include_final: bool = False,
    loader=None,
    model_registry_fn: Callable[..., dict[str, BaseForecaster]] | None = None,
) -> dict:
    """All validation tests (+ optional final forecast) for one state."""
    if loader is None:
        from mosqlimate_ai.data.loader import CompetitionDataLoader

        loader = CompetitionDataLoader()
    config = get_validation_config()

    state_result = {"state": uf, "disease": disease, "tests": {}}
    for test_config in config.validation_tests:
        registry = (
            model_registry_fn() if model_registry_fn else None
        )
        state_result["tests"][str(test_config.test_number)] = run_single_backtest(
            uf, test_config, disease=disease, loader=loader, model_registry=registry
        )

    if include_final:
        final_config = _final_test_config()
        registry = model_registry_fn() if model_registry_fn else None
        state_result["final"] = run_single_backtest(
            uf, final_config, disease=disease, loader=loader, model_registry=registry
        )
    return state_result


def _final_test_config():
    from mosqlimate_ai.validation.config import ValidationTestConfig, get_validation_config

    cfg = get_validation_config()
    return ValidationTestConfig(
        test_number=5,
        season="2026-2027",
        train_end=cfg.final_forecast_train_end,
        target_start=cfg.final_forecast_target_start,
        target_end=cfg.final_forecast_target_end,
        description="Final forecast: 2026-2027 season",
    )


def summarize_backtest(results: dict) -> pd.DataFrame:
    """Flat model-level WIS table across tests for one state result.

    Accepts either a full state pipeline result (with ``tests``) or a
    single ``run_single_backtest`` result.
    """
    if "tests" in results:
        buckets = dict(results["tests"])
    else:
        buckets = {str(results.get("test_number", "?")): results}

    rows = []
    for test_key, test_res in buckets.items():
        for model_name, model_res in test_res.get("models", {}).items():
            metrics = model_res.get("metrics", {})
            rows.append(
                {
                    "state": test_res.get("state", results.get("state")),
                    "disease": test_res.get("disease", results.get("disease")),
                    "test": test_key,
                    "season": test_res.get("season"),
                    "model": model_name,
                    "wis": metrics.get("wis_total"),
                    "crps": metrics.get("crps"),
                    "mae": metrics.get("mae"),
                    "coverage_50": metrics.get("coverage_50"),
                    "coverage_95": metrics.get("coverage_95"),
                    "n_eval_weeks": model_res.get("n_eval_weeks"),
                    "runtime_s": model_res.get("runtime_s"),
                }
            )
    return pd.DataFrame(rows)


def save_backtest_results(result: dict, out_dir: Path, forecasts: bool = True) -> Path:
    """Persist one state's backtest result (JSON + forecast parquets)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = _strip_forecasts(result)
    path = out_dir / f"{result['state']}_{result['disease']}_backtest.json"
    path.write_text(json.dumps(serializable, indent=2, default=str))

    if forecasts:
        fc_dir = out_dir / "forecasts"
        fc_dir.mkdir(exist_ok=True)
        buckets: dict[str, dict] = dict(result.get("tests", {}))
        if result.get("final"):
            buckets["final"] = result["final"]
        for test_key, test_res in buckets.items():
            for model_name, model_res in test_res.get("models", {}).items():
                fc = model_res.get("forecast")
                if fc is not None and len(fc):
                    fc.to_csv(
                        fc_dir / f"{result['state']}_{result['disease']}_{test_key}_{model_name}.csv.gz",
                        index=False,
                    )
    return path


def _strip_forecasts(result: dict) -> dict:
    """Deep copy without heavy forecast frames (keeps metrics)."""
    import copy

    slim = copy.deepcopy(result)
    buckets: list[dict] = list(slim.get("tests", {}).values())
    if slim.get("final"):
        buckets.append(slim["final"])
    for test_res in buckets:
        if not isinstance(test_res, dict):
            continue
        for model_res in test_res.get("models", {}).values():
            model_res.pop("forecast", None)
            model_res.pop("by_horizon", None)
    return slim
