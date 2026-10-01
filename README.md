# Mosqlimate AI Competitor 🤖🦟

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered **dengue and chikungunya forecasting system** for the
[Infodengue-Mosqlimate Dengue Challenge (3rd IMDC, 2026)](https://sprint.mosqlimate.org).

The pipeline produces weekly state-level probabilistic forecasts
(median + 50/80/90/95% prediction intervals), validates them honestly
against four historical seasons plus the final 2026-27 target, and
packages submissions in the competition format.

---

## 🚀 Step-by-step workflow

### Step 0 — Install

Requires Python 3.12+ and Git. [uv](https://docs.astral.sh/uv/) is recommended:

```bash
git clone git@github.com:fccoelho/mosqlimate-ai-competitor.git
cd mosqlimate-ai-competitor
uv sync            # or: pip install -e .
```

### Step 1 — Download the competition data

```bash
mosqlimate-ai download-data
```

Files land in `data/` (FTP: `info.dengue.mat.br/data_imdc_2026`):

| File | Content |
|------|---------|
| `dengue.csv.gz` | Weekly dengue cases by municipality (2010–2026) |
| `chikungunya.csv.gz` | Weekly chikungunya cases by municipality |
| `climate.csv.gz` | Weekly ERA5 climate reanalysis |
| `forecasting_climate.csv.gz` | Monthly ECMWF climate forecasts (issue/lead) |
| `datasus_population_2001_*.csv.gz` | Population by municipality and year |
| `environ_vars.csv.gz` | Köppen climate, biome |
| `ocean_climate_oscillations*.csv.gz` | ENSO / IOD / PDO indices |
| `*_update_2025/2026.csv.gz` | Partial-season update segments (merged automatically) |

### Step 2 — Verify data completeness (always do this before training)

The IMDC publishes data as a base file plus partial update segments; if
vintages drift apart, weeks can go missing. Missing weeks are never
silently zero-filled — the loader reindexes each state to a complete
weekly grid (gaps become NaN) and warns you.

```bash
mosqlimate-ai check-data                  # all states, both diseases
mosqlimate-ai check-data --states SP,RJ   # subset
```

- Exit code 0 + `✓ ... no missing weeks` → data is complete.
- Exit code 1 → re-download with `mosqlimate-ai download-data --force`
  (or use the refresh script, Step 2b) and re-check.

On success, `check-data` writes `data/completeness_verified.json`
recording the verified file fingerprints.

### Step 2b — Refresh stale data without losing verified files

```bash
python scripts/refresh_data.py          # fetch only what is stale
python scripts/refresh_data.py --force  # full manual refresh
```

Policy:
1. Files recorded as **verified complete** in the manifest are
   **skipped** — the saved version is authoritative while unchanged
   (`skip dengue.csv.gz: verified complete — using saved version`).
2. Other files are fetched only when the remote size differs from the
   local copy.
3. `--force` overrides everything.

### Step 3 — (Optional) tune hyperparameters per state

Hyperparameter tuning is built into the validation run (Step 4) via the
`--tune N` flag. It runs a seeded random search per (state, disease,
model family), scored by WIS on a held-out 67-week window that mirrors
deployment (15-week gap + 52 target weeks). Results are cached under
`validation_results/backtest/hyperparams/` and reused across splits and
future runs — so tuning cost is paid once.

In our runs, tuning improved **104/104** GBM configurations with a mean
WIS gain of **18.6%** over defaults.

### Step 4 — Run the validation backtests

```bash
# full protocol: 4 validation seasons + final 2026-27 forecast,
# 26 states (ES excluded), dengue + chikungunya, with per-state tuning
mosqlimate-ai validate --full-pipeline --tune 12

# or via the driver script (equivalent, extra flags)
python scripts/run_backtests.py 4 - 12     # workers, states("-"=all), tune trials

# quick subset
mosqlimate-ai validate --test 1 --states SP,RJ --diseases dengue
```

For every state × test, models train **strictly on data up to EW25**
and produce a 67-step probabilistic forecast (15-week unobserved gap +
52 target weeks). Pre-training completeness warnings list any missing
weeks per window. Results land in `validation_results/backtest/`:

| Output | Content |
|--------|---------|
| `summary.csv` | WIS / MAE / coverage per state × disease × test × model |
| `<UF>_<disease>_backtest.json` | full metrics incl. per-horizon WIS |
| `forecasts/` | calibrated quantile forecasts (all models) |
| `hyperparams/` | cached tuned parameters |
| `plots/` | forecast-vs-observed figures (Step 5) |

### Step 5 — Generate the validation report

```bash
python scripts/make_validation_report.py
```

Writes `validation_results/backtest/VALIDATION_REPORT.md` containing:

- mean WIS per model with **skill vs the seasonal-naive baseline** (the
  skill floor) and coverage diagnostics,
- the **per-state hyperparameter tuning summary** (gain over defaults),
- the **skill-gated model selection** per state (states where nothing
  beats the naive baseline deploy the baseline),
- **52 forecast-vs-observed figures** (one per state × disease):
  observed training tail, calibrated median with 50%/95% bands, the
  shaded 15-week gap, observed target season, per-panel WIS.

### Step 6 — Generate forecasts for the target season

Two equivalent routes produce the calibrated 52-week forecasts for the
2026-27 season (train on all data ≤ EW25 2026 → predict across the
15-week gap → conformally calibrate → slice the target window):

**Route A — dedicated forecast script (recommended)**

```bash
python scripts/generate_forecast.py                       # all states, both diseases
python scripts/generate_forecast.py --states SP,RJ --model ens_qavg
python scripts/generate_forecast.py --help                # all options
```

- Per state, uses the **skill-gated model selection** from the latest
  backtest run (override with `--model`), applies the per-state tuned
  hyperparameters when cached, and warns if the training window has
  missing weeks.
- Writes one quantile CSV per state:
  `forecasts/final/<disease>/<UF>.csv` with columns
  `date, q025, q050, ..., q975` (52 weekly rows from EW41 2026).

**Route B — via the validation harness**

```bash
mosqlimate-ai validate --final-forecast
```

Runs the same machinery through the backtest harness (results also
saved under `validation_results/backtest/`).

### Step 7 — Build the submission package

```bash
python scripts/make_submission.py [MODEL_ID]
```

- Selects the best model per state/disease from the backtest results
  (skill-gated: falls back to the seasonal-naive baseline when nothing
  beats it),
- loads that model's calibrated 52-week forecasts for every split
  (tests 1–4 + final),- enforces median-inside-intervals, non-negativity and completeness
  (26 UFs × 52 weeks × 5 splits × diseases),
- writes JSON payloads to `submissions/<disease>/<split>/<UF>.json`
  plus `submissions/model_selection.csv`.

Submit via the Mosqlimate API:

```bash
export MOSQLIMATE_API_KEY="your-key"
mosqlimate-ai submit --model-id 123 --dry-run   # preview first
```

> **Note (final forecast):** the 2026-27 final target requires data
> through EW25 2026. Re-run Steps 1-2 and 6 after each data update so the
> final forecasts always use the freshest verified data.

---

## 📅 Competition protocol

| Split | Training cutoff | Target season |
|-------|-----------------|---------------|
| Test 1 | EW25 2022 | EW41 2022 – EW40 2023 |
| Test 2 | EW25 2023 | EW41 2023 – EW40 2024 |
| Test 3 | EW25 2024 | EW41 2024 – EW40 2025 |
| Test 4 | EW25 2025 | EW41 2025 – EW40 2026 |
| Final | EW25 2026 | EW41 2026 – EW40 2027 |

Geography: 27 Brazilian state units minus Espírito Santo.
Scoring: **Weighted Interval Score** (Bracher et al. 2021) on the
50/80/90/95% intervals, computed weekly.

## 🏗️ Architecture

```
src/mosqlimate_ai/
├── data/
│   ├── downloader.py        # FTP download / size-checked cache
│   ├── loader.py            # merge, weekly reindex (gaps -> NaN), state aggregation
│   ├── completeness.py      # missing-week detection, user warnings, verification manifest
│   ├── future_exog.py       # ExogLookup: ECMWF issue/lead semantics, ENSO/IOD/PDO persistence
│   ├── features.py          # legacy feature engineering (unused by the pipeline)
│   └── preprocessor.py      # legacy preprocessing (deprecated)
├── models/
│   ├── base.py              # unified interface: fit(df) / predict(horizon) -> q025..q975
│   ├── gbm_direct.py        # direct multi-horizon XGBoost/LightGBM (log1p target)
│   ├── tft_direct.py        # Temporal Fusion Transformer, native 9-quantile loss (GPU)
│   ├── baselines.py         # seasonal-naive, flat, log-linear trend
│   └── ensemble.py, ...     # legacy models (retired from the pipeline)
├── evaluation/
│   ├── quantiles.py         # canonical quantile format, WIS/CRPS (Bracher et al.)
│   ├── calibration.py       # conformal quantile recalibration
│   └── metrics.py           # RMSE/MAE/coverage/... + evaluate_by_horizon
├── validation/
│   ├── config.py            # IMDC calendar (4 tests + final 2026-27)
│   ├── backtest.py          # deterministic parallel OOS harness
│   ├── tuning.py            # per-state random-search tuning (cached)
│   ├── selection.py         # skill-gated per-state model selection
│   └── report_plots.py      # forecast-vs-observed figures
└── submission/
    ├── imdc.py              # submission package builder + completeness checks
    ├── formatter.py         # Mosqlimate API payload formatting
    └── api_client.py        # API client
```

## 🤖 Models

| Model | Type | Exogenous | Uncertainty |
|-------|------|-----------|-------------|
| `xgb_direct` | Direct multi-horizon XGBoost, log1p target | origin reanalysis + target-time ECMWF/ocean | quantile regression |
| `lgbm_direct` | Same with LightGBM | same | quantile regression |
| `tft_direct` | TFT (GPU), 67-step direct | calendar | native 9-quantile loss |
| `loglin_trend` | Log-linear trend + harmonics | — | Gaussian residual quantiles |
| `seas_naive` | Seasonal naive (lag 52) | — | additive seasonal error quantiles |
| `ens_qavg` / `ens_median` | Ensembles of calibrated members | — | averaged |

Key properties:

- **Leak-free by construction** — lags/rolling features are computed
  inside the models from the training window only; future covariates
  are supplied explicitly via `ExogLookup`, which honors the ECMWF
  issue/lead semantics (a forecast for target month M uses the latest
  issue available at the origin month, lead capped at 6).
- **Conformal calibration** — per-quantile additive offsets from a
  held-out 67-week window fix the systematic overconfidence of
  quantile regression at long horizons.
- **Data-completeness aware** — missing weeks are NaN (never zeros),
  skipped in training, surfaced as warnings, and drawn as line breaks
  in plots.

## 📜 Scripts reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run_backtests.py` | Full parallel backtest run | `run_backtests.py [WORKERS] [STATES] [TUNE_TRIALS]` (`STATES` `-` = all) |
| `scripts/generate_forecast.py` | Generate calibrated forecasts for the target season | `generate_forecast.py [--states ..] [--model ..] [--cutoff ..]` |
| `scripts/make_validation_report.py` | Markdown report + plots + tuning summary + selection | `make_validation_report.py [BACKTEST_DIR]` |
| `scripts/make_submission.py` | Build + validate the IMDC submission package | `make_submission.py [MODEL_ID]` |
| `scripts/refresh_data.py` | Refresh stale data, skip verified files | `refresh_data.py [--force]` |

## 🧪 Development

```bash
pytest tests/ -q            # 95 tests
ruff check src/ scripts/    # lint
mosqlimate-ai check-data    # always run before training
```

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [Mosqlimate Platform](https://mosqlimate.org/) for organizing the challenge
- [Infodengue](https://info.dengue.mat.br/) for epidemiological data
