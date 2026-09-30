# Mosqlimate AI Competitor 🤖🦟

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered dengue and chikungunya forecasting system for the
[Infodengue-Mosqlimate Dengue Challenge (3rd IMDC, 2026)](https://sprint.mosqlimate.org).

## 🚀 Quick Start

```bash
# Install (requires Python 3.12+; recommended: uv)
uv sync

# Download competition data (FTP) into ./data
mosqlimate-ai download-data

# Run the deterministic out-of-sample validation pipeline
# (4 validation seasons + final 2026-27 forecast, all states, both diseases)
mosqlimate-ai validate --full-pipeline

# Or a quick single test on a few states
mosqlimate-ai validate --test 1 --states SP,RJ --diseases dengue

# Generate the validation report (mean WIS, skill vs baseline, selection)
python scripts/make_validation_report.py

# Build the IMDC submission package from backtest forecasts
python scripts/make_submission.py [MODEL_ID]
```

## 📊 The challenge

The IMDC scores **weekly state-level forecasts with the Weighted Interval
Score (WIS, Bracher et al. 2021)** over the 50/80/90/95% prediction
intervals. Each season runs from EW41 of one year to EW40 of the next;
models train on data up to EW25 — a **15-week unobserved gap** separates
the training cutoff from the forecast start, so every model must be a
genuine 67-step probabilistic forecaster (15 gap weeks + 52 target
weeks).

| Test | Training cutoff | Target season |
|------|-----------------|---------------|
| 1 | EW25 2022 | EW41 2022 – EW40 2023 |
| 2 | EW25 2023 | EW41 2023 – EW40 2024 |
| 3 | EW25 2024 | EW41 2024 – EW40 2025 |
| 4 | EW25 2025 | EW41 2025 – EW40 2026 |
| Final | EW25 2026 | EW41 2026 – EW40 2027 |

Geography: all 27 Brazilian state units (UF) except Espírito Santo.
Diseases: dengue (mandatory) and chikungunya (optional state-level).

## 🏗️ Architecture

```
src/mosqlimate_ai/
├── data/
│   ├── downloader.py        # FTP download/cache
│   ├── loader.py            # merge + state aggregation (dengue/chikungunya)
│   ├── future_exog.py       # known-covariate lookup (ECMWF by issue/lead, ENSO/IOD/PDO)
│   ├── features.py          # legacy feature engineering
│   └── preprocessor.py      # legacy preprocessing
├── models/
│   ├── base.py              # unified interface: fit(df) / predict(horizon) -> quantiles
│   ├── gbm_direct.py        # direct multi-horizon XGBoost/LightGBM (workhorse)
│   ├── tft_direct.py        # TFT with native 9-quantile loss (GPU)
│   └── baselines.py         # seasonal-naive, flat, log-linear trend
├── evaluation/
│   ├── quantiles.py         # canonical quantile format, WIS/CRPS (Bracher et al.)
│   ├── metrics.py           # RMSE/MAE/coverage/... + evaluate_by_horizon
│   └── calibration.py       # conformal quantile recalibration
├── validation/
│   ├── config.py            # IMDC calendar (4 tests + final)
│   ├── backtest.py          # deterministic OOS backtest harness (parallel)
│   └── selection.py         # per-state model selection (skill-gated)
└── submission/
    ├── imdc.py              # IMDC submission package + completeness checks
    ├── formatter.py         # Mosqlimate API payload formatting
    └── api_client.py        # API client
```

### Key design decisions

1. **Unified model interface** — every model implements
   `fit(df)` / `predict(horizon) -> quantile DataFrame` (canonical
   columns `q025..q975`, weekly date index). This eliminates the
   interface-mismatch bug class that silently produced fake metrics in
   earlier versions.
2. **Leak-free by construction** — target lags/rolling features are
   computed inside the models from the training window only; future
   covariates must be supplied explicitly via
   `ExogLookup`, which honors the ECMWF issue/lead semantics
   (a forecast for target month M is taken from the latest issue
   available at the origin month, lead capped at 6).
3. **Honest out-of-sample validation** — models train strictly on
   data ≤ EW25 and are evaluated on the full 52-week target season,
   per-horizon. The **seasonal-naive baseline is the skill floor**;
   selection deploys the baseline for states where nothing beats it.
4. **Conformal calibration** — per-quantile additive offsets estimated
   on a held-out 67-week calibration window (gap + season structure)
   fix the systematic overconfidence of quantile regression at long
   horizons.
5. **Ensembles** — quantile-averaged (`ens_qavg`) and per-quantile
   median (`ens_median`) combinations of the calibrated members.

## 🤖 Models

| Model | Type | Exogenous | Uncertainty |
|-------|------|-----------|-------------|
| `xgb_direct` | Direct multi-horizon XGBoost, log1p target | origin reanalysis + target-time ECMWF/ocean | quantile regression |
| `lgbm_direct` | Same with LightGBM | same | quantile regression |
| `tft_direct` | Temporal Fusion Transformer (GPU), 67-step direct | calendar | native 9-quantile loss |
| `loglin_trend` | Log-linear trend + harmonics | — | Gaussian residual quantiles |
| `seas_naive` | Seasonal naive (lag 52) | — | empirical seasonal error quantiles |
| `ens_qavg` / `ens_median` | Ensembles of calibrated members | — | inherited + averaged |

The legacy LSTM/N-BEATS models and the LLM multi-agent layer are
retired from the pipeline (kept under `models/` and `agents/` for
reference only).

## 📈 Evaluation

`mosqlimate-ai validate` writes per-state JSON results and forecast
CSVs to `validation_results/backtest/`:

- `summary.csv` — model-level WIS/MAE/coverage per state, disease, test
- `<UF>_<disease>_backtest.json` — full metrics incl. per-horizon WIS
- `forecasts/` — calibrated quantile forecasts per state/test/model
- `scripts/make_validation_report.py` — markdown report with skill vs
  baseline and per-state model selection

## 📤 Submission

`scripts/make_submission.py` builds the complete submission package:
selected model per state/disease, 52 weekly quantiles per split
(tests 1-4 + final), median forced inside all intervals, non-negativity,
completeness check (26 UFs × 52 weeks × 5 splits × diseases), saved as
JSON payloads under `submissions/` ready for the Mosqlimate API.

> **Note (final forecast):** the 2026-27 final forecast requires data
> through EW25 2026. Re-run `download-data` and `validate --final-forecast`
> when updated data is published.

## 🛠️ Development

```bash
pytest tests/ -v            # test suite
ruff check src/ tests/      # lint
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Mosqlimate Platform](https://mosqlimate.org/) for organizing the challenge
- [Infodengue](https://info.dengue.mat.br/) for epidemiological data
