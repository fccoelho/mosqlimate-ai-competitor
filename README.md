# Mosqlimate AI Competitor 🤖🦟

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered dengue forecasting system for the [Infodengue-Mosqlimate Dengue Challenge](https://sprint.mosqlimate.org), leveraging multi-agent coding with **Karl DBot**.

## 🚀 Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download competition data
mosqlimate-ai download-data

# Run full validation pipeline
mosqlimate-ai validate --full-pipeline

# Train models
mosqlimate-ai train --states SP,RJ

# Generate forecasts
mosqlimate-ai forecast --weeks 52

# Evaluate and report
mosqlimate-ai report --output report.md
```

## 📦 Installation

Requires Python 3.9+ and Git:

```bash
git clone git@github.com:fccoelho/mosqlimate-ai-competitor.git
cd mosqlimate-ai-competitor

# Recommended: use uv
uv sync

# Or pip
pip install -e ".[dev]"
```

## 📊 Get the Data

Download competition data from the Mosqlimate Platform:

```bash
# Download all required data files
mosqlimate-ai download-data

# View cache status
mosqlimate-ai cache-info

# Clear cache if needed
mosqlimate-ai clear-cache
```

**Available Data:**

| File | Description |
|------|-------------|
| `dengue.csv.gz` | Weekly dengue cases by municipality (2010-2025) |
| `climate.csv.gz` | Weekly climate reanalysis data (ERA5) |
| `climate_forecast.csv.gz` | Monthly climate forecasts (ECMWF) |
| `datasus_population_2001_2024.csv.gz` | Population by municipality and year |
| `environ_vars.csv.gz` | Environmental variables (Koppen, Biome) |
| `ocean_climate_oscillations.csv.gz` | ENSO, IOD, PDO ocean indices |

## ⚙️ Configuration

Initialize a configuration file for reusable settings:

```bash
mosqlimate-ai init-config
```

**Example `mosqlimate.yaml`:**

```yaml
models:
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    epochs: 100

paths:
  models_dir: "./models"
  forecasts_dir: "./forecasts"

states: []  # Empty = all states
```

Use configuration in any command:

```bash
mosqlimate-ai train --config myconfig.yaml
mosqlimate-ai validate --full-pipeline --config myconfig.yaml
```

## 🔄 Run Validation Tests

The validation pipeline trains models 4 times following competition rules: 3 validation tests (2022-2025) + 1 final forecast (2025-2026).

| Test | Training Data | Forecast Period |
|------|---------------|-----------------|
| **Test 1** | 2010-EW25 2022 | EW41 2022 - EW40 2023 |
| **Test 2** | 2010-EW25 2023 | EW41 2023 - EW40 2024 |
| **Test 3** | 2010-EW25 2024 | EW41 2024 - EW40 2025 |
| **Final** | 2010-EW25 2025 | EW41 2025 - EW40 2026 |

### Run Validation

```bash
# Complete 4-stage validation for all states
mosqlimate-ai validate --full-pipeline

# Validation for specific states
mosqlimate-ai validate --full-pipeline --states SP,RJ,MG

# Run single test
mosqlimate-ai validate --test 1 --states SP,RJ

# Final forecast only
mosqlimate-ai validate --final-forecast
```

### View Logs

```bash
# Show agent communication logs
mosqlimate-ai validate --show-logs

# Export audit trail
mosqlimate-ai validate --export-audit --output audit_report.md
```

### Validation Features

- **Multi-Agent System**: StateValidationAgent per state with cross-state learning
- **Hyperparameter Tuning**: Bayesian optimization with warm start from similar states
- **Model Selection**: Top 3 models by weighted composite score (CRPS, WIS, MAE, Coverage)
- **Audit Logging**: Complete log of all agent decisions in `logs/agent_communications/`

## 🏋️ Train Models

Train forecasting models per state:

```bash
# Train all states
mosqlimate-ai train

# Train specific states
mosqlimate-ai train --states SP,RJ

# Train only XGBoost
mosqlimate-ai train --models xgboost

# With custom config
mosqlimate-ai train --config myconfig.yaml
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--output, -o` | Save directory | `models/` |
| `--states, -s` | State UFs (comma-separated) | all states |
| `--models, -m` | Models to train | `xgboost,lstm` |
| `--val-size` | Validation fraction | `0.1` |
| `--verbose, -v` | Detailed output | `false` |

## 🔮 Generate Forecasts

Generate forecasts with prediction intervals:

```bash
# Generate 52-week forecasts for all states
mosqlimate-ai forecast

# Forecast specific states
mosqlimate-ai forecast --states SP,RJ

# Custom period
mosqlimate-ai forecast --weeks 26 --start-date 2026-01-01

# Output to specific directory
mosqlimate-ai forecast --output ./my_forecasts
```

**Output Format:**

```csv
date,median,lower_50,upper_50,lower_80,upper_80,lower_95,upper_95
2026-01-05,1250,1000,1500,850,1650,700,1800
...
```

## 📈 Evaluate & Report

### Evaluate Forecasts

```bash
# Evaluate specific state
mosqlimate-ai evaluate --state SP

# Evaluate up to date
mosqlimate-ai evaluate --state RJ --end-date 2025-12-31
```

**Metrics:** CRPS, WIS, Coverage 95%/50%, MAE, RMSE

### Generate Report

```bash
# Full report with visualizations
mosqlimate-ai report

# Text-only report (faster)
mosqlimate-ai report --no-plots

# Specific states
mosqlimate-ai report --states SP,RJ --output my_report.md
```

**Report includes:**
- Executive summary with average metrics
- Best model by metric
- Detailed results per state
- Visualizations: timeseries, residuals, calibration, heatmaps

## 📤 Submit to Competition

Submit forecasts to the Mosqlimate API:

```bash
# Submit all forecasts (requires API key)
mosqlimate-ai submit --model-id 123

# Dry run to preview
mosqlimate-ai submit --model-id 123 --dry-run

# Save to JSON for review
mosqlimate-ai submit --model-id 123 --dry-run --output-json submissions.json
```

**Setup API Key:**

```bash
export MOSQLIMATE_API_KEY="your-api-key-here"
```

## 🤖 Multi-Agent System

The system uses specialized agents working together:

| Agent | Responsibility |
|-------|---------------|
| **Data Engineer** | Data collection, cleaning, feature engineering |
| **Model Architect** | ML/DL architectures, hyperparameter tuning |
| **Forecaster** | Predictions, uncertainty quantification |
| **Validator** | Cross-validation, backtesting, metrics |
| **Ensembler** | Combine models for robust forecasts |
| **StateValidationAgent** | Manages 4-run validation pipeline |
| **CrossStateKnowledgeBase** | Shares insights between states |

### Architecture

```
mosqlimate-ai-competitor/
├── src/mosqlimate_ai/
│   ├── data/           # Data ingestion & preprocessing
│   ├── models/         # XGBoost, LSTM, Ensemble
│   ├── agents/         # Karl DBot multi-agent system
│   ├── evaluation/     # Validation & metrics
│   └── validation/     # Competition validation pipeline
├── notebooks/          # Exploratory analysis
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🔬 Models

### Ensemble Approach

1. **XGBoost**: Quantile regression with feature importance
2. **LSTM**: Deep learning with Monte Carlo dropout
3. **Ensemble**: Weighted average based on validation performance

## 🛠️ Development

```bash
# Run tests
pytest tests/ -v --cov=src/mosqlimate_ai

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/mosqlimate_ai

# Feature caching
mosqlimate-ai feature-cache-info
mosqlimate-ai clear-feature-cache
```

## 📚 Documentation

- [Competition Rules](docs/competition_rules.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Model Documentation](docs/models.md)
- [Agent Configuration](docs/agents.md)
- [Validation Pipeline](VALIDATION_PIPELINE_SUMMARY.md)
- [Report Enhancements](REPORT_ENHANCEMENTS.md)

## 🏆 Competition Results

| Round | Model | RMSE | Rank |
|-------|-------|------|------|
| Validation | - | - | - |
| Final | - | - | - |

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Mosqlimate Platform](https://mosqlimate.org/) for organizing the challenge
- [Infodengue](https://info.dengue.mat.br/) for epidemiological data
- [Karl DBot](https://github.com/Deeplearn-PeD/KarlDBot) for multi-agent AI support

---

<p align="center">Built with ❤️ for better dengue forecasting in Brazil 🇧🇷</p>
