# Mosqlimate AI Competitor 🤖🦟

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-blue.svg)](https://github.com/astral-sh/ruff)

AI-powered dengue forecasting system for the [Infodengue-Mosqlimate Dengue Challenge (IMDC)](https://sprint.mosqlimate.org), leveraging multi-agent coding with **Karl DBot**.

## 🎯 Objective

Predict probable dengue cases in Brazil (nationally and by state) for upcoming epidemic seasons using machine learning and multi-agent AI systems.

## 🏗️ Architecture

```
mosqlimate-ai-competitor/
├── src/mosqlimate_ai/
│   ├── data/           # Data ingestion & preprocessing
│   ├── models/         # ML/DL forecasting models
│   ├── agents/         # Karl DBot multi-agent system
│   ├── evaluation/     # Model validation & metrics
│   └── submission/     # Competition submission format
├── notebooks/          # Exploratory analysis
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- [Karl DBot](https://github.com/Deeplearn-PeD/KarlDBot) (multi-agent coding tool)

### Installation

```bash
# Clone repository
git clone git@github.com:fccoelho/mosqlimate-ai-competitor.git
cd mosqlimate-ai-competitor

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### CLI Quick Reference

```bash
# Download competition data
mosqlimate-ai download-data

# Train models
mosqlimate-ai train --states SP,RJ

# Generate forecasts
mosqlimate-ai forecast --weeks 52

# Evaluate forecasts
mosqlimate-ai evaluate --state SP

# Generate performance report
mosqlimate-ai report --output report.md

# Submit to competition
mosqlimate-ai submit --model-id 123

# View all commands
mosqlimate-ai --help
```

## 📊 Data Sources

The system integrates data from [Mosqlimate Platform](https://mosqlimate.org/):

- **Epidemiological**: SINAN dengue cases (weekly)
- **Climate**: Temperature, precipitation, humidity
- **Demographic**: Population data by municipality
- **Historical**: 5-10 years of time series data

### Downloading Competition Data

Download the Sprint 2025 competition data using the CLI:

```bash
# Download all required data files
mosqlimate-ai download-data

# Force re-download (overwrite existing files)
mosqlimate-ai download-data --force

# Specify custom cache directory
mosqlimate-ai download-data --cache-dir /path/to/data
```

**Available Data Files:**

| File | Description |
|------|-------------|
| `dengue.csv.gz` | Weekly dengue cases by municipality (2010-2025) |
| `climate.csv.gz` | Weekly climate reanalysis data (ERA5) |
| `climate_forecast.csv.gz` | Monthly climate forecasts (ECMWF) |
| `datasus_population_2001_2024.csv.gz` | Population by municipality and year |
| `environ_vars.csv.gz` | Environmental variables (Koppen, Biome) |
| `map_regional_health.csv` | City to health region mapping |
| `shape_muni.gpkg` | Municipality geometries |
| `shape_regional_health.gpkg` | Regional health geometries |
| `shape_macroregional_health.gpkg` | Macroregional health geometries |
| `ocean_climate_oscillations.csv.gz` | ENSO, IOD, PDO ocean indices |

**Other CLI commands:**

```bash
# View cached data status
mosqlimate-ai cache-info

# Clear the data cache
mosqlimate-ai clear-cache
```

## 🤖 Multi-Agent System (Karl DBot)

Our AI competitor uses specialized agents:

| Agent | Responsibility |
|-------|---------------|
| **Data Engineer** | Data collection, cleaning, feature engineering |
| **Model Architect** | Design ML/DL architectures, hyperparameter tuning |
| **Forecaster** | Generate predictions, uncertainty quantification |
| **Validator** | Cross-validation, backtesting, metrics calculation |
| **Ensembler** | Combine multiple models for robust forecasts |

### Agent Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Data      │────▶│   Model      │────▶│  Forecast   │
│  Engineer   │     │  Architect   │     │     Agent   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Submission │◀────│  Ensembler   │◀────│  Validator  │
│   Formatter │     │     Agent    │     │    Agent    │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 🏋️ Model Training

Train forecasting models using the CLI. Models are trained per state and saved for later use.

### Train All States

```bash
# Train XGBoost and LSTM models for all states
mosqlimate-ai train

# Train only XGBoost
mosqlimate-ai train --models xgboost

# Train only LSTM
mosqlimate-ai train --models lstm

# Specify output directory
mosqlimate-ai train --output ./my_models
```

### Train Specific States

```bash
# Train models for São Paulo and Rio de Janeiro only
mosqlimate-ai train --states SP,RJ

# Train single state with verbose output
mosqlimate-ai train --states SP --verbose
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output, -o` | Directory to save trained models | `models/` |
| `--states, -s` | Comma-separated state UFs | all states |
| `--models, -m` | Models to train (xgboost,lstm) | `xgboost,lstm` |
| `--val-size` | Validation data fraction | `0.1` |
| `--verbose, -v` | Show detailed training output | `false` |

### Model Output Structure

```
models/
├── SP/
│   ├── xgboost/
│   │   ├── model.json
│   │   └── scaler.pkl
│   └── lstm/
│       ├── model.pt
│       └── scaler.pkl
├── RJ/
│   ├── xgboost/
│   └── lstm/
└── ...
```

## 🔮 Generating Forecasts

After training, generate forecasts with prediction intervals.

### Generate Forecasts

```bash
# Generate 52-week forecasts for all trained states
mosqlimate-ai forecast

# Forecast specific states
mosqlimate-ai forecast --states SP,RJ

# Custom forecast period
mosqlimate-ai forecast --weeks 26 --start-date 2026-01-01

# Output to specific directory
mosqlimate-ai forecast --output ./my_forecasts
```

### Forecast Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-dir, -m` | Directory with trained models | `models/` |
| `--output, -o` | Directory for forecast files | `forecasts/` |
| `--states, -s` | Comma-separated state UFs | all states |
| `--start-date` | Start date (YYYY-MM-DD) | next Sunday |
| `--weeks, -w` | Number of weeks to forecast | `52` |
| `--ensemble/--no-ensemble` | Create ensemble from models | `true` |

### Forecast Output Format

Each state generates a CSV file with prediction intervals:

```csv
date,median,lower_50,upper_50,lower_80,upper_80,lower_95,upper_95
2026-01-05,1250,1000,1500,850,1650,700,1800
2026-01-12,1180,950,1410,800,1560,650,1710
...
```

## 📤 Submitting Forecasts

Submit forecasts to the Mosqlimate API for competition evaluation.

### Submit to API

```bash
# Submit all forecasts (requires API key)
mosqlimate-ai submit --model-id 123

# Dry run to preview submissions
mosqlimate-ai submit --model-id 123 --dry-run

# Save submissions to JSON for review
mosqlimate-ai submit --model-id 123 --dry-run --output-json submissions.json

# Specify forecast directory
mosqlimate-ai submit --forecast-dir ./my_forecasts --model-id 123
```

### Submit Options

| Option | Description | Required |
|--------|-------------|----------|
| `--model-id, -m` | Registered model ID from Mosqlimate | Yes |
| `--forecast-dir, -f` | Directory with forecast files | No (`forecasts/`) |
| `--predict-date` | Prediction date (YYYY-MM-DD) | No (today) |
| `--description, -d` | Prediction description | No |
| `--dry-run` | Prepare without submitting | No |
| `--output-json` | Save submissions to JSON | No |

### API Key Setup

Set your Mosqlimate API key as an environment variable:

```bash
export MOSQLIMATE_API_KEY="your-api-key-here"
```

Or enter it interactively when prompted.

## 📈 Evaluating Forecasts

Evaluate forecast accuracy against historical data for backtesting.

### Run Evaluation

```bash
# Evaluate forecasts for a specific state
mosqlimate-ai evaluate --state SP

# Evaluate against data up to a specific date
mosqlimate-ai evaluate --state RJ --end-date 2025-12-31
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CRPS** | Continuous Ranked Probability Score |
| **WIS** | Weighted Interval Score |
| **Coverage 95%** | Percentage of true values in 95% interval |
| **Coverage 50%** | Percentage of true values in 50% interval |
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Square Error |

## 📊 Performance Reports

Generate comprehensive markdown reports comparing model performance across states.

### Generate Report

```bash
# Generate report for all states
mosqlimate-ai report

# Generate report for specific states
mosqlimate-ai report --states SP,RJ

# Specify custom output path
mosqlimate-ai report --output my_report.md

# Use custom model and forecast directories
mosqlimate-ai report --model-dir ./my_models --forecast-dir ./my_forecasts
```

### Report Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-dir, -m` | Directory with trained models | `models/` |
| `--forecast-dir, -f` | Directory with forecast files | `forecasts/` |
| `--output, -o` | Output markdown file path | `forecast_report.md` |
| `--states, -s` | Comma-separated state UFs | all states |
| `--end-date` | End date for evaluation data | all data |

### Report Contents

The generated report includes:

1. **Executive Summary** - Average performance metrics across all states
2. **Best Model by Metric** - Which model performs best for each metric
3. **Detailed Results by State** - Per-state breakdown for all models
4. **Metric Definitions** - Explanations of each evaluation metric

## 🔬 Models

### Ensemble Approach

1. **ML Models**
   - XGBoost with quantile regression
   - Feature importance analysis

2. **Deep Learning**
   - LSTM with Monte Carlo dropout for uncertainty

3. **Ensemble**
   - Weighted average of model predictions
   - Automatic model selection based on validation performance

## 📝 Submission Format

Following [Mosqlimate submission template](https://github.com/Mosqlimate/mosqlimate-competition):

```csv
date,uf,cases,lower_95,upper_95
2026-01-01,SP,1500,1200,1800
2026-01-01,RJ,800,600,1000
...
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v --cov=src/mosqlimate_ai
```

### Code Quality

```bash
# Linting and formatting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/mosqlimate_ai
```

## 📚 Documentation

- [Competition Rules](docs/competition_rules.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Model Documentation](docs/models.md)
- [Agent Configuration](docs/agents.md)

## 🏆 Competition Results

| Round | Model | RMSE | Rank |
|-------|-------|------|------|
| Validation | - | - | - |
| Final | - | - | - |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Mosqlimate Platform](https://mosqlimate.org/) for organizing the challenge
- [Infodengue](https://info.dengue.mat.br/) for epidemiological data
- [Karl DBot](https://github.com/Deeplearn-PeD/KarlDBot) for multi-agent AI support

## 📧 Contact

- **Project Lead**: [Flávio Codeço Coelho](https://github.com/fccoelho)
- **Issues**: [GitHub Issues](https://github.com/fccoelho/mosqlimate-ai-competitor/issues)

---

<p align="center">Built with ❤️ for better dengue forecasting in Brazil 🇧🇷</p>
