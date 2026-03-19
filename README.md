# Mosqlimate AI Competitor 🤖🦟

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

## 🔬 Models

### Ensemble Approach

1. **Baseline Models**
   - ARIMA/SARIMA (time series)
   - Prophet (Facebook)
   - Exponential Smoothing

2. **ML Models**
   - XGBoost / LightGBM
   - Random Forest
   - Gradient Boosting

3. **Deep Learning**
   - LSTM / GRU (temporal patterns)
   - Transformer (attention mechanisms)
   - Temporal Fusion Transformer (TFT)

4. **Hybrid Models**
   - Climate-epidemiological coupled models
   - Mechanistic + ML ensemble

## 📈 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **CRPS**: Continuous Ranked Probability Score
- **Interval Score**: Prediction interval calibration

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
# Formatting
black src/ tests/
ruff check src/ tests/

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
