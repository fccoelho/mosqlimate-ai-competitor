# Data Pipeline Documentation

## Overview

The Mosqlimate AI Competitor uses a comprehensive data pipeline to ingest, process, and transform epidemiological and environmental data for dengue forecasting. This document describes the complete data flow from raw sources to model-ready features.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Data Processing │────▶│  Feature Store  │
│  (FTP/Local)    │     │  (Pipeline)      │     │  (Features)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Competition    │     │  Preprocessing   │     │  Model Input    │
│  Data Files     │     │  & Cleaning      │     │  (X, y)         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Data Sources

### Primary Data Files

| File | Description | Frequency | Size |
|------|-------------|-----------|------|
| `dengue.csv.gz` | Weekly dengue cases by municipality | Weekly | ~4.5M records |
| `chikungunya.csv.gz` | Weekly chikungunya cases by municipality | Weekly | ~2M records |
| `climate.csv.gz` | ERA5 climate reanalysis data | Weekly | ~4.5M records |
| `forecasting_climate.csv.gz` | ECMWF monthly forecasts (issue month × lead 1–6) | Monthly | >6M rows / >1.5GB in memory |
| `datasus_population_2001_*.csv.gz` | Municipal population data | Yearly | ~150K records |
| `environ_vars.csv.gz` | Environmental variables (Koppen, Biome) | Static | 5.5K records |
| `ocean_climate_oscillations*.csv.gz` | ENSO, IOD, PDO indices | Monthly | Time series |
| `*_update_2025/2026.csv.gz` | Partial-season update segments (merged automatically, latest vintage wins on duplicate keys) | — | — |

### Data Schema

#### Dengue Data
```python
{
    "date": "datetime64[ns]",           # Week start date
    "geocode": "int64",                 # Municipality IBGE code
    "casos": "float64",                 # Reported dengue cases
    "uf": "str",                        # State abbreviation
    "epiweek": "int64",                 # Epidemiological week
    "year": "int64"                     # Year
}
```

#### Climate Data
```python
{
    "date": "datetime64[ns]",
    "geocode": "int64",
    "temp_mean": "float64",             # Mean temperature (°C)
    "temp_max": "float64",              # Maximum temperature
    "temp_min": "float64",              # Minimum temperature
    "precipitation": "float64",         # Precipitation (mm)
    "humidity": "float64",              # Relative humidity (%)
    "pressure": "float64"               # Atmospheric pressure
}
```

## Data Flow

### 1. Download & Cache

**Module**: `mosqlimate_ai.data.downloader`

```python
from mosqlimate_ai.data.downloader import DataDownloader

downloader = DataDownloader()
downloader.download_all(force=False)  # fetches only what is missing/stale
```

**Features**:
- FTP download from `info.dengue.mat.br`
- **Never re-downloads a complete file**: a local copy is trusted when
  it is recorded in the completeness-verified manifest
  (`data/completeness_verified.json`, written by `mosqlimate-ai
  check-data`), when its size matches the remote size, or when the
  server does not report a size
- **Resume of interrupted transfers** via FTP `REST`: a partial
  `<file>.csv.gz.tmp` is continued from its byte offset instead of
  restarting; a `.tmp` that already holds the full transfer is simply
  finalized (zero bytes re-fetched)
- Size verification before the temp file is renamed into place — an
  incomplete transfer stays as `.tmp` and resumes on the next attempt
- Stale `.tmp` files are cleaned up when the target file is already valid
- Progress bars with Rich; `--force` overrides every skip rule

### 2. Data Loading

**Module**: `mosqlimate_ai.data.loader`

```python
from mosqlimate_ai.data.loader import CompetitionDataLoader

loader = CompetitionDataLoader()
df = loader.load_state_data(uf="SP", aggregate=True)
```

**Key Classes**:
- `CompetitionDataLoader`: Main entry point
- Loads and merges multiple data sources
- Aggregates municipality data to state level
- Handles temporal joins
- Every source table is lazily loaded **once per loader instance** and
  cached (pass the same loader around instead of constructing new ones)

**Loading Methods**:

| Method | Purpose |
|--------|---------|
| `load_state_data()` | State-specific data (merged + weekly reindexed) |
| `load_all_states()` | All 27 Brazilian states in one pass |
| `load_merged_data()` | Join all sources |
| `pop_weighted_climate_forecast(uf)` | Population-weighted monthly ECMWF aggregate for a state — computed for **all states in a single streaming pass** over `forecasting_climate*.csv.gz` (the >6M-row / >1.5GB table is never held in memory, and the CSVs are read once per loader instead of once per state) |
| `load_climate_forecast_for_state()` | Raw per-state climate-forecast rows (filtered chunked read; prefer the method above) |
| `load_ocean_data()` | ENSO / IOD / PDO series (base + update vintages merged, latest wins) |

The exogenous-covariate providers (`future_exog.ExogLookup` and
`future_exog.build_future_exog`) use
`pop_weighted_climate_forecast()` when available, so a validation run
reads the large climate-forecast CSVs exactly once per process
(regardless of how many states/windows are requested) and never loads
the full table.

### 3. Preprocessing

**Module**: `mosqlimate_ai.data.preprocessor`

```python
from mosqlimate_ai.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean(df)
df_imputed = preprocessor.impute_missing(df_clean)
df_features = preprocessor.add_epidemiological_features(df_imputed)
```

**Preprocessing Steps**:

1. **Data Cleaning** (`clean()`)
   - Remove duplicates
   - Handle outliers (IQR method)
   - Validate date ranges
   - Standardize column names

2. **Missing Value Imputation** (`impute_missing()`)
   - Seasonal decomposition for trends
   - Linear interpolation for short gaps
   - Spline interpolation for longer gaps
   - Forward/backward fill for edges

3. **Epidemiological Features** (`add_epidemiological_features()`)
   - Incidence rates (cases per 100k)
   - Log-transformed cases
   - Lag features (1-52 weeks)
   - Rolling statistics

### 4. Feature Engineering

**Module**: `mosqlimate_ai.data.features`

```python
from mosqlimate_ai.data.features import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.build_feature_set(df, target_col="casos")
```

**Feature Categories**:

#### Temporal Features
- Cyclical encoding: `sin/cos(epiweek/52 * 2π)`
- Season indicators: rainy/dry season
- Holiday flags (carnival, new year)
- Year-over-year changes

#### Lag Features
```python
# Example: 1, 2, 4, 8 week lags
lag_features = ["casos_lag_1", "casos_lag_2", "casos_lag_4", "casos_lag_8"]
```

#### Rolling Statistics
- Moving averages (4, 8, 12 weeks)
- Standard deviations
- Min/max over windows
- Exponential moving averages

#### Climate Interactions
- Heat index calculation
- Temperature-humidity interactions
- Precipitation anomalies
- Climate indices (ENSO, IOD, PDO)

#### Spatial Features
- Neighboring state lags
- Regional aggregates
- Population-weighted features

## Usage Examples

### Complete Pipeline

```python
from mosqlimate_ai.data.downloader import DataDownloader
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.data.preprocessor import DataPreprocessor
from mosqlimate_ai.data.features import FeatureEngineer

# 1. Download data
downloader = DataDownloader()
downloader.download_all()

# 2. Load data for São Paulo
loader = CompetitionDataLoader()
df = loader.load_state_data(uf="SP", aggregate=True)

# 3. Preprocess
preprocessor = DataPreprocessor()
df = preprocessor.clean(df)
df = preprocessor.impute_missing(df)
df = preprocessor.add_epidemiological_features(df)

# 4. Build features
engineer = FeatureEngineer()
ocean_df = loader.load_ocean_data()
features = engineer.build_feature_set(
    df, 
    target_col="casos",
    ocean_df=ocean_df
)
```

### CLI Commands

```bash
# Download all competition data (never re-downloads complete files)
mosqlimate-ai download-data

# Check cache status
mosqlimate-ai cache-info

# Clear cache
mosqlimate-ai clear-cache

# Verify weekly completeness (writes the verified manifest on success)
mosqlimate-ai check-data

# Refresh only stale files; honors the verified manifest
python scripts/refresh_data.py
```

## Data Quality

### Validation Checks

The pipeline performs several validation checks:

1. **Completeness**: Missing data percentage by column
2. **Consistency**: Date ranges, geographic codes
3. **Accuracy**: Outlier detection, logical constraints
4. **Timeliness**: Recent data availability

### Monitoring

```python
from mosqlimate_ai.data.loader import CompetitionDataLoader

loader = CompetitionDataLoader()
print(f"Available states: {loader.get_available_states()}")
print(f"Date range: {loader.get_date_range()}")
```

Weekly completeness is checked explicitly (missing weeks are reported,
never silently filled) — see `mosqlimate_ai.data.completeness` and the
`check-data` CLI command.

## Performance Considerations

### No redundant dataset reads
- Each CSV is read **once per process**: the loader caches every table
  (cases, climate, ocean, population, climate-forecast aggregates) and
  is shared across the validation pipeline
- The climate-forecast files are aggregated for all states in a single
  chunked pass (sufficient statistics), never re-scanned per state and
  never fully materialized
- In the backtest harness the main process builds the heavy merged
  municipality table once (shared by tuning and test preparation, then
  released), and each worker process reuses one shared loader for all
  its (state, window) jobs
- On disk there is no redundancy either: complete files are never
  re-downloaded and interrupted downloads resume (see
  *Download & Cache* above)

### Memory Usage
- Data loaded in chunks for large files
- Automatic garbage collection
- Efficient pandas operations

### Optimization Tips
1. Use `aggregate=True` to reduce data size
2. Filter by date range when possible
3. Cache processed features
4. Use parquet format for intermediate storage

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```bash
   mosqlimate-ai download-data --force
   ```

2. **Memory Errors**
   ```python
   # Process states individually
   for uf in ["SP", "RJ", "MG"]:
       df = loader.load_state_data(uf=uf)
       # Process each state separately
   ```

3. **Date Parsing Errors**
   ```python
   # Ensure date column is properly parsed
   df["date"] = pd.to_datetime(df["date"])
   ```

## References

- [Mosqlimate Platform](https://mosqlimate.org/)
- [InfoDengue](https://info.dengue.mat.br/)
- [IBGE Geocodes](https://www.ibge.gov.br/)
- [ERA5 Climate Data](https://cds.climate.copernicus.eu/)
