# Models Documentation

## Overview

The Mosqlimate AI Competitor implements a multi-model ensemble approach for dengue forecasting. This document describes the model architecture, training procedures, and usage guidelines.

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ensemble Forecaster                       │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   XGBoost       │    │     LSTM        │                │
│  │  (Quantile)     │    │ (MC Dropout)    │                │
│  │                 │    │                 │                │
│  │ 9 quantile      │    │ 100 MC samples  │                │
│  │ models          │    │ for uncertainty │                │
│  └────────┬────────┘    └────────┬────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                       │
│           CRPS-Optimized Weighting                           │
│                      │                                       │
│                      ▼                                       │
│           Final Predictions                                  │
│    (median + prediction intervals)                           │
└─────────────────────────────────────────────────────────────┘
```

## Models

### 1. XGBoost Quantile Regression

**Module**: `mosqlimate_ai.models.xgboost_model`

**Purpose**: Gradient boosting with quantile regression for prediction intervals

**Key Features**:
- Trains 9 separate models for different quantiles
- Supports quantiles: [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
- Early stopping with validation monitoring
- Feature importance tracking
- Cross-validation support

**Default Hyperparameters**:
```python
{
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "early_stopping_rounds": 50,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

**Usage**:
```python
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster

# Initialize model
model = XGBoostForecaster(
    target_col="casos",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05
)

# Train
model.fit(df, validation_size=0.1, verbose=True)

# Predict with intervals
predictions = model.predict(df, levels=[0.50, 0.80, 0.95])

# Save/Load
model.save("models/SP/xgboost")
model.load("models/SP/xgboost")
```

**Output Format**:
```python
{
    "median": [...],           # 0.50 quantile
    "lower_50": [...],         # 0.25 quantile
    "upper_50": [...],         # 0.75 quantile
    "lower_80": [...],         # 0.10 quantile
    "upper_80": [...],         # 0.90 quantile
    "lower_95": [...],         # 0.025 quantile
    "upper_95": [...],         # 0.975 quantile
    "date": [...],             # Forecast dates
    "uf": [...]                # State code
}
```

### 2. LSTM with Monte Carlo Dropout

**Module**: `mosqlimate_ai.models.lstm_model`

**Purpose**: Deep learning sequence model with uncertainty quantification

**Key Features**:
- 2-layer LSTM architecture
- Sequence-to-sequence prediction
- Monte Carlo Dropout for uncertainty (100 samples)
- Automatic feature scaling
- Early stopping with model checkpointing

**Architecture**:
```
Input (52 timesteps × n_features)
    ↓
LSTM Layer 1 (hidden_size=128)
    ↓
Dropout (p=0.2)
    ↓
LSTM Layer 2 (hidden_size=128)
    ↓
Dropout (p=0.2)
    ↓
Linear Layer (output_size=1)
    ↓
Output (1 prediction)
```

**Default Hyperparameters**:
```python
{
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "sequence_length": 52,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "patience": 10
}
```

**Usage**:
```python
from mosqlimate_ai.models.lstm_model import LSTMForecaster

# Initialize model
model = LSTMForecaster(
    target_col="casos",
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    epochs=100
)

# Train
model.fit(df, validation_size=0.1, verbose=True)

# Predict with uncertainty
predictions = model.predict(df, n_mc_samples=100)

# Save/Load
model.save("models/SP/lstm")
model.load("models/SP/lstm")
```

**Training Considerations**:
- Requires PyTorch
- GPU acceleration supported but not required
- Sequence length of 52 weeks (1 year lookback)
- Handles missing values with forward fill

### 3. Ensemble Forecaster

**Module**: `mosqlimate_ai.models.ensemble`

**Purpose**: Combine multiple models for robust predictions

**Methods**:
1. **Weighted Average**: CRPS-optimized weights
2. **Median**: Median-of-medians ensemble
3. **Stacking**: Meta-learner (planned)

**Key Features**:
- CRPS-based weight optimization
- Conformal calibration for intervals
- Individual model tracking
- Automatic model selection

**Usage**:
```python
from mosqlimate_ai.models.ensemble import EnsembleForecaster

# Create ensemble
ensemble = EnsembleForecaster(method="weighted_average")

# Add models
ensemble.add_model("xgboost", xgb_predictions, weight=0.6)
ensemble.add_model("lstm", lstm_predictions, weight=0.4)

# Optimize weights (optional)
ensemble.fit_weights(y_validation)

# Generate ensemble prediction
final_predictions = ensemble.predict()

# With individual models
final_predictions, individual_models = ensemble.predict(return_individual=True)
```

**Weight Optimization**:
```python
# Automatic weight optimization based on validation CRPS
ensemble.fit_weights(y_true=y_val, optimize=True)

# Results in optimized weights:
# ensemble.weights = {'xgboost': 0.7, 'lstm': 0.3}
```

## Training Pipeline

### CLI Training

```bash
# Train all models for all states
mosqlimate-ai train

# Train specific models
mosqlimate-ai train --models xgboost
mosqlimate-ai train --models lstm

# Train specific states
mosqlimate-ai train --states SP,RJ,MG

# Custom output directory
mosqlimate-ai train --output ./my_models

# Verbose training output
mosqlimate-ai train --verbose
```

### Programmatic Training

```python
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.data.preprocessor import DataPreprocessor
from mosqlimate_ai.data.features import FeatureEngineer
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster
from mosqlimate_ai.models.lstm_model import LSTMForecaster

# Load and prepare data
loader = CompetitionDataLoader()
df = loader.load_state_data(uf="SP", aggregate=True)

preprocessor = DataPreprocessor()
df = preprocessor.clean(df)
df = preprocessor.impute_missing(df)
df = preprocessor.add_epidemiological_features(df)

engineer = FeatureEngineer()
ocean_df = loader.load_ocean_data()
df = engineer.build_feature_set(df, target_col="casos", ocean_df=ocean_df)

# Train XGBoost
xgb_model = XGBoostForecaster(
    target_col="casos",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05
)
xgb_model.fit(df, validation_size=0.1)
xgb_model.save("models/SP/xgboost")

# Train LSTM
lstm_model = LSTMForecaster(
    target_col="casos",
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    epochs=100
)
lstm_model.fit(df, validation_size=0.1)
lstm_model.save("models/SP/lstm")
```

## Model Storage

### Directory Structure

```
models/
├── SP/
│   ├── xgboost/
│   │   ├── model.json          # XGBoost model
│   │   ├── quantile_0.025.json # Individual quantile models
│   │   ├── quantile_0.050.json
│   │   ├── ...
│   │   └── forecaster_config.pkl
│   └── lstm/
│       ├── model.pt            # PyTorch state dict
│       ├── scaler.pkl          # Feature scaler
│       └── forecaster_config.pkl
├── RJ/
│   ├── xgboost/
│   └── lstm/
└── ...
```

### Serialization

**XGBoost**:
- Uses XGBoost's native JSON format
- Separate file per quantile model
- Config includes target_col and feature_cols

**LSTM**:
- PyTorch state dict (`.pt` file)
- Scikit-learn scaler for features
- Config includes architecture parameters

## Model Evaluation

### Individual Model Evaluation

```python
from mosqlimate_ai.evaluation.metrics import evaluate_forecast

# Evaluate XGBoost
y_true = df["casos"].values
xgb_pred = xgb_model.predict(df, levels=[0.50, 0.80, 0.95])
xgb_metrics = evaluate_forecast(y_true, xgb_pred)

print(f"XGBoost CRPS: {xgb_metrics['crps']:.4f}")
print(f"XGBoost WIS: {xgb_metrics['wis_total']:.4f}")

# Evaluate LSTM
lstm_pred = lstm_model.predict(df, n_mc_samples=100)
lstm_metrics = evaluate_forecast(y_true, lstm_pred)

print(f"LSTM CRPS: {lstm_metrics['crps']:.4f}")
print(f"LSTM WIS: {lstm_metrics['wis_total']:.4f}")
```

### Ensemble Evaluation

```python
# Create ensemble
ensemble = EnsembleForecaster()
ensemble.add_model("xgboost", xgb_pred)
ensemble.add_model("lstm", lstm_pred)
ensemble_pred = ensemble.predict()

# Evaluate
ensemble_metrics = evaluate_forecast(y_true, ensemble_pred)

# Compare models
import pandas as pd
comparison = pd.DataFrame({
    "XGBoost": xgb_metrics,
    "LSTM": lstm_metrics,
    "Ensemble": ensemble_metrics
})
print(comparison.T)
```

### CLI Evaluation

```bash
# Evaluate forecasts for a specific state
mosqlimate-ai evaluate --state SP

# Evaluate with custom end date
mosqlimate-ai evaluate --state SP --end-date 2025-12-31

# Generate comprehensive report
mosqlimate-ai report --output model_comparison.md
```

## Hyperparameter Tuning

### Configuration File

```yaml
# config.yaml
models:
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    early_stopping_rounds: 50
    
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    epochs: 100
    learning_rate: 0.001
    
  ensemble:
    method: weighted_average
    weight_metric: crps
    calibrate_intervals: true
```

### Grid Search (Example)

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 500, 700]
}

best_score = float("inf")
best_params = None

for params in ParameterGrid(param_grid):
    model = XGBoostForecaster(**params)
    model.fit(df, validation_size=0.1)
    pred = model.predict(df_val)
    score = crps(y_val, pred)
    
    if score < best_score:
        best_score = score
        best_params = params

print(f"Best params: {best_params}")
print(f"Best CRPS: {best_score:.4f}")
```

## Best Practices

### 1. Feature Engineering
- Include lag features (1-52 weeks)
- Add temporal features (seasonality)
- Consider climate interactions
- Use population normalization

### 2. Validation Strategy
- Use time-based splits (not random)
- Reserve recent data for validation
- Consider multiple validation periods

### 3. Model Selection
- Start with XGBoost (faster, good baseline)
- Add LSTM for complex temporal patterns
- Use ensemble for best performance

### 4. Prediction Intervals
- Target coverage: 50%, 80%, 95%
- Calibrate intervals if coverage is off
- Monitor interval width (should be reasonable)

### 5. State-Specific Models
- Train separate models per state
- States have different epidemic patterns
- Share features but not model parameters

## Troubleshooting

### XGBoost Issues

**Issue**: Poor quantile calibration
```python
# Increase number of trees
model = XGBoostForecaster(n_estimators=1000, learning_rate=0.01)

# Or adjust quantile-specific parameters
# Lower learning rate for extreme quantiles
```

**Issue**: Overfitting
```python
# Reduce model complexity
model = XGBoostForecaster(
    max_depth=4,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7
)
```

### LSTM Issues

**Issue**: Not converging (high loss)
```python
# Reduce learning rate
model = LSTMForecaster(learning_rate=0.0001)

# Or simplify architecture
model = LSTMForecaster(hidden_size=64, num_layers=1)
```

**Issue**: Overfitting
```python
# Increase dropout
model = LSTMForecaster(dropout=0.5)

# Or add regularization
model = LSTMForecaster(weight_decay=0.01)
```

### Ensemble Issues

**Issue**: Ensemble worse than individual models
```python
# Check model diversity
# Models should make different errors

# Use median instead of weighted average
ensemble = EnsembleForecaster(method="median")

# Or optimize weights more carefully
ensemble.fit_weights(y_val, optimize=True)
```

## Performance Benchmarks

Based on evaluation on São Paulo (SP) data:

| Model | CRPS | WIS | RMSE | Coverage 95% | Coverage 50% |
|-------|------|-----|------|--------------|--------------|
| XGBoost | 1,143,628 | 1,177 | 341.7 | 0.00% | 60.61% |
| LSTM | 483,597 | 114,033 | 10,539 | 2.74% | 2.05% |
| Ensemble | 2,706,737 | 16,548 | 5,332 | 0.00% | 0.68% |

*Note: Results vary by state and time period*

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression)
- [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142)
- [CRPS Scoring](https://en.wikipedia.org/wiki/Scoring_rule)
