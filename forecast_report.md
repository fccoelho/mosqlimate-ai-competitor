# 🎯 Forecast Performance Report

**Generated:** 2026-03-19 16:20:58
**States Evaluated:** 1
**Models Compared:** xgboost, lstm, ensemble

---

## 📊 Executive Summary

### Average Performance Across All States

| Model | CRPS | WIS Total | RMSE | MAE | MAPE | Bias | Coverage 95% | Coverage 50% |
|-------|------|-----------|------|-----|------|------|--------------|--------------|
| XGBOOST | 1143628.0910 | 1177.0239 | 341.7219 | 103.0242 | 4.46% | 13.8890 | 0.00% | 60.61% |
| LSTM | 497588.8929 | 113913.4299 | 10540.5089 | 6316.3335 | 555.31% | -4569.7359 | 2.74% | 2.05% |
| ENSEMBLE | 2707894.5485 | 16566.3672 | 5335.3134 | 3243.4778 | 199.07% | -2386.0090 | 0.00% | 0.68% |

### 🏆 Best Model by Metric

| Metric | Best Model | Value |
|--------|------------|-------|
| CRPS | LSTM | 497588.8929 |
| WIS_TOTAL | XGBOOST | 1177.0239 |
| RMSE | XGBOOST | 341.7219 |
| MAE | XGBOOST | 103.0242 |
| MAPE | XGBOOST | 4.4573 |

---

## 📈 Visualizations

### Performance Overview

#### RMSE Comparison Across States

![RMSE Comparison](figures/metrics_comparison_rmse.png)

#### Error Distribution by Model

![Error Distribution](figures/error_distribution.png)

#### Performance Heatmap

![Performance Heatmap](figures/performance_heatmap.png)

### Prediction Interval Coverage

![Coverage Analysis](figures/coverage_analysis.png)

*Note: Bars colored green when within 10% of target, orange within 20%, red otherwise.*

---

## 📋 Detailed Results by State

### SP

#### Performance Metrics

| Model | CRPS | WIS Total | RMSE | MAE | MAPE | Bias | Coverage 95% | Coverage 50% |
|-------|------|-----------|------|-----|------|------|--------------|--------------|
| XGBOOST | 1143628.0910 | 1177.0239 | 341.7219 | 103.0242 | 4.46% | 13.8890 | 0.00% | 60.61% |
| LSTM | 497588.8929 | 113913.4299 | 10540.5089 | 6316.3335 | 555.31% | -4569.7359 | 2.74% | 2.05% |
| ENSEMBLE | 2707894.5485 | 16566.3672 | 5335.3134 | 3243.4778 | 199.07% | -2386.0090 | 0.00% | 0.68% |

#### Forecast Visualizations

**Model Comparison:**

![SP Model Comparison](figures/SP_model_comparison.png)

**Xgboost Forecast:**

![SP xgboost Timeseries](figures/SP_xgboost_timeseries.png)

**Residual Analysis:**

![SP xgboost Residuals](figures/SP_xgboost_residuals.png)

**Calibration Curve:**

![SP xgboost Calibration](figures/SP_xgboost_calibration.png)

**Lstm Forecast:**

![SP lstm Timeseries](figures/SP_lstm_timeseries.png)

**Residual Analysis:**

![SP lstm Residuals](figures/SP_lstm_residuals.png)

**Calibration Curve:**

![SP lstm Calibration](figures/SP_lstm_calibration.png)

**Ensemble Forecast:**

![SP ensemble Timeseries](figures/SP_ensemble_timeseries.png)

**Residual Analysis:**

![SP ensemble Residuals](figures/SP_ensemble_residuals.png)

**Calibration Curve:**

![SP ensemble Calibration](figures/SP_ensemble_calibration.png)


---

## 📖 Metric Definitions

| Metric | Description |
|--------|-------------|
| **CRPS** | Continuous Ranked Probability Score - measures the integrated squared difference between the empirical CDF and the predicted CDF. Lower is better. |
| **WIS Total** | Weighted Interval Score - combines interval sharpness with penalty for misses across multiple confidence levels. Lower is better. |
| **RMSE** | Root Mean Square Error - square root of the average of squared errors. Lower is better. |
| **MAE** | Mean Absolute Error - average of absolute prediction errors. Lower is better. |
| **MAPE** | Mean Absolute Percentage Error - percentage error relative to actual values. Lower is better. |
| **Bias** | Systematic error (mean of predictions minus actual). Positive = overestimation, Negative = underestimation. |
| **Coverage 95%** | Percentage of true values within the 95% prediction interval. Ideal: ~95%. |
| **Coverage 50%** | Percentage of true values within the 50% prediction interval. Ideal: ~50%. |
