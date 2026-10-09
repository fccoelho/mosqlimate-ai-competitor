# Enhanced Report Command - Summary

## Overview

The `mosqlimate-ai report` command has been significantly enhanced with rich visualizations that transform the previously "arid and boring" text-only report into a comprehensive visual dashboard.

## New Features

### 1. 📈 Rich Visualizations

The report now includes **13 different plot types** automatically generated for each evaluation:

#### Time Series Forecasts
- **Forecast vs Observed Plots**: Line charts showing predicted vs actual values
- **Prediction Intervals**: Shaded areas for 50%, 80%, and 95% confidence intervals
- **Multi-Model Comparison**: Side-by-side comparison of all models

#### Residual Analysis
- **Residuals vs Predicted**: Scatter plot to check for patterns
- **Residuals Over Time**: Time series of errors
- **Histogram**: Distribution of residuals
- **Q-Q Plot**: Normality check for residuals

#### Performance Analysis
- **Metrics Comparison**: Bar charts comparing RMSE, MAE across states
- **Error Distribution**: Box plots showing error spread by model
- **Performance Heatmap**: Color-coded performance matrix

#### Calibration Analysis
- **Coverage Analysis**: Bar charts showing prediction interval coverage
- **Calibration Curves**: Nominal vs empirical coverage

### 2. 🎨 Visual Improvements

- **Emojis and Formatting**: Rich markdown with emojis (🎯 📊 🏆 📈 📋 📖)
- **Better Organization**: Clear sections with visual separators
- **Color Coding**: Heatmaps use color to highlight best/worst performers
- **Professional Styling**: Clean matplotlib styling with seaborn

### 3. ⚙️ New CLI Option

```bash
# Generate report with plots (default)
mosqlimate-ai report --plots

# Generate report without plots (text only)
mosqlimate-ai report --no-plots

# Specify output location
mosqlimate-ai report --output my_report.md
```

### 4. 📁 Output Structure

```
output_directory/
├── forecast_report.md          # Main markdown report
└── figures/                    # Generated plots
    ├── metrics_comparison_rmse.png
    ├── metrics_comparison_mae.png
    ├── coverage_analysis.png
    ├── error_distribution.png
    ├── performance_heatmap.png
    ├── SP_model_comparison.png
    ├── SP_xgboost_timeseries.png
    ├── SP_xgboost_residuals.png
    ├── SP_xgboost_calibration.png
    ├── SP_lstm_timeseries.png
    ├── SP_lstm_residuals.png
    ├── SP_lstm_calibration.png
    ├── SP_ensemble_timeseries.png
    └── ... (for each state and model)
```

## Implementation Details

### New Module: `mosqlimate_ai.visualization.report_plots`

The visualization functionality is encapsulated in a new module with the `ReportVisualizer` class:

```python
from mosqlimate_ai.visualization.report_plots import ReportVisualizer

# Create visualizer
viz = ReportVisualizer(output_dir=Path("figures"))

# Generate plots
viz.plot_forecast_timeseries(df_obs, forecast, state, model)
viz.plot_residuals(y_true, forecast, state, model)
viz.plot_calibration_curve(y_true, forecast, state, model)
viz.plot_metrics_comparison(all_results, metric="rmse")
viz.plot_coverage_analysis(all_results)
viz.plot_error_distribution(all_results)
viz.create_summary_heatmap(all_results)
```

### Key Methods

1. **`plot_forecast_timeseries()`**: Shows forecast with prediction intervals overlaid on observed data
2. **`plot_residuals()`**: 4-panel residual diagnostic plots
3. **`plot_calibration_curve()`**: Checks if prediction intervals are well-calibrated
4. **`plot_metrics_comparison()`**: Bar charts comparing models across states
5. **`plot_coverage_analysis()`**: Visualizes prediction interval coverage with color coding
6. **`plot_error_distribution()`**: Box plots of error metrics
7. **`create_summary_heatmap()`**: Normalized performance heatmap
8. **`plot_multi_model_comparison()`**: Side-by-side model comparison for a state

### Features

- **High Resolution**: All plots saved at 150 DPI for crisp display
- **Consistent Styling**: Seaborn whitegrid style for professional look
- **Automatic Colors**: Models get distinct colors automatically
- **Smart Scaling**: Axis limits and figure sizes adapt to data
- **Error Handling**: Graceful handling of missing data or failed plots

## Example Output

The enhanced report includes:

```markdown
# 🎯 Forecast Performance Report

**Generated:** 2026-03-19 15:53:56
**States Evaluated:** 1
**Models Compared:** xgboost, lstm, ensemble

---

## 📊 Executive Summary

### Average Performance Across All States

| Model | CRPS | WIS Total | RMSE | MAE | MAPE | Bias | Coverage 95% | Coverage 50% |
|-------|------|-----------|------|-----|------|------|--------------|--------------|
| XGBOOST | 1143628.09 | 1177.02 | 341.72 | 103.02 | 4.46% | 13.89 | 0.00% | 60.61% |

### 🏆 Best Model by Metric

| Metric | Best Model | Value |
|--------|------------|-------|
| CRPS | LSTM | 503638.87 |

---

## 📈 Visualizations

### Performance Overview

#### RMSE Comparison Across States
![RMSE Comparison](figures/metrics_comparison_rmse.png)

#### Error Distribution by Model
![Error Distribution](figures/error_distribution.png)

... and more visualizations
```

## Usage

```bash
# Basic usage with plots (recommended)
mosqlimate-ai report

# Generate report for specific states
mosqlimate-ai report --states SP,RJ,MG

# Generate report without plots (faster)
mosqlimate-ai report --no-plots

# Custom output location
mosqlimate-ai report --output reports/my_analysis.md

# View the report
cat forecast_report.md
# or open in a markdown viewer
```

## Benefits

1. **Faster Insights**: Visual patterns are immediately apparent
2. **Better Communication**: Share visual reports with stakeholders
3. **Quality Control**: Residual plots help identify model issues
4. **Model Comparison**: Easy visual comparison of different approaches
5. **Calibration Check**: Coverage analysis shows if uncertainty is well-estimated

## Technical Notes

- Uses `matplotlib` and `seaborn` for plotting
- Plots saved as PNG files in `figures/` subdirectory
- Each plot is 150 DPI for high quality
- Total runtime increases slightly due to plot generation (~2-5 seconds per state)
- Reports are self-contained with relative paths to images

## Dependencies

The visualization features require:
- matplotlib
- seaborn
- numpy
- pandas
- scipy (for Q-Q plots)

All already included in the project dependencies.
