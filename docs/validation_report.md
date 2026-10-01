# Validation PDF Report

## Overview

The Validation PDF Report system generates comprehensive 6-page PDF reports for each state's validation results in the Mosqlimate Sprint 2025 competition. These reports provide detailed analysis of model performance across the 4-run validation pipeline.

## Features

### Report Contents (6 Pages)

1. **Executive Summary**
   - State identification and test period
   - Overall performance metrics (CRPS, WIS)
   - Top model rankings
   - Key insights

2. **Time Series Analysis**
   - Full-page visualization of observed vs predicted cases
   - All 3 validation test predictions overlaid
   - 50%, 80%, and 95% prediction intervals
   - Training period boundaries marked

3. **CRPS Analysis**
   - CRPS progression across tests
   - Per-model comparison charts
   - Detailed CRPS values table

4. **WIS Analysis**
   - WIS progression across tests
   - Weighted Interval Score breakdown
   - Comprehensive WIS values table

5. **Model Performance Comparison**
   - Performance heatmap across tests
   - Model architecture details
   - Hyperparameter summaries

6. **Coverage & Calibration**
   - Prediction interval coverage analysis
   - Coverage statistics for 50%, 80%, 95% intervals
   - Calibration assessment

## Usage

### CLI Command

Generate a validation report from command line:

```bash
# Generate report for a specific state
mosqlimate-ai validation-report SP

# Specify custom results directory
mosqlimate-ai validation-report RJ --results-dir ./my_validation_results

# Custom output path
mosqlimate-ai validation-report MG -o ./reports/mg_validation.pdf
```

### Python API

Generate reports programmatically:

```python
from mosqlimate_ai.visualization import ValidationPDFReport

# Create report generator
report = ValidationPDFReport(state="SP", output_dir="validation_results")

# Generate from saved validation results
pdf_path = report.generate_from_files()
print(f"Report saved to: {pdf_path}")
```

### Advanced Usage

Generate report with data already in memory:

```python
import pandas as pd
from mosqlimate_ai.visualization import ValidationPDFReport

# Your validation results
validation_results = {
    "state": "SP",
    "validation_tests": {...},
    "top_models": [...]
}

# Your observed data
observed_data = pd.DataFrame({
    "date": [...],
    "casos": [...]
})

# Your forecast data
forecast_data = {
    1: pd.DataFrame({...}),  # Test 1 forecasts
    2: pd.DataFrame({...}),  # Test 2 forecasts
    3: pd.DataFrame({...}),  # Test 3 forecasts
}

# Generate report
report = ValidationPDFReport("SP")
pdf_path = report.generate_report(
    validation_results=validation_results,
    observed_data=observed_data,
    forecast_data=forecast_data
)
```

## File Structure

### Source Files

```
src/mosqlimate_ai/visualization/
├── validation_report.py    # Main PDF report generator
├── validation_plots.py     # Visualization functions
└── __init__.py            # Module exports
```

### Output Structure

```
validation_results/
├── SP/
│   ├── validation_results.json
│   ├── SP_validation_report.pdf     # ← Generated report
│   └── report_figures/
│       ├── SP_timeseries.png
│       ├── SP_crps_progression.png
│       ├── SP_wis_progression.png
│       ├── SP_heatmap_crps.png
│       └── SP_coverage.png
├── RJ/
│   └── ...
└── validation_summary.json
```

## Visualization Functions

The `validation_plots.py` module provides specialized plotting functions:

### `plot_validation_timeseries_full()`

Creates the primary time series visualization with all test predictions overlaid.

```python
from mosqlimate_ai.visualization import plot_validation_timeseries_full

fig = plot_validation_timeseries_full(
    observed_df=df_observed,
    test_forecasts={
        1: df_test1,
        2: df_test2,
        3: df_test3
    },
    state="SP",
    train_end_dates={
        1: "2022-06-26",
        2: "2023-06-25",
        3: "2024-06-23"
    }
)
```

### `plot_crps_progression()`

Shows CRPS metric improvement across validation tests.

```python
from mosqlimate_ai.visualization import plot_crps_progression

fig = plot_crps_progression(
    results_by_test={
        1: {"metrics": {"xgboost": {"crps": 0.452}}},
        2: {"metrics": {"xgboost": {"crps": 0.421}}},
        3: {"metrics": {"xgboost": {"crps": 0.398}}}
    },
    models=["xgboost", "lstm", "ensemble"]
)
```

### `plot_wis_progression()`

Shows WIS metric improvement across validation tests.

### `plot_coverage_analysis()`

Analyzes prediction interval coverage for all confidence levels.

### `plot_model_performance_heatmap()`

Creates a heatmap comparing model performance across tests.

## Requirements

- Python 3.9+
- reportlab >= 4.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0
- numpy >= 1.24.0

## Example

See `examples/generate_validation_report.py` for a complete example:

```bash
# Run the example
python examples/generate_validation_report.py

# Then generate the actual PDF
mosqlimate-ai validation-report SP
```

## Report Design

### Visual Style

- **Clean, professional design** (no logos)
- **Colorblind-friendly palette**
- **A4 page size** (210 × 297 mm)
- **300 DPI** for print quality
- **Consistent typography** (Helvetica/DejaVu Sans)

### Color Scheme

- **Observed data**: Black
- **Test 1 predictions**: Blue (#1f77b4)
- **Test 2 predictions**: Orange (#ff7f0e)
- **Test 3 predictions**: Green (#2ca02c)
- **Prediction intervals**: Matching color with transparency

### Tables

All tables use:
- Professional styling with alternating row colors
- Clear headers with dark background
- Consistent alignment and spacing
- Grid lines for readability

## Troubleshooting

### Missing Validation Results

If you get an error about missing validation results:

```bash
# First run validation
mosqlimate-ai validate --states SP --full-pipeline

# Then generate report
mosqlimate-ai validation-report SP
```

### Custom Data Paths

If your validation results are in a non-standard location:

```python
report = ValidationPDFReport(
    state="SP",
    output_dir=Path("/path/to/custom/results")
)
```

### Report Not Generated

Check that:
1. Validation has completed successfully
2. `validation_results.json` exists in the state directory
3. You have write permissions in the output directory

## Integration with Validation Pipeline

The report generator integrates seamlessly with the validation orchestrator:

```python
from mosqlimate_ai.validation import ValidationOrchestrator

# Run validation
orchestrator = ValidationOrchestrator()
results = orchestrator.run_full_pipeline(states=["SP"])

# Generate report
from mosqlimate_ai.visualization import ValidationPDFReport
report = ValidationPDFReport("SP")
pdf_path = report.generate_from_files()
```

## License

MIT License - See project root for details.
