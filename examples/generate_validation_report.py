"""Example: Generate validation PDF report with sample data.

This script demonstrates how to use the ValidationPDFReport class
to generate a comprehensive PDF report from validation results.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def create_sample_validation_results(state: str = "SP") -> dict:
    """Create sample validation results for testing."""

    return {
        "state": state,
        "timestamp": datetime.now().isoformat(),
        "validation_tests": {
            "1": {
                "test_number": 1,
                "season": "2022-2023",
                "state": state,
                "status": "success",
                "metrics": {
                    "xgboost": {
                        "crps": 0.452,
                        "wis_total": 142.3,
                        "mae": 45.2,
                        "rmse": 67.8,
                        "mape": 18.5,
                        "bias": 12.3,
                        "coverage_50": 0.48,
                        "coverage_80": 0.78,
                        "coverage_95": 0.94,
                    },
                    "lstm": {
                        "crps": 0.485,
                        "wis_total": 158.7,
                        "mae": 48.9,
                        "rmse": 72.4,
                        "mape": 20.1,
                        "bias": 15.6,
                        "coverage_50": 0.51,
                        "coverage_80": 0.81,
                        "coverage_95": 0.93,
                    },
                    "ensemble": {
                        "crps": 0.438,
                        "wis_total": 138.2,
                        "mae": 43.1,
                        "rmse": 65.2,
                        "mape": 17.8,
                        "bias": 10.5,
                        "coverage_50": 0.50,
                        "coverage_80": 0.80,
                        "coverage_95": 0.95,
                    },
                },
                "hyperparameters": {
                    "xgboost": {
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "n_estimators": 500,
                    },
                    "lstm": {
                        "hidden_size": 128,
                        "num_layers": 2,
                        "dropout": 0.2,
                    },
                },
            },
            "2": {
                "test_number": 2,
                "season": "2023-2024",
                "state": state,
                "status": "success",
                "metrics": {
                    "xgboost": {
                        "crps": 0.421,
                        "wis_total": 135.8,
                        "mae": 42.1,
                        "rmse": 63.5,
                        "mape": 16.9,
                        "bias": 8.7,
                        "coverage_50": 0.49,
                        "coverage_80": 0.79,
                        "coverage_95": 0.94,
                    },
                    "lstm": {
                        "crps": 0.456,
                        "wis_total": 149.3,
                        "mae": 46.2,
                        "rmse": 68.9,
                        "mape": 19.2,
                        "bias": 13.4,
                        "coverage_50": 0.52,
                        "coverage_80": 0.82,
                        "coverage_95": 0.94,
                    },
                    "ensemble": {
                        "crps": 0.412,
                        "wis_total": 131.5,
                        "mae": 40.8,
                        "rmse": 61.8,
                        "mape": 16.2,
                        "bias": 7.2,
                        "coverage_50": 0.51,
                        "coverage_80": 0.81,
                        "coverage_95": 0.95,
                    },
                },
                "hyperparameters": {
                    "xgboost": {
                        "learning_rate": 0.045,
                        "max_depth": 7,
                        "n_estimators": 550,
                    },
                    "lstm": {
                        "hidden_size": 144,
                        "num_layers": 2,
                        "dropout": 0.18,
                    },
                },
            },
            "3": {
                "test_number": 3,
                "season": "2024-2025",
                "state": state,
                "status": "success",
                "metrics": {
                    "xgboost": {
                        "crps": 0.398,
                        "wis_total": 128.4,
                        "mae": 39.5,
                        "rmse": 59.2,
                        "mape": 15.3,
                        "bias": 6.1,
                        "coverage_50": 0.51,
                        "coverage_80": 0.80,
                        "coverage_95": 0.95,
                    },
                    "lstm": {
                        "crps": 0.432,
                        "wis_total": 142.1,
                        "mae": 43.8,
                        "rmse": 65.4,
                        "mape": 17.8,
                        "bias": 11.2,
                        "coverage_50": 0.50,
                        "coverage_80": 0.80,
                        "coverage_95": 0.94,
                    },
                    "ensemble": {
                        "crps": 0.389,
                        "wis_total": 124.7,
                        "mae": 38.2,
                        "rmse": 57.6,
                        "mape": 14.8,
                        "bias": 5.3,
                        "coverage_50": 0.51,
                        "coverage_80": 0.81,
                        "coverage_95": 0.96,
                    },
                },
                "hyperparameters": {
                    "xgboost": {
                        "learning_rate": 0.04,
                        "max_depth": 7,
                        "n_estimators": 600,
                    },
                    "lstm": {
                        "hidden_size": 160,
                        "num_layers": 3,
                        "dropout": 0.15,
                    },
                },
            },
        },
        "top_models": [
            {
                "model_name": "ensemble",
                "rank": 1,
                "composite_score": 0.923,
                "avg_crps": 0.413,
                "avg_wis": 131.5,
            },
            {
                "model_name": "xgboost",
                "rank": 2,
                "composite_score": 0.891,
                "avg_crps": 0.424,
                "avg_wis": 135.5,
            },
            {
                "model_name": "lstm",
                "rank": 3,
                "composite_score": 0.856,
                "avg_crps": 0.458,
                "avg_wis": 150.0,
            },
        ],
    }


def create_sample_observed_data() -> pd.DataFrame:
    """Create sample observed data."""
    # Generate 5 years of weekly data
    dates = pd.date_range(start="2020-01-01", periods=260, freq="W")

    # Create seasonal pattern with trend
    t = np.arange(len(dates))
    seasonal = 50 * np.sin(2 * np.pi * t / 52) + 30  # Annual seasonality
    trend = 0.1 * t  # Slight upward trend
    noise = np.random.normal(0, 10, len(dates))

    cases = 100 + seasonal + trend + noise
    cases = np.maximum(cases, 0)  # Ensure non-negative

    return pd.DataFrame(
        {
            "date": dates,
            "casos": cases.astype(int),
        }
    )


def create_sample_forecasts() -> dict:
    """Create sample forecast data for 3 validation tests."""
    forecasts = {}

    for test_num in [1, 2, 3]:
        # Each test predicts 52 weeks
        if test_num == 1:
            start_date = pd.to_datetime("2022-10-09")
        elif test_num == 2:
            start_date = pd.to_datetime("2023-10-08")
        else:
            start_date = pd.to_datetime("2024-10-06")

        dates = pd.date_range(start=start_date, periods=52, freq="W")

        # Generate predictions with uncertainty
        base_trend = np.linspace(120, 150, 52)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(52) / 52)

        median = base_trend + seasonal + np.random.normal(0, 5, 52)

        # Prediction intervals
        forecasts[test_num] = pd.DataFrame(
            {
                "date": dates,
                "median": median,
                "lower_50": median - 15,
                "upper_50": median + 15,
                "lower_80": median - 30,
                "upper_80": median + 30,
                "lower_95": median - 50,
                "upper_95": median + 50,
            }
        )

    return forecasts


def main():
    """Generate a sample validation report."""
    print("=" * 60)
    print("Validation PDF Report Generator - Example")
    print("=" * 60)

    state = "SP"
    output_dir = Path("validation_results")

    # Create directory structure
    state_dir = output_dir / state
    state_dir.mkdir(parents=True, exist_ok=True)

    # Create sample data
    print("\n[1/4] Creating sample validation results...")
    validation_results = create_sample_validation_results(state)

    print("[2/4] Creating sample observed data...")
    observed_data = create_sample_observed_data()

    print("[3/4] Creating sample forecasts...")
    forecast_data = create_sample_forecasts()

    # Save validation results to file
    print("[4/4] Saving validation results...")
    results_file = state_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)

    print(f"\n✓ Sample data created in: {output_dir}")
    print(f"✓ Validation results saved to: {results_file}")

    print("\n" + "=" * 60)
    print("To generate the PDF report, run:")
    print(f"  mosqlimate-ai validation-report {state}")
    print("=" * 60)

    print("\nAlternatively, use Python:")
    print("  from mosqlimate_ai.visualization import ValidationPDFReport")
    print(f"  report = ValidationPDFReport('{state}')")
    print("  pdf_path = report.generate_from_files()")
    print(f"  print(f'Report saved to: {{pdf_path}}')")


if __name__ == "__main__":
    main()
