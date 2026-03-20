"""Validation-specific visualization functions for PDF reports.

Generates specialized plots for the validation pipeline including:
- Multi-test time series overlays
- CRPS/WIS progression charts
- Model comparison visualizations
- Coverage analysis plots
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set professional style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 300


def plot_validation_test_timeseries(
    observed_df: pd.DataFrame,
    forecasts_by_model: Dict[str, pd.DataFrame],
    state: str,
    test_num: int,
    train_end_date: str,
    season: str,
    figsize: Tuple[float, float] = (14, 6),
) -> Figure:
    """Create time series plot for a single validation test.

    Shows the training period, prediction period, forecasts from all models
    (XGBoost, LSTM, Ensemble), and observed data for comparison.

    Args:
        observed_df: DataFrame with observed data (date, casos columns)
        forecasts_by_model: Dictionary mapping model names to forecast DataFrames
            Each forecast DataFrame should have columns: date, median, lower_50/80/95, upper_50/80/95
        state: State name/code
        test_num: Test number (1, 2, or 3)
        train_end_date: Training end date (e.g., "2022-06-26")
        season: Season string (e.g., "2022-2023")
        figsize: Figure dimensions in inches (width, height)

    Returns:
        Matplotlib Figure object ready for PDF embedding
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Parse dates
    train_cutoff = pd.to_datetime(train_end_date)

    # Get forecast dates from first available model
    first_model = list(forecasts_by_model.keys())[0]
    forecast_df = forecasts_by_model[first_model]
    forecast_start = forecast_df["date"].min()
    forecast_end = forecast_df["date"].max()

    # Define time window for plot (show 1 year before training end + forecast period)
    plot_start = train_cutoff - pd.Timedelta(days=365)
    plot_end = forecast_end + pd.Timedelta(days=30)

    # Filter observed data to plot window
    mask = (observed_df["date"] >= plot_start) & (observed_df["date"] <= plot_end)
    observed_plot = observed_df[mask].copy()

    # Split observed data into training and prediction periods
    obs_training = observed_plot[observed_plot["date"] <= train_cutoff]
    obs_prediction = observed_plot[
        (observed_plot["date"] > train_cutoff) & (observed_plot["date"] <= forecast_end)
    ]

    # Plot training period (observed data)
    if len(obs_training) > 0:
        ax.plot(
            obs_training["date"],
            obs_training["casos"],
            "k-",
            label="Training Data",
            linewidth=1.5,
            alpha=0.9,
            zorder=10,
        )

    # Plot prediction period (observed data)
    if len(obs_prediction) > 0:
        ax.plot(
            obs_prediction["date"],
            obs_prediction["casos"],
            "k--",
            label="Observed (Prediction Period)",
            linewidth=2,
            alpha=0.9,
            zorder=10,
        )

    # Model colors
    model_colors = {
        "xgboost": "#1f77b4",  # Blue
        "lstm": "#ff7f0e",  # Orange
        "ensemble": "#2ca02c",  # Green
    }

    # Plot forecast for each model
    for model_name, forecast_df in forecasts_by_model.items():
        color = model_colors.get(model_name.lower(), "#9467bd")

        if "median" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                forecast_df["median"],
                color=color,
                linewidth=2.5,
                label=f"{model_name.title()} Forecast",
                zorder=8,
            )

    # Plot prediction intervals for ensemble model only (to avoid clutter)
    if "ensemble" in forecasts_by_model:
        ensemble_forecast = forecasts_by_model["ensemble"]
        interval_specs = [
            ("95", "#d62728", 0.15),  # Red, light
            ("80", "#ff7f0e", 0.25),  # Orange, medium
            ("50", "#2ca02c", 0.4),  # Green, darker
        ]

        for level, color, alpha in interval_specs:
            lower_col = f"lower_{level}"
            upper_col = f"upper_{level}"

            if lower_col in ensemble_forecast.columns and upper_col in ensemble_forecast.columns:
                ax.fill_between(
                    ensemble_forecast["date"],
                    ensemble_forecast[lower_col],
                    ensemble_forecast[upper_col],
                    alpha=alpha,
                    color=color,
                    label=f"Ensemble {level}% PI",
                    zorder=5,
                )

    # Add training cutoff line
    ax.axvline(
        x=train_cutoff,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Training Cutoff",
        zorder=1,
    )

    # Formatting
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dengue Cases", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{state} - Validation Test {test_num}: {season}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Legend - use 3 columns to fit all entries
    ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=3,
        framealpha=0.95,
        fancybox=True,
        shadow=True,
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Format x-axis
    fig.autofmt_xdate()

    # Tight layout
    plt.tight_layout()

    return fig


def plot_all_validation_tests(
    observed_df: pd.DataFrame,
    test_forecasts: Dict[int, Dict[str, pd.DataFrame]],
    state: str,
    train_end_dates: Dict[int, str],
    seasons: Dict[int, str],
    figsize: Tuple[float, float] = (16, 12),
) -> Figure:
    """Create figure with 3 subplots, one for each validation test.

    Args:
        observed_df: DataFrame with observed data
        test_forecasts: Dictionary mapping test numbers (1, 2, 3) to
                       dictionaries of {model_name: forecast_df}
        state: State name/code
        train_end_dates: Dictionary of training end dates for each test
        seasons: Dictionary of season strings for each test
        figsize: Figure dimensions in inches (width, height)

    Returns:
        Matplotlib Figure object with 3 subplots
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharey=True)

    # Model colors
    model_colors = {
        "xgboost": "#1f77b4",  # Blue
        "lstm": "#ff7f0e",  # Orange
        "ensemble": "#2ca02c",  # Green
    }

    for idx, test_num in enumerate([1, 2, 3]):
        ax = axes[idx]

        if test_num not in test_forecasts:
            ax.text(
                0.5,
                0.5,
                f"Test {test_num} data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        forecasts_by_model = test_forecasts[test_num]
        train_end_date = train_end_dates.get(test_num)
        season = seasons.get(test_num, f"Test {test_num}")

        if train_end_date is None:
            continue

        # Parse dates
        train_cutoff = pd.to_datetime(train_end_date)

        # Get forecast dates from first available model
        first_model = list(forecasts_by_model.keys())[0]
        forecast_df = forecasts_by_model[first_model]
        forecast_start = forecast_df["date"].min()
        forecast_end = forecast_df["date"].max()

        # Define time window for plot
        plot_start = train_cutoff - pd.Timedelta(days=365)
        plot_end = forecast_end + pd.Timedelta(days=30)

        # Filter observed data
        mask = (observed_df["date"] >= plot_start) & (observed_df["date"] <= plot_end)
        observed_plot = observed_df[mask].copy()

        # Split observed data
        obs_training = observed_plot[observed_plot["date"] <= train_cutoff]
        obs_prediction = observed_plot[
            (observed_plot["date"] > train_cutoff) & (observed_plot["date"] <= forecast_end)
        ]

        # Plot training period
        if len(obs_training) > 0:
            ax.plot(
                obs_training["date"],
                obs_training["casos"],
                "k-",
                label="Training" if idx == 0 else "",
                linewidth=1.5,
                alpha=0.9,
                zorder=10,
            )

        # Plot prediction period (observed)
        if len(obs_prediction) > 0:
            ax.plot(
                obs_prediction["date"],
                obs_prediction["casos"],
                "k--",
                label="Observed" if idx == 0 else "",
                linewidth=2,
                alpha=0.9,
                zorder=10,
            )

        # Plot forecast for each model
        for model_name, forecast_df in forecasts_by_model.items():
            color = model_colors.get(model_name.lower(), "#9467bd")

            if "median" in forecast_df.columns:
                ax.plot(
                    forecast_df["date"],
                    forecast_df["median"],
                    color=color,
                    linewidth=2.5,
                    label=f"{model_name.title()}" if idx == 0 else "",
                    zorder=8,
                )

        # Plot prediction intervals for ensemble model only
        if "ensemble" in forecasts_by_model:
            ensemble_forecast = forecasts_by_model["ensemble"]
            interval_specs = [
                ("95", "#d62728", 0.15),
                ("80", "#ff7f0e", 0.25),
                ("50", "#2ca02c", 0.4),
            ]

            for level, color, alpha in interval_specs:
                lower_col = f"lower_{level}"
                upper_col = f"upper_{level}"

                if (
                    lower_col in ensemble_forecast.columns
                    and upper_col in ensemble_forecast.columns
                ):
                    ax.fill_between(
                        ensemble_forecast["date"],
                        ensemble_forecast[lower_col],
                        ensemble_forecast[upper_col],
                        alpha=alpha,
                        color=color,
                        label=f"{level}% PI" if idx == 0 else "",
                        zorder=5,
                    )

        # Training cutoff line
        ax.axvline(
            x=train_cutoff,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            zorder=1,
        )

        # Subplot title
        ax.set_title(f"Test {test_num}: {season}", fontsize=12, fontweight="bold")

        # Legend for first subplot only
        if idx == 0:
            ax.legend(
                loc="upper left",
                fontsize=7,
                ncol=3,
                framealpha=0.95,
            )

        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Y-label only on leftmost
        if idx == 1:
            ax.set_ylabel("Dengue Cases", fontsize=11, fontweight="bold")

    # X-label on bottom subplot
    axes[-1].set_xlabel("Date", fontsize=11, fontweight="bold")

    # Main title
    fig.suptitle(
        f"{state} - Validation Tests: Observed vs Predicted with Prediction Intervals",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig


def plot_crps_progression(
    results_by_test: Dict[int, Dict[str, Any]],
    models: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Create line plot showing CRPS progression across validation tests.

    Args:
        results_by_test: Dictionary mapping test numbers to results dicts
        models: List of model names to plot (None = all found)
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract CRPS values
    test_nums = sorted(results_by_test.keys())

    if not models:
        # Auto-detect models from first test
        first_test = results_by_test.get(test_nums[0], {})
        metrics = first_test.get("metrics", {})
        models = list(metrics.keys())

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        crps_values = []
        valid_tests = []

        for test_num in test_nums:
            test_results = results_by_test.get(test_num, {})
            metrics = test_results.get("metrics", {})
            model_metrics = metrics.get(model, {})
            crps_val = model_metrics.get("crps")

            if crps_val is not None and not np.isnan(crps_val):
                crps_values.append(crps_val)
                valid_tests.append(test_num)

        if crps_values:
            ax.plot(
                valid_tests,
                crps_values,
                "o-",
                color=colors[idx],
                linewidth=2.5,
                markersize=10,
                label=model.title(),
            )

    ax.set_xlabel("Validation Test", fontsize=12, fontweight="bold")
    ax.set_ylabel("CRPS (Continuous Ranked Probability Score)", fontsize=12, fontweight="bold")
    ax.set_title(
        "CRPS Progression Across Validation Tests",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(test_nums)
    ax.set_xticklabels([f"Test {t}" for t in test_nums])
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining CRPS
    ax.text(
        0.02,
        0.98,
        "Lower CRPS = Better Forecast Accuracy",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_wis_progression(
    results_by_test: Dict[int, Dict[str, Any]],
    models: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Create line plot showing WIS progression across validation tests.

    Args:
        results_by_test: Dictionary mapping test numbers to results dicts
        models: List of model names to plot (None = all found)
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract WIS values
    test_nums = sorted(results_by_test.keys())

    if not models:
        first_test = results_by_test.get(test_nums[0], {})
        metrics = first_test.get("metrics", {})
        models = list(metrics.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        wis_values = []
        valid_tests = []

        for test_num in test_nums:
            test_results = results_by_test.get(test_num, {})
            metrics = test_results.get("metrics", {})
            model_metrics = metrics.get(model, {})
            wis_val = model_metrics.get("wis_total")

            if wis_val is not None and not np.isnan(wis_val):
                wis_values.append(wis_val)
                valid_tests.append(test_num)

        if wis_values:
            ax.plot(
                valid_tests,
                wis_values,
                "s-",
                color=colors[idx],
                linewidth=2.5,
                markersize=10,
                label=model.title(),
            )

    ax.set_xlabel("Validation Test", fontsize=12, fontweight="bold")
    ax.set_ylabel("WIS (Weighted Interval Score)", fontsize=12, fontweight="bold")
    ax.set_title(
        "WIS Progression Across Validation Tests",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(test_nums)
    ax.set_xticklabels([f"Test {t}" for t in test_nums])
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.02,
        0.98,
        "Lower WIS = Better Probabilistic Forecasts",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_model_performance_heatmap(
    results_by_test: Dict[int, Dict[str, Any]],
    metric: str = "crps",
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Create heatmap showing model performance across tests.

    Args:
        results_by_test: Results by test number
        metric: Metric to visualize ("crps" or "wis_total")
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure
    """
    # Build data matrix
    test_nums = sorted(results_by_test.keys())

    # Get all models
    all_models = set()
    for test_num in test_nums:
        metrics = results_by_test[test_num].get("metrics", {})
        all_models.update(metrics.keys())
    models = sorted(all_models)

    # Create data matrix
    data = np.full((len(models), len(test_nums)), np.nan)

    for j, test_num in enumerate(test_nums):
        test_metrics = results_by_test[test_num].get("metrics", {})
        for i, model in enumerate(models):
            model_metrics = test_metrics.get(model, {})
            val = model_metrics.get(metric)
            if val is not None:
                data[i, j] = val

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Use reversed colormap (lower is better)
    cmap = sns.color_palette("YlOrRd_r", as_cmap=True)

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Labels
    ax.set_xticks(np.arange(len(test_nums)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f"Test {t}" for t in test_nums])
    ax.set_yticklabels([m.title() for m in models])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper(), fontsize=11, fontweight="bold")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(test_nums)):
            if not np.isnan(data[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] > np.nanmean(data) else "black",
                    fontweight="bold",
                    fontsize=10,
                )

    ax.set_title(
        f"Model Performance Heatmap: {metric.upper()}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Validation Test", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_coverage_analysis(
    results_by_test: Dict[int, Dict[str, Any]],
    models: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> Figure:
    """Create improved bar chart showing prediction interval coverage analysis.

    Coverage measures what percentage of actual observed values fell within the
    prediction intervals. A well-calibrated model should have coverage close to
    the nominal interval level (e.g., 95% of observations should fall within the
    95% prediction interval).

    Args:
        results_by_test: Results by test number
        models: Models to include
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure with annotated coverage analysis
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], hspace=0.4, wspace=0.3)

    # Main coverage plots
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    test_nums = sorted(results_by_test.keys())
    coverage_levels = ["50", "80", "95"]
    target_coverage = [0.50, 0.80, 0.95]

    if not models:
        first_test = results_by_test.get(test_nums[0], {})
        all_models = set(first_test.get("metrics", {}).keys())
        models = sorted(all_models)

    # Use distinct colors for models
    model_colors = {"xgboost": "#1f77b4", "lstm": "#ff7f0e", "ensemble": "#2ca02c"}

    for idx, (level, target) in enumerate(zip(coverage_levels, target_coverage)):
        ax = axes[idx]
        metric_name = f"coverage_{level}"
        target_pct = target * 100

        x = np.arange(len(test_nums))
        width = 0.25

        for model_idx, model in enumerate(models):
            coverage_values = []

            for test_num in test_nums:
                test_results = results_by_test.get(test_num, {})
                metrics = test_results.get("metrics", {})
                model_metrics = metrics.get(model, {})
                cov_val = model_metrics.get(metric_name)

                if cov_val is not None:
                    coverage_values.append(cov_val * 100)  # Convert to percentage
                else:
                    coverage_values.append(0)

            offset = (model_idx - len(models) / 2 + 0.5) * width
            color = model_colors.get(model.lower(), plt.cm.Set2(model_idx))

            bars = ax.bar(
                x + offset,
                coverage_values,
                width,
                label=model.title(),
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=1,
            )

            # Add value labels on bars
            for bar, val in zip(bars, coverage_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

                # Color code bars based on coverage quality
                deviation = abs(val - target_pct)
                if deviation < 3:  # Excellent: within 3%
                    bar.set_edgecolor("darkgreen")
                    bar.set_linewidth(2.5)
                elif deviation < 7:  # Good: within 7%
                    bar.set_edgecolor("orange")
                    bar.set_linewidth(2)
                else:  # Poor: more than 7% off
                    bar.set_edgecolor("darkred")
                    bar.set_linewidth(2)

        # Target line with clear label
        ax.axhline(
            y=target_pct,
            color="red",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label=f"Target: {target:.0%}",
        )

        # Add shaded "good coverage" zone
        ax.axhspan(
            target_pct - 5,
            target_pct + 5,
            alpha=0.1,
            color="green",
            label="Good Coverage Zone (±5%)",
        )

        # Improved labels
        ax.set_xlabel("Validation Test Number", fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel(
                "Coverage (% of observations\nwithin prediction interval)",
                fontsize=11,
                fontweight="bold",
            )

        # Clear title with explanation
        ax.set_title(
            f"{level}% Prediction Interval\n({level}% of observations should fall within)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Test {t}" for t in test_nums], fontsize=10)
        ax.set_ylim(0, 105)

        # Legend with better placement
        if idx == 2:  # Only on last plot
            ax.legend(
                fontsize=9, loc="upper left", framealpha=0.95, title="Models", title_fontsize=10
            )

        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add annotation explaining this panel
        ax.text(
            0.98,
            0.02,
            f"Target: {target:.0%}\nObservations should\nfall in this interval",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray"
            ),
        )

    # Add coverage summary table below
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    # Create summary statistics
    table_data = []
    table_data.append(
        ["Model", "Test", "50% Interval", "80% Interval", "95% Interval", "Avg Error"]
    )

    for model in models:
        for test_num in test_nums:
            test_results = results_by_test.get(test_num, {})
            metrics = test_results.get("metrics", {})
            model_metrics = metrics.get(model, {})

            cov_50 = model_metrics.get("coverage_50", 0) * 100
            cov_80 = model_metrics.get("coverage_80", 0) * 100
            cov_95 = model_metrics.get("coverage_95", 0) * 100

            # Calculate average deviation from targets
            errors = [abs(cov_50 - 50), abs(cov_80 - 80), abs(cov_95 - 95)]
            avg_error = sum(errors) / len(errors)

            table_data.append(
                [
                    model.title(),
                    f"Test {test_num}",
                    f"{cov_50:.1f}%",
                    f"{cov_80:.1f}%",
                    f"{cov_95:.1f}%",
                    f"±{avg_error:.1f}%",
                ]
            )

    # Create table
    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.12, 0.18, 0.18, 0.18, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style cells
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("white")

    ax_table.set_title(
        "Coverage Summary by Model and Test\n(Lower avg error = better calibration)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # Add explanation text at bottom
    ax_explain = fig.add_subplot(gs[2, :])
    ax_explain.axis("off")

    explanation = (
        "WHAT IS COVERAGE?  Coverage measures what percentage of actual dengue cases fell within the predicted "
        "intervals. For example, if the 95% prediction interval has 90% coverage, it means 90% of actual cases "
        "fell within the predicted 95% interval. Well-calibrated models should have coverage close to the nominal "
        "interval level (e.g., ~95% for 95% intervals).  "
        "COLOR CODING:  Green border = Excellent (within 3% of target)  |  "
        "Orange border = Good (within 7% of target)  |  "
        "Red border = Poor (more than 7% off target)  |  "
        "Red dashed line = Target coverage level"
    )

    ax_explain.text(
        0.5,
        0.5,
        explanation,
        transform=ax_explain.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="lightblue",
            alpha=0.3,
            edgecolor="navy",
            linewidth=2,
        ),
    )

    # Main title
    fig.suptitle(
        "Prediction Interval Coverage Analysis: How Well Calibrated Are the Forecasts?",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    return fig


def save_figure_for_pdf(fig: Figure, filepath: Path, dpi: int = 300) -> None:
    """Save matplotlib figure optimized for PDF embedding.

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution for raster elements
    """
    fig.savefig(
        filepath,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info(f"Saved figure: {filepath}")


def create_validation_figure_set(
    state: str,
    observed_df: pd.DataFrame,
    test_forecasts: Dict[int, Any],
    results_by_test: Dict[int, Dict[str, Any]],
    output_dir: Path,
    train_end_dates: Optional[Dict[int, str]] = None,
    seasons: Optional[Dict[int, str]] = None,
) -> Dict[str, Path]:
    """Generate complete set of validation figures.

    Creates individual plots for each validation test showing training period,
    prediction period, forecast from all models, and prediction intervals.

    Args:
        state: State code
        observed_df: Observed data
        test_forecasts: Dictionary mapping test numbers to either:
                       - Dict[str, pd.DataFrame]: forecasts by model (new format)
                       - pd.DataFrame: single forecast (old format, will be treated as ensemble)
        results_by_test: Results by test number
        output_dir: Directory to save figures
        train_end_dates: Training end dates for each test
        seasons: Season labels for each test (e.g., {1: "2022-2023"})

    Returns:
        Dictionary mapping figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Default seasons if not provided
    if seasons is None:
        seasons = {
            1: "2022-2023",
            2: "2023-2024",
            3: "2024-2025",
        }

    # Default train end dates if not provided
    if train_end_dates is None:
        train_end_dates = {
            1: "2022-06-26",
            2: "2023-06-25",
            3: "2024-06-23",
        }

    # Generate individual test plots (Pages 2-4)
    for test_num in [1, 2, 3]:
        if test_num in test_forecasts:
            # Handle both old format (single DataFrame) and new format (dict of DataFrames)
            forecast_data = test_forecasts[test_num]
            if isinstance(forecast_data, pd.DataFrame):
                # Old format: wrap in dict as ensemble
                forecasts_by_model = {"ensemble": forecast_data}
            else:
                # New format: already a dict
                forecasts_by_model = forecast_data

            fig = plot_validation_test_timeseries(
                observed_df=observed_df,
                forecasts_by_model=forecasts_by_model,
                state=state,
                test_num=test_num,
                train_end_date=train_end_dates[test_num],
                season=seasons[test_num],
            )
            filepath = output_dir / f"{state}_test{test_num}_timeseries.png"
            save_figure_for_pdf(fig, filepath)
            figures[f"test{test_num}_timeseries"] = filepath
            logger.info(f"Generated Test {test_num} time series plot")

    # Combined view with all 3 tests (for overview)
    # Convert test_forecasts to new format if needed
    test_forecasts_by_model = {}
    for test_num, forecast_data in test_forecasts.items():
        if isinstance(forecast_data, pd.DataFrame):
            test_forecasts_by_model[test_num] = {"ensemble": forecast_data}
        else:
            test_forecasts_by_model[test_num] = forecast_data

    fig = plot_all_validation_tests(
        observed_df=observed_df,
        test_forecasts=test_forecasts_by_model,
        state=state,
        train_end_dates=train_end_dates,
        seasons=seasons,
    )
    filepath = output_dir / f"{state}_all_tests_timeseries.png"
    save_figure_for_pdf(fig, filepath)
    figures["all_tests"] = filepath

    # CRPS progression (Page 5)
    fig = plot_crps_progression(results_by_test)
    filepath = output_dir / f"{state}_crps_progression.png"
    save_figure_for_pdf(fig, filepath)
    figures["crps"] = filepath

    # WIS progression (Page 6)
    fig = plot_wis_progression(results_by_test)
    filepath = output_dir / f"{state}_wis_progression.png"
    save_figure_for_pdf(fig, filepath)
    figures["wis"] = filepath

    # Performance heatmap (Page 7)
    fig = plot_model_performance_heatmap(results_by_test, metric="crps")
    filepath = output_dir / f"{state}_heatmap_crps.png"
    save_figure_for_pdf(fig, filepath)
    figures["heatmap"] = filepath

    # Coverage analysis (Page 8)
    fig = plot_coverage_analysis(results_by_test)
    filepath = output_dir / f"{state}_coverage.png"
    save_figure_for_pdf(fig, filepath)
    figures["coverage"] = filepath

    return figures
