"""Visualization module for forecast reports.

Generates rich visualizations for forecast evaluation including:
- Time series plots with observed vs forecasted values
- Prediction interval visualizations
- Residual analysis plots
- Metric comparison charts
- Coverage analysis plots
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class ReportVisualizer:
    """Generates visualizations for forecast reports.

    Creates plots showing forecast performance and saves them as PNG files
    that can be embedded in markdown reports.
    """

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plot images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_generated: list[str] = []

    def _save_plot(self, filename: str) -> str:
        """Save current plot and return relative path.

        Args:
            filename: Name of the file (without extension)

        Returns:
            Relative path to the saved plot
        """
        filepath = self.output_dir / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        rel_path = f"figures/{filename}.png"
        self.plots_generated.append(rel_path)
        logger.info(f"Saved plot: {filepath}")
        return rel_path

    def plot_forecast_timeseries(
        self,
        df_observed: pd.DataFrame,
        forecast_df: pd.DataFrame,
        state: str,
        model_name: str,
        title_suffix: str = "",
    ) -> str:
        """Plot time series with forecast and prediction intervals.

        Args:
            df_observed: DataFrame with observed data (date, casos columns)
            forecast_df: DataFrame with forecast (date, median, intervals)
            state: State abbreviation
            model_name: Name of the model
            title_suffix: Optional suffix for title

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot observed data
        if "date" in df_observed.columns and "casos" in df_observed.columns:
            ax.plot(
                df_observed["date"],
                df_observed["casos"],
                "k-",
                label="Observed",
                linewidth=1.5,
                alpha=0.8,
            )

        # Plot forecast median
        if "median" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                forecast_df["median"],
                "b-",
                label=f"{model_name.title()} Forecast",
                linewidth=2,
            )

        # Plot prediction intervals
        colors = {"95": "#ffcccc", "80": "#ff9999", "50": "#ff6666"}
        alphas = {"95": 0.3, "80": 0.4, "50": 0.5}

        for level in ["95", "80", "50"]:
            lower_col = f"lower_{level}"
            upper_col = f"upper_{level}"
            if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                ax.fill_between(
                    forecast_df["date"],
                    forecast_df[lower_col],
                    forecast_df[upper_col],
                    alpha=alphas[level],
                    color=colors[level],
                    label=f"{level}% Prediction Interval",
                )

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Dengue Cases", fontsize=11)
        title = f"{state} - {model_name.title()} Forecast"
        if title_suffix:
            title += f" {title_suffix}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        fig.autofmt_xdate()

        return self._save_plot(f"{state}_{model_name}_timeseries")

    def plot_multi_model_comparison(
        self,
        df_observed: pd.DataFrame,
        forecasts: dict[str, pd.DataFrame],
        state: str,
    ) -> str:
        """Plot comparison of multiple models.

        Args:
            df_observed: DataFrame with observed data
            forecasts: Dictionary mapping model names to forecast DataFrames
            state: State abbreviation

        Returns:
            Path to saved plot
        """
        n_models = len(forecasts)
        if n_models == 0:
            return ""

        fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=True)
        if n_models == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
            ax = axes[idx]

            # Observed data
            if "date" in df_observed.columns and "casos" in df_observed.columns:
                ax.plot(
                    df_observed["date"],
                    df_observed["casos"],
                    "k-",
                    label="Observed",
                    linewidth=1.5,
                    alpha=0.7,
                )

            # Forecast
            if "median" in forecast_df.columns:
                ax.plot(
                    forecast_df["date"],
                    forecast_df["median"],
                    color=colors[idx],
                    label=f"{model_name.title()} Forecast",
                    linewidth=2,
                )

            # 95% interval
            if "lower_95" in forecast_df.columns and "upper_95" in forecast_df.columns:
                ax.fill_between(
                    forecast_df["date"],
                    forecast_df["lower_95"],
                    forecast_df["upper_95"],
                    alpha=0.2,
                    color=colors[idx],
                    label="95% Interval",
                )

            ax.set_ylabel("Cases", fontsize=10)
            ax.set_title(f"{model_name.title()}", fontsize=11, fontweight="bold")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.suptitle(f"{state} - Model Comparison", fontsize=14, fontweight="bold", y=0.995)

        return self._save_plot(f"{state}_model_comparison")

    def plot_residuals(
        self,
        y_true: np.ndarray,
        forecast_df: pd.DataFrame,
        state: str,
        model_name: str,
    ) -> str:
        """Plot residual analysis.

        Args:
            y_true: True values
            forecast_df: Forecast DataFrame
            state: State abbreviation
            model_name: Model name

        Returns:
            Path to saved plot
        """
        if "median" not in forecast_df.columns:
            return ""

        y_pred = forecast_df["median"].values
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Residuals vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted")
        ax.grid(True, alpha=0.3)

        # 2. Residuals over time
        ax = axes[0, 1]
        if "date" in forecast_df.columns:
            ax.plot(forecast_df["date"], residuals, "b-", alpha=0.7)
            ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
            ax.set_xlabel("Date")
            fig.autofmt_xdate()
        else:
            ax.plot(residuals, "b-", alpha=0.7)
            ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
            ax.set_xlabel("Observation")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Over Time")
        ax.grid(True, alpha=0.3)

        # 3. Histogram of residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normality Check)")
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{state} - {model_name.title()} Residual Analysis",
            fontsize=14,
            fontweight="bold",
        )

        return self._save_plot(f"{state}_{model_name}_residuals")

    def plot_metrics_comparison(
        self,
        all_results: dict[str, dict[str, dict[str, float]]],
        metric: str = "rmse",
    ) -> str:
        """Plot bar chart comparing models across states.

        Args:
            all_results: Nested dict of results by state and model
            metric: Metric to plot

        Returns:
            Path to saved plot
        """
        # Collect data
        states = []
        models = []
        values = []

        for state, state_results in all_results.items():
            for model_name, metrics in state_results.items():
                if metric in metrics:
                    states.append(state)
                    models.append(model_name)
                    values.append(metrics[metric])

        if not values:
            return ""

        # Create DataFrame for plotting
        df_plot = pd.DataFrame(
            {
                "State": states,
                "Model": models,
                "Value": values,
            }
        )

        # Pivot for grouped bar chart
        pivot_df = df_plot.pivot(index="State", columns="Model", values="Value")

        fig, ax = plt.subplots(figsize=(max(10, len(pivot_df) * 0.8), 6))
        pivot_df.plot(kind="bar", ax=ax, width=0.8)

        ax.set_xlabel("State", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} Comparison Across States", fontsize=13, fontweight="bold")
        ax.legend(title="Model", loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")

        return self._save_plot(f"metrics_comparison_{metric}")

    def plot_coverage_analysis(
        self,
        all_results: dict[str, dict[str, dict[str, float]]],
    ) -> str:
        """Plot coverage analysis across states and models.

        Args:
            all_results: Nested dict of results

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        coverage_levels = ["50", "95"]
        target_coverage = [0.50, 0.95]
        colors = ["#3498db", "#e74c3c"]

        for idx, (level, target, color) in enumerate(zip(coverage_levels, target_coverage, colors)):
            ax = axes[idx]
            metric_name = f"coverage_{level}"

            # Collect coverage data
            data = []
            labels = []

            for state, state_results in all_results.items():
                for model_name, metrics in state_results.items():
                    if metric_name in metrics:
                        data.append(metrics[metric_name] * 100)  # Convert to percentage
                        labels.append(f"{state}\n{model_name}")

            if data:
                x_pos = np.arange(len(data))
                bars = ax.bar(x_pos, data, color=color, alpha=0.7, edgecolor="black")

                # Add target line
                ax.axhline(
                    y=target * 100,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Target ({target:.0%})",
                )

                # Color bars based on how close to target
                for bar, val in zip(bars, data):
                    if abs(val - target * 100) < 10:  # Within 10%
                        bar.set_color("#2ecc71")  # Green for good
                    elif abs(val - target * 100) < 20:  # Within 20%
                        bar.set_color("#f39c12")  # Orange for okay
                    else:
                        bar.set_color("#e74c3c")  # Red for poor

                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Coverage (%)", fontsize=11)
                ax.set_title(
                    f"{level}% Prediction Interval Coverage",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
                ax.set_ylim(0, 100)

        fig.suptitle(
            "Prediction Interval Coverage Analysis",
            fontsize=14,
            fontweight="bold",
        )

        return self._save_plot("coverage_analysis")

    def plot_error_distribution(
        self,
        all_results: dict[str, dict[str, dict[str, float]]],
    ) -> str:
        """Plot error metrics distribution.

        Args:
            all_results: Nested dict of results

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_to_plot = [
            ("rmse", "RMSE", axes[0, 0]),
            ("mae", "MAE", axes[0, 1]),
            ("mape", "MAPE (%)", axes[1, 0]),
            ("bias", "Bias", axes[1, 1]),
        ]

        for metric_key, metric_label, ax in metrics_to_plot:
            data_by_model: dict[str, list[float]] = {}

            for state, state_results in all_results.items():
                for model_name, metrics in state_results.items():
                    if metric_key in metrics:
                        if model_name not in data_by_model:
                            data_by_model[model_name] = []
                        data_by_model[model_name].append(metrics[metric_key])

            if data_by_model:
                # Create box plot
                bp = ax.boxplot(
                    data_by_model.values(),
                    labels=[m.title() for m in data_by_model.keys()],
                    patch_artist=True,
                )

                # Color boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel(metric_label, fontsize=10)
                ax.set_title(f"{metric_label} Distribution", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")

                if metric_key == "bias":
                    ax.axhline(y=0, color="r", linestyle="--", linewidth=2, alpha=0.5)

        fig.suptitle("Error Metrics Distribution by Model", fontsize=14, fontweight="bold")

        return self._save_plot("error_distribution")

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        forecast_df: pd.DataFrame,
        state: str,
        model_name: str,
    ) -> str:
        """Plot calibration curve for prediction intervals.

        Args:
            y_true: True values
            forecast_df: Forecast DataFrame with intervals
            state: State abbreviation
            model_name: Model name

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Define confidence levels to check
        confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        empirical_coverage = []

        for level in confidence_levels:
            lower_col = f"lower_{int(level * 100)}"
            upper_col = f"upper_{int(level * 100)}"

            if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                lower = forecast_df[lower_col].values
                upper = forecast_df[upper_col].values
                coverage = np.mean((y_true >= lower) & (y_true <= upper))
                empirical_coverage.append(coverage)
            else:
                # Interpolate or skip
                empirical_coverage.append(None)

        # Plot
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

        valid_points = [
            (c, e) for c, e in zip(confidence_levels, empirical_coverage) if e is not None
        ]
        if valid_points:
            conf, emp = zip(*valid_points)
            ax.plot(conf, emp, "bo-", label=f"{model_name.title()}", markersize=8, linewidth=2)

        ax.set_xlabel("Nominal Confidence Level", fontsize=11)
        ax.set_ylabel("Empirical Coverage", fontsize=11)
        ax.set_title(
            f"{state} - {model_name.title()} Calibration Curve",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return self._save_plot(f"{state}_{model_name}_calibration")

    def create_summary_heatmap(
        self,
        all_results: dict[str, dict[str, dict[str, float]]],
    ) -> str:
        """Create heatmap of all metrics.

        Args:
            all_results: Nested dict of results

        Returns:
            Path to saved plot
        """
        # Prepare data for heatmap
        metrics_list = ["rmse", "mae", "mape", "crps", "wis_total"]

        # Create matrix: rows = state_model combinations, cols = metrics
        data_matrix = []
        labels = []

        for state in sorted(all_results.keys()):
            for model in ["xgboost", "lstm", "ensemble"]:
                if model in all_results[state]:
                    row = []
                    for metric in metrics_list:
                        val = all_results[state][model].get(metric, np.nan)
                        row.append(val)
                    data_matrix.append(row)
                    labels.append(f"{state}-{model[:3].upper()}")

        if not data_matrix:
            return ""

        # Normalize each column for better visualization
        data_array = np.array(data_matrix)
        data_normalized = np.zeros_like(data_array)

        for col in range(data_array.shape[1]):
            col_data = data_array[:, col]
            col_min, col_max = np.nanmin(col_data), np.nanmax(col_data)
            if col_max > col_min:
                data_normalized[:, col] = (col_data - col_min) / (col_max - col_min)
            else:
                data_normalized[:, col] = 0.5

        fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.3)))

        im = ax.imshow(data_normalized, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metrics_list)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels([m.upper() for m in metrics_list])
        ax.set_yticklabels(labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Score (Lower is Better)", rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(metrics_list)):
                if not np.isnan(data_array[i, j]):
                    text = ax.text(
                        j,
                        i,
                        f"{data_array[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if data_normalized[i, j] < 0.5 else "white",
                        fontsize=7,
                    )

        ax.set_title("Performance Heatmap by State-Model", fontsize=13, fontweight="bold")

        return self._save_plot("performance_heatmap")

    def get_plots_list(self) -> list[str]:
        """Get list of all generated plot paths."""
        return self.plots_generated.copy()


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string for inline embedding.

    Args:
        fig: Matplotlib figure

    Returns:
        Base64 encoded PNG string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64
