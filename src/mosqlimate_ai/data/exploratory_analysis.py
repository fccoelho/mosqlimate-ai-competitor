"""Exploratory data analysis module for validating downloaded time-series data.

Provides visualization tools to inspect raw observed data including:
- Dengue case time series
- Climate variables (temperature, precipitation, humidity, pressure)
- Ocean climate oscillations (ENSO, IOD, PDO)
- Data quality checks (missing values, date coverage, outliers)
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 9


class ExploratoryDataAnalyzer:
    """Analyze and visualize raw time-series data for validation.

    This class provides methods to generate diagnostic plots for the
    downloaded competition data, helping users verify data quality
    and understand time-series patterns before modeling.

    Args:
        loader: CompetitionDataLoader instance with loaded data
        state: Optional state abbreviation to filter data

    Example:
        >>> from mosqlimate_ai.data import CompetitionDataLoader
        >>> loader = CompetitionDataLoader()
        >>> analyzer = ExploratoryDataAnalyzer(loader, state="SP")
        >>> analyzer.plot_dengue_timeseries()
        >>> analyzer.plot_climate_panel()
        >>> analyzer.plot_data_quality_summary()
    """

    def __init__(
        self,
        loader,
        state: Optional[str] = None,
        figsize: tuple[float, float] = (12, 5),
    ):
        self.loader = loader
        self.state = state
        self.figsize = figsize
        self._dengue_df: Optional[pd.DataFrame] = None
        self._climate_df: Optional[pd.DataFrame] = None
        self._ocean_df: Optional[pd.DataFrame] = None

    @property
    def dengue_df(self) -> pd.DataFrame:
        if self._dengue_df is None:
            if self.state:
                self._dengue_df = self.loader.load_state_data(self.state, aggregate=True)
            else:
                self._dengue_df = self.loader.load_merged_data()
        return self._dengue_df

    @property
    def climate_df(self) -> pd.DataFrame:
        if self._climate_df is None:
            df = self.loader.climate_df.copy()
            if self.state:
                dengue = self.loader.dengue_df
                geocodes = dengue[dengue["uf"] == self.state]["geocode"].unique()
                df = df[df["geocode"].isin(geocodes)]
            self._climate_df = df
        return self._climate_df

    @property
    def ocean_df(self) -> pd.DataFrame:
        if self._ocean_df is None:
            self._ocean_df = self.loader.load_ocean_data()
        return self._ocean_df

    def plot_dengue_timeseries(
        self,
        date_col: str = "date",
        value_col: str = "casos",
        include_incidence: bool = True,
        rolling_window: int = 4,
    ) -> Figure:
        """Create dengue cases time series plot with trend analysis.

        Args:
            date_col: Column name for dates
            value_col: Column name for case counts
            include_incidence: Whether to show incidence rate if available
            rolling_window: Window size for rolling mean

        Returns:
            Matplotlib Figure with dengue time series
        """
        df = self.dengue_df.copy()
        if df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No dengue data available", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(
            2 if include_incidence and "incidence_rate" in df.columns else 1,
            1,
            figsize=(
                self.figsize[0],
                self.figsize[1]
                * (2 if include_incidence and "incidence_rate" in df.columns else 1),
            ),
        )

        if include_incidence and "incidence_rate" in df.columns:
            ax_cases, ax_incidence = axes
        else:
            ax_cases = axes

        ax_cases.plot(
            df[date_col], df[value_col], "b-", alpha=0.4, linewidth=0.8, label="Weekly Cases"
        )
        rolling = df[value_col].rolling(window=rolling_window, center=True).mean()
        ax_cases.plot(
            df[date_col], rolling, "b-", linewidth=2, label=f"{rolling_window}-Week Rolling Mean"
        )

        ax_cases.set_xlabel("Date")
        ax_cases.set_ylabel("Dengue Cases")
        title_state = f" ({self.state})" if self.state else ""
        ax_cases.set_title(f"Dengue Cases Time Series{title_state}")
        ax_cases.legend(loc="upper right")
        ax_cases.grid(True, alpha=0.3)

        if include_incidence and "incidence_rate" in df.columns:
            ax_incidence.plot(df[date_col], df["incidence_rate"], "r-", linewidth=1.5)
            ax_incidence.set_xlabel("Date")
            ax_incidence.set_ylabel("Incidence Rate (per 100k)")
            ax_incidence.set_title("Incidence Rate")
            ax_incidence.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_climate_panel(self) -> Figure:
        """Create panel of climate variable time series.

        Shows temperature (min, med, max), precipitation, pressure,
        and humidity variables as subplots.

        Returns:
            Matplotlib Figure with climate panel
        """
        df = self.climate_df.copy()
        if df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No climate data available", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 2.5))

        df = df.sort_values("date")

        axes[0].plot(df["date"], df["temp_min"], "b-", alpha=0.5, linewidth=0.7, label="Min")
        axes[0].plot(df["date"], df["temp_med"], "g-", alpha=0.7, linewidth=1, label="Median")
        axes[0].plot(df["date"], df["temp_max"], "r-", alpha=0.5, linewidth=0.7, label="Max")
        if "thermal_range" in df.columns:
            axes[0].fill_between(
                df["date"], df["temp_min"], df["temp_max"], alpha=0.2, color="orange", label="Range"
            )
        axes[0].set_ylabel("Temperature")
        axes[0].set_title("Temperature (°C)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        precip_cols = [
            c for c in ["precip_min", "precip_med", "precip_max", "precip_tot"] if c in df.columns
        ]
        if precip_cols:
            for col in precip_cols:
                axes[1].plot(
                    df["date"],
                    df[col],
                    label=col.replace("precip_", "").replace("_", " ").title(),
                    linewidth=1,
                )
        axes[1].set_ylabel("Precipitation (mm)")
        axes[1].set_title("Precipitation")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        humid_cols = [
            c for c in ["rel_humid_min", "rel_humid_med", "rel_humid_max"] if c in df.columns
        ]
        if humid_cols:
            for col in humid_cols:
                axes[2].plot(
                    df["date"],
                    df[col],
                    label=col.replace("rel_humid_", "").replace("_", " ").title(),
                    linewidth=1,
                )
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Relative Humidity (%)")
        axes[2].set_title("Relative Humidity")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        title_state = f" ({self.state})" if self.state else ""
        fig.suptitle(
            f"Climate Variables Time Series{title_state}", fontsize=12, fontweight="bold", y=0.998
        )
        plt.tight_layout()
        return fig

    def plot_ocean_oscillations(self) -> Figure:
        """Create ocean climate oscillation indices plot.

        Shows ENSO, IOD, and PDO indices over time.

        Returns:
            Matplotlib Figure with ocean oscillations
        """
        df = self.ocean_df.copy()
        if df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No ocean oscillation data available", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=self.figsize)

        df = df.sort_values("date")

        osc_cols = {
            "ENSO": "enso",
            "IOD": "iod",
            "PDO": "pdo",
        }

        colors = {"ENSO": "#1f77b4", "IOD": "#ff7f0e", "PDO": "#2ca02c"}

        for name, col in osc_cols.items():
            if col in df.columns:
                ax.plot(df["date"], df[col], label=name, linewidth=1.2, color=colors[name])

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Value")
        ax.set_title("Ocean Climate Oscillations Indices")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_data_quality_summary(self) -> Figure:
        """Create data quality summary with missing value analysis.

        Returns:
            Matplotlib Figure with data quality dashboard
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5))

        dengue = self.dengue_df
        climate = self.climate_df

        ax = axes[0, 0]
        if not dengue.empty:
            yearly_counts = dengue.groupby(dengue["date"].dt.year)["casos"].sum()
            yearly_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Cases")
            ax.set_title("Annual Dengue Cases")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax = axes[0, 1]
        if not dengue.empty:
            missing_pct = dengue.isnull().sum() / len(dengue) * 100
            missing_pct = missing_pct[missing_pct > 0]
            if len(missing_pct) > 0:
                missing_pct.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
                ax.set_xlabel("Missing %")
                ax.set_title("Missing Values in Dengue Data")
            else:
                ax.text(
                    0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes
                )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3, axis="x")

        ax = axes[1, 0]
        if not climate.empty:
            date_range = pd.date_range(
                start=climate["date"].min(), end=climate["date"].max(), freq="W"
            )
            expected_weeks = len(date_range)
            actual_weeks = climate["date"].nunique()
            coverage = actual_weeks / expected_weeks * 100 if expected_weeks > 0 else 0

            categories = ["Expected\nWeeks", "Actual\nWeeks"]
            values = [expected_weeks, actual_weeks]
            colors_bar = ["lightgray", "steelblue"]
            ax.bar(categories, values, color=colors_bar, edgecolor="black", width=0.5)
            for i, v in enumerate(values):
                ax.text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")
            ax.set_ylabel("Number of Weeks")
            ax.set_title(f"Climate Data Coverage: {coverage:.1f}%")
            ax.set_ylim(0, max(values) * 1.1)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 1]
        if not dengue.empty:
            weekly_cases = dengue.groupby(["date", "uf"])["casos"].sum().reset_index()
            weekly_pivot = weekly_cases.pivot(index="date", columns="uf", values="casos")
            weekly_pivot = weekly_pivot.fillna(0)

            n_states = min(6, len(weekly_pivot.columns))
            sample_states = weekly_pivot.columns[:n_states]

            for uf in sample_states:
                ax.plot(weekly_pivot.index, weekly_pivot[uf], label=uf, linewidth=1, alpha=0.8)
            ax.set_xlabel("Date")
            ax.set_ylabel("Cases")
            ax.set_title(f"Top {n_states} States by Cases Over Time")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        fig.suptitle("Data Quality Summary", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        return fig

    def plot_seasonal_pattern(self, variable: str = "casos") -> Figure:
        """Create seasonal decomposition plot showing yearly patterns.

        Args:
            variable: Column to analyze (casos, temp_med, precip_med, etc.)

        Returns:
            Matplotlib Figure with seasonal patterns
        """
        if variable == "casos":
            df = self.dengue_df.copy()
        elif variable in ["temp_med", "precip_med", "rel_humid_med"]:
            df = self.climate_df.copy()
        else:
            df = self.dengue_df.copy()

        if df.empty or variable not in df.columns:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f"No data for {variable}", ha="center", va="center")
            return fig

        df = df.copy()
        df["week"] = df["date"].dt.isocalendar().week
        df["month"] = df["date"].dt.month

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        weekly_avg = df.groupby("week")[variable].mean()
        axes[0].plot(weekly_avg.index, weekly_avg.values, "b-o", linewidth=2, markersize=4)
        axes[0].axhline(
            y=weekly_avg.mean(), color="red", linestyle="--", label=f"Mean: {weekly_avg.mean():.1f}"
        )
        axes[0].fill_between(weekly_avg.index, weekly_avg.values, alpha=0.3)
        axes[0].set_xlabel("Epidemiological Week")
        axes[0].set_ylabel(variable.title())
        axes[0].set_title(f"Average {variable.title()} by Epidemiological Week")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(1, 53, 4))

        monthly_avg = df.groupby("month")[variable].mean()
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        axes[1].bar(
            monthly_avg.index, monthly_avg.values, color="steelblue", edgecolor="black", alpha=0.8
        )
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(month_names)
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel(variable.title())
        axes[1].set_title(f"Average {variable.title()} by Month")
        axes[1].grid(True, alpha=0.3, axis="y")

        title_state = f" ({self.state})" if self.state else ""
        fig.suptitle(f"Seasonal Patterns{title_state}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, variables: Optional[list[str]] = None) -> Figure:
        """Create correlation matrix heatmap for numerical variables.

        Args:
            variables: List of columns to include. If None, uses default set.

        Returns:
            Matplotlib Figure with correlation matrix
        """
        df = self.dengue_df.copy()

        if df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        if variables is None:
            variables = ["casos", "temp_med", "precip_med", "rel_humid_med"]
            variables = [v for v in variables if v in df.columns]

        if len(variables) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Not enough variables for correlation", ha="center", va="center")
            return fig

        corr_df = df[variables].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        sns.heatmap(
            corr_df,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold")

        plt.tight_layout()
        return fig

    def plot_complete_dashboard(self, output_path: Optional[Path] = None) -> dict[str, Figure]:
        """Generate complete exploratory analysis dashboard.

        Args:
            output_path: Optional path to save figures. If provided, saves all plots.

        Returns:
            Dictionary mapping plot names to Figure objects
        """
        figures = {}

        logger.info("Generating dengue time series plot...")
        figures["dengue_timeseries"] = self.plot_dengue_timeseries()

        logger.info("Generating climate panel plot...")
        figures["climate_panel"] = self.plot_climate_panel()

        logger.info("Generating ocean oscillations plot...")
        figures["ocean_oscillations"] = self.plot_ocean_oscillations()

        logger.info("Generating data quality summary...")
        figures["data_quality"] = self.plot_data_quality_summary()

        logger.info("Generating seasonal patterns...")
        figures["seasonal_cases"] = self.plot_seasonal_pattern("casos")
        if "temp_med" in self.climate_df.columns:
            figures["seasonal_temp"] = self.plot_seasonal_pattern("temp_med")

        logger.info("Generating correlation matrix...")
        figures["correlation"] = self.plot_correlation_matrix()

        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                filepath = output_path / f"exploratory_{name}.png"
                fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
                logger.info(f"Saved {filepath}")

        return figures


def analyze_data(
    state: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Figure]:
    """Convenience function to run full exploratory analysis.

    Args:
        state: State abbreviation to filter data (e.g., "SP")
        output_dir: Directory to save generated plots

    Returns:
        Dictionary of generated figures
    """
    from mosqlimate_ai.data import CompetitionDataLoader

    loader = CompetitionDataLoader()
    analyzer = ExploratoryDataAnalyzer(loader, state=state)

    return analyzer.plot_complete_dashboard(output_path=output_dir)
