"""PDF report generator for validation pipeline results.

Generates comprehensive 6-page PDF reports per state with:
- Executive summary
- Time series analysis (observed vs predicted)
- CRPS and WIS metric analysis
- Model performance comparison
- Coverage and calibration analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from mosqlimate_ai.visualization.validation_plots import (
    create_validation_figure_set,
    plot_all_validation_tests,
    plot_coverage_analysis,
    plot_crps_progression,
    plot_model_performance_heatmap,
    plot_validation_test_timeseries,
    plot_wis_progression,
    save_figure_for_pdf,
)

logger = logging.getLogger(__name__)


class ValidationPDFReport:
    """Generate comprehensive PDF validation reports per state.

    Creates professional 6-page PDF reports including time series visualizations,
    CRPS/WIS analysis, model comparisons, and coverage analysis.

    Example:
        >>> report = ValidationPDFReport("SP")
        >>> pdf_path = report.generate_from_files()
        >>> print(f"Report saved to: {pdf_path}")

    Attributes:
        state: State UF code (e.g., "SP", "RJ")
        output_dir: Directory containing validation results
        report_data: Loaded validation results
    """

    def __init__(
        self,
        state: str,
        output_dir: Path = Path("validation_results"),
    ):
        """Initialize PDF report generator.

        Args:
            state: State UF code (e.g., "SP")
            output_dir: Directory containing validation results
        """
        self.state = state.upper()
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / self.state
        self.report_data: Optional[Dict[str, Any]] = None

        logger.info(f"Initialized ValidationPDFReport for {self.state}")

    def generate_from_files(self) -> Path:
        """Generate PDF report from saved validation result files.

        Loads results from validation_results/{state}/validation_results.json
        and generates a complete PDF report.

        Returns:
            Path to generated PDF file

        Raises:
            FileNotFoundError: If validation results file not found
        """
        # Load validation results
        results_file = self.state_dir / "validation_results.json"
        if not results_file.exists():
            raise FileNotFoundError(
                f"Validation results not found: {results_file}\n"
                f"Run validation first: mosqlimate-ai validate --states {self.state}"
            )

        with open(results_file) as f:
            self.report_data = json.load(f)

        logger.info(f"Loaded validation results for {self.state}")

        # Generate report
        return self.generate_report()

    def generate_report(
        self,
        validation_results: Optional[Dict[str, Any]] = None,
        observed_data: Optional[pd.DataFrame] = None,
        forecast_data: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> Path:
        """Generate complete 6-page PDF validation report.

        Args:
            validation_results: Validation results dict (loaded from file if None)
            observed_data: Observed data DataFrame (optional)
            forecast_data: Forecast data by test number (optional)

        Returns:
            Path to generated PDF file
        """
        if validation_results:
            self.report_data = validation_results

        if not self.report_data:
            raise ValueError("No validation data available. Load from files or pass data.")

        # Create figures directory
        figures_dir = self.state_dir / "report_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Generate or load figures
        if observed_data is not None and forecast_data is not None:
            figures = self._generate_figures(observed_data, forecast_data, figures_dir)
        else:
            figures = self._create_placeholder_figures(figures_dir)

        # Build PDF
        pdf_path = self._build_pdf(figures)

        logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path

    def _generate_figures(
        self,
        observed_data: pd.DataFrame,
        forecast_data: Dict[int, pd.DataFrame],
        figures_dir: Path,
    ) -> Dict[str, Path]:
        """Generate all figures for the report.

        Args:
            observed_data: Observed dengue cases
            forecast_data: Forecasts by test number
            figures_dir: Directory to save figures

        Returns:
            Dictionary mapping figure names to file paths
        """
        # Extract results by test
        results_by_test = self.report_data.get("validation_tests", {})

        # Convert string keys back to integers
        results_by_test = {int(k): v for k, v in results_by_test.items()}

        # Create figures using validation_plots module
        figures = create_validation_figure_set(
            state=self.state,
            observed_df=observed_data,
            test_forecasts=forecast_data,
            results_by_test=results_by_test,
            output_dir=figures_dir,
        )

        return figures

    def _create_placeholder_figures(self, figures_dir: Path) -> Dict[str, Path]:
        """Create placeholder/example figures when data is not available.

        Args:
            figures_dir: Directory to save figures

        Returns:
            Dictionary mapping figure names to file paths
        """
        import matplotlib.pyplot as plt
        import numpy as np

        figures = {}

        # Create a simple placeholder time series
        fig, ax = plt.subplots(figsize=(14, 8))
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        values = np.random.poisson(100, 100)
        ax.plot(dates, values, "k-", label="Observed (Example)")
        ax.set_title(f"{self.state} - Validation Forecasts (Example Data)")
        ax.legend()

        filepath = figures_dir / f"{self.state}_timeseries.png"
        save_figure_for_pdf(fig, filepath)
        figures["timeseries"] = filepath

        # Create placeholder CRPS plot
        fig, ax = plt.subplots(figsize=(12, 6))
        tests = [1, 2, 3]
        ax.plot(tests, [0.5, 0.45, 0.42], "o-", label="XGBoost")
        ax.plot(tests, [0.55, 0.48, 0.44], "s-", label="LSTM")
        ax.set_xlabel("Validation Test")
        ax.set_ylabel("CRPS")
        ax.set_title("CRPS Progression (Example)")
        ax.legend()

        filepath = figures_dir / f"{self.state}_crps_progression.png"
        save_figure_for_pdf(fig, filepath)
        figures["crps"] = filepath

        # Similar for WIS
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(tests, [150, 140, 135], "o-", label="XGBoost")
        ax.plot(tests, [160, 145, 138], "s-", label="LSTM")
        ax.set_xlabel("Validation Test")
        ax.set_ylabel("WIS")
        ax.set_title("WIS Progression (Example)")
        ax.legend()

        filepath = figures_dir / f"{self.state}_wis_progression.png"
        save_figure_for_pdf(fig, filepath)
        figures["wis"] = filepath

        # Coverage analysis
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        for idx, (level, target) in enumerate([(50, 0.5), (80, 0.8), (95, 0.95)]):
            ax = axes[idx]
            ax.bar([1, 2, 3], [48, 51, 49], alpha=0.7, label="XGBoost")
            ax.bar([1, 2, 3], [52, 49, 51], alpha=0.7, label="LSTM")
            ax.axhline(y=target * 100, color="red", linestyle="--")
            ax.set_title(f"{level}% Coverage")
            ax.set_ylim(0, 100)

        filepath = figures_dir / f"{self.state}_coverage.png"
        save_figure_for_pdf(fig, filepath)
        figures["coverage"] = filepath

        # Performance heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [[0.42, 0.40, 0.38], [0.45, 0.43, 0.41]]
        im = ax.imshow(data, cmap="YlOrRd_r", aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Test 1", "Test 2", "Test 3"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["XGBoost", "LSTM"])
        plt.colorbar(im, ax=ax)
        ax.set_title("Performance Heatmap (Example)")

        filepath = figures_dir / f"{self.state}_heatmap_crps.png"
        save_figure_for_pdf(fig, filepath)
        figures["heatmap"] = filepath

        return figures

    def _build_pdf(self, figures: Dict[str, Path]) -> Path:
        """Build the PDF document from generated figures and data.

        Args:
            figures: Dictionary mapping figure names to file paths

        Returns:
            Path to generated PDF
        """
        # Setup PDF document
        pdf_path = self.state_dir / f"{self.state}_validation_report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        # Get styles
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor("#1a5490"),
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor("#2c3e50"),
        )

        subheading_style = ParagraphStyle(
            "CustomSubheading",
            parent=styles["Heading3"],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.HexColor("#34495e"),
        )

        normal_style = styles["Normal"]
        normal_style.fontSize = 10
        normal_style.spaceAfter = 6

        # Build content
        story = []

        # Page 1: Executive Summary
        story.append(
            Paragraph(f"MOSQLIMATE SPRINT 2025<br/>Validation Report: {self.state}", title_style)
        )
        story.append(Spacer(1, 0.5 * cm))

        # Report metadata
        story.append(Paragraph("<b>Executive Summary</b>", heading_style))
        story.append(Spacer(1, 0.3 * cm))

        # Generation info
        story.append(
            Paragraph(
                f"<b>State:</b> {self.state}<br/>"
                f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
                f"<b>Validation Tests:</b> 3 (2022-2023, 2023-2024, 2024-2025)",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.5 * cm))

        # Performance summary table
        story.append(Paragraph("<b>Overall Performance Summary</b>", subheading_style))

        summary_data = self._extract_summary_data()
        if summary_data:
            table = self._create_performance_table(summary_data)
            story.append(table)
            story.append(Spacer(1, 0.5 * cm))

        # Best model info
        top_models = self.report_data.get("top_models", [])
        if top_models:
            story.append(Paragraph("<b>Top Performing Models</b>", subheading_style))
            for idx, model in enumerate(top_models[:3], 1):
                model_name = model.get("model_name", "Unknown")
                score = model.get("composite_score", 0)
                story.append(
                    Paragraph(
                        f"{idx}. <b>{model_name.title()}</b> - Composite Score: {score:.3f}",
                        normal_style,
                    )
                )
            story.append(Spacer(1, 0.5 * cm))

        # Key insights
        story.append(Paragraph("<b>Key Insights</b>", subheading_style))
        insights = self._generate_insights()
        for insight in insights:
            story.append(Paragraph(f"• {insight}", normal_style))

        story.append(PageBreak())

        # Pages 2-4: Individual Test Time Series
        story.append(Paragraph("Validation Test Time Series", heading_style))
        story.append(
            Paragraph(
                "Individual time series plots for each validation test showing training period, "
                "prediction period, forecast with prediction intervals (50%, 80%, 95%), and observed values.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        # Test 1
        if "test1_timeseries" in figures:
            story.append(Paragraph("<b>Test 1: 2022-2023 Season</b>", subheading_style))
            img = Image(str(figures["test1_timeseries"]), width=17 * cm, height=8 * cm)
            story.append(img)
            story.append(
                Paragraph(
                    "Training period ends EW25 2022. Forecast period: EW41 2022 to EW40 2023.",
                    normal_style,
                )
            )
            story.append(Spacer(1, 0.3 * cm))
            story.append(PageBreak())

        # Test 2
        if "test2_timeseries" in figures:
            story.append(Paragraph("<b>Test 2: 2023-2024 Season</b>", subheading_style))
            img = Image(str(figures["test2_timeseries"]), width=17 * cm, height=8 * cm)
            story.append(img)
            story.append(
                Paragraph(
                    "Training period ends EW25 2023. Forecast period: EW41 2023 to EW40 2024.",
                    normal_style,
                )
            )
            story.append(Spacer(1, 0.3 * cm))
            story.append(PageBreak())

        # Test 3
        if "test3_timeseries" in figures:
            story.append(Paragraph("<b>Test 3: 2024-2025 Season</b>", subheading_style))
            img = Image(str(figures["test3_timeseries"]), width=17 * cm, height=8 * cm)
            story.append(img)
            story.append(
                Paragraph(
                    "Training period ends EW25 2024. Forecast period: EW41 2024 to EW40 2025.",
                    normal_style,
                )
            )
            story.append(Spacer(1, 0.3 * cm))
            story.append(PageBreak())

        # All Tests Overview
        if "all_tests" in figures:
            story.append(Paragraph("<b>All Tests Overview</b>", subheading_style))
            img = Image(str(figures["all_tests"]), width=16 * cm, height=12 * cm)
            story.append(img)
            story.append(PageBreak())

        # CRPS Analysis
        story.append(Paragraph("CRPS Analysis", heading_style))
        story.append(
            Paragraph(
                "Continuous Ranked Probability Score (CRPS) measures probabilistic forecast accuracy. "
                "Lower values indicate better performance.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "crps" in figures:
            img = Image(str(figures["crps"]), width=16 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        # CRPS values table
        story.append(Paragraph("<b>CRPS Values by Test</b>", subheading_style))
        crps_table = self._create_metric_table("crps")
        if crps_table:
            story.append(crps_table)

        story.append(PageBreak())

        # WIS Analysis
        story.append(Paragraph("WIS Analysis", heading_style))
        story.append(
            Paragraph(
                "Weighted Interval Score (WIS) evaluates probabilistic forecasts across multiple "
                "prediction intervals. Lower values are better.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "wis" in figures:
            img = Image(str(figures["wis"]), width=16 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("<b>WIS Values by Test</b>", subheading_style))
        wis_table = self._create_metric_table("wis_total")
        if wis_table:
            story.append(wis_table)

        story.append(PageBreak())

        # Model Performance
        story.append(Paragraph("Model Performance Comparison", heading_style))
        story.append(
            Paragraph(
                "Performance heatmap comparing all models across validation tests.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "heatmap" in figures:
            img = Image(str(figures["heatmap"]), width=14 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        # Model details
        story.append(Paragraph("<b>Model Details</b>", subheading_style))
        story.append(self._create_model_details_table())

        story.append(PageBreak())

        # Coverage Analysis
        story.append(Paragraph("Prediction Interval Coverage", heading_style))
        story.append(
            Paragraph(
                "Coverage analysis shows how often observed values fall within prediction intervals. "
                "Values close to target percentages (50%, 80%, 95%) indicate well-calibrated forecasts.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "coverage" in figures:
            img = Image(str(figures["coverage"]), width=17 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("<b>Coverage Statistics</b>", subheading_style))
        coverage_table = self._create_coverage_table()
        if coverage_table:
            story.append(coverage_table)

        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                "<b>Note:</b> Green bars indicate coverage within 5% of target. "
                "Red dashed lines show target coverage levels.",
                normal_style,
            )
        )

        story.append(PageBreak())

        # Page 3: CRPS Analysis
        story.append(Paragraph("CRPS Analysis", heading_style))
        story.append(
            Paragraph(
                "Continuous Ranked Probability Score (CRPS) measures probabilistic forecast accuracy. "
                "Lower values indicate better performance.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "crps" in figures:
            img = Image(str(figures["crps"]), width=16 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        # CRPS values table
        story.append(Paragraph("<b>CRPS Values by Test</b>", subheading_style))
        crps_table = self._create_metric_table("crps")
        if crps_table:
            story.append(crps_table)

        story.append(PageBreak())

        # Page 4: WIS Analysis
        story.append(Paragraph("WIS Analysis", heading_style))
        story.append(
            Paragraph(
                "Weighted Interval Score (WIS) evaluates probabilistic forecasts across multiple "
                "prediction intervals. Lower values are better.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "wis" in figures:
            img = Image(str(figures["wis"]), width=16 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("<b>WIS Values by Test</b>", subheading_style))
        wis_table = self._create_metric_table("wis_total")
        if wis_table:
            story.append(wis_table)

        story.append(PageBreak())

        # Page 5: Model Performance
        story.append(Paragraph("Model Performance Comparison", heading_style))
        story.append(
            Paragraph(
                "Performance heatmap comparing all models across validation tests.", normal_style
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "heatmap" in figures:
            img = Image(str(figures["heatmap"]), width=14 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        # Model details
        story.append(Paragraph("<b>Model Details</b>", subheading_style))
        story.append(self._create_model_details_table())

        story.append(PageBreak())

        # Page 6: Coverage Analysis
        story.append(Paragraph("Prediction Interval Coverage", heading_style))
        story.append(
            Paragraph(
                "Coverage analysis shows how often observed values fall within prediction intervals. "
                "Values close to target percentages (50%, 80%, 95%) indicate well-calibrated forecasts.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        if "coverage" in figures:
            img = Image(str(figures["coverage"]), width=17 * cm, height=8 * cm)
            story.append(img)

        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("<b>Coverage Statistics</b>", subheading_style))
        coverage_table = self._create_coverage_table()
        if coverage_table:
            story.append(coverage_table)

        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                "<b>Note:</b> Green bars indicate coverage within 5% of target. "
                "Red dashed lines show target coverage levels.",
                normal_style,
            )
        )

        # Build PDF
        doc.build(story)

        return pdf_path

    def _extract_summary_data(self) -> Dict[str, Any]:
        """Extract summary statistics from validation results."""
        summary = {}

        validation_tests = self.report_data.get("validation_tests", {})

        # Calculate average metrics across tests
        crps_values = []
        wis_values = []

        for test_num, test_results in validation_tests.items():
            metrics = test_results.get("metrics", {})
            for model_name, model_metrics in metrics.items():
                if "crps" in model_metrics:
                    crps_values.append(model_metrics["crps"])
                if "wis_total" in model_metrics:
                    wis_values.append(model_metrics["wis_total"])

        if crps_values:
            summary["avg_crps"] = sum(crps_values) / len(crps_values)
            summary["best_crps"] = min(crps_values)

        if wis_values:
            summary["avg_wis"] = sum(wis_values) / len(wis_values)
            summary["best_wis"] = min(wis_values)

        return summary

    def _create_performance_table(self, summary: Dict[str, Any]) -> Table:
        """Create performance summary table."""
        data = [
            ["Metric", "Average", "Best"],
            ["CRPS", f"{summary.get('avg_crps', 0):.3f}", f"{summary.get('best_crps', 0):.3f}"],
            ["WIS", f"{summary.get('avg_wis', 0):.1f}", f"{summary.get('best_wis', 0):.1f}"],
        ]

        table = Table(data, colWidths=[6 * cm, 4 * cm, 4 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ecf0f1")],
                    ),
                ]
            )
        )

        return table

    def _create_metric_table(self, metric_name: str) -> Optional[Table]:
        """Create metric comparison table."""
        validation_tests = self.report_data.get("validation_tests", {})

        if not validation_tests:
            return None

        # Get all models
        all_models = set()
        for test_results in validation_tests.values():
            metrics = test_results.get("metrics", {})
            all_models.update(metrics.keys())
        models = sorted(all_models)

        if not models:
            return None

        # Build table data
        headers = ["Model", "Test 1", "Test 2", "Test 3", "Average"]
        data = [headers]

        for model in models:
            row = [model.title()]
            values = []

            for test_num in [1, 2, 3]:
                test_results = validation_tests.get(str(test_num), {})
                metrics = test_results.get("metrics", {})
                model_metrics = metrics.get(model, {})
                val = model_metrics.get(metric_name)

                if val is not None:
                    row.append(f"{val:.3f}")
                    values.append(val)
                else:
                    row.append("N/A")

            if values:
                row.append(f"{sum(values)/len(values):.3f}")
            else:
                row.append("N/A")

            data.append(row)

        # Create table
        col_widths = [4 * cm] + [3 * cm] * 4
        table = Table(data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ecf0f1")],
                    ),
                ]
            )
        )

        return table

    def _create_model_details_table(self) -> Table:
        """Create model hyperparameters and details table."""
        data = [
            ["Model", "Type", "Description"],
            ["XGBoost", "Tree-based", "Gradient boosting with temporal features"],
            ["LSTM", "Neural Network", "Long Short-Term Memory recurrent network"],
            ["Ensemble", "Combined", "Weighted average of XGBoost and LSTM"],
        ]

        table = Table(data, colWidths=[4 * cm, 4 * cm, 8 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ecf0f1")],
                    ),
                ]
            )
        )

        return table

    def _create_coverage_table(self) -> Optional[Table]:
        """Create coverage statistics table."""
        validation_tests = self.report_data.get("validation_tests", {})

        if not validation_tests:
            return None

        headers = [
            "Model",
            "50% Target",
            "50% Actual",
            "80% Target",
            "80% Actual",
            "95% Target",
            "95% Actual",
        ]
        data = [headers]

        # Aggregate coverage across tests
        coverage_levels = ["50", "80", "95"]

        # Get models from first test
        first_test = list(validation_tests.values())[0]
        models = sorted(first_test.get("metrics", {}).keys())

        for model in models:
            row = [model.title()]

            for level in coverage_levels:
                target = float(level)
                row.append(f"{target:.0f}%")

                # Calculate average actual coverage
                coverages = []
                for test_results in validation_tests.values():
                    metrics = test_results.get("metrics", {})
                    model_metrics = metrics.get(model, {})
                    cov = model_metrics.get(f"coverage_{level}")
                    if cov is not None:
                        coverages.append(cov * 100)

                if coverages:
                    avg_cov = sum(coverages) / len(coverages)
                    row.append(f"{avg_cov:.1f}%")
                else:
                    row.append("N/A")

            data.append(row)

        col_widths = [3.5 * cm] + [2.3 * cm] * 6
        table = Table(data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ecf0f1")],
                    ),
                ]
            )
        )

        return table

    def _generate_insights(self) -> List[str]:
        """Generate key insights from validation results."""
        insights = []

        validation_tests = self.report_data.get("validation_tests", {})
        top_models = self.report_data.get("top_models", [])

        if not validation_tests:
            insights.append("No validation data available for analysis.")
            return insights

        # Check for improvement trend
        crps_trend = []
        for test_num in [1, 2, 3]:
            test_results = validation_tests.get(str(test_num), {})
            metrics = test_results.get("metrics", {})
            for model_metrics in metrics.values():
                if "crps" in model_metrics:
                    crps_trend.append(model_metrics["crps"])
                    break

        if len(crps_trend) >= 2 and crps_trend[-1] < crps_trend[0]:
            improvement = ((crps_trend[0] - crps_trend[-1]) / crps_trend[0]) * 100
            insights.append(
                f"Models showed {improvement:.1f}% improvement in CRPS from Test 1 to Test 3, "
                f"indicating learning across validation periods."
            )

        # Best model info
        if top_models:
            best_model = top_models[0]
            model_name = best_model.get("model_name", "Unknown")
            insights.append(
                f"{model_name.title()} emerged as the top performer based on composite scoring "
                f"of CRPS, WIS, and coverage metrics."
            )

        # Coverage calibration
        coverage_good = True
        for test_results in validation_tests.values():
            metrics = test_results.get("metrics", {})
            for model_metrics in metrics.values():
                cov_95 = model_metrics.get("coverage_95", 0)
                if cov_95 < 0.90:  # Less than 90% for 95% interval
                    coverage_good = False
                    break

        if coverage_good:
            insights.append(
                "Prediction intervals are well-calibrated with 95% coverage close to target, "
                f"indicating reliable uncertainty quantification."
            )
        else:
            insights.append(
                "Some prediction intervals show under-coverage, suggesting potential for "
                "improved uncertainty estimation in future iterations."
            )

        # Add default if too few insights
        if len(insights) < 2:
            insights.append(
                f"Validation completed successfully across {len(validation_tests)} test periods "
                f"with multiple model architectures evaluated."
            )

        return insights[:3]  # Limit to 3 insights


def generate_validation_report(
    state: str,
    output_dir: Path = Path("validation_results"),
    validation_results: Optional[Dict[str, Any]] = None,
    observed_data: Optional[pd.DataFrame] = None,
    forecast_data: Optional[Dict[int, pd.DataFrame]] = None,
) -> Path:
    """Convenience function to generate a validation report.

    Args:
        state: State UF code
        output_dir: Directory containing validation results
        validation_results: Optional pre-loaded validation results
        observed_data: Optional observed data DataFrame
        forecast_data: Optional forecast data by test number

    Returns:
        Path to generated PDF file
    """
    report = ValidationPDFReport(state, output_dir)

    if validation_results is not None:
        return report.generate_report(validation_results, observed_data, forecast_data)
    else:
        return report.generate_from_files()
