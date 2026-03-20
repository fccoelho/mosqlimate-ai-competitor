"""Visualization module for Mosqlimate AI Competitor."""

from mosqlimate_ai.visualization.report_plots import ReportVisualizer, fig_to_base64
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
from mosqlimate_ai.visualization.validation_report import (
    ValidationPDFReport,
    generate_validation_report,
)

__all__ = [
    "ReportVisualizer",
    "fig_to_base64",
    "ValidationPDFReport",
    "generate_validation_report",
    "plot_validation_test_timeseries",
    "plot_all_validation_tests",
    "plot_crps_progression",
    "plot_wis_progression",
    "plot_coverage_analysis",
    "plot_model_performance_heatmap",
    "create_validation_figure_set",
    "save_figure_for_pdf",
]
