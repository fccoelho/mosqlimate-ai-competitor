"""Model selection agent for validation pipeline.

Selects top N models based on validation performance across all tests.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TopNModelSelectionAgent:
    """Select top N models based on validation performance.

        This agent analyzes performance across all 3 validation tests and selects
    the best N models based on a weighted composite score.
    """

    def __init__(
        self,
        n_top: int = 3,
        min_coverage_threshold: float = 0.85,
        max_bias_threshold: float = 500,
    ):
        """Initialize model selection agent.

        Args:
            n_top: Number of top models to select
            min_coverage_threshold: Minimum acceptable 95% coverage
            max_bias_threshold: Maximum acceptable absolute bias
        """
        self.n_top = n_top
        self.min_coverage_threshold = min_coverage_threshold
        self.max_bias_threshold = max_bias_threshold

        # Weights for composite score
        self.weights = {
            "crps": 0.35,
            "wis_total": 0.25,
            "mae": 0.20,
            "coverage_95": 0.15,
            "bias": 0.05,
        }

    def select_top_models(
        self,
        all_results: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Select top N models from validation results.

        Args:
            all_results: Nested dict of results by model and validation test
                         Format: {model_name: {test_num: metrics}}

        Returns:
            List of top model configurations
        """
        model_scores = []

        for model_name, test_results in all_results.items():
            # Must have results from all 3 validation tests
            if len(test_results) < 3:
                logger.warning(f"Model {model_name} missing validation tests, skipping")
                continue

            # Calculate average performance across tests
            avg_metrics = self._calculate_average_metrics(test_results)

            # Check minimum criteria
            if not self._meets_minimum_criteria(avg_metrics):
                logger.info(f"Model {model_name} does not meet minimum criteria")
                continue

            # Calculate composite score
            score = self._calculate_composite_score(avg_metrics)

            # Calculate consistency score (low variance across tests is good)
            consistency = self._calculate_consistency(test_results)

            # Combined score (lower is better)
            combined_score = score * (1 + (1 - consistency) * 0.2)  # Penalty for inconsistency

            model_scores.append(
                {
                    "model_name": model_name,
                    "combined_score": combined_score,
                    "performance_score": score,
                    "consistency": consistency,
                    "avg_metrics": avg_metrics,
                    "test_results": test_results,
                }
            )

        if not model_scores:
            logger.error("No models met selection criteria")
            return []

        # Sort by combined score (ascending - lower is better)
        model_scores.sort(key=lambda x: x["combined_score"])

        # Select top N
        selected = model_scores[: self.n_top]

        logger.info(f"Selected top {len(selected)} models:")
        for i, model in enumerate(selected, 1):
            logger.info(
                f"  {i}. {model['model_name']}: "
                f"score={model['combined_score']:.2f}, "
                f"consistency={model['consistency']:.2%}"
            )

        return selected

    def _calculate_average_metrics(
        self,
        test_results: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate average metrics across validation tests."""
        metrics_to_average = [
            "crps",
            "wis_total",
            "rmse",
            "mae",
            "mape",
            "bias",
            "coverage_50",
            "coverage_80",
            "coverage_90",
            "coverage_95",
        ]

        avg_metrics = {}
        for metric in metrics_to_average:
            values = [
                test_results[test][metric] for test in test_results if metric in test_results[test]
            ]
            if values:
                avg_metrics[metric] = np.mean(values)

        return avg_metrics

    def _meets_minimum_criteria(self, avg_metrics: Dict[str, float]) -> bool:
        """Check if model meets minimum quality criteria."""
        # Check coverage
        if avg_metrics.get("coverage_95", 0) < self.min_coverage_threshold:
            return False

        # Check bias
        if abs(avg_metrics.get("bias", 0)) > self.max_bias_threshold:
            return False

        return True

    def _calculate_composite_score(self, avg_metrics: Dict[str, float]) -> float:
        """Calculate weighted composite score (lower is better)."""
        score = 0.0

        # CRPS (lower is better)
        score += self.weights["crps"] * avg_metrics.get("crps", 1e6)

        # WIS (lower is better)
        score += self.weights["wis_total"] * avg_metrics.get("wis_total", 1e6)

        # MAE (lower is better)
        score += self.weights["mae"] * avg_metrics.get("mae", 1e6)

        # Coverage (penalize deviation from 95%)
        coverage_penalty = abs(avg_metrics.get("coverage_95", 0.95) - 0.95)
        score += self.weights["coverage_95"] * coverage_penalty * 1000

        # Bias (penalize large bias)
        bias_penalty = abs(avg_metrics.get("bias", 0)) / 1000
        score += self.weights["bias"] * bias_penalty

        return score

    def _calculate_consistency(self, test_results: Dict[int, Dict[str, Any]]) -> float:
        """Calculate consistency score across validation tests.

        Returns a score between 0 and 1, where 1 means perfectly consistent.
        """
        primary_metric = "crps"
        values = [test_results[test][primary_metric] for test in sorted(test_results.keys())]

        if len(values) < 2:
            return 1.0

        # Coefficient of variation (lower is more consistent)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 1.0

        cv = std_val / mean_val

        # Convert to consistency score (1 - normalized CV)
        consistency = max(0, 1 - cv)

        return consistency

    def generate_model_report(self, selected_models: List[Dict[str, Any]]) -> str:
        """Generate human-readable report of selected models.

        Args:
            selected_models: List of selected model dictionaries

        Returns:
            Markdown-formatted report
        """
        lines = [
            "# Top Model Selection Report",
            "",
            f"**Selection Criteria:**",
            f"- Minimum 95% coverage: {self.min_coverage_threshold:.0%}",
            f"- Maximum absolute bias: {self.max_bias_threshold}",
            f"- Number of models selected: {self.n_top}",
            "",
            "## Selected Models",
            "",
        ]

        for i, model in enumerate(selected_models, 1):
            lines.extend(
                [
                    f"### {i}. {model['model_name']}",
                    "",
                    "**Performance Summary:**",
                    "",
                    f"| Metric | Average | Test 1 | Test 2 | Test 3 |",
                    f"|--------|---------|--------|--------|--------|",
                ]
            )

            metrics = ["crps", "wis_total", "rmse", "mae", "mape", "bias", "coverage_95"]
            for metric in metrics:
                avg_val = model["avg_metrics"].get(metric, 0)
                test_vals = [model["test_results"][test].get(metric, 0) for test in [1, 2, 3]]

                if metric == "coverage_95":
                    lines.append(
                        f"| {metric} | {avg_val:.1%} | {test_vals[0]:.1%} | {test_vals[1]:.1%} | {test_vals[2]:.1%} |"
                    )
                else:
                    lines.append(
                        f"| {metric} | {avg_val:.2f} | {test_vals[0]:.2f} | {test_vals[1]:.2f} | {test_vals[2]:.2f} |"
                    )

            lines.extend(
                [
                    "",
                    f"**Composite Score:** {model['combined_score']:.2f}",
                    f"**Consistency:** {model['consistency']:.2%}",
                    "",
                    "---",
                    "",
                ]
            )

        lines.extend(
            [
                "## Selection Weights",
                "",
                "The composite score is calculated using the following weights:",
                "",
            ]
        )

        for metric, weight in self.weights.items():
            lines.append(f"- **{metric}:** {weight:.0%}")

        lines.append("")

        return "\n".join(lines)
