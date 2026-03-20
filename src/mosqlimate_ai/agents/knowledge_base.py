"""Cross-state knowledge base for validation pipeline.

Allows state agents to learn from each other's experiences.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StateProfile:
    """Profile of a state's characteristics for similarity matching."""

    state_uf: str
    population: float
    climate_zone: str
    biome: str
    avg_cases: float
    peak_season: str
    historical_pattern: str  # 'endemic', 'epidemic', 'epidemic_outliers'


@dataclass
class ValidationResult:
    """Results from a single validation test."""

    state: str
    validation_test: int
    model_name: str
    crps: float
    wis_total: float
    rmse: float
    mae: float
    mape: float
    bias: float
    coverage_50: float
    coverage_80: float
    coverage_90: float
    coverage_95: float
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "validation_test": self.validation_test,
            "model_name": self.model_name,
            "crps": self.crps,
            "wis_total": self.wis_total,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "bias": self.bias,
            "coverage_50": self.coverage_50,
            "coverage_80": self.coverage_80,
            "coverage_90": self.coverage_90,
            "coverage_95": self.coverage_95,
            "hyperparameters": self.hyperparameters,
            "timestamp": self.timestamp,
        }


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration."""

    model_type: str  # 'xgboost' or 'lstm'
    params: Dict[str, Any]
    performance_score: float  # Weighted composite score
    source_state: str
    validation_tests: List[int]


class CrossStateKnowledgeBase:
    """Knowledge base for sharing insights between state agents.

    This class allows agents to learn from similar states and
    apply successful strategies across the federation.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize knowledge base.

        Args:
            cache_dir: Directory for caching knowledge base data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/knowledge_base")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State profiles for similarity matching
        self.state_profiles: Dict[str, StateProfile] = self._load_state_profiles()

        # Validation results by state and test
        self.validation_results: Dict[str, List[ValidationResult]] = {}

        # Best hyperparameters by state
        self.best_hyperparameters: Dict[str, Dict[str, HyperparameterConfig]] = {}

        # Cross-state insights
        self.insights: List[Dict[str, Any]] = []

        logger.info("CrossStateKnowledgeBase initialized")

    def _load_state_profiles(self) -> Dict[str, StateProfile]:
        """Load state profiles with demographic and climate info."""
        # This would ideally come from a database or config file
        # For now, using hardcoded representative data
        profiles = {
            "SP": StateProfile(
                "SP", 46000000, "tropical", "atlantic_forest", 1500, "summer", "endemic"
            ),
            "RJ": StateProfile(
                "RJ", 17000000, "tropical", "atlantic_forest", 1200, "summer", "endemic"
            ),
            "MG": StateProfile(
                "MG", 21000000, "tropical", "atlantic_forest", 800, "summer", "endemic"
            ),
            "BA": StateProfile(
                "BA", 15000000, "tropical", "atlantic_forest", 600, "year_round", "endemic"
            ),
            "PR": StateProfile(
                "PR", 11000000, "subtropical", "atlantic_forest", 400, "summer", "epidemic"
            ),
            "SC": StateProfile(
                "SC", 7000000, "subtropical", "atlantic_forest", 300, "summer", "epidemic"
            ),
            "RS": StateProfile("RS", 11000000, "subtropical", "pampas", 250, "summer", "epidemic"),
            "AM": StateProfile("AM", 4000000, "tropical", "amazon", 900, "year_round", "endemic"),
            "PA": StateProfile("PA", 8000000, "tropical", "amazon", 700, "year_round", "endemic"),
        }
        return profiles

    def get_similar_states(
        self,
        target_state: str,
        n_similar: int = 3,
        criteria: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find states similar to the target state.

        Args:
            target_state: Target state UF
            n_similar: Number of similar states to return
            criteria: Similarity criteria (climate, biome, population, pattern)

        Returns:
            List of (state_uf, similarity_score) tuples
        """
        if target_state not in self.state_profiles:
            logger.warning(f"No profile for state {target_state}")
            return []

        target = self.state_profiles[target_state]
        criteria = criteria or ["climate", "biome", "pattern"]

        similarities = []
        for state_uf, profile in self.state_profiles.items():
            if state_uf == target_state:
                continue

            score = 0.0
            weights = {"climate": 0.3, "biome": 0.3, "population": 0.2, "pattern": 0.2}

            if "climate" in criteria and profile.climate_zone == target.climate_zone:
                score += weights["climate"]

            if "biome" in criteria and profile.biome == target.biome:
                score += weights["biome"]

            if "population" in criteria:
                # Similar population (within 50%)
                pop_ratio = min(profile.population, target.population) / max(
                    profile.population, target.population
                )
                score += weights["population"] * pop_ratio

            if "pattern" in criteria and profile.historical_pattern == target.historical_pattern:
                score += weights["pattern"]

            # Bonus if target state has validation results to learn from
            if state_uf in self.validation_results:
                score *= 1.2  # Boost states with proven results

            similarities.append((state_uf, score))

        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

    def share_results(self, result: ValidationResult) -> None:
        """Share validation results from a state agent.

        Args:
            result: Validation results
        """
        state = result.state

        if state not in self.validation_results:
            self.validation_results[state] = []

        self.validation_results[state].append(result)

        # Update best hyperparameters
        self._update_best_hyperparameters(result)

        logger.info(
            f"Shared results from {state} test {result.validation_test}: " f"CRPS={result.crps:.2f}"
        )

    def _update_best_hyperparameters(self, result: ValidationResult) -> None:
        """Update best hyperparameters based on new result."""
        state = result.state
        model_type = result.model_name

        if state not in self.best_hyperparameters:
            self.best_hyperparameters[state] = {}

        # Calculate composite performance score (lower is better)
        # Weight: CRPS 50%, WIS 30%, coverage penalty 20%
        coverage_penalty = abs(result.coverage_95 - 0.95) + abs(result.coverage_50 - 0.50)
        score = 0.5 * result.crps + 0.3 * result.wis_total + 0.2 * coverage_penalty * 1000

        config = HyperparameterConfig(
            model_type=model_type,
            params=result.hyperparameters,
            performance_score=score,
            source_state=state,
            validation_tests=[result.validation_test],
        )

        # Keep only the best config per model type
        if model_type not in self.best_hyperparameters[state]:
            self.best_hyperparameters[state][model_type] = config
        elif score < self.best_hyperparameters[state][model_type].performance_score:
            self.best_hyperparameters[state][model_type] = config
            logger.info(f"Updated best {model_type} config for {state} (score: {score:.2f})")

    def get_best_params(
        self,
        states: List[str],
        model_type: str = "xgboost",
        metric: str = "crps",
    ) -> Dict[str, Any]:
        """Get best hyperparameters from similar states.

        Args:
            states: List of state UFs to consider
            model_type: Type of model ('xgboost' or 'lstm')
            metric: Primary metric for ranking

        Returns:
            Dictionary of recommended hyperparameters
        """
        configs = []
        for state in states:
            if state in self.best_hyperparameters:
                if model_type in self.best_hyperparameters[state]:
                    configs.append(self.best_hyperparameters[state][model_type])

        if not configs:
            logger.warning(f"No hyperparameter configs found for {states}")
            return {}

        # Sort by performance score
        configs.sort(key=lambda c: c.performance_score)

        # Return best config
        best = configs[0]
        logger.info(f"Best params from {best.source_state}: {best.params}")
        return best.params.copy()

    def get_tuning_recommendations(
        self,
        state: str,
        validation_results: List[ValidationResult],
    ) -> Dict[str, Any]:
        """Get tuning recommendations based on cross-state analysis.

        Args:
            state: Target state
            validation_results: Current validation results for this state

        Returns:
            Recommendations for hyperparameter adjustments
        """
        similar_states = self.get_similar_states(state)
        recommendations = {
            "similar_states": [s[0] for s in similar_states],
            "suggested_adjustments": {},
            "patterns_observed": [],
        }

        # Analyze similar states' successful strategies
        for similar_state, similarity in similar_states:
            if similar_state not in self.validation_results:
                continue

            similar_results = self.validation_results[similar_state]

            # Look for improvement patterns
            if len(similar_results) >= 2:
                # Check if performance improved from test 1 to test 3
                test1 = [r for r in similar_results if r.validation_test == 1]
                test3 = [r for r in similar_results if r.validation_test == 3]

                if test1 and test3:
                    crps_improvement = test1[0].crps - test3[0].crps
                    if crps_improvement > 0:
                        recommendations["patterns_observed"].append(
                            f"{similar_state} improved CRPS by {crps_improvement:.2f} "
                            f"through validation (similarity: {similarity:.2f})"
                        )

        # Analyze current state's performance
        if validation_results:
            latest = validation_results[-1]

            # Coverage-based recommendations
            if latest.coverage_95 < 0.90:
                recommendations["suggested_adjustments"]["coverage_issue"] = (
                    "95% coverage below target. Consider widening prediction intervals."
                )

            if latest.coverage_50 < 0.45 or latest.coverage_50 > 0.55:
                recommendations["suggested_adjustments"]["calibration_issue"] = (
                    f"50% coverage is {latest.coverage_50:.1%}. "
                    "Consider recalibrating quantiles."
                )

            # Bias-based recommendations
            if abs(latest.bias) > 100:
                direction = "overestimating" if latest.bias > 0 else "underestimating"
                recommendations["suggested_adjustments"]["bias_issue"] = (
                    f"Model is {direction} by {abs(latest.bias):.0f} cases on average."
                )

        return recommendations

    def get_aggregate_insights(self) -> Dict[str, Any]:
        """Get aggregate insights across all states.

        Returns:
            Dictionary with aggregate statistics and insights
        """
        insights = {
            "total_states_with_results": len(self.validation_results),
            "validation_tests_completed": sum(
                len(results) for results in self.validation_results.values()
            ),
            "best_performing_states": [],
            "common_strategies": [],
            "regional_patterns": {},
        }

        # Find best performing states by climate zone
        climate_performance = {}
        for state, results in self.validation_results.items():
            if state not in self.state_profiles:
                continue

            climate = self.state_profiles[state].climate_zone
            if climate not in climate_performance:
                climate_performance[climate] = []

            # Average CRPS across all tests
            avg_crps = np.mean([r.crps for r in results])
            climate_performance[climate].append((state, avg_crps))

        for climate, performances in climate_performance.items():
            performances.sort(key=lambda x: x[1])
            insights["regional_patterns"][climate] = {
                "best_state": performances[0][0],
                "best_crps": performances[0][1],
                "avg_crps": np.mean([p[1] for p in performances]),
            }

        return insights

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save knowledge base to disk.

        Args:
            filepath: Path to save file
        """
        filepath = filepath or self.cache_dir / "knowledge_base.json"

        data = {
            "validation_results": {
                state: [r.to_dict() for r in results]
                for state, results in self.validation_results.items()
            },
            "best_hyperparameters": {
                state: {
                    model: {
                        "model_type": config.model_type,
                        "params": config.params,
                        "performance_score": config.performance_score,
                        "source_state": config.source_state,
                        "validation_tests": config.validation_tests,
                    }
                    for model, config in models.items()
                }
                for state, models in self.best_hyperparameters.items()
            },
            "insights": self.insights,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Knowledge base saved to {filepath}")

    def load(self, filepath: Optional[Path] = None) -> None:
        """Load knowledge base from disk.

        Args:
            filepath: Path to load file
        """
        filepath = filepath or self.cache_dir / "knowledge_base.json"

        if not filepath.exists():
            logger.warning(f"No knowledge base file found at {filepath}")
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Restore validation results
            for state, results_data in data.get("validation_results", {}).items():
                self.validation_results[state] = [ValidationResult(**r) for r in results_data]

            # Restore best hyperparameters
            for state, models_data in data.get("best_hyperparameters", {}).items():
                self.best_hyperparameters[state] = {}
                for model, config_data in models_data.items():
                    self.best_hyperparameters[state][model] = HyperparameterConfig(
                        model_type=config_data["model_type"],
                        params=config_data["params"],
                        performance_score=config_data["performance_score"],
                        source_state=config_data["source_state"],
                        validation_tests=config_data["validation_tests"],
                    )

            self.insights = data.get("insights", [])

            logger.info(f"Knowledge base loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
