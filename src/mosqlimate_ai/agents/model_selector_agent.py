"""Model pre-selection agent for validation pipeline.

Selects top N model types to train per state based on historical
performance from the knowledge base and state characteristics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["xgboost", "lstm", "prophet", "tft", "nbeats"]

MODEL_CHARACTERISTICS = {
    "xgboost": {
        "strengths": ["seasonality", "trend", "feature_importance"],
        "data_requirements": "moderate",
        "training_speed": "fast",
        "interpretability": "high",
    },
    "lstm": {
        "strengths": ["long_term_patterns", "complex_dynamics"],
        "data_requirements": "high",
        "training_speed": "slow",
        "interpretability": "low",
    },
    "prophet": {
        "strengths": ["seasonality", "trend_changepoints", "holidays"],
        "data_requirements": "low",
        "training_speed": "fast",
        "interpretability": "high",
    },
    "tft": {
        "strengths": ["attention", "interpretability", "variable_selection"],
        "data_requirements": "high",
        "training_speed": "slow",
        "interpretability": "high",
    },
    "nbeats": {
        "strengths": ["trend_decomposition", "seasonality"],
        "data_requirements": "moderate",
        "training_speed": "moderate",
        "interpretability": "moderate",
    },
}


@dataclass
class ModelRecommendation:
    """Recommendation for a model type."""

    model_name: str
    priority: int
    score: float
    reasoning: str


class ModelPreSelector:
    """Pre-selects best model types for a given state.

    Uses historical performance from knowledge base and state
    characteristics to recommend which models to train.

    Args:
        max_models: Maximum number of models to select
        knowledge_base: Cross-state knowledge base (optional)
    """

    def __init__(
        self,
        max_models: int = 3,
        knowledge_base: Optional[Any] = None,
    ):
        self.max_models = max_models
        self.knowledge_base = knowledge_base

    def select_models_for_state(
        self,
        state: str,
        data_characteristics: Optional[Dict[str, Any]] = None,
    ) -> List[ModelRecommendation]:
        """Select top models for a state.

        Args:
            state: State UF code
            data_characteristics: Optional dict with data info (size, features, etc.)

        Returns:
            List of model recommendations sorted by priority
        """
        scores = {}

        if self.knowledge_base is not None:
            scores = self._get_knowledge_based_scores(state)
        else:
            scores = self._get_default_scores(state)

        if data_characteristics:
            scores = self._adjust_for_data_characteristics(scores, data_characteristics)

        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for priority, (model_name, score) in enumerate(sorted_models[: self.max_models], 1):
            reasoning = self._generate_reasoning(model_name, state, score)
            recommendations.append(
                ModelRecommendation(
                    model_name=model_name,
                    priority=priority,
                    score=score,
                    reasoning=reasoning,
                )
            )

        logger.info(
            f"Selected models for {state}: "
            f"{[r.model_name for r in recommendations]} "
            f"(scores: {[f'{r.score:.2f}' for r in recommendations]})"
        )

        return recommendations

    def _get_knowledge_based_scores(self, state: str) -> Dict[str, float]:
        """Get model scores from knowledge base history."""
        scores = {model: 0.5 for model in AVAILABLE_MODELS}
        counts = {model: 0 for model in AVAILABLE_MODELS}

        if not hasattr(self.knowledge_base, "results"):
            return scores

        for result in self.knowledge_base.results.values():
            if not isinstance(result, list):
                continue

            for r in result:
                if not hasattr(r, "state") or r.state != state:
                    continue

                model_name = getattr(r, "model_name", "").lower()
                if model_name not in scores:
                    continue

                crps = getattr(r, "crps", float("inf"))
                wis = getattr(r, "wis_total", float("inf"))

                if crps > 0 and crps < float("inf"):
                    normalized_score = 1.0 / (1.0 + crps / 100.0)
                    scores[model_name] += normalized_score
                    counts[model_name] += 1

        for model in scores:
            if counts[model] > 0:
                scores[model] /= counts[model] + 1

        return scores

    def _get_default_scores(self, state: str) -> Dict[str, float]:
        """Get default model scores based on heuristics."""
        scores = {
            "xgboost": 0.85,
            "prophet": 0.75,
            "tft": 0.70,
            "nbeats": 0.65,
            "lstm": 0.60,
        }

        high_seasonality_states = ["SP", "RJ", "CE", "PE", "BA"]
        if state in high_seasonality_states:
            scores["prophet"] += 0.1
            scores["nbeats"] += 0.05

        high_complexity_states = ["SP", "RJ", "MG"]
        if state in high_complexity_states:
            scores["tft"] += 0.1
            scores["lstm"] += 0.05

        small_data_states = ["AC", "AP", "RR", "RO"]
        if state in small_data_states:
            scores["prophet"] += 0.1
            scores["lstm"] -= 0.1
            scores["tft"] -= 0.05

        return scores

    def _adjust_for_data_characteristics(
        self,
        scores: Dict[str, float],
        characteristics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Adjust scores based on data characteristics."""
        data_size = characteristics.get("data_size", 500)
        n_features = characteristics.get("n_features", 10)
        has_missing = characteristics.get("has_missing", False)

        if data_size < 300:
            scores["lstm"] *= 0.7
            scores["tft"] *= 0.7
            scores["prophet"] *= 1.1
        elif data_size > 1000:
            scores["lstm"] *= 1.1
            scores["tft"] *= 1.1

        if n_features > 20:
            scores["xgboost"] *= 1.15
            scores["tft"] *= 1.1

        if has_missing:
            scores["xgboost"] *= 1.1
            scores["prophet"] *= 1.05

        return scores

    def _generate_reasoning(
        self,
        model_name: str,
        state: str,
        score: float,
    ) -> str:
        """Generate human-readable reasoning for model selection."""
        chars = MODEL_CHARACTERISTICS.get(model_name, {})
        strengths = chars.get("strengths", [])

        reasons = []

        if score > 0.8:
            reasons.append(f"High historical performance in {state}")
        elif score > 0.6:
            reasons.append(f"Good match for {state}'s characteristics")

        if "seasonality" in strengths:
            reasons.append("handles seasonality well")
        if "trend" in strengths or "trend_decomposition" in strengths:
            reasons.append("captures trend patterns")
        if "interpretability" in strengths and chars.get("interpretability") == "high":
            reasons.append("highly interpretable")

        if not reasons:
            reasons.append("general purpose model")

        return f"{model_name.upper()}: " + ", ".join(reasons)

    def get_all_model_classes(self) -> Dict[str, type]:
        """Get dictionary of all available model classes."""
        from mosqlimate_ai.models import (
            XGBoostForecaster,
            LSTMForecaster,
            ProphetForecaster,
            TFTForecaster,
            NBEATSForecaster,
        )

        return {
            "xgboost": XGBoostForecaster,
            "lstm": LSTMForecaster,
            "prophet": ProphetForecaster,
            "tft": TFTForecaster,
            "nbeats": NBEATSForecaster,
        }

    def get_model_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
            },
            "prophet": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "seasonality_mode": "multiplicative",
            },
            "tft": {
                "hidden_size": 64,
                "attention_head_size": 4,
                "dropout": 0.1,
                "learning_rate": 0.001,
            },
            "nbeats": {
                "hidden_size": 256,
                "learning_rate": 0.001,
            },
        }
        return defaults.get(model_name.lower(), {})
