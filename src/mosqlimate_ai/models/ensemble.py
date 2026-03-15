"""Ensemble forecaster combining multiple models."""

from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """Combine multiple forecasting models.
    
    This class implements ensemble methods for dengue forecasting,
    including weighted averaging and stacking.
    
    Example:
        >>> ensemble = EnsembleForecaster()
        >>> ensemble.add_model("xgboost", xgb_model, weight=0.4)
        >>> ensemble.add_model("lstm", lstm_model, weight=0.6)
        >>> forecast = ensemble.predict(X_test)
    """
    
    def __init__(self, method: str = "weighted_average"):
        """Initialize ensemble.
        
        Args:
            method: Ensemble method (weighted_average, stacking)
        """
        self.method = method
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.meta_model: Optional[Any] = None  # For stacking
        logger.info(f"EnsembleForecaster initialized with {method}")
    
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add model to ensemble.
        
        Args:
            name: Model identifier
            model: Trained model object
            weight: Model weight for averaging
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model {name} with weight {weight}")
    
    def fit_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Optimize model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        # Simple inverse RMSE weighting
        performance = {}
        for name, model in self.models.items():
            pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((pred - y_val) ** 2))
            performance[name] = 1 / rmse  # Inverse RMSE
        
        # Normalize to sum to 1
        total = sum(performance.values())
        for name in performance:
            self.weights[name] = performance[name] / total
        
        logger.info(f"Optimized weights: {self.weights}")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_individual: bool = False
    ) -> Dict[str, Any]:
        """Generate ensemble prediction.
        
        Args:
            X: Feature matrix
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary with ensemble and individual predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            ensemble_pred += pred * (weight / total_weight)
        
        result = {
            "ensemble": ensemble_pred,
            "weights_used": self.weights
        }
        
        if return_individual:
            result["individual"] = predictions
        
        return result
    
    def predict_with_intervals(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """Generate predictions with uncertainty intervals.
        
        Args:
            X: Feature matrix
            confidence: Confidence level for intervals
            
        Returns:
            DataFrame with point estimates and intervals
        """
        # Get predictions from all models
        all_preds = []
        for model in self.models.values():
            pred = model.predict(X)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)
        
        # Calculate statistics
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        # Confidence intervals (normal approximation)
        alpha = 1 - confidence
        z_score = 1.96  # for 95% CI
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return pd.DataFrame({
            "forecast": mean_pred,
            f"lower_{int(confidence*100)}": lower,
            f"upper_{int(confidence*100)}": upper
        })
