"""XGBoost model with quantile regression for prediction intervals.

Implements gradient boosting models for dengue forecasting with
proper quantile regression for uncertainty quantification.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
QUANTILE_MAP = {
    0.025: "lower_95",
    0.05: "lower_90",
    0.10: "lower_80",
    0.25: "lower_50",
    0.50: "median",
    0.75: "upper_50",
    0.90: "upper_80",
    0.95: "upper_90",
    0.975: "upper_95",
}


class XGBoostQuantileModel:
    """XGBoost model for quantile regression forecasting.

    This model trains separate gradient boosting models for each quantile,
    enabling proper prediction interval estimation.

    Args:
        quantiles: List of quantiles to predict
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        early_stopping_rounds: Early stopping patience
        random_state: Random seed

    Example:
        >>> model = XGBoostQuantileModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict_quantiles(X_test)
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.quantiles = quantiles or QUANTILES
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.models_: Dict[float, Any] = {}
        self.feature_importances_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_ = False

        self._check_xgboost()

    def _check_xgboost(self) -> None:
        """Check if XGBoost is available."""
        try:
            import xgboost as xgb

            self._xgb = xgb
        except ImportError as e:
            raise ImportError("XGBoost is required. Install with: pip install xgboost") from e

    def _create_model(self, quantile: float) -> Any:
        """Create XGBoost model for a specific quantile.

        Args:
            quantile: Quantile to estimate

        Returns:
            XGBoost regressor configured for quantile regression
        """
        return self._xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            tree_method="hist",
            enable_categorical=True,
            missing=np.nan,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = False,
    ) -> "XGBoostQuantileModel":
        """Fit quantile regression models.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features for early stopping
            y_val: Validation target for early stopping
            verbose: Whether to print training progress

        Returns:
            Fitted model
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        y = np.asarray(y).ravel()

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, np.asarray(y_val).ravel())]

        logger.info(f"Training XGBoost models for {len(self.quantiles)} quantiles...")

        for quantile in self.quantiles:
            model = self._create_model(quantile)

            if eval_set:
                model.fit(X, y, eval_set=eval_set, verbose=verbose)
            else:
                model.fit(X, y, verbose=verbose)

            self.models_[quantile] = model
            logger.debug(f"Trained model for quantile {quantile}")

        self._compute_feature_importance()
        self.is_fitted_ = True

        logger.info("XGBoost training completed")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict median values.

        Args:
            X: Features

        Returns:
            Median predictions
        """
        self._check_fitted()
        return self.models_[0.5].predict(X)

    def predict_quantiles(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Predict all quantiles.

        Args:
            X: Features

        Returns:
            DataFrame with quantile predictions
        """
        self._check_fitted()

        predictions = {}
        for quantile in self.quantiles:
            col_name = QUANTILE_MAP.get(quantile, f"q{quantile}")
            predictions[col_name] = self.models_[quantile].predict(X)

        return pd.DataFrame(predictions)

    def predict_with_intervals(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Predict with prediction intervals.

        Args:
            X: Features
            levels: Confidence levels (e.g., [0.50, 0.80, 0.95])

        Returns:
            DataFrame with predictions and intervals
        """
        levels = levels or [0.50, 0.80, 0.95]
        quantile_pred = self.predict_quantiles(X)

        result = pd.DataFrame()
        result["median"] = quantile_pred["median"]

        for level in levels:
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            lower_col = QUANTILE_MAP.get(lower_q, f"q{lower_q}")
            upper_col = QUANTILE_MAP.get(upper_q, f"q{upper_q}")

            if lower_col in quantile_pred.columns and upper_col in quantile_pred.columns:
                result[f"lower_{int(level*100)}"] = quantile_pred[lower_col]
                result[f"upper_{int(level*100)}"] = quantile_pred[upper_col]

        return result

    def _compute_feature_importance(self) -> None:
        """Compute feature importance across all quantile models."""
        importance_data = []

        for quantile, model in self.models_.items():
            importance = model.feature_importances_
            for name, imp in zip(self.feature_names_, importance):
                importance_data.append({"feature": name, "quantile": quantile, "importance": imp})

        self.feature_importances_ = pd.DataFrame(importance_data)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance summary.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        self._check_fitted()

        importance = self.feature_importances_.groupby("feature")["importance"].mean()
        importance = importance.sort_values(ascending=False).head(top_n)

        return pd.DataFrame({"feature": importance.index, "importance": importance.values})

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        gap: int = 4,
    ) -> Dict[str, Any]:
        """Perform time-series cross-validation.

        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits
            gap: Gap between train and test (in samples)

        Returns:
            Dictionary with CV results
        """
        y = np.asarray(y).ravel()

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        cv_results = {
            "rmse": [],
            "mae": [],
            "coverage_95": [],
            "coverage_80": [],
            "coverage_50": [],
        }

        logger.info(f"Running {n_splits}-fold time-series cross-validation...")

        for _fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            val_size = len(test_idx) // 4
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]

            self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            pred = self.predict_quantiles(X_test)

            cv_results["rmse"].append(np.sqrt(mean_squared_error(y_test, pred["median"])))
            cv_results["mae"].append(mean_absolute_error(y_test, pred["median"]))

            for _level, col in [(0.95, "95"), (0.80, "80"), (0.50, "50")]:
                lower = pred[f"lower_{col}"]
                upper = pred[f"upper_{col}"]
                coverage = np.mean((y_test >= lower) & (y_test <= upper))
                cv_results[f"coverage_{col}"].append(coverage)

        for metric, values in cv_results.items():
            cv_results[metric] = {"mean": np.mean(values), "std": np.std(values), "values": values}

        logger.info(
            f"CV Results - RMSE: {cv_results['rmse']['mean']:.2f} "
            f"(±{cv_results['rmse']['std']:.2f})"
        )

        return cv_results

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for quantile, model in self.models_.items():
            model.save_model(str(path / f"model_q{quantile}.json"))

        config = {
            "quantiles": self.quantiles,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_names": self.feature_names_,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "XGBoostQuantileModel":
        """Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        self.quantiles = config["quantiles"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.learning_rate = config["learning_rate"]
        self.feature_names_ = config["feature_names"]

        for quantile in self.quantiles:
            model = self._create_model(quantile)
            model.load_model(str(path / f"model_q{quantile}.json"))
            self.models_[quantile] = model

        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")


class XGBoostForecaster:
    """High-level forecaster using XGBoost quantile models.

    This class wraps XGBoostQuantileModel for easy use in the
    multi-agent forecasting pipeline.

    Args:
        target_col: Target column name
        feature_cols: Feature column names
        **model_kwargs: Arguments passed to XGBoostQuantileModel
    """

    def __init__(
        self,
        target_col: str = "casos",
        feature_cols: Optional[List[str]] = None,
        **model_kwargs,
    ):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.model = XGBoostQuantileModel(**model_kwargs)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        validation_size: float = 0.1,
        verbose: bool = False,
    ) -> "XGBoostForecaster":
        """Fit model on DataFrame.

        Args:
            df: DataFrame with features and target
            validation_size: Fraction for validation
            verbose: Training verbosity

        Returns:
            Fitted forecaster
        """
        if self.feature_cols is None:
            self.feature_cols = self._infer_features(df)

        X = df[self.feature_cols].values.copy()
        y = df[self.target_col].values.copy()

        X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        split_idx = int(len(X) * (1 - validation_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)
        self.is_fitted = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
        levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with prediction intervals.

        Args:
            df: DataFrame with features
            levels: Confidence levels for intervals

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        X = df[self.feature_cols].values.copy()
        X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
        predictions = self.model.predict_with_intervals(X, levels=levels)

        if "date" in df.columns:
            predictions["date"] = df["date"].values
        if "uf" in df.columns:
            predictions["uf"] = df["uf"].values

        return predictions

    def _infer_features(self, df: pd.DataFrame) -> List[str]:
        """Infer feature columns from DataFrame."""
        exclude = [
            "date",
            "uf",
            "geocode",
            "epiweek",
            "year",
            "train_1",
            "train_2",
            "train_3",
            "target_1",
            "target_2",
            "target_3",
            self.target_col,
        ]
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]

    def save(self, path: Union[str, Path]) -> None:
        """Save forecaster."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        with open(path / "forecaster_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "target_col": self.target_col,
                    "feature_cols": self.feature_cols,
                },
                f,
            )

    def load(self, path: Union[str, Path]) -> "XGBoostForecaster":
        """Load forecaster."""
        path = Path(path)

        self.model.load(path)
        with open(path / "forecaster_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.target_col = config["target_col"]
        self.feature_cols = config["feature_cols"]
        self.is_fitted = True

        return self
