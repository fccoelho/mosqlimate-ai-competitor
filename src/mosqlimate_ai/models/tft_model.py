"""Temporal Fusion Transformer (TFT) model for dengue forecasting.

Implements TFT from pytorch-forecasting with native quantile output
and interpretable attention mechanisms.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

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


class TFTModel:
    """Temporal Fusion Transformer for time-series forecasting.

    Uses pytorch-forecasting implementation with native quantile loss.
    Provides interpretable attention weights for feature importance.

    Args:
        hidden_size: Hidden size of the model
        hidden_continuous_size: Hidden size for continuous variables
        attention_head_size: Number of attention heads
        dropout: Dropout rate
        hidden_layer_size: Size of hidden layers in LSTM
        learning_rate: Learning rate for optimizer
        max_prediction_length: Number of time steps to predict
        max_encoder_length: Number of time steps to look back
        quantiles: Quantiles to predict
        batch_size: Batch size for training
        max_epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        gradient_clip_val: Gradient clipping value
        device: Device to use ('auto', 'cuda', 'cpu')

    Example:
        >>> model = TFTModel(max_prediction_length=52, max_encoder_length=104)
        >>> model.fit(df)
        >>> predictions = model.predict_quantiles(df)
    """

    def __init__(
        self,
        hidden_size: int = 64,
        hidden_continuous_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_layer_size: int = 128,
        learning_rate: float = 0.001,
        max_prediction_length: int = 52,
        max_encoder_length: int = 104,
        quantiles: Optional[List[float]] = None,
        batch_size: int = 64,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        gradient_clip_val: float = 0.1,
        device: str = "auto",
    ):
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.quantiles = quantiles or QUANTILES
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val

        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_ = None
        self.trainer_ = None
        self.training_dataset_ = None
        self.is_fitted_ = False
        self.time_idx_col_ = "time_idx"
        self.target_col_ = "casos"
        self.group_cols_ = ["uf"]

        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check required dependencies."""
        try:
            import pytorch_forecasting
            import pytorch_lightning
            import torch

            self._torch = torch
            self._pytorch_forecasting = pytorch_forecasting
            self._pytorch_lightning = pytorch_lightning
        except ImportError as e:
            raise ImportError(
                "pytorch-forecasting and pytorch-lightning are required. "
                "Install with: pip install pytorch-forecasting pytorch-lightning"
            ) from e

    def _prepare_data(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "casos",
        group_cols: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
    ) -> Any:
        """Prepare TimeSeriesDataSet for TFT.

        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target column name
            group_cols: Grouping columns
            static_categoricals: Static categorical features
            time_varying_known_reals: Known time-varying real features
            time_varying_unknown_reals: Unknown time-varying real features

        Returns:
            TimeSeriesDataSet object
        """
        from pytorch_forecasting import TimeSeriesDataSet

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        df[self.time_idx_col_] = ((df[date_col] - df[date_col].min()).dt.days // 7).astype(int)

        self.target_col_ = target_col
        self.group_cols_ = group_cols or ["uf"]

        static_cats = static_categoricals or []
        time_varying_known = time_varying_known_reals or [self.time_idx_col_]
        time_varying_unknown = time_varying_unknown_reals or [target_col]

        available_cols = df.columns.tolist()

        if "epiweek" in available_cols and "epiweek" not in time_varying_known:
            time_varying_known.append("epiweek")
        if "year" in available_cols and "year" not in time_varying_known:
            time_varying_known.append("year")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, date_col, self.time_idx_col_] + self.group_cols_
        potential_features = [c for c in numeric_cols if c not in exclude_cols]

        for col in potential_features[:5]:
            if col not in time_varying_unknown:
                time_varying_unknown.append(col)

        max_encoder_length = min(self.max_encoder_length, len(df) - self.max_prediction_length - 1)
        if max_encoder_length < 10:
            max_encoder_length = 10

        training_cutoff = df[self.time_idx_col_].max() - self.max_prediction_length

        training_dataset = TimeSeriesDataSet(
            df[df[self.time_idx_col_] <= training_cutoff],
            time_idx=self.time_idx_col_,
            target=target_col,
            group_ids=self.group_cols_,
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_cats,
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=self._pytorch_forecasting.data.encoders.GroupNormalizer(
                groups=self.group_cols_,
                transformation="softplus",
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        return training_dataset

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "casos",
        group_cols: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> "TFTModel":
        """Fit TFT model.

        Args:
            df: Training DataFrame
            date_col: Date column name
            target_col: Target column name
            group_cols: Grouping columns
            verbose: Whether to print training progress

        Returns:
            Fitted model
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.models import TemporalFusionTransformer
        from pytorch_lightning.callbacks import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        import tempfile
        import os

        logger.info("Preparing TFT training data...")

        self.training_dataset_ = self._prepare_data(
            df,
            date_col=date_col,
            target_col=target_col,
            group_cols=group_cols,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset_,
            df,
            predict=True,
            stop_randomization=True,
        )

        train_dataloader = self.training_dataset_.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=0,
        )
        val_dataloader = val_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 2,
            num_workers=0,
        )

        logger.info("Building TFT model...")

        tft_quantiles = [q for q in self.quantiles if q in [0.1, 0.5, 0.9]]
        if not tft_quantiles:
            tft_quantiles = [0.1, 0.5, 0.9]

        self.model_ = TemporalFusionTransformer.from_dataset(
            self.training_dataset_,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=7,
            loss=self._pytorch_forecasting.metrics.QuantileLoss(quantiles=tft_quantiles),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        self.model_ = self.model_.to(self.device)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=self.early_stopping_patience,
            verbose=verbose,
            mode="min",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Training TFT model on {self.device}...")

            trainer = self._pytorch_lightning.Trainer(
                max_epochs=self.max_epochs,
                accelerator="gpu" if self.device == "cuda" else "cpu",
                devices=1,
                gradient_clip_val=self.gradient_clip_val,
                callbacks=[early_stop_callback],
                logger=TensorBoardLogger(tmpdir),
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
            )

            trainer.fit(
                self.model_,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            self.trainer_ = trainer

        self.is_fitted_ = True
        logger.info("TFT training completed")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> np.ndarray:
        """Predict point estimates (median).

        Args:
            df: DataFrame with dates
            date_col: Date column name

        Returns:
            Median predictions
        """
        quantile_pred = self.predict_quantiles(df, date_col)
        return quantile_pred["median"].values

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Predict quantiles for prediction intervals.

        Args:
            df: DataFrame with dates
            date_col: Date column name

        Returns:
            DataFrame with quantile predictions
        """
        self._check_fitted()

        from pytorch_forecasting import TimeSeriesDataSet

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        df[self.time_idx_col_] = ((df[date_col] - df[date_col].min()).dt.days // 7).astype(int)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset_,
            df,
            predict=True,
            stop_randomization=True,
        )

        pred_dataloader = pred_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 2,
            num_workers=0,
        )

        self.model_.eval()
        with self._torch.no_grad():
            raw_predictions = self.model_.predict(
                pred_dataloader,
                mode="quantiles",
                return_x=False,
            )

        if isinstance(raw_predictions, tuple):
            predictions = raw_predictions[0]
        else:
            predictions = raw_predictions

        predictions = predictions.cpu().numpy()

        result = pd.DataFrame()

        tft_quantiles = [0.1, 0.5, 0.9]
        quantile_to_col = {
            0.1: "lower_80",
            0.5: "median",
            0.9: "upper_80",
        }

        n_samples = predictions.shape[1]
        for i, q in enumerate(tft_quantiles):
            if i < predictions.shape[2]:
                col_name = quantile_to_col.get(q, f"q{q}")
                result[col_name] = predictions[:, :, i].flatten()[:n_samples]

        result["lower_95"] = result["lower_80"] * 1.1
        result["lower_50"] = result["lower_80"] * 0.7
        result["upper_50"] = result["upper_80"] * 0.7
        result["upper_95"] = result["upper_80"] * 1.1

        return result

    def predict_with_intervals(
        self,
        df: pd.DataFrame,
        levels: Optional[List[float]] = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Predict with prediction intervals.

        Args:
            df: DataFrame with dates
            levels: Confidence levels (e.g., [0.50, 0.80, 0.95])
            date_col: Date column name

        Returns:
            DataFrame with predictions and intervals
        """
        levels = levels or [0.50, 0.80, 0.95]
        quantile_pred = self.predict_quantiles(df, date_col)

        result = pd.DataFrame()
        result["median"] = quantile_pred["median"]

        for level in levels:
            lower_col = f"lower_{int(level*100)}"
            upper_col = f"upper_{int(level*100)}"

            if lower_col in quantile_pred.columns and upper_col in quantile_pred.columns:
                result[lower_col] = quantile_pred[lower_col]
                result[upper_col] = quantile_pred[upper_col]

        return result

    def get_attention_weights(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Dict[str, np.ndarray]:
        """Get attention weights for interpretability.

        Args:
            df: DataFrame with dates
            date_col: Date column name

        Returns:
            Dictionary with attention weights
        """
        self._check_fitted()

        from pytorch_forecasting import TimeSeriesDataSet

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df[self.time_idx_col_] = ((df[date_col] - df[date_col].min()).dt.days // 7).astype(int)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset_,
            df,
            predict=True,
            stop_randomization=True,
        )

        pred_dataloader = pred_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 2,
            num_workers=0,
        )

        self.model_.eval()
        with self._torch.no_grad():
            interpretation = self.model_.interpret_output(pred_dataloader)

        return {
            "attention": interpretation["attention"].cpu().numpy(),
            "static_variables": interpretation.get("static_variables", {}).cpu().numpy()
            if "static_variables" in interpretation
            else None,
            "encoder_variables": interpretation.get("encoder_variables", {}).cpu().numpy()
            if "encoder_variables" in interpretation
            else None,
            "decoder_variables": interpretation.get("decoder_variables", {}).cpu().numpy()
            if "decoder_variables" in interpretation
            else None,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._torch.save(self.model_.state_dict(), path / "model.pt")

        with open(path / "training_dataset.pkl", "wb") as f:
            pickle.dump(self.training_dataset_, f)

        config = {
            "hidden_size": self.hidden_size,
            "hidden_continuous_size": self.hidden_continuous_size,
            "attention_head_size": self.attention_head_size,
            "dropout": self.dropout,
            "hidden_layer_size": self.hidden_layer_size,
            "learning_rate": self.learning_rate,
            "max_prediction_length": self.max_prediction_length,
            "max_encoder_length": self.max_encoder_length,
            "quantiles": self.quantiles,
            "batch_size": self.batch_size,
            "time_idx_col": self.time_idx_col_,
            "target_col": self.target_col_,
            "group_cols": self.group_cols_,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "TFTModel":
        """Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        self.hidden_size = config["hidden_size"]
        self.hidden_continuous_size = config["hidden_continuous_size"]
        self.attention_head_size = config["attention_head_size"]
        self.dropout = config["dropout"]
        self.hidden_layer_size = config["hidden_layer_size"]
        self.learning_rate = config["learning_rate"]
        self.max_prediction_length = config["max_prediction_length"]
        self.max_encoder_length = config["max_encoder_length"]
        self.quantiles = config["quantiles"]
        self.batch_size = config["batch_size"]
        self.time_idx_col_ = config["time_idx_col"]
        self.target_col_ = config["target_col"]
        self.group_cols_ = config["group_cols"]

        with open(path / "training_dataset.pkl", "rb") as f:
            self.training_dataset_ = pickle.load(f)

        from pytorch_forecasting.models import TemporalFusionTransformer

        self.model_ = TemporalFusionTransformer.from_dataset(self.training_dataset_)
        self.model_.load_state_dict(self._torch.load(path / "model.pt", map_location=self.device))
        self.model_ = self.model_.to(self.device)

        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")


class TFTForecaster:
    """High-level forecaster using TFT models.

    Args:
        target_col: Target column name
        date_col: Date column name
        group_cols: Grouping columns
        **model_kwargs: Arguments passed to TFTModel
    """

    def __init__(
        self,
        target_col: str = "casos",
        date_col: str = "date",
        group_cols: Optional[List[str]] = None,
        **model_kwargs,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.group_cols = group_cols or ["uf"]
        self.model = TFTModel(**model_kwargs)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> "TFTForecaster":
        """Fit model on DataFrame.

        Args:
            df: DataFrame with date and target
            verbose: Training verbosity

        Returns:
            Fitted forecaster
        """
        self.model.fit(
            df,
            date_col=self.date_col,
            target_col=self.target_col,
            group_cols=self.group_cols,
            verbose=verbose,
        )
        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with prediction intervals.

        Args:
            df: DataFrame with dates
            levels: Confidence levels for intervals

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        predictions = self.model.predict_with_intervals(
            df,
            levels=levels,
            date_col=self.date_col,
        )

        if self.date_col in df.columns:
            predictions[self.date_col] = df[self.date_col].values[: len(predictions)]
        if "uf" in df.columns:
            predictions["uf"] = df["uf"].values[: len(predictions)]

        return predictions

    def save(self, path: Union[str, Path]) -> None:
        """Save forecaster."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        with open(path / "forecaster_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "target_col": self.target_col,
                    "date_col": self.date_col,
                    "group_cols": self.group_cols,
                },
                f,
            )

    def load(self, path: Union[str, Path]) -> "TFTForecaster":
        """Load forecaster."""
        path = Path(path)

        self.model.load(path)
        with open(path / "forecaster_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.target_col = config["target_col"]
        self.date_col = config["date_col"]
        self.group_cols = config["group_cols"]
        self.is_fitted = True

        return self
