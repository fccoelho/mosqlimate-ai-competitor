"""N-BEATS model for dengue forecasting with uncertainty quantification.

Implements N-BEATS from pytorch-forecasting with MC Dropout for
uncertainty estimation.
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


class NBEATSModel:
    """N-BEATS model for time-series forecasting with uncertainty.

    Uses pytorch-forecasting implementation with MC Dropout for
    probabilistic predictions.

    Args:
        stack_types: Types of stacks ('generic', 'trend', 'seasonality')
        num_blocks: Number of blocks per stack
        num_block_layers: Number of layers per block
        hidden_size: Hidden layer size
        learning_rate: Learning rate
        max_prediction_length: Number of time steps to predict
        max_encoder_length: Number of time steps to look back
        quantiles: Quantiles to predict
        batch_size: Batch size for training
        max_epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        gradient_clip_val: Gradient clipping value
        mc_samples: Number of MC Dropout samples for uncertainty
        device: Device to use ('auto', 'cuda', 'cpu')

    Example:
        >>> model = NBEATSModel(max_prediction_length=52, max_encoder_length=104)
        >>> model.fit(df)
        >>> predictions = model.predict_quantiles(df)
    """

    def __init__(
        self,
        stack_types: Optional[List[str]] = None,
        num_blocks: List[int] = [3],
        num_block_layers: List[int] = [4],
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        max_prediction_length: int = 52,
        max_encoder_length: int = 104,
        quantiles: Optional[List[float]] = None,
        batch_size: int = 64,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        gradient_clip_val: float = 1.0,
        mc_samples: int = 100,
        device: str = "auto",
        dropout: float = 0.1,
    ):
        self.stack_types = stack_types or ["generic"]
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.quantiles = quantiles or QUANTILES
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.mc_samples = mc_samples
        self.dropout = dropout

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
    ) -> Any:
        """Prepare TimeSeriesDataSet for N-BEATS.

        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target column name
            group_cols: Grouping columns

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
    ) -> "NBEATSModel":
        """Fit N-BEATS model.

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
        from pytorch_forecasting.models import NBeats
        from pytorch_lightning.callbacks import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        import tempfile

        logger.info("Preparing N-BEATS training data...")

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

        logger.info("Building N-BEATS model...")

        self.model_ = NBeats.from_dataset(
            self.training_dataset_,
            learning_rate=self.learning_rate,
            log_interval=10,
            log_val_interval=1,
            weight_decay=1e-2,
            widths=[self.hidden_size] * 4,
            backcast_loss_ratio=0.1,
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
            logger.info(f"Training N-BEATS model on {self.device}...")

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
        logger.info("N-BEATS training completed")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> np.ndarray:
        """Predict point estimates.

        Args:
            df: DataFrame with dates
            date_col: Date column name

        Returns:
            Predictions
        """
        quantile_pred = self.predict_quantiles(df, date_col)
        return quantile_pred["median"].values

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict quantiles using MC Dropout.

        Args:
            df: DataFrame with dates
            date_col: Date column name
            n_samples: Number of MC samples

        Returns:
            DataFrame with quantile predictions
        """
        self._check_fitted()
        n_samples = n_samples or self.mc_samples

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

        self.model_.train()

        all_predictions = []
        with self._torch.no_grad():
            for _ in range(n_samples):
                pred = self.model_.predict(pred_dataloader, return_x=False)
                if isinstance(pred, self._torch.Tensor):
                    all_predictions.append(pred.cpu().numpy())
                else:
                    all_predictions.append(np.array(pred))

        all_predictions = np.array(all_predictions)

        result = pd.DataFrame()

        n_obs = all_predictions.shape[1] if all_predictions.ndim > 1 else 1

        for q in self.quantiles:
            col_name = QUANTILE_MAP.get(q, f"q{q}")
            result[col_name] = np.quantile(all_predictions, q, axis=0).flatten()[:n_obs]

        result = result.clip(lower=0)

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
            "stack_types": self.stack_types,
            "num_blocks": self.num_blocks,
            "num_block_layers": self.num_block_layers,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "max_prediction_length": self.max_prediction_length,
            "max_encoder_length": self.max_encoder_length,
            "quantiles": self.quantiles,
            "batch_size": self.batch_size,
            "mc_samples": self.mc_samples,
            "time_idx_col": self.time_idx_col_,
            "target_col": self.target_col_,
            "group_cols": self.group_cols_,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "NBEATSModel":
        """Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        self.stack_types = config["stack_types"]
        self.num_blocks = config["num_blocks"]
        self.num_block_layers = config["num_block_layers"]
        self.hidden_size = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.max_prediction_length = config["max_prediction_length"]
        self.max_encoder_length = config["max_encoder_length"]
        self.quantiles = config["quantiles"]
        self.batch_size = config["batch_size"]
        self.mc_samples = config["mc_samples"]
        self.time_idx_col_ = config["time_idx_col"]
        self.target_col_ = config["target_col"]
        self.group_cols_ = config["group_cols"]

        with open(path / "training_dataset.pkl", "rb") as f:
            self.training_dataset_ = pickle.load(f)

        from pytorch_forecasting.models import NBeats

        self.model_ = NBeats.from_dataset(self.training_dataset_)
        self.model_.load_state_dict(self._torch.load(path / "model.pt", map_location=self.device))
        self.model_ = self.model_.to(self.device)

        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")


class NBEATSForecaster:
    """High-level forecaster using N-BEATS models.

    Args:
        target_col: Target column name
        date_col: Date column name
        group_cols: Grouping columns
        **model_kwargs: Arguments passed to NBEATSModel
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
        self.model = NBEATSModel(**model_kwargs)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> "NBEATSForecaster":
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
        n_mc_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with prediction intervals.

        Args:
            df: DataFrame with dates
            levels: Confidence levels for intervals
            n_mc_samples: Number of MC Dropout samples

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        if n_mc_samples:
            quantile_pred = self.model.predict_quantiles(df, self.date_col, n_mc_samples)
        else:
            quantile_pred = self.model.predict_quantiles(df, self.date_col)

        levels = levels or [0.50, 0.80, 0.95]

        result = pd.DataFrame()
        result["median"] = quantile_pred["median"]

        for level in levels:
            lower_col = f"lower_{int(level*100)}"
            upper_col = f"upper_{int(level*100)}"

            if lower_col in quantile_pred.columns and upper_col in quantile_pred.columns:
                result[lower_col] = quantile_pred[lower_col]
                result[upper_col] = quantile_pred[upper_col]

        if self.date_col in df.columns:
            result[self.date_col] = df[self.date_col].values[: len(result)]
        if "uf" in df.columns:
            result["uf"] = df["uf"].values[: len(result)]

        return result

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

    def load(self, path: Union[str, Path]) -> "NBEATSForecaster":
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
