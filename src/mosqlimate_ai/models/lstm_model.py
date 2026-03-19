"""LSTM model with Monte Carlo Dropout for uncertainty quantification.

Implements deep learning models for dengue forecasting with
probabilistic predictions via MC Dropout at inference time.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]


class LSTMModel:
    """LSTM model for time-series forecasting with uncertainty.

    Uses Monte Carlo Dropout for probabilistic predictions.
    Implements sequence-to-sequence architecture for multi-horizon forecasting.

    Args:
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate (used at inference for MC Dropout)
        output_size: Number of output neurons (1 for single step)
        quantiles: Quantiles to estimate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        device: Device to use ('auto', 'cuda', 'cpu')

    Example:
        >>> model = LSTMModel(hidden_size=128, num_layers=2)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict_with_uncertainty(X_test)
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        quantiles: Optional[List[float]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        device: str = "auto",
        mc_samples: int = 100,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.quantiles = quantiles or QUANTILES
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.mc_samples = mc_samples

        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_ = None
        self.optimizer_ = None
        self.scaler_params_ = None
        self.is_fitted_ = False

        self._check_torch()

    def _check_torch(self) -> None:
        """Check if PyTorch is available."""
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
        except ImportError as e:
            raise ImportError("PyTorch is required. Install with: pip install torch") from e

    def _build_model(self, input_size: int) -> Any:
        """Build LSTM model architecture.

        Args:
            input_size: Number of input features

        Returns:
            PyTorch model
        """
        import torch.nn as nn

        class LSTMForecaster(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )

                self.dropout = nn.Dropout(dropout)

                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, output_size),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = lstm_out[:, -1, :]
                out = self.dropout(out)
                out = self.fc(out)
                return out

        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.output_size,
        )
        model = model.to(self.device)

        return model

    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sequence_length: int = 52,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for LSTM.

        Args:
            X: Feature array (n_samples, n_features) or (n_samples,)
            y: Target array (n_samples,)
            sequence_length: Length of input sequences

        Returns:
            Tuple of (X_seq, y_seq) where X_seq is (n_sequences, sequence_length, n_features)
        """
        X = np.asarray(X, dtype=np.float32)

        logger.info(f"_prepare_sequences input: X.shape={X.shape}, X.ndim={X.ndim}")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            logger.info(f"Reshaped 1D X to 2D: {X.shape}")

        if len(X) <= sequence_length:
            raise ValueError(
                f"Not enough data for sequence_length={sequence_length}. "
                f"Got {len(X)} samples, need at least {sequence_length + 1}."
            )

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length : i])

        X_seq = np.array(X_seq, dtype=np.float32)
        logger.info(f"X_seq shape after loop: {X_seq.shape}")

        if y is not None:
            y = np.asarray(y, dtype=np.float32)
            for i in range(sequence_length, len(y)):
                y_seq.append(y[i])
            y_seq = np.array(y_seq, dtype=np.float32)
            logger.info(f"y_seq shape: {y_seq.shape}")
            return X_seq, y_seq

        return X_seq, None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sequence_length: int = 52,
        verbose: bool = True,
    ) -> "LSTMModel":
        """Fit LSTM model.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (if None, uses last portion of training data)
            y_val: Validation target
            sequence_length: Length of input sequences
            verbose: Print training progress

        Returns:
            Fitted model
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        logger.info(f"LSTM received X shape: {X.shape}, y shape: {y.shape}")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            logger.info(f"Reshaped X to 2D: {X.shape}")

        X_seq, y_seq = self._prepare_sequences(X, y, sequence_length)
        logger.info(f"Prepared sequences X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(
                np.asarray(X_val, dtype=np.float32),
                np.asarray(y_val, dtype=np.float32).ravel(),
                sequence_length,
            )
        else:
            val_size = max(1, int(len(X_seq) * 0.1))
            X_seq_train, X_val_seq = X_seq[:-val_size], X_seq[-val_size:]
            y_seq_train, y_val_seq = y_seq[:-val_size], y_seq[-val_size:]
            X_seq, y_seq = X_seq_train, y_seq_train
            logger.info(f"Split sequences: train={X_seq.shape}, val={X_val_seq.shape}")

        input_size = X_seq.shape[2]
        self.input_size_ = input_size
        self.model_ = self._build_model(input_size)

        X_tensor = self._torch.tensor(X_seq, dtype=self._torch.float32)
        y_tensor = self._torch.tensor(y_seq, dtype=self._torch.float32).unsqueeze(1)

        if X_val_seq is not None:
            X_val_tensor = self._torch.tensor(X_val_seq, dtype=self._torch.float32)
            y_val_tensor = self._torch.tensor(y_val_seq, dtype=self._torch.float32).unsqueeze(1)

        criterion = self._nn.MSELoss()
        optimizer = self._torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training LSTM model on {self.device}...")

        train_loader = self._create_dataloader(X_tensor, y_tensor)

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            if X_val_seq is not None:
                self.model_.eval()
                with self._torch.no_grad():
                    X_val_tensor = X_val_tensor.to(self.device)
                    y_val_tensor = y_val_tensor.to(self.device)
                    val_outputs = self.model_(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model_.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    self.model_.load_state_dict(best_model_state)
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}")

        self.sequence_length_ = sequence_length
        self.is_fitted_ = True
        logger.info("LSTM training completed")
        return self

    def _create_dataloader(self, X, y):
        """Create PyTorch DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict point estimates.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        self._check_fitted()

        X_seq, _ = self._prepare_sequences(X, sequence_length=self.sequence_length_)
        X_tensor = self._torch.tensor(X_seq, dtype=self._torch.float32).to(self.device)

        self.model_.eval()
        with self._torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy()

        padding = np.full(self.sequence_length_, np.nan)
        return np.concatenate([padding, predictions.flatten()])

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Predict with MC Dropout uncertainty estimation.

        Args:
            X: Input features
            n_samples: Number of MC samples (default: self.mc_samples)

        Returns:
            Dictionary with mean, std, and quantile predictions
        """
        self._check_fitted()
        n_samples = n_samples or self.mc_samples

        X_seq, _ = self._prepare_sequences(X, sequence_length=self.sequence_length_)
        X_tensor = self._torch.tensor(X_seq, dtype=self._torch.float32).to(self.device)

        self.model_.train()

        predictions = []
        with self._torch.no_grad():
            for _ in range(n_samples):
                pred = self.model_(X_tensor).cpu().numpy()
                predictions.append(pred)

        predictions = np.array(predictions)

        mean_pred = predictions.mean(axis=0).flatten()
        std_pred = predictions.std(axis=0).flatten()

        quantile_pred = {}
        for q in self.quantiles:
            q_val = np.quantile(predictions, q, axis=0).flatten()
            quantile_pred[f"q{q}"] = q_val

        padding_len = self.sequence_length_
        mean_pred = np.concatenate([np.full(padding_len, np.nan), mean_pred])
        std_pred = np.concatenate([np.full(padding_len, np.nan), std_pred])

        for key in quantile_pred:
            quantile_pred[key] = np.concatenate([np.full(padding_len, np.nan), quantile_pred[key]])

        return {"mean": mean_pred, "std": std_pred, **quantile_pred}

    def predict_quantiles(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict quantiles for prediction intervals.

        Args:
            X: Input features
            n_samples: Number of MC samples

        Returns:
            DataFrame with quantile predictions
        """
        results = self.predict_with_uncertainty(X, n_samples)

        quantile_map = {
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

        df = pd.DataFrame()
        for q, col_name in quantile_map.items():
            df[col_name] = results[f"q{q}"]

        return df

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._torch.save(self.model_.state_dict(), path / "model.pt")

        config = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "quantiles": self.quantiles,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length_,
            "input_size": self.input_size_,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "LSTMModel":
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
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.output_size = config["output_size"]
        self.quantiles = config["quantiles"]
        self.learning_rate = config["learning_rate"]
        self.sequence_length_ = config["sequence_length"]
        self.input_size_ = config.get("input_size", 1)

        self.model_ = self._build_model(input_size=self.input_size_)
        self.model_.load_state_dict(self._torch.load(path / "model.pt", map_location=self.device))

        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")


class LSTMForecaster:
    """High-level forecaster using LSTM models.

    Args:
        target_col: Target column name
        feature_cols: Feature column names
        sequence_length: Length of input sequences
        **model_kwargs: Arguments passed to LSTMModel
    """

    def __init__(
        self,
        target_col: str = "casos",
        feature_cols: Optional[List[str]] = None,
        sequence_length: int = 52,
        **model_kwargs,
    ):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.model = LSTMModel(**model_kwargs)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        validation_size: float = 0.1,
        verbose: bool = True,
    ) -> "LSTMForecaster":
        """Fit model on DataFrame.

        Args:
            df: DataFrame with features and target
            validation_size: Fraction for validation (applied after sequence creation)
            verbose: Training verbosity

        Returns:
            Fitted forecaster
        """
        if self.feature_cols is None:
            self.feature_cols = self._infer_features(df)

        if not self.feature_cols:
            raise ValueError(
                "No feature columns found. Check that DataFrame contains numeric columns."
            )

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.target_col].values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"LSTM input shape: {X.shape}, target shape: {y.shape}")

        self.model.fit(
            X,
            y,
            X_val=None,
            y_val=None,
            sequence_length=self.sequence_length,
            verbose=verbose,
        )
        self.is_fitted = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
        n_mc_samples: int = 100,
    ) -> pd.DataFrame:
        """Generate forecasts with prediction intervals.

        Args:
            df: DataFrame with features
            n_mc_samples: Number of MC Dropout samples

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        X = df[self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = self.model.predict_quantiles(X, n_samples=n_mc_samples)

        predictions = predictions.reset_index(drop=True)
        valid_mask = ~predictions["median"].isna()
        valid_indices = predictions.index[valid_mask].tolist()

        if "date" in df.columns and valid_indices:
            dates = df["date"].values
            date_vals = [
                dates[i + self.sequence_length]
                for i in valid_indices
                if i + self.sequence_length < len(dates)
            ]
            predictions.loc[valid_indices[: len(date_vals)], "date"] = date_vals
        if "uf" in df.columns and valid_indices:
            ufs = df["uf"].values
            uf_vals = [
                ufs[i + self.sequence_length]
                for i in valid_indices
                if i + self.sequence_length < len(ufs)
            ]
            predictions.loc[valid_indices[: len(uf_vals)], "uf"] = uf_vals

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
        features = [
            c
            for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        if not features:
            logger.warning(f"No numeric features found. Columns: {df.columns.tolist()}")
            logger.warning(f"Column dtypes: {df.dtypes.to_dict()}")
        return features

    def save(self, path: Union[str, Path]) -> None:
        """Save forecaster."""
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        with open(path / "forecaster_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "target_col": self.target_col,
                    "feature_cols": self.feature_cols,
                    "sequence_length": self.sequence_length,
                },
                f,
            )

    def load(self, path: Union[str, Path]) -> "LSTMForecaster":
        """Load forecaster."""
        import pickle

        path = Path(path)

        self.model.load(path)
        with open(path / "forecaster_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.target_col = config["target_col"]
        self.feature_cols = config["feature_cols"]
        self.sequence_length = config["sequence_length"]
        self.is_fitted = True

        return self
