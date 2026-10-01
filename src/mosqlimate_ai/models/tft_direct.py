"""Temporal Fusion Transformer on the unified forecaster interface.

Role in the model zoo: a neural multi-horizon model with a *native*
nine-quantile output (no fabricated interval scaling). v1 uses the
target series plus calendar known-reals; the GBM models carry the
exogenous information. Trained on ``log1p(cases)``.

Requires ``pytorch_forecasting`` (pytorch-lightning). Uses the GPU when
available.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS, QUANTILE_LEVELS
from mosqlimate_ai.models.base import BaseForecaster, make_forecast_dates

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class TFTDirectForecaster(BaseForecaster):
    """Direct 67-week TFT with native multi-quantile loss.

    Args:
        max_horizon: Direct prediction length (default 67 = gap + season).
        encoder_length: Weeks of history fed to the encoder (default 104).
        quantiles: Quantile levels (default: nine IMDC levels).
        hidden_size / hidden_continuous_size / attention_head_size /
        dropout / learning_rate: TFT capacity knobs.
        epochs / batch_size / num_workers: Training loop knobs.
        use_gpu: Auto-detect CUDA when None.
    """

    def __init__(
        self,
        max_horizon: int = 80,
        encoder_length: int = 104,
        quantiles: list[float] | None = None,
        hidden_size: int = 32,
        hidden_continuous_size: int = 16,
        attention_head_size: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 0.03,
        epochs: int = 40,
        batch_size: int = 64,
        num_workers: int = 0,
        use_gpu: bool | None = None,
    ):
        super().__init__(quantiles=quantiles or list(QUANTILE_LEVELS))
        self.max_horizon = max_horizon
        self.encoder_length = encoder_length
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = _cuda_available() if use_gpu is None else use_gpu

    # ------------------------------------------------------------------
    def _build_frame(self, df: pd.DataFrame, include_future_to: pd.Timestamp | None = None) -> pd.DataFrame:
        """Weekly frame with time index, group id, log target and calendar reals.

        When ``include_future_to`` is given, rows are extended to that date
        with NaN targets and computed calendar covariates (predict mode).
        """
        df = df.sort_values("date").reset_index(drop=True)
        start = df["date"].min()
        out = pd.DataFrame({"date": pd.to_datetime(df["date"])})
        out["time_idx"] = ((out["date"] - start).dt.days // 7).astype(int)
        out["group_id"] = "0"
        out["target"] = np.log1p(np.maximum(df[self.target_col].astype(float).values, 0.0))

        dates = out["date"]
        out["doy_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
        out["doy_sin2"] = np.sin(4 * np.pi * dates.dt.dayofyear / 365.25)
        out["doy_cos2"] = np.cos(4 * np.pi * dates.dt.dayofyear / 365.25)

        if include_future_to is not None:
            last_date = out["date"].max()
            future_dates = pd.date_range(
                last_date + pd.Timedelta(days=7), pd.Timestamp(include_future_to), freq="7D"
            )
            if len(future_dates):
                fut = pd.DataFrame({"date": future_dates})
                fut["time_idx"] = ((fut["date"] - start).dt.days // 7).astype(int)
                fut["group_id"] = "0"
                fut["target"] = np.nan
                fut["doy_sin"] = np.sin(2 * np.pi * fut["date"].dt.dayofyear / 365.25)
                fut["doy_cos"] = np.cos(2 * np.pi * fut["date"].dt.dayofyear / 365.25)
                fut["doy_sin2"] = np.sin(4 * np.pi * fut["date"].dt.dayofyear / 365.25)
                fut["doy_cos2"] = np.cos(4 * np.pi * fut["date"].dt.dayofyear / 365.25)
                out = pd.concat([out, fut], ignore_index=True)
        return out

    def _fit(self, df: pd.DataFrame) -> None:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss

        pl.seed_everything(42)
        frame = self._build_frame(df)

        training = TimeSeriesDataSet(
            frame,
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=self.encoder_length // 2,
            max_encoder_length=self.encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_horizon,
            static_categoricals=["group_id"],
            time_varying_known_reals=["time_idx", "doy_sin", "doy_cos", "doy_sin2", "doy_cos2"],
            time_varying_unknown_reals=["target"],
            target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        frame.iloc[-self.max_horizon :]
        validation = TimeSeriesDataSet.from_dataset(training, frame, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=self.num_workers
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=self.num_workers
        )

        self.loss_quantiles_ = [float(q) for q in self.quantiles]
        self.model_ = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            hidden_continuous_size=self.hidden_continuous_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            loss=QuantileLoss(quantiles=self.loss_quantiles_),
            optimizer="adam",
            reduce_on_plateau_patience=4,
        )

        accelerator = "gpu" if self.use_gpu else "cpu"
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-4)
            ],
        )
        trainer.fit(self.model_, train_dataloader, val_dataloader)
        self.training_dataset_ = training
        self.frame_start_ = frame["date"].min()

    def _predict(self, horizon: int) -> pd.DataFrame:
        from pytorch_forecasting import TimeSeriesDataSet

        last_train = pd.Timestamp(self.last_train_date_)
        future_to = last_train + pd.Timedelta(days=7 * horizon)

        df_hist = pd.DataFrame(
            {
                "date": self._history_.index,
                self.target_col: self._history_.values,
            }
        )
        frame = self._build_frame(df_hist, include_future_to=future_to)
        # predict-mode datasets require finite targets; decoder ground truth
        # rows are ignored for prediction, so forward-fill is safe.
        frame["target"] = frame["target"].ffill()

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset_, frame, predict=True, stop_randomization=True
        )
        pred_dataloader = pred_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=self.num_workers
        )

        import torch

        device = next(self.model_.parameters()).device
        x, _ = next(iter(pred_dataloader))
        x = {k: v.to(device) if hasattr(v, "to") else v for k, v in x.items()}
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x)
        if hasattr(out, "prediction"):
            arr = out.prediction
        elif isinstance(out, dict):
            arr = out["prediction"]
        else:
            arr = out
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        # shape: (n_series, decoder_steps, n_quantiles)
        if arr.ndim == 3:
            arr = arr[0]

        n_q = len(self.loss_quantiles_)
        pred_log = arr[:, :n_q]
        counts = np.expm1(pred_log)

        dates = make_forecast_dates(self.last_train_date_, min(horizon, counts.shape[0]))
        qf = pd.DataFrame(
            counts[: len(dates), :], index=dates, columns=[QUANTILE_COLS[q] for q in self.quantiles]
        )
        return qf.clip(lower=0.0)

    def fit(self, df: pd.DataFrame, target_col: str = "casos", **kwargs):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        self._history_ = (
            pd.Series(df[target_col].astype(float).values, index=df["date"]).sort_index()
        )
        return super().fit(df, target_col=target_col)
