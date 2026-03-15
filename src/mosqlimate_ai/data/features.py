"""Feature engineering for dengue forecasting."""

from typing import List, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for ML models."""
    
    def __init__(self):
        self.lag_periods = [1, 2, 3, 4, 8, 12]  # weeks
        self.rolling_windows = [4, 8, 12]  # weeks
        logger.info("FeatureEngineer initialized")
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "cases",
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Create lagged features."""
        df = df.copy()
        lags = lags or self.lag_periods
        
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = "cases",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Create rolling window statistics."""
        df = df.copy()
        windows = windows or self.rolling_windows
        
        for window in windows:
            df[f"{target_col}_roll_mean_{window}"] = df[target_col].rolling(window=window).mean()
            df[f"{target_col}_roll_std_{window}"] = df[target_col].rolling(window=window).std()
        
        return df
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df["week_of_year"] = df[date_col].dt.isocalendar().week
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df["day_of_year"] = df[date_col].dt.dayofyear
        
        # Cyclical encoding for week/month
        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        
        return df
    
    def add_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add climate interaction features."""
        df = df.copy()
        
        if "temp_mean" in df.columns and "precip" in df.columns:
            # Temperature-precipitation interaction
            df["temp_precip"] = df["temp_mean"] * df["precip"]
            
            # Heat index approximation
            df["heat_index"] = df["temp_mean"] + 0.5555 * (df["humidity"] / 100 - 0.5) \
                if "humidity" in df.columns else df["temp_mean"]
        
        return df
    
    def build_feature_set(
        self,
        df: pd.DataFrame,
        target_col: str = "cases"
    ) -> pd.DataFrame:
        """Create complete feature set."""
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.add_climate_features(df)
        
        # Remove rows with NaN from lag features
        df = df.dropna()
        
        logger.info(f"Feature set created: {df.shape[1]} features")
        return df
