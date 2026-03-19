"""Feature engineering for dengue forecasting.

Creates temporal, lag, rolling, climate, and spatial features
for machine learning models.
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for ML forecasting models.

    This class generates various types of features:
    - Lag features (previous weeks' cases)
    - Rolling statistics (moving averages, std)
    - Temporal features (week, month, seasonality)
    - Climate interaction features
    - Spatial lag features (neighboring states)
    - Ocean oscillation features (ENSO, IOD, PDO)

    Example:
        >>> fe = FeatureEngineer()
        >>> df_features = fe.build_feature_set(df, target_col="casos")
    """

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        include_spatial: bool = True,
        include_ocean: bool = True,
    ):
        """Initialize feature engineer.

        Args:
            lag_periods: List of lag periods in weeks
            rolling_windows: List of rolling window sizes in weeks
            include_spatial: Whether to compute spatial lag features
            include_ocean: Whether to include ocean oscillation features
        """
        self.lag_periods = lag_periods or [1, 2, 3, 4, 8, 12, 16, 20, 24, 52]
        self.rolling_windows = rolling_windows or [2, 4, 8, 12, 24]
        self.include_spatial = include_spatial
        self.include_ocean = include_ocean

        logger.info(
            f"FeatureEngineer initialized with {len(self.lag_periods)} lags, "
            f"{len(self.rolling_windows)} rolling windows"
        )

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        lags: Optional[List[int]] = None,
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create lagged features.

        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lags: List of lag periods
            group_cols: Columns to group by before shifting

        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        lags = lags or self.lag_periods
        group_cols = group_cols or ["uf"] if "uf" in df.columns else None

        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            if group_cols:
                df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
            else:
                df[col_name] = df[target_col].shift(lag)

        logger.info(f"Created {len(lags)} lag features for {target_col}")
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None,
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create rolling window statistics.

        Args:
            df: Input DataFrame
            target_col: Column to compute statistics for
            windows: List of window sizes
            stats: Statistics to compute ('mean', 'std', 'min', 'max', 'median')
            group_cols: Columns to group by

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        windows = windows or self.rolling_windows
        stats = stats or ["mean", "std", "min", "max"]
        group_cols = group_cols or ["uf"] if "uf" in df.columns else None

        for window in windows:
            for stat in stats:
                col_name = f"{target_col}_roll_{stat}_{window}"

                if group_cols:
                    grouped = df.groupby(group_cols)[target_col]
                    if stat == "mean":
                        df[col_name] = grouped.transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                        )
                    elif stat == "std":
                        df[col_name] = grouped.transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).std()
                        )
                    elif stat == "min":
                        df[col_name] = grouped.transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).min()
                        )
                    elif stat == "max":
                        df[col_name] = grouped.transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).max()
                        )
                    elif stat == "median":
                        df[col_name] = grouped.transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).median()
                        )

        n_features = len(windows) * len(stats)
        logger.info(f"Created {n_features} rolling features for {target_col}")
        return df

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Create time-based features.

        Args:
            df: Input DataFrame
            date_col: Date column name

        Returns:
            DataFrame with time features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["quarter"] = df[date_col].dt.quarter

        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["season"] = df["month"].map(self._get_season)
        df["is_dengue_season"] = df["month"].isin([1, 2, 3, 4, 5]).astype(int)

        logger.info("Created time features")
        return df

    def _get_season(self, month: int) -> str:
        """Get season from month (Southern Hemisphere)."""
        if month in [12, 1, 2]:
            return "summer"
        elif month in [3, 4, 5]:
            return "autumn"
        elif month in [6, 7, 8]:
            return "winter"
        else:
            return "spring"

    def create_climate_features(
        self,
        df: pd.DataFrame,
        climate_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create climate-derived features.

        Args:
            df: Input DataFrame with climate variables
            climate_cols: List of climate column names

        Returns:
            DataFrame with climate features
        """
        df = df.copy()
        climate_cols = climate_cols or [
            "temp_med",
            "precip_tot",
            "rel_humid_med",
            "temp_min",
            "temp_max",
        ]

        if "temp_med" in df.columns and "rel_humid_med" in df.columns:
            df["heat_index"] = self._calculate_heat_index(df["temp_med"], df["rel_humid_med"])

        if "temp_max" in df.columns and "temp_min" in df.columns:
            df["temp_range"] = df["temp_max"] - df["temp_min"]

        if "temp_med" in df.columns and "precip_tot" in df.columns:
            df["temp_precip_interaction"] = df["temp_med"] * df["precip_tot"]

        if "rel_humid_med" in df.columns and "precip_tot" in df.columns:
            df["humidity_precip_interaction"] = df["rel_humid_med"] * df["precip_tot"]

        if "temp_med" in df.columns:
            df["temp_squared"] = df["temp_med"] ** 2

        logger.info("Created climate interaction features")
        return df

    def _calculate_heat_index(
        self,
        temp: pd.Series,
        humidity: pd.Series,
    ) -> pd.Series:
        """Calculate heat index (feels-like temperature).

        Args:
            temp: Temperature in Celsius
            humidity: Relative humidity in %

        Returns:
            Heat index values
        """
        T = temp * 9 / 5 + 32
        R = humidity

        HI = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * R
            - 0.22475541 * T * R
            - 0.00683783 * T**2
            - 0.05481717 * R**2
            + 0.00122874 * T**2 * R
            + 0.00085282 * T * R**2
            - 0.00000199 * T**2 * R**2
        )

        return (HI - 32) * 5 / 9

    def create_spatial_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        neighbor_map: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """Create spatial lag features from neighboring states.

        Args:
            df: Input DataFrame with 'uf' column
            target_col: Column to create spatial lags for
            neighbor_map: Dictionary mapping states to their neighbors

        Returns:
            DataFrame with spatial lag features
        """
        if not self.include_spatial:
            return df

        if "uf" not in df.columns:
            warnings.warn("No 'uf' column found, skipping spatial features")
            return df

        df = df.copy()

        neighbor_map = neighbor_map or self._get_default_neighbor_map()

        df_pivot = df.pivot(index="date", columns="uf", values=target_col)

        for uf in df["uf"].unique():
            neighbors = neighbor_map.get(uf, [])
            if not neighbors:
                continue

            valid_neighbors = [n for n in neighbors if n in df_pivot.columns]
            if not valid_neighbors:
                continue

            neighbor_mean = df_pivot[valid_neighbors].mean(axis=1)
            df.loc[df["uf"] == uf, f"{target_col}_spatial_lag"] = neighbor_mean.values

        logger.info("Created spatial lag features")
        return df

    def _get_default_neighbor_map(self) -> Dict[str, List[str]]:
        """Get default neighboring states map for Brazil."""
        return {
            "AC": ["AM", "RO"],
            "AL": ["PE", "SE", "BA"],
            "AP": ["PA"],
            "AM": ["AC", "RO", "MT", "PA", "RR"],
            "BA": ["SE", "AL", "PE", "PI", "TO", "GO", "MG", "ES"],
            "CE": ["PI", "PE", "PB", "RN"],
            "DF": ["GO", "MG"],
            "ES": ["BA", "MG", "RJ"],
            "GO": ["DF", "TO", "BA", "MG", "MS", "MT"],
            "MA": ["PA", "TO", "PI"],
            "MT": ["RO", "AM", "PA", "TO", "GO", "MS"],
            "MS": ["MT", "GO", "MG", "SP", "PR"],
            "MG": ["BA", "ES", "RJ", "SP", "GO", "DF", "MS"],
            "PA": ["AP", "AM", "RR", "MA", "TO", "MT"],
            "PB": ["RN", "CE", "PE"],
            "PR": ["SP", "MS", "SC"],
            "PE": ["AL", "SE", "BA", "PI", "CE", "PB"],
            "PI": ["MA", "TO", "BA", "PE", "CE"],
            "RJ": ["ES", "MG", "SP"],
            "RN": ["PB", "CE"],
            "RS": ["SC"],
            "RO": ["AC", "AM", "MT"],
            "RR": ["AM", "PA"],
            "SC": ["PR", "RS"],
            "SP": ["RJ", "MG", "MS", "PR"],
            "SE": ["AL", "BA"],
            "TO": ["MA", "PA", "MT", "GO", "BA", "PI"],
        }

    def create_ocean_features(
        self,
        df: pd.DataFrame,
        ocean_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge ocean oscillation features (ENSO, IOD, PDO).

        Args:
            df: Input DataFrame with 'date' column
            ocean_df: DataFrame with ocean indices

        Returns:
            DataFrame with ocean features
        """
        if not self.include_ocean:
            return df

        df = df.copy()

        if ocean_df is None or ocean_df.empty:
            warnings.warn("Ocean data not available, skipping ocean features")
            return df

        ocean_df = ocean_df.copy()
        ocean_df["date"] = pd.to_datetime(ocean_df["date"])

        for lag in [0, 4, 8, 12]:
            ocean_shifted = ocean_df.copy()
            if lag > 0:
                ocean_shifted["date"] = ocean_shifted["date"] + pd.Timedelta(weeks=lag)

            suffix = f"_lag{lag}" if lag > 0 else ""

            if "enso" in ocean_shifted.columns:
                ocean_shifted[f"enso{suffix}"] = ocean_shifted["enso"]
            if "iod" in ocean_shifted.columns:
                ocean_shifted[f"iod{suffix}"] = ocean_shifted["iod"]
            if "pdo" in ocean_shifted.columns:
                ocean_shifted[f"pdo{suffix}"] = ocean_shifted["pdo"]

        df = df.merge(ocean_df[["date", "enso", "iod", "pdo"]], on="date", how="left")

        for col in ["enso", "iod", "pdo"]:
            if col in df.columns:
                df[f"{col}_lag4"] = df[col].shift(4)
                df[f"{col}_lag8"] = df[col].shift(8)

        logger.info("Created ocean oscillation features")
        return df

    def create_diff_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        periods: Optional[List[int]] = None,
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create differenced features.

        Args:
            df: Input DataFrame
            target_col: Column to difference
            periods: List of differencing periods
            group_cols: Columns to group by

        Returns:
            DataFrame with diff features
        """
        df = df.copy()
        periods = periods or [1, 4, 52]
        group_cols = group_cols or ["uf"] if "uf" in df.columns else None

        for period in periods:
            col_name = f"{target_col}_diff_{period}"
            if group_cols:
                df[col_name] = df.groupby(group_cols)[target_col].diff(period)
            else:
                df[col_name] = df[target_col].diff(period)

            df[f"{target_col}_pct_change_{period}"] = df[col_name] / df[target_col].shift(period)

        logger.info(f"Created differencing features for {target_col}")
        return df

    def create_target_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        horizon: int = 4,
    ) -> pd.DataFrame:
        """Create target variables for different horizons.

        Args:
            df: Input DataFrame
            target_col: Base target column
            horizon: Number of weeks ahead to predict

        Returns:
            DataFrame with target variables
        """
        df = df.copy()
        group_cols = ["uf"] if "uf" in df.columns else None

        for h in range(1, horizon + 1):
            col_name = f"{target_col}_horizon_{h}"
            if group_cols:
                df[col_name] = df.groupby(group_cols)[target_col].shift(-h)
            else:
                df[col_name] = df[target_col].shift(-h)

        logger.info(f"Created {horizon} horizon target features")
        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        exclude_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """Select feature columns for modeling.

        Args:
            df: Input DataFrame
            target_col: Target column to exclude
            exclude_cols: Additional columns to exclude

        Returns:
            List of feature column names
        """
        exclude_cols = exclude_cols or []
        exclude_cols.extend(
            [
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
                "regional_geocode",
                "macroregional_geocode",
                target_col,
                f"{target_col}_horizon_1",
            ]
        )

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, float, int]
        ]

        logger.info(f"Selected {len(feature_cols)} feature columns")
        return feature_cols

    def build_feature_set(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        ocean_df: Optional[pd.DataFrame] = None,
        neighbor_map: Optional[Dict[str, List[str]]] = None,
        horizon: int = 4,
    ) -> pd.DataFrame:
        """Create complete feature set for modeling.

        Args:
            df: Input DataFrame
            target_col: Target column
            ocean_df: Ocean oscillation data
            neighbor_map: State neighbor mapping
            horizon: Forecasting horizon in weeks

        Returns:
            DataFrame with all features
        """
        df = df.copy()

        df = self.create_time_features(df)

        df = self.create_lag_features(df, target_col)

        df = self.create_rolling_features(df, target_col)

        df = self.create_climate_features(df)

        df = self.create_diff_features(df, target_col)

        if self.include_spatial:
            df = self.create_spatial_lag_features(df, target_col, neighbor_map)

        if self.include_ocean and ocean_df is not None:
            df = self.create_ocean_features(df, ocean_df)

        df = self.create_target_features(df, target_col, horizon)

        df = df.dropna(subset=[target_col])

        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")

        logger.info(f"Built feature set: {df.shape[1]} columns, {len(df)} rows")
        return df
