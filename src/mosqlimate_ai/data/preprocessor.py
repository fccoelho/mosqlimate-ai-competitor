"""Data preprocessing for dengue time-series forecasting."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess dengue and climate time-series data.

    This class handles:
    - Data cleaning and validation
    - Missing value imputation
    - Outlier detection and handling
    - Time-series specific preprocessing
    - Epidemiological data quality checks

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> df_clean = preprocessor.clean(df)
        >>> df_filled = preprocessor.impute_missing(df_clean)
    """

    def __init__(
        self,
        max_gap_weeks: int = 4,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
    ):
        """Initialize preprocessor.

        Args:
            max_gap_weeks: Maximum gap for interpolation (in weeks)
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            outlier_threshold: Threshold for outlier detection
        """
        self.max_gap_weeks = max_gap_weeks
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        logger.info(f"DataPreprocessor initialized with outlier_method={outlier_method}")

    def clean(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_outliers: bool = True,
    ) -> pd.DataFrame:
        """Clean raw data.

        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: Whether to handle outliers

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        if remove_duplicates:
            subset = ["date", "uf"] if "uf" in df.columns else ["date"]
            df = df.drop_duplicates(subset=subset, keep="last")

        df = self._validate_dtypes(df)

        df = df.sort_values("date").reset_index(drop=True)

        if handle_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ["casos", "incidence_rate"]:
                    df = self._handle_outliers(df, col, treatment="cap")

        logger.info(f"Cleaned data: {len(df)} rows")
        return df

    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if "epiweek" in df.columns:
            df["epiweek"] = df["epiweek"].astype(int)

        if "casos" in df.columns:
            df["casos"] = df["casos"].clip(lower=0).fillna(0).astype(int)

        if "population" in df.columns:
            df["population"] = df["population"].fillna(method="ffill").fillna(method="bfill")

        return df

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        treatment: str = "cap",
    ) -> pd.DataFrame:
        """Handle outliers in a column.

        Args:
            df: Input DataFrame
            column: Column name
            treatment: Treatment method ('cap', 'remove', 'nan')

        Returns:
            DataFrame with outliers handled
        """
        if column not in df.columns:
            return df

        if self.outlier_method == "iqr":
            lower, upper = self._iqr_bounds(df[column])
        elif self.outlier_method == "zscore":
            lower, upper = self._zscore_bounds(df[column])
        else:
            return df

        outlier_mask = (df[column] < lower) | (df[column] > upper)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            logger.info(f"Found {n_outliers} outliers in {column}")

            if treatment == "cap":
                df[column] = df[column].clip(lower=lower, upper=upper)
            elif treatment == "nan":
                df.loc[outlier_mask, column] = np.nan
            elif treatment == "remove":
                df = df[~outlier_mask]

        return df

    def _iqr_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate IQR-based bounds for outlier detection."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - self.outlier_threshold * IQR
        upper = Q3 + self.outlier_threshold * IQR
        return lower, upper

    def _zscore_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate z-score based bounds for outlier detection."""
        mean = series.mean()
        std = series.std()
        lower = mean - self.outlier_threshold * std
        upper = mean + self.outlier_threshold * std
        return lower, upper

    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "seasonal",
        seasonal_period: int = 52,
    ) -> pd.DataFrame:
        """Impute missing values in time-series data.

        Args:
            df: Input DataFrame
            method: Imputation method ('seasonal', 'linear', 'spline', 'forward')
            seasonal_period: Period for seasonal imputation (weeks)

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["epiweek", "year"]]

        for col in numeric_cols:
            if df[col].isna().sum() == 0:
                continue

            if method == "seasonal":
                df = self._seasonal_impute(df, col, seasonal_period)
            elif method == "linear":
                df[col] = df[col].interpolate(method="linear")
            elif method == "spline":
                df[col] = df[col].interpolate(method="spline", order=3)
            elif method == "forward":
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

            remaining_na = df[col].isna().sum()
            if remaining_na > 0:
                df[col] = df[col].fillna(df[col].median())

        logger.info(f"Imputed missing values using {method} method")
        return df

    def _seasonal_impute(
        self,
        df: pd.DataFrame,
        column: str,
        period: int,
    ) -> pd.DataFrame:
        """Seasonal imputation using historical patterns."""
        if "epiweek" not in df.columns:
            df["epiweek"] = df["date"].dt.isocalendar().week

        seasonal_values = df.groupby("epiweek")[column].transform("median")
        df[column] = df[column].fillna(seasonal_values)

        remaining_na = df[column].isna()
        if remaining_na.any():
            df.loc[remaining_na, column] = df[column].median()

        return df

    def fill_date_gaps(
        self,
        df: pd.DataFrame,
        freq: str = "W-SUN",
    ) -> pd.DataFrame:
        """Fill missing dates in time-series.

        Args:
            df: Input DataFrame with 'date' column
            freq: Frequency string (default: weekly on Sunday)

        Returns:
            DataFrame with complete date range
        """
        if "date" not in df.columns:
            return df

        df = df.copy()
        df = df.sort_values("date")

        min_date = df["date"].min()
        max_date = df["date"].max()

        full_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        full_df = pd.DataFrame({"date": full_range})

        df = full_df.merge(df, on="date", how="left")

        group_cols = [c for c in ["uf", "geocode"] if c in df.columns]
        if group_cols:
            df = (
                df.groupby(group_cols, group_keys=False)
                .apply(lambda g: self._fill_group_dates(g, freq))
                .reset_index(drop=True)
            )

        logger.info(f"Filled date gaps: {len(df)} rows")
        return df

    def _fill_group_dates(self, group: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Fill dates for a single group."""
        if len(group) == 0:
            return group

        min_date = group["date"].min()
        max_date = group["date"].max()

        full_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        full_df = pd.DataFrame({"date": full_range})

        for col in group.columns:
            if col == "date":
                continue
            if col in ["uf", "geocode", "epiweek"]:
                if group[col].notna().any():
                    full_df[col] = group[col].dropna().iloc[0]

        full_df = full_df.merge(
            group.drop(columns=["uf", "geocode"], errors="ignore"), on="date", how="left"
        )

        return full_df

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = "target_3",
        train_col: str = "train_3",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets using competition flags.

        Args:
            df: Input DataFrame with train/target flags
            target_col: Target flag column name
            train_col: Train flag column name

        Returns:
            Tuple of (train_df, test_df)
        """
        if train_col in df.columns and target_col in df.columns:
            train_df = df[df[train_col] is True].copy()
            test_df = df[df[target_col] is True].copy()
        else:
            raise ValueError(f"Columns {train_col} and {target_col} not found in DataFrame")

        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
        return train_df, test_df

    def add_epidemiological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add epidemiological derived features.

        Args:
            df: Input DataFrame with 'casos' column

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        if "casos" in df.columns and "population" in df.columns:
            df["incidence_rate"] = df["casos"] / df["population"] * 100000

        if "incidence_rate" in df.columns:
            df["log_incidence"] = np.log1p(df["incidence_rate"])

        if "casos" in df.columns:
            df["log_cases"] = np.log1p(df["casos"])

        if "epiweek" in df.columns:
            df["week_of_year"] = df["epiweek"] % 100

        logger.info("Added epidemiological features")
        return df

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> pd.DataFrame:
        """Normalize numeric columns.

        Args:
            df: Input DataFrame
            columns: Columns to normalize (default: all numeric)
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c not in ["epiweek", "year", "casos"]]

        for col in columns:
            if col not in df.columns:
                continue

            if method == "standard":
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            elif method == "minmax":
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            elif method == "robust":
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                df[col] = (df[col] - median) / (iqr + 1e-8)

        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df

    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        target_col: str = "casos",
        impute: bool = True,
        add_epi_features: bool = True,
        normalize_features: bool = False,
    ) -> pd.DataFrame:
        """Full preprocessing pipeline for modeling.

        Args:
            df: Raw input DataFrame
            target_col: Target column name
            impute: Whether to impute missing values
            add_epi_features: Whether to add epidemiological features
            normalize_features: Whether to normalize features

        Returns:
            Preprocessed DataFrame ready for modeling
        """
        df = self.clean(df)

        if impute:
            df = self.impute_missing(df)

        if add_epi_features:
            df = self.add_epidemiological_features(df)

        if normalize_features:
            feature_cols = [
                c
                for c in df.columns
                if c
                not in [
                    "date",
                    "uf",
                    "geocode",
                    "epiweek",
                    "year",
                    target_col,
                    "train_1",
                    "train_2",
                    "train_3",
                    "target_1",
                    "target_2",
                    "target_3",
                ]
            ]
            df = self.normalize(df, columns=feature_cols)

        df = df.dropna(subset=[target_col])

        logger.info(f"Prepared data for modeling: {df.shape}")
        return df
