"""Data preprocessing module."""

from typing import Any, Dict, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess dengue and climate data."""
    
    def __init__(self):
        logger.info("DataPreprocessor initialized")
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method="ffill").fillna(method="bfill")
        
        # Remove outliers (simple IQR method)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        
        return df
    
    def normalize(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """Normalize numeric columns."""
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
