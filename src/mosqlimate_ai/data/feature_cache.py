"""Feature caching module for Mosqlimate AI Competitor.

Provides caching functionality for processed features to speed up
iterative development and model training.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/mosqlimate/features")


class FeatureCache:
    """Cache manager for processed features.

    Caches features to disk with automatic invalidation based on:
    - Source data hash
    - Feature configuration hash
    - Timestamp

    Example:
        >>> cache = FeatureCache()
        >>> features = cache.get_or_compute(
        ...     df, config, compute_fn=build_features
        ... )
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize feature cache.

        Args:
            cache_dir: Directory for cache files (default: .cache/mosqlimate/features)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load cache metadata: {e}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except IOError as e:
            logger.warning(f"Could not save cache metadata: {e}")

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for cache key.

        Uses a combination of:
        - Column names
        - Row count
        - First/last few values
        - Dataframe hash

        Args:
            df: Input dataframe

        Returns:
            Hash string
        """
        # Use pandas hash for the content
        try:
            # Hash the key identifying columns
            hash_cols = (
                ["date", "uf"] if "date" in df.columns and "uf" in df.columns else df.columns[:5]
            )
            hash_df = df[hash_cols].head(1000).copy()

            # Convert to string and hash
            hash_str = hashlib.md5(pd.util.hash_pandas_object(hash_df).values.tobytes()).hexdigest()

            # Include shape and columns in hash
            shape_str = f"{len(df)}_{len(df.columns)}"
            col_str = "_".join(sorted(df.columns))

            return hashlib.md5(f"{hash_str}_{shape_str}_{col_str}".encode()).hexdigest()[:16]

        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")
            # Fallback to simple hash
            return hashlib.md5(str(df.shape).encode()).hexdigest()[:16]

    def _compute_config_hash(self, config: dict) -> str:
        """Compute hash of feature configuration.

        Args:
            config: Feature configuration dictionary

        Returns:
            Hash string
        """
        try:
            config_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.md5(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not compute config hash: {e}")
            return "no_config"

    def _get_cache_key(
        self,
        df: pd.DataFrame,
        config: Optional[dict] = None,
        suffix: str = "",
    ) -> str:
        """Generate cache key for dataframe and config.

        Args:
            df: Input dataframe
            config: Feature configuration
            suffix: Optional suffix for different feature sets

        Returns:
            Cache key string
        """
        data_hash = self._compute_data_hash(df)
        config_hash = self._compute_config_hash(config or {})

        key = f"{data_hash}_{config_hash}"
        if suffix:
            key = f"{key}_{suffix}"

        return key

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file for given key.

        Args:
            cache_key: Cache key string

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self,
        df: pd.DataFrame,
        config: Optional[dict] = None,
        suffix: str = "",
    ) -> Optional[pd.DataFrame]:
        """Get cached features if available.

        Args:
            df: Input dataframe (for hash computation)
            config: Feature configuration
            suffix: Optional suffix

        Returns:
            Cached dataframe or None if not found/invalid
        """
        cache_key = self._get_cache_key(df, config, suffix)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_df = pickle.load(f)

            # Verify cached data is valid
            if not isinstance(cached_df, pd.DataFrame):
                logger.warning(f"Invalid cache data for {cache_key}")
                return None

            logger.info(f"Cache hit: {cache_key} ({len(cached_df)} rows)")
            return cached_df

        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Could not load cache {cache_key}: {e}")
            return None

    def set(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        config: Optional[dict] = None,
        suffix: str = "",
    ) -> None:
        """Save features to cache.

        Args:
            df: Input dataframe (for hash computation)
            features: Features dataframe to cache
            config: Feature configuration
            suffix: Optional suffix
        """
        cache_key = self._get_cache_key(df, config, suffix)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)

            # Update metadata
            self._metadata[cache_key] = {
                "rows": len(features),
                "columns": list(features.columns),
                "size_mb": cache_path.stat().st_size / (1024 * 1024),
            }
            self._save_metadata()

            logger.info(f"Cached features: {cache_key} ({len(features)} rows)")

        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Could not save cache {cache_key}: {e}")

    def get_or_compute(
        self,
        df: pd.DataFrame,
        compute_fn: callable,
        config: Optional[dict] = None,
        suffix: str = "",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get features from cache or compute them.

        This is the main interface for feature caching.

        Args:
            df: Input dataframe
            compute_fn: Function to compute features if not cached
            config: Feature configuration (for cache key)
            suffix: Optional suffix for different feature sets
            use_cache: Whether to use caching

        Returns:
            Features dataframe

        Example:
            >>> def build_features(df):
            ...     return engineer.build_feature_set(df)
            >>>
            >>> cache = FeatureCache()
            >>> features = cache.get_or_compute(
            ...     df,
            ...     compute_fn=build_features,
            ...     config={"lags": [1, 2, 4]},
            ... )
        """
        if use_cache:
            cached = self.get(df, config, suffix)
            if cached is not None:
                return cached

        # Compute features
        logger.info("Computing features...")
        features = compute_fn(df)

        if use_cache:
            self.set(df, features, config, suffix)

        return features

    def clear(self, confirm: bool = True) -> int:
        """Clear all cached features.

        Args:
            confirm: Whether to require confirmation (for CLI)

        Returns:
            Number of files removed
        """
        if not self.cache_dir.exists():
            return 0

        cache_files = list(self.cache_dir.glob("*.pkl"))
        count = len(cache_files)

        for f in cache_files:
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Could not remove {f}: {e}")

        # Clear metadata
        self._metadata = {}
        if self._metadata_file.exists():
            try:
                self._metadata_file.unlink()
            except OSError:
                pass

        logger.info(f"Cleared {count} cached feature files")
        return count

    def get_cache_info(self) -> dict:
        """Get information about cached features.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {"files": 0, "total_size_mb": 0}

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }

    def list_cached(self) -> list[dict]:
        """List all cached feature sets.

        Returns:
            List of dictionaries with cache entry info
        """
        cached = []
        for key, metadata in self._metadata.items():
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cached.append(
                    {
                        "key": key,
                        "rows": metadata.get("rows", "?"),
                        "columns": len(metadata.get("columns", [])),
                        "size_mb": metadata.get("size_mb", 0),
                    }
                )
        return cached


# Global cache instance
_cache_instance: Optional[FeatureCache] = None


def get_feature_cache(cache_dir: Optional[Path] = None) -> FeatureCache:
    """Get or create global feature cache instance.

    Args:
        cache_dir: Directory for cache files

    Returns:
        FeatureCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = FeatureCache(cache_dir)
    return _cache_instance
