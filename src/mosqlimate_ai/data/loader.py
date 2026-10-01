"""Data loader for Mosqlimate competition data.

Loads and merges cached competition data from the local data/{challenge}/ directory.
Supports both 2nd IMDC (2025) and 3rd IMDC (2026) challenges.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from mosqlimate_ai.data.completeness import warn_missing_weeks

logger = logging.getLogger(__name__)

DEFAULT_CHALLENGE = "3rd_IMDC"


def _get_default_data_dir(challenge: str = DEFAULT_CHALLENGE) -> Path:
    """Get default data directory for the given challenge.

    Prefers ``data/{challenge}/`` when it exists; falls back to the flat
    ``data/`` directory where the FTP files are usually cached.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    challenge_dir = project_root / "data" / challenge
    if challenge_dir.exists():
        return challenge_dir
    return project_root / "data"


def _merge_with_update(base: pd.DataFrame, update: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Merge a base dataset with a newer partial update.

    Rows from ``update`` take precedence over base rows sharing the same key.
    """
    if update.empty:
        return base
    combined = pd.concat([base, update], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined.sort_values(key_cols).reset_index(drop=True)


def reindex_weekly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Reindex a state series onto a complete weekly grid.

    Missing weeks become explicit NaN rows (never zero-filled), so
    downstream lags stay date-aligned and gaps are visible to the user.
    """
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.drop_duplicates(subset=[date_col]).set_index(date_col).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="7D")
    if len(full) == len(df):
        return df.reset_index()
    df = df.reindex(full)
    df.index.name = date_col
    return df.reset_index()


BRAZILIAN_STATES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

STATE_CODES = {
    "AC": 12,
    "AL": 27,
    "AP": 16,
    "AM": 13,
    "BA": 29,
    "CE": 23,
    "DF": 53,
    "ES": 32,
    "GO": 52,
    "MA": 21,
    "MT": 51,
    "MS": 50,
    "MG": 31,
    "PA": 15,
    "PB": 25,
    "PR": 41,
    "PE": 26,
    "PI": 22,
    "RJ": 33,
    "RN": 24,
    "RS": 43,
    "RO": 11,
    "RR": 14,
    "SC": 42,
    "SP": 35,
    "SE": 28,
    "TO": 17,
}


class CompetitionDataLoader:
    """Load and merge competition data for dengue forecasting.

    This class handles loading all competition datasets from the local cache,
    merging them by geocode and date, and aggregating to state level.

    Attributes:
        data_dir: Path to data directory
        dengue_df: Dengue cases DataFrame
        climate_df: Climate data DataFrame
        climate_forecast_df: Climate forecast DataFrame
        population_df: Population data DataFrame
        environ_df: Environmental variables DataFrame
        ocean_df: Ocean climate oscillations DataFrame
        regional_map_df: Regional health mapping DataFrame

    Example:
        >>> loader = CompetitionDataLoader()
        >>> df = loader.load_state_data("SP")
        >>> df_aggregated = loader.aggregate_to_state(df)
    """

    def __init__(self, data_dir: Optional[Path] = None, challenge: str = DEFAULT_CHALLENGE):
        """Initialize data loader.

        Args:
            data_dir: Path to data directory. Defaults to project's data/{challenge}/ folder.
            challenge: Which challenge dataset to use ("2nd_IMDC" or "3rd_IMDC").
        """
        self.challenge = challenge
        self.data_dir = Path(data_dir) if data_dir else _get_default_data_dir(challenge)
        self._dengue_df: Optional[pd.DataFrame] = None
        self._chikungunya_df: Optional[pd.DataFrame] = None
        self._climate_df: Optional[pd.DataFrame] = None
        self._climate_forecast_df: Optional[pd.DataFrame] = None
        self._population_df: Optional[pd.DataFrame] = None
        self._environ_df: Optional[pd.DataFrame] = None
        self._ocean_df: Optional[pd.DataFrame] = None
        self._regional_map_df: Optional[pd.DataFrame] = None
        self._merged_cache: dict[str, pd.DataFrame] = {}

        logger.info(f"CompetitionDataLoader initialized with data_dir: {self.data_dir}")

    def _merged_cache_get(self, disease: str) -> pd.DataFrame:
        """Build (once) and return the full merged municipality table."""
        if disease not in self._merged_cache:
            base = self.chikungunya_df if disease == "chikungunya" else self.dengue_df
            if base.empty:
                raise FileNotFoundError(
                    f"{disease} data is not available in {self.data_dir}; "
                    "run 'mosqlimate-ai download-data' first."
                )
            merged = self._merge_auxiliary(base)
            self._merged_cache[disease] = merged
        return self._merged_cache[disease].copy()

    def _merge_auxiliary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge climate, population and environmental tables into a case table."""
        if not self.climate_df.empty:
            climate_cols = [
                "date",
                "epiweek",
                "geocode",
                "temp_min",
                "temp_med",
                "temp_max",
                "precip_min",
                "precip_med",
                "precip_max",
                "pressure_min",
                "pressure_med",
                "pressure_max",
                "rel_humid_min",
                "rel_humid_med",
                "rel_humid_max",
                "thermal_range",
                "rainy_days",
            ]
            climate_merge = self.climate_df[climate_cols].drop_duplicates(
                subset=["date", "geocode"]
            )
            df = df.merge(climate_merge, on=["date", "geocode", "epiweek"], how="left")

        if not self.population_df.empty:
            df["year"] = df["date"].dt.year
            pop_merge = self.population_df[["geocode", "year", "population"]].drop_duplicates(
                subset=["geocode", "year"]
            )
            df = df.merge(pop_merge, on=["geocode", "year"], how="left")

        if not self.environ_df.empty:
            environ_merge = self.environ_df[["geocode", "koppen", "biome"]].drop_duplicates(
                subset=["geocode"]
            )
            df = df.merge(environ_merge, on=["geocode"], how="left")

        logger.info(f"Merged data shape: {df.shape}")
        return df

    @property
    def dengue_df(self) -> pd.DataFrame:
        """Lazy load dengue data."""
        if self._dengue_df is None:
            self._dengue_df = self._load_dengue_data()
        return self._dengue_df

    @property
    def chikungunya_df(self) -> pd.DataFrame:
        """Lazy load chikungunya data."""
        if self._chikungunya_df is None:
            self._chikungunya_df = self._load_chikungunya_data()
        return self._chikungunya_df

    @property
    def climate_df(self) -> pd.DataFrame:
        """Lazy load climate data."""
        if self._climate_df is None:
            self._climate_df = self._load_climate_data()
        return self._climate_df

    @property
    def climate_forecast_df(self) -> pd.DataFrame:
        """Lazy load climate forecast data."""
        if self._climate_forecast_df is None:
            self._climate_forecast_df = self._load_climate_forecast_data()
        return self._climate_forecast_df

    @property
    def population_df(self) -> pd.DataFrame:
        """Lazy load population data."""
        if self._population_df is None:
            self._population_df = self._load_population_data()
        return self._population_df

    @property
    def environ_df(self) -> pd.DataFrame:
        """Lazy load environmental data."""
        if self._environ_df is None:
            self._environ_df = self._load_environmental_data()
        return self._environ_df

    @property
    def ocean_df(self) -> pd.DataFrame:
        """Lazy load ocean oscillation data."""
        if self._ocean_df is None:
            self._ocean_df = self._load_ocean_data()
        return self._ocean_df

    @property
    def regional_map_df(self) -> pd.DataFrame:
        """Lazy load regional health mapping."""
        if self._regional_map_df is None:
            self._regional_map_df = self._load_regional_map()
        return self._regional_map_df

    def _find_update_files(self, stem_patterns: list[str]) -> list[Path]:
        """All ``*update*`` files for the given stems, oldest vintage first."""
        updates: list[Path] = []
        for pat in stem_patterns:
            updates.extend(self.data_dir.glob(f"{pat}_updated_*.csv.gz"))
            updates.extend(self.data_dir.glob(f"{pat}_update_*.csv.gz"))
        return sorted(set(updates))

    def _load_epiweek_case_data(self, disease: str) -> pd.DataFrame:
        """Load case data for a disease, merging newer update files when present.

        Args:
            disease: "dengue" or "chikungunya"

        Returns:
            DataFrame with date, epiweek, geocode, casos columns
        """
        base_path = self.data_dir / f"{disease}.csv.gz"
        if not base_path.exists():
            if disease == "chikungunya":
                warnings.warn(f"Chikungunya data not found at {base_path}")
                return pd.DataFrame()
            raise FileNotFoundError(
                f"{disease} data not found at {base_path}. "
                "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading {disease} data from {base_path}")
        df = pd.read_csv(base_path, compression="gzip")

        for update_path in self._find_update_files([disease]):
            logger.info(f"Merging {disease} update file {update_path.name}")
            update_df = pd.read_csv(update_path, compression="gzip")
            df = _merge_with_update(df, update_df, key_cols=["date", "geocode"])

        df["date"] = pd.to_datetime(df["date"])
        df["epiweek"] = df["epiweek"].astype(int)
        df["geocode"] = df["geocode"].astype(int)
        df["casos"] = df["casos"].fillna(0).astype(int)
        if "uf_code" in df.columns:
            # only newer update vintages carry uf_code; keep it nullable
            df["uf_code"] = df["uf_code"].astype("Int64")

        logger.info(f"Loaded {len(df)} {disease} records")
        return df

    def _load_dengue_data(self) -> pd.DataFrame:
        """Load dengue cases data."""
        return self._load_epiweek_case_data("dengue")

    def _load_chikungunya_data(self) -> pd.DataFrame:
        """Load chikungunya cases data."""
        return self._load_epiweek_case_data("chikungunya")

    def _load_climate_data(self) -> pd.DataFrame:
        """Load climate reanalysis data (merging newer update files when present)."""
        filepath = self.data_dir / "climate.csv.gz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Climate data not found at {filepath}. " "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading climate data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        for update_path in self._find_update_files(["climate"]):
            logger.info(f"Merging climate update file {update_path.name}")
            update_df = pd.read_csv(update_path, compression="gzip")
            df = _merge_with_update(df, update_df, key_cols=["date", "geocode"])

        df["date"] = pd.to_datetime(df["date"])
        df["epiweek"] = df["epiweek"].astype(int)
        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} climate records")
        return df

    def _load_climate_forecast_data(self) -> pd.DataFrame:
        """Load monthly climate forecast data (ECMWF).

        The 3rd IMDC FTP server publishes this as ``forecasting_climate.csv.gz``
        (with a ``*_updated_2025`` extension); the 2nd IMDC name was
        ``climate_forecast.csv.gz``. Both are supported.
        """
        candidates = [
            self.data_dir / "forecasting_climate.csv.gz",
            self.data_dir / "climate_forecast.csv.gz",
        ]
        filepath = next((p for p in candidates if p.exists()), None)
        if filepath is None:
            warnings.warn(
                f"Climate forecast data not found at {self.data_dir} "
                "(expected forecasting_climate.csv.gz or climate_forecast.csv.gz)"
            )
            return pd.DataFrame()

        logger.info(f"Loading climate forecast data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        update_paths = self._find_update_files(
            ["forecasting_climate", "climate_forecast"]
        )
        for update_path in update_paths:
            logger.info(f"Merging climate forecast update file {update_path.name}")
            update_df = pd.read_csv(update_path, compression="gzip")
            # The 2025+ updates renamed umid_med -> rel_umid_med
            if "umid_med" not in update_df.columns and "rel_umid_med" in update_df.columns:
                update_df = update_df.rename(columns={"rel_umid_med": "umid_med"})
            key = ["geocode", "reference_month"]
            if "forecast_months_ahead" in df.columns and "forecast_months_ahead" in update_df.columns:
                # leads are distinct rows; keep all of them
                key = key + ["forecast_months_ahead"]
            df = _merge_with_update(df, update_df, key_cols=key)

        # Keep a single humidity column name regardless of source vintage
        if "umid_med" not in df.columns and "rel_umid_med" in df.columns:
            df = df.rename(columns={"rel_umid_med": "umid_med"})

        df["reference_month"] = pd.to_datetime(df["reference_month"])
        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} climate forecast records")
        return df

    def _load_population_data(self) -> pd.DataFrame:
        """Load population data."""
        filepath_2025 = self.data_dir / "datasus_population_2001_2025.csv.gz"
        filepath_2024 = self.data_dir / "datasus_population_2001_2024.csv.gz"

        filepath = filepath_2025 if filepath_2025.exists() else filepath_2024
        if not filepath.exists():
            raise FileNotFoundError(
                f"Population data not found at {filepath}. "
                "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading population data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["geocode"] = df["geocode"].astype(int)
        df["year"] = df["year"].astype(int)

        # Forward-fill population up to 2027 so forecast years (2025-2027)
        # inherit the most recent official estimate instead of NaN/0.
        latest_year = df["year"].max()
        if latest_year < 2027:
            geocodes = df["geocode"].unique()
            full = pd.MultiIndex.from_product(
                [geocodes, range(latest_year + 1, 2028)], names=["geocode", "year"]
            ).to_frame(index=False)
            df = pd.concat([df, full], ignore_index=True)
            df = df.sort_values(["geocode", "year"])
            df["population"] = df.groupby("geocode")["population"].ffill()
            df = df.dropna(subset=["population"])

        logger.info(f"Loaded {len(df)} population records")
        return df

    def load_climate_forecast_for_state(self, state_code: Optional[int] = None) -> pd.DataFrame:
        """Climate forecast rows for one state (or all when ``state_code`` is None).

        Result is cached per state code. When a state code is given, the
        CSV is read in filtered chunks so worker memory stays bounded
        (the full table is >6M rows / >1.5GB).
        """
        key = int(state_code) if state_code is not None else None
        cache = getattr(self, "_cf_state_cache", None)
        if cache is None:
            cache = self._cf_state_cache = {}
        if key in cache:
            return cache[key]

        if key is None:
            cache[key] = self.climate_forecast_df
            return cache[key]

        filepath = next(
            (
                p
                for p in [
                    self.data_dir / "forecasting_climate.csv.gz",
                    self.data_dir / "climate_forecast.csv.gz",
                ]
                if p.exists()
            ),
            None,
        )
        if filepath is None:
            cache[key] = pd.DataFrame()
            return cache[key]

        chunks = []
        for chunk in pd.read_csv(filepath, compression="gzip", chunksize=500_000):
            sel = chunk[chunk["geocode"] // 100000 == key]
            if len(sel):
                chunks.append(sel)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        for update_path in self._find_update_files(["forecasting_climate", "climate_forecast"]):
            update_df = pd.read_csv(update_path, compression="gzip")
            if "umid_med" not in update_df.columns and "rel_umid_med" in update_df.columns:
                update_df = update_df.rename(columns={"rel_umid_med": "umid_med"})
            update_df = update_df[update_df["geocode"] // 100000 == key]
            df = _merge_with_update(
                df,
                update_df,
                key_cols=["geocode", "reference_month", "forecast_months_ahead"],
            )

        if "umid_med" not in df.columns and "rel_umid_med" in df.columns:
            df = df.rename(columns={"rel_umid_med": "umid_med"})
        df["reference_month"] = pd.to_datetime(df["reference_month"])
        df["geocode"] = df["geocode"].astype(int)
        cache[key] = df
        return cache[key]

    def _load_environmental_data(self) -> pd.DataFrame:
        """Load environmental variables."""
        filepath = self.data_dir / "environ_vars.csv.gz"
        if not filepath.exists():
            warnings.warn(f"Environmental data not found at {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading environmental data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} environmental records")
        return df

    def _load_ocean_data(self) -> pd.DataFrame:
        """Load ocean climate oscillation data.

        Handles both 2nd IMDC (single file) and 3rd IMDC (separate files)
        formats. Update files (``*update*.csv.gz``) are revised full
        series and take precedence over the base combined file.
        """
        enso_file = self.data_dir / "enso.csv.gz"
        iod_file = self.data_dir / "iod.csv.gz"
        pdo_file = self.data_dir / "pdo.csv.gz"
        combined_file = self.data_dir / "ocean_climate_oscillations.csv.gz"

        update_files = self._find_update_files(["ocean_climate_oscillations"])
        if combined_file.exists() or update_files:
            # Update files are append segments: merge base + updates,
            # latest values win on duplicate dates.
            frames = []
            if combined_file.exists():
                logger.info(f"Loading ocean oscillation data from {combined_file.name}")
                frames.append(pd.read_csv(combined_file, compression="gzip"))
            for update_file in update_files:
                logger.info(f"Merging ocean oscillation update file {update_file.name}")
                frames.append(pd.read_csv(update_file, compression="gzip"))
            df = _merge_with_update(frames[0], pd.concat(frames[1:], ignore_index=True), key_cols=["date"])
            df["date"] = pd.to_datetime(df["date"])
            logger.info(f"Loaded {len(df)} ocean oscillation records")
            return df

        if not enso_file.exists() and not iod_file.exists() and not pdo_file.exists():
            warnings.warn(f"Ocean oscillation data not found at {self.data_dir}")
            return pd.DataFrame()

        dfs = []
        if enso_file.exists():
            logger.info(f"Loading ENSO data from {enso_file}")
            enso_df = pd.read_csv(enso_file, compression="gzip")
            enso_df = enso_df[["date", "enso"]].rename(columns={"enso": "enso"})
            dfs.append(enso_df)

        if iod_file.exists():
            logger.info(f"Loading IOD data from {iod_file}")
            iod_df = pd.read_csv(iod_file, compression="gzip")
            iod_df = iod_df[["date", "iod"]].rename(columns={"iod": "iod"})
            dfs.append(iod_df)

        if pdo_file.exists():
            logger.info(f"Loading PDO data from {pdo_file}")
            pdo_df = pd.read_csv(pdo_file, compression="gzip")
            pdo_df = pdo_df[["date", "pdo"]].rename(columns={"pdo": "pdo"})
            dfs.append(pdo_df)

        if not dfs:
            return pd.DataFrame()

        df = dfs[0]
        for other_df in dfs[1:]:
            df = df.merge(other_df, on="date", how="outer")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} ocean oscillation records")
        return df

    def _load_regional_map(self) -> pd.DataFrame:
        """Load regional health mapping."""
        filepath = self.data_dir / "map_regional_health.csv"
        if not filepath.exists():
            warnings.warn(f"Regional health mapping not found at {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading regional health mapping from {filepath}")
        df = pd.read_csv(filepath)

        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} regional mapping records")
        return df

    def get_state_from_geocode(self, geocode: int) -> str:
        """Extract state abbreviation from IBGE geocode.

        Args:
            geocode: 7-digit IBGE municipality code

        Returns:
            2-letter state abbreviation
        """
        state_code = int(str(geocode)[:2])
        for uf, code in STATE_CODES.items():
            if code == state_code:
                return uf
        raise ValueError(f"Unknown state code: {state_code}")

    def load_merged_data(
        self,
        uf: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_climate: bool = True,
        include_population: bool = True,
        include_environmental: bool = True,
        disease: str = "dengue",
    ) -> pd.DataFrame:
        """Load and merge all data sources.

        Args:
            uf: Filter by state abbreviation (e.g., "SP"). None for all states.
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            include_climate: Whether to merge climate data
            include_population: Whether to merge population data
            include_environmental: Whether to merge environmental data
            disease: Which disease cases to load ("dengue" or "chikungunya")

        Returns:
            Merged DataFrame with all data sources
        """
        if disease == "chikungunya":
            df = self._merged_cache_get("chikungunya")
        else:
            df = self._merged_cache_get("dengue")

        if uf:
            df = df[df["uf"] == uf]

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        logger.info(f"Merged data shape: {df.shape}")
        return df

    def aggregate_to_state(self, df: pd.DataFrame, aggregation: str = "sum") -> pd.DataFrame:
        """Aggregate municipality-level data to state level.

        Args:
            df: DataFrame with municipality-level data
            aggregation: Aggregation method for numeric columns

        Returns:
            DataFrame aggregated by state and date
        """
        group_cols = ["date", "uf"]
        if "epiweek" in df.columns:
            group_cols.append("epiweek")

        agg_dict = {"casos": "sum"}

        climate_cols = [
            "temp_min",
            "temp_med",
            "temp_max",
            "precip_min",
            "precip_med",
            "precip_max",
            "precip_tot",
            "pressure_min",
            "pressure_med",
            "pressure_max",
            "rel_humid_min",
            "rel_humid_med",
            "rel_humid_max",
            "thermal_range",
            "rainy_days",
        ]
        for col in climate_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

        if "population" in df.columns:
            agg_dict["population"] = "sum"

        train_cols = ["train_1", "train_2", "train_3", "train_4"]
        for col in train_cols:
            if col in df.columns:
                agg_dict[col] = "max"

        target_cols = ["target_1", "target_2", "target_3", "target_4"]
        for col in target_cols:
            if col in df.columns:
                agg_dict[col] = "max"

        df_state = df.groupby(group_cols, as_index=False).agg(agg_dict)

        if "population" in df_state.columns:
            df_state["incidence_rate"] = df_state["casos"] / df_state["population"] * 100000

        logger.info(f"Aggregated to state level: {len(df_state)} records")
        return df_state

    def load_state_data(
        self,
        uf: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregate: bool = True,
        disease: str = "dengue",
    ) -> pd.DataFrame:
        """Load data for a specific state.

        Args:
            uf: State abbreviation (e.g., "SP")
            start_date: Start date filter
            end_date: End date filter
            aggregate: Whether to aggregate municipalities to state level
            disease: Which disease cases to load ("dengue" or "chikungunya")

        Returns:
            DataFrame for the specified state
        """
        df = self.load_merged_data(
            uf=uf, start_date=start_date, end_date=end_date, disease=disease
        )

        if aggregate:
            df = self.aggregate_to_state(df)

        df = df.sort_values("date").reset_index(drop=True)

        # Reindex to a complete weekly grid: missing weeks become NaN
        # (never zero-filled), so feature lags stay date-aligned and
        # gaps are explicit. Completeness is reported to the user.
        uf_name = f"{uf}/{disease}"
        df = reindex_weekly(df)
        warn_missing_weeks(df, uf_name, context="loaded state series")

        logger.info(f"Loaded {uf} data: {len(df)} records")
        return df

    def load_all_states(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregate: bool = True,
        disease: str = "dengue",
    ) -> dict[str, pd.DataFrame]:
        """Load data for all states.

        Aggregates the merged municipality table in a single pass
        (memory-bounded: no full-table copy).

        Args:
            start_date: Start date filter
            end_date: End date filter
            aggregate: Whether to aggregate municipalities to state level
            disease: Which disease cases to load ("dengue" or "chikungunya")

        Returns:
            Dictionary mapping state abbreviations to DataFrames
        """
        states_data: dict[str, pd.DataFrame] = {}

        if aggregate:
            if disease not in self._merged_cache:
                base = self.chikungunya_df if disease == "chikungunya" else self.dengue_df
                self._merged_cache[disease] = self._merge_auxiliary(base)
            df = self._merged_cache[disease]
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date)]
            aggregated = self.aggregate_to_state(df)
            for uf, group in aggregated.groupby("uf"):
                g = group.sort_values("date").reset_index(drop=True)
                g = reindex_weekly(g)
                warn_missing_weeks(g, f"{uf}/{disease}", context="loaded state series")
                states_data[uf] = g
        else:
            df = self.load_merged_data(
                start_date=start_date, end_date=end_date, disease=disease
            )
            for uf in df["uf"].unique():
                uf_df = df[df["uf"] == uf].copy()
                uf_df = uf_df.sort_values("date").reset_index(drop=True)
                states_data[uf] = uf_df

        logger.info(f"Loaded data for {len(states_data)} states")
        return states_data

    def get_available_states(self) -> list[str]:
        """Get list of states with available data.

        Returns:
            List of state abbreviations
        """
        return sorted(self.dengue_df["uf"].unique().tolist())

    def get_date_range(self) -> dict[str, str]:
        """Get the date range of available data.

        Returns:
            Dictionary with min_date and max_date
        """
        return {
            "min_date": self.dengue_df["date"].min().strftime("%Y-%m-%d"),
            "max_date": self.dengue_df["date"].max().strftime("%Y-%m-%d"),
        }

    def load_ocean_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load ocean oscillation data.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with ENSO, IOD, PDO indices
        """
        df = self.ocean_df.copy()

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return df.sort_values("date").reset_index(drop=True)


DataLoader = CompetitionDataLoader
