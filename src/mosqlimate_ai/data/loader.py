"""Data loader for Mosqlimate Sprint 2025 competition data.

Loads and merges cached competition data from the local data/ directory.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

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

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader.

        Args:
            data_dir: Path to data directory. Defaults to project's data/ folder.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._dengue_df: Optional[pd.DataFrame] = None
        self._climate_df: Optional[pd.DataFrame] = None
        self._climate_forecast_df: Optional[pd.DataFrame] = None
        self._population_df: Optional[pd.DataFrame] = None
        self._environ_df: Optional[pd.DataFrame] = None
        self._ocean_df: Optional[pd.DataFrame] = None
        self._regional_map_df: Optional[pd.DataFrame] = None

        logger.info(f"CompetitionDataLoader initialized with data_dir: {self.data_dir}")

    @property
    def dengue_df(self) -> pd.DataFrame:
        """Lazy load dengue data."""
        if self._dengue_df is None:
            self._dengue_df = self._load_dengue_data()
        return self._dengue_df

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

    def _load_dengue_data(self) -> pd.DataFrame:
        """Load dengue cases data."""
        filepath = self.data_dir / "dengue.csv.gz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dengue data not found at {filepath}. " "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading dengue data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["date"] = pd.to_datetime(df["date"])
        df["epiweek"] = df["epiweek"].astype(int)
        df["geocode"] = df["geocode"].astype(int)
        df["casos"] = df["casos"].fillna(0).astype(int)

        logger.info(f"Loaded {len(df)} dengue records")
        return df

    def _load_climate_data(self) -> pd.DataFrame:
        """Load climate reanalysis data."""
        filepath = self.data_dir / "climate.csv.gz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Climate data not found at {filepath}. " "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading climate data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["date"] = pd.to_datetime(df["date"])
        df["epiweek"] = df["epiweek"].astype(int)
        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} climate records")
        return df

    def _load_climate_forecast_data(self) -> pd.DataFrame:
        """Load climate forecast data."""
        filepath = self.data_dir / "climate_forecast.csv.gz"
        if not filepath.exists():
            warnings.warn(f"Climate forecast data not found at {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading climate forecast data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["reference_month"] = pd.to_datetime(df["reference_month"])
        df["geocode"] = df["geocode"].astype(int)

        logger.info(f"Loaded {len(df)} climate forecast records")
        return df

    def _load_population_data(self) -> pd.DataFrame:
        """Load population data."""
        filepath = self.data_dir / "datasus_population_2001_2024.csv.gz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Population data not found at {filepath}. "
                "Run 'mosqlimate-ai download-data' first."
            )

        logger.info(f"Loading population data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["geocode"] = df["geocode"].astype(int)
        df["year"] = df["year"].astype(int)

        logger.info(f"Loaded {len(df)} population records")
        return df

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
        """Load ocean climate oscillation data."""
        filepath = self.data_dir / "ocean_climate_oscillations.csv.gz"
        if not filepath.exists():
            warnings.warn(f"Ocean oscillation data not found at {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading ocean oscillation data from {filepath}")
        df = pd.read_csv(filepath, compression="gzip")

        df["date"] = pd.to_datetime(df["date"])

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
    ) -> pd.DataFrame:
        """Load and merge all data sources.

        Args:
            uf: Filter by state abbreviation (e.g., "SP"). None for all states.
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            include_climate: Whether to merge climate data
            include_population: Whether to merge population data
            include_environmental: Whether to merge environmental data

        Returns:
            Merged DataFrame with all data sources
        """
        df = self.dengue_df.copy()

        if uf:
            df = df[df["uf"] == uf]

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if include_climate and not self.climate_df.empty:
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

        if include_population and not self.population_df.empty:
            df["year"] = df["date"].dt.year
            pop_merge = self.population_df[["geocode", "year", "population"]].drop_duplicates(
                subset=["geocode", "year"]
            )
            df = df.merge(pop_merge, on=["geocode", "year"], how="left")

        if include_environmental and not self.environ_df.empty:
            environ_merge = self.environ_df[["geocode", "koppen", "biome"]].drop_duplicates(
                subset=["geocode"]
            )
            df = df.merge(environ_merge, on=["geocode"], how="left")

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

        train_cols = ["train_1", "train_2", "train_3"]
        for col in train_cols:
            if col in df.columns:
                agg_dict[col] = "max"

        target_cols = ["target_1", "target_2", "target_3"]
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
    ) -> pd.DataFrame:
        """Load data for a specific state.

        Args:
            uf: State abbreviation (e.g., "SP")
            start_date: Start date filter
            end_date: End date filter
            aggregate: Whether to aggregate municipalities to state level

        Returns:
            DataFrame for the specified state
        """
        df = self.load_merged_data(uf=uf, start_date=start_date, end_date=end_date)

        if aggregate:
            df = self.aggregate_to_state(df)

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def load_all_states(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregate: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for all states.

        Args:
            start_date: Start date filter
            end_date: End date filter
            aggregate: Whether to aggregate municipalities to state level

        Returns:
            Dictionary mapping state abbreviations to DataFrames
        """
        states_data = {}

        df = self.load_merged_data(start_date=start_date, end_date=end_date)

        for uf in df["uf"].unique():
            uf_df = df[df["uf"] == uf].copy()
            if aggregate:
                uf_df = self.aggregate_to_state(uf_df)
            uf_df = uf_df.sort_values("date").reset_index(drop=True)
            states_data[uf] = uf_df

        logger.info(f"Loaded data for {len(states_data)} states")
        return states_data

    def get_available_states(self) -> List[str]:
        """Get list of states with available data.

        Returns:
            List of state abbreviations
        """
        return sorted(self.dengue_df["uf"].unique().tolist())

    def get_date_range(self) -> Dict[str, str]:
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
