"""Data downloader for Mosqlimate competition data.

Downloads and caches data from the Mosqlimate FTP server (info.dengue.mat.br).
Supports both 2nd IMDC (2025) and 3rd IMDC (2026) challenges.
"""

import ftplib
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

CHALLENGE_2ND_IMDC = "2nd_IMDC"
CHALLENGE_3RD_IMDC = "3rd_IMDC"
DEFAULT_CHALLENGE = CHALLENGE_3RD_IMDC

DATA_FILES_2ND = {
    "dengue.csv.gz": {
        "description": "Weekly dengue cases by municipality (2010-2025)",
        "required": True,
    },
    "climate.csv.gz": {
        "description": "Weekly climate reanalysis data (ERA5)",
        "required": True,
    },
    "climate_forecast.csv.gz": {
        "description": "Monthly climate forecasts (ECMWF)",
        "required": True,
    },
    "datasus_population_2001_2024.csv.gz": {
        "description": "Population by municipality and year",
        "required": True,
    },
    "environ_vars.csv.gz": {
        "description": "Environmental variables (Koppen, Biome)",
        "required": True,
    },
    "map_regional_health.csv": {
        "description": "City to health region mapping",
        "required": True,
    },
    "shape_muni.gpkg": {
        "description": "Municipality geometries",
        "required": True,
    },
    "shape_regional_health.gpkg": {
        "description": "Regional health geometries",
        "required": True,
    },
    "shape_macroregional_health.gpkg": {
        "description": "Macroregional health geometries",
        "required": True,
    },
    "ocean_climate_oscillations.csv.gz": {
        "description": "ENSO, IOD, PDO ocean indices",
        "required": True,
    },
}

DATA_FILES_3RD = {
    "dengue.csv.gz": {
        "description": "Weekly dengue cases by municipality (2010-2026)",
        "required": True,
    },
    "chikungunya.csv.gz": {
        "description": "Weekly chikungunya cases by municipality (2014-2026)",
        "required": True,
    },
    "climate.csv.gz": {
        "description": "Weekly climate reanalysis data (ERA5)",
        "required": True,
    },
    "climate_forecast.csv.gz": {
        "description": "Monthly climate forecasts (ECMWF)",
        "required": True,
    },
    "datasus_population_2001_2025.csv.gz": {
        "description": "Population by municipality and year",
        "required": True,
    },
    "environ_vars.csv.gz": {
        "description": "Environmental variables (Koppen, Biome)",
        "required": True,
    },
    "map_regional_health.csv": {
        "description": "City to health region mapping",
        "required": True,
    },
    "shape_muni.gpkg": {
        "description": "Municipality geometries",
        "required": True,
    },
    "shape_regional_health.gpkg": {
        "description": "Regional health geometries",
        "required": True,
    },
    "shape_macroregional_health.gpkg": {
        "description": "Macroregional health geometries",
        "required": True,
    },
    "enso.csv.gz": {
        "description": "ENSO ocean index",
        "required": True,
    },
    "iod.csv.gz": {
        "description": "IOD ocean index",
        "required": True,
    },
    "pdo.csv.gz": {
        "description": "PDO ocean index",
        "required": True,
    },
}

CHALLENGE_DATA_FILES = {
    CHALLENGE_2ND_IMDC: DATA_FILES_2ND,
    CHALLENGE_3RD_IMDC: DATA_FILES_3RD,
}

CHALLENGE_REMOTE_DIRS = {
    CHALLENGE_2ND_IMDC: "data_sprint_2025",
    CHALLENGE_3RD_IMDC: "data_imdc_2026",
}


class DownloadConfig(BaseModel):
    """Configuration for data download."""

    ftp_host: str = "info.dengue.mat.br"
    ftp_user: str = "anonymous"
    ftp_password: str = "anonymous@domain.com"
    timeout: int = 300


class DataDownloader:
    """Download and cache Mosqlimate competition data.

    This class handles downloading data from the Mosqlimate FTP server
    and caching it locally. Files are kept compressed (.gz) to save space.

    Args:
        cache_dir: Directory to store downloaded data. Defaults to project's data/{challenge}/ folder.
        config: FTP connection configuration.
        challenge: Which challenge dataset to use ("2nd_IMDC" or "3rd_IMDC").

    Example:
        >>> downloader = DataDownloader(challenge="3rd_IMDC")
        >>> downloader.download_all()
        >>> df = downloader.load_data("dengue.csv.gz")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[DownloadConfig] = None,
        challenge: str = DEFAULT_CHALLENGE,
    ):
        self.challenge = challenge
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir(challenge)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or DownloadConfig()
        self._ftp: Optional[ftplib.FTP] = None

    @staticmethod
    def _get_default_cache_dir(challenge: str = DEFAULT_CHALLENGE) -> Path:
        """Get default cache directory.

        Prefers ``data/{challenge}/`` when it exists; falls back to the flat
        ``data/`` directory so downloads land next to already-cached files.
        """
        project_root = Path(__file__).parent.parent.parent.parent
        challenge_dir = project_root / "data" / challenge
        if challenge_dir.exists():
            return challenge_dir
        return project_root / "data"

    @property
    def remote_dir(self) -> str:
        """Get the remote FTP directory for the current challenge."""
        return CHALLENGE_REMOTE_DIRS.get(self.challenge, CHALLENGE_REMOTE_DIRS[DEFAULT_CHALLENGE])

    @property
    def data_files(self) -> dict:
        """Get the data files list for the current challenge."""
        return CHALLENGE_DATA_FILES.get(self.challenge, CHALLENGE_DATA_FILES[DEFAULT_CHALLENGE])

    def connect(self) -> None:
        """Connect to FTP server."""
        if self._ftp is not None:
            return

        console.print(f"[cyan]Connecting to {self.config.ftp_host}...[/cyan]")
        self._ftp = ftplib.FTP()
        self._ftp.connect(self.config.ftp_host, timeout=self.config.timeout)
        self._ftp.login(self.config.ftp_user, self.config.ftp_password)
        self._ftp.voidcmd("TYPE I")
        self._ftp.cwd(self.remote_dir)
        console.print("[green]Connected successfully![/green]")

    def disconnect(self) -> None:
        """Disconnect from FTP server."""
        if self._ftp is not None:
            try:
                self._ftp.quit()
            except Exception:
                self._ftp.close()
            self._ftp = None

    def _get_remote_file_size(self, filename: str) -> int:
        """Get file size from FTP server."""
        if self._ftp is None:
            self.connect()
        assert self._ftp is not None
        size = self._ftp.size(filename)
        return size if size else 0

    def _file_exists_and_valid(self, filename: str) -> bool:
        """Check if local file exists and has correct size."""
        local_path = self.cache_dir / filename
        if not local_path.exists():
            return False

        try:
            remote_size = self._get_remote_file_size(filename)
            local_size = local_path.stat().st_size
            return local_size == remote_size
        except Exception:
            return local_path.exists()

    def download_file(
        self,
        filename: str,
        force: bool = False,
    ) -> Path:
        """Download a single file from FTP server.

        Args:
            filename: Name of the file to download.
            force: Force re-download even if file exists.

        Returns:
            Path to the downloaded file.

        Raises:
            FileNotFoundError: If file doesn't exist on server.
        """
        local_path = self.cache_dir / filename

        if not force and self._file_exists_and_valid(filename):
            console.print(f"[green]✓[/green] {filename} already cached")
            return local_path

        if self._ftp is None:
            self.connect()

        remote_size = self._get_remote_file_size(filename)

        console.print(
            f"[yellow]↓[/yellow] Downloading {filename} ({self._format_size(remote_size)})..."
        )

        temp_path = local_path.with_suffix(local_path.suffix + ".tmp")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(filename, total=remote_size)
            downloaded = [0]

            def callback(data: bytes) -> None:
                temp_path.write_bytes(temp_path.read_bytes() + data)
                downloaded[0] += len(data)
                progress.update(task, completed=downloaded[0])

            temp_path.write_bytes(b"")
            assert self._ftp is not None
            self._ftp.retrbinary(f"RETR {filename}", callback)

        temp_path.rename(local_path)
        console.print(f"[green]✓[/green] Downloaded {filename}")

        return local_path

    def download_all(self, force: bool = False) -> dict:
        """Download all required data files for the current challenge.

        Args:
            force: Force re-download even if files exist.

        Returns:
            Dictionary mapping filenames to local paths.
        """
        self.connect()

        results = {}
        failed = []

        table = Table(title=f"Data Download Summary ({self.challenge})")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description")

        for filename, info in self.data_files.items():
            try:
                path = self.download_file(filename, force=force)
                results[filename] = path
                table.add_row(filename, "✓ Downloaded", info["description"])
            except Exception as e:
                failed.append(filename)
                table.add_row(filename, f"✗ Failed: {e}", info["description"])
                logger.error(f"Failed to download {filename}: {e}")

        console.print(table)

        if failed:
            console.print(
                f"\n[red]Failed to download {len(failed)} files: {', '.join(failed)}[/red]"
            )

        self.disconnect()
        return results

    def get_local_path(self, filename: str) -> Optional[Path]:
        """Get local path for a file if it exists.

        Args:
            filename: Name of the file.

        Returns:
            Path to the file if it exists, None otherwise.
        """
        local_path = self.cache_dir / filename
        return local_path if local_path.exists() else None

    def list_cached_files(self) -> list:
        """List all cached data files.

        Returns:
            List of paths to cached files.
        """
        return list(self.cache_dir.glob("*"))

    def clear_cache(self) -> None:
        """Clear all cached data files."""
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        console.print(f"[yellow]Cache cleared: {self.cache_dir}[/yellow]")

    def get_cache_info(self) -> dict:
        """Get information about cached files.

        Returns:
            Dictionary with file info including size and status.
        """
        info = {}
        for filename, meta in self.data_files.items():
            local_path = self.cache_dir / filename
            if local_path.exists():
                size = local_path.stat().st_size
                info[filename] = {
                    "cached": True,
                    "size": size,
                    "size_formatted": self._format_size(size),
                    "description": meta["description"],
                }
            else:
                info[filename] = {
                    "cached": False,
                    "size": 0,
                    "size_formatted": "N/A",
                    "description": meta["description"],
                }
        return info

    @staticmethod
    def _format_size(size: int) -> str:
        """Format file size in human-readable format."""
        s = float(size)
        for unit in ["B", "KB", "MB", "GB"]:
            if s < 1024:
                return f"{s:.1f} {unit}"
            s /= 1024
        return f"{s:.1f} TB"

    def __enter__(self) -> "DataDownloader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def download_data(
    cache_dir: Optional[Path] = None,
    force: bool = False,
    challenge: str = DEFAULT_CHALLENGE,
) -> dict:
    """Convenience function to download all data.

    Args:
        cache_dir: Directory to store downloaded data.
        force: Force re-download even if files exist.
        challenge: Which challenge dataset to use ("2nd_IMDC" or "3rd_IMDC").

    Returns:
        Dictionary mapping filenames to local paths.
    """
    with DataDownloader(cache_dir=cache_dir, challenge=challenge) as downloader:
        return downloader.download_all(force=force)
