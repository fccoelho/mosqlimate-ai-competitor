"""Refresh stale competition data — without re-downloading verified files.

Policy:
1. Files recorded as *verified complete* in
   ``data/completeness_verified.json`` (written by ``check-data`` after
   a successful completeness pass) are skipped — the saved version is
   authoritative while its fingerprint (size + mtime) is unchanged.
2. Remaining files are fetched only when the remote size differs from
   the local copy (the downloader's own cache check).
3. ``--force`` overrides everything (manual full refresh).

Usage:
    python scripts/refresh_data.py            # refresh stale files only
    python scripts/refresh_data.py --force    # full refresh
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refresh_data")

from mosqlimate_ai.data.completeness import file_is_verified

DATA_DIR = "data"

FILES = [
    # case bases — subject to the completeness verification manifest
    "dengue.csv.gz",
    "chikungunya.csv.gz",
    # exogenous bases — freshness tracked by remote-size comparison
    "climate.csv.gz",
    "forecasting_climate.csv.gz",
    "ocean_climate_oscillations.csv.gz",
]


def main() -> None:
    force = "--force" in sys.argv
    from mosqlimate_ai.data.downloader import DataDownloader

    downloader = DataDownloader()
    downloader.connect()
    skipped_verified, skipped_fresh, downloaded = [], [], []
    try:
        for filename in FILES:
            if not force and file_is_verified(DATA_DIR, filename):
                logger.info("skip %s: verified complete — using saved version", filename)
                skipped_verified.append(filename)
                continue
            try:
                if not force and downloader._file_exists_and_valid(filename):
                    logger.info("skip %s: local copy matches remote", filename)
                    skipped_fresh.append(filename)
                    continue
            except Exception as exc:
                logger.warning("remote size check failed for %s (%s); downloading", filename, exc)
            downloader.download_file(filename, force=force)
            downloaded.append(filename)
    finally:
        downloader.disconnect()

    logger.info(
        "refresh summary: %d downloaded, %d skipped (verified), %d skipped (fresh)",
        len(downloaded), len(skipped_verified), len(skipped_fresh),
    )


if __name__ == "__main__":
    main()
