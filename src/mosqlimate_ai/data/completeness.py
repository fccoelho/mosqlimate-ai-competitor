"""Weekly time-series completeness checks.

The IMDC data is published as a cumulative base file plus partial
season-update segments; when vintages drift apart, entire weeks can be
absent from the merged series. Missing weeks silently corrupt
positional feature engineering (lag ``k`` no longer means ``k`` weeks)
and can hide evaluation weeks, so they are checked explicitly and
reported to the user as warnings — never silently zero-filled, never
fatal (the models handle NaN and the pipeline degrades gracefully but
visibly).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

WEEK = pd.Timedelta(days=7)


def expected_weekly_index(
    start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Complete weekly (Sunday-anchored) index covering ``start``..``end``."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return pd.date_range(start, end + WEEK - pd.Timedelta(days=1), freq="7D")


def find_missing_weeks(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_col: str = "date",
    value_col: Optional[str] = "casos",
) -> List[pd.Timestamp]:
    """Missing weekly dates inside ``[start, end]`` (default: data range).

    A week counts as missing when its row does not exist **or** its
    value in ``value_col`` is NaN (e.g. after weekly reindexing).

    Args:
        df: Frame with a (weekly) date column or DatetimeIndex.
        start/end: Optional bounds (inclusive). Default: series range.
        value_col: Value column whose NaN marks a missing week
            (``None`` disables the value check).

    Returns:
        Sorted list of the missing weekly timestamps.
    """
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        values = df[value_col] if value_col and value_col in df.columns else None
    else:
        dates = pd.to_datetime(df.index)
        values = df[value_col] if value_col and value_col in df.columns else None

    if len(dates) == 0:
        return []

    first = pd.Timestamp(dates.min()) if start is None else pd.Timestamp(start)
    last = pd.Timestamp(dates.max()) if end is None else pd.Timestamp(end)

    if values is not None:
        present = set(pd.DatetimeIndex(dates[values.notna()]))
    else:
        present = set(pd.DatetimeIndex(dates))

    missing = [d for d in expected_weekly_index(first, last) if d not in present]
    return missing


def warn_missing_weeks(
    df: pd.DataFrame,
    name: str,
    context: str = "",
    start: Optional[str] = None,
    end: Optional[str] = None,
    value_col: Optional[str] = "casos",
) -> List[pd.Timestamp]:
    """Emit a user-facing warning listing missing weeks, if any.

    Args:
        df: Frame to check.
        name: Label for messages (e.g. ``"SP/dengue"``).
        context: Optional context suffix (e.g. ``"training window"``).
        start/end: Optional explicit bounds.
        value_col: Value column whose NaN marks a missing week.

    Returns:
        The missing-week list (empty when the series is complete), so
        callers can act on it programmatically.
    """
    missing = find_missing_weeks(
        df, start=start, end=end, value_col=value_col
    )
    if not missing:
        return []

    ctx = f" ({context})" if context else ""
    preview = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing[:10])
    more = f" ... +{len(missing) - 10} more" if len(missing) > 10 else ""
    message = (
        f"{name}{ctx}: time series is INCOMPLETE — {len(missing)} missing "
        f"weekly value(s): {preview}{more}. Forecasts and metrics that "
        f"depend on these weeks are degraded. Try re-downloading the data "
        f"('mosqlimate-ai download-data --force')."
    )
    warnings.warn(message, stacklevel=2)
    logger.warning(message)
    return missing


def summarize_missing_by_state(
    states_data: dict[str, pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, List[pd.Timestamp]]:
    """Check every state frame; returns only the states with gaps."""
    return {
        uf: missing
        for uf, df in states_data.items()
        if (missing := find_missing_weeks(df, start=start, end=end))
    }


# ---------------------------------------------------------------------------
# Verification manifest
# ---------------------------------------------------------------------------
# A small manifest records which cached base files have been verified
# complete (no missing weeks) and *when*, together with the local file
# fingerprint at verification time. Refresh tooling uses it to skip
# re-downloading data that was already verified — the saved version is
# authoritative until the file itself changes on disk.
VERIFICATION_MANIFEST_NAME = "completeness_verified.json"


def _manifest_path(data_dir) -> Path:
    from pathlib import Path

    return Path(data_dir) / VERIFICATION_MANIFEST_NAME


def _file_fingerprint(path: Path) -> dict:

    st = path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def load_verification_manifest(data_dir) -> dict:
    """Return ``{filename: {size, mtime, verified_at, status}}`` ({} if none)."""
    import json

    path = _manifest_path(data_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # pragma: no cover - corrupted manifest
        return {}


def file_is_verified(data_dir, filename: str) -> bool:
    """True when ``filename`` passed a completeness check and has not
    changed on disk since (same size and mtime)."""
    from pathlib import Path

    path = Path(data_dir) / filename
    if not path.exists():
        return False
    entry = load_verification_manifest(data_dir).get(filename)
    if not entry or entry.get("status") != "complete":
        return False
    try:
        current = _file_fingerprint(path)
    except OSError:  # pragma: no cover - defensive
        return False
    return current.get("size") == entry.get("size") and current.get("mtime") == entry.get("mtime")


def mark_files_verified(data_dir, filenames: List[str], status: str = "complete") -> Path:
    """Record the current local fingerprint of ``filenames`` as verified.

    Written by the completeness check (CLI ``check-data``) after a
    successful pass; consumed by refresh tooling to avoid re-downloading
    verified files.
    """
    import datetime as _dt
    import json
    from pathlib import Path

    data_dir = Path(data_dir)
    manifest = load_verification_manifest(data_dir)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for filename in filenames:
        path = data_dir / filename
        if path.exists():
            entry = _file_fingerprint(path)
            entry.update({"verified_at": now, "status": status})
            manifest[filename] = entry
    path = _manifest_path(data_dir)
    path.write_text(json.dumps(manifest, indent=2))
    return path
