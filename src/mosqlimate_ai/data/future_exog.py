"""Build future-known exogenous covariates for forecast horizons.

For the IMDC setup, models forecast from EW25 (train cutoff) through
EW40 of the following year. Two families of covariates are known (or
projected) for those future weeks:

- **ECMWF monthly climate forecasts** (``forecasting_climate.csv.gz``):
  municipality-level monthly ``temp_med``, ``umid_med``, ``precip_tot``
  with a lead time (``forecast_months_ahead``). These are aggregated to
  the state with population weights and mapped to weeks.

- **Ocean indices** (ENSO/IOD/PDO, weekly): observed up to the data
  cutoff; extended by persistence beyond it (ocean indices have strong
  multi-month autocorrelation; persistence is the standard baseline).

Leakage control: for a target month ``M`` with forecast origin ``O``,
only forecast leads ``>= ceil(months(O -> M))`` are eligible, so that
the value was actually available at forecast issue time. When no such
lead exists the shortest available lead is used (relevant only for the
final, beyond-dataset months).
"""

from __future__ import annotations

from typing import Dict, List, Optional  # noqa: F401 - used in annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLIMATE_FORECAST_COLS = ["temp_med", "umid_med", "precip_tot"]
OCEAN_COLS = ["enso", "iod", "pdo"]


def _pop_weighted_monthly_state(
    climate_forecast: pd.DataFrame,
    population: pd.DataFrame,
    uf: str,
    uf_codes: dict,
) -> pd.DataFrame:
    """Population-weighted state series per (issue month, forecast lead).

    ``reference_month`` in the IMDC climate-forecast file is the *issue*
    month; each issue carries leads 1..6 (target = issue + lead months).
    Returns a wide frame indexed by issue month with columns
    ``<col>_lead<k>``.
    """
    state_code = uf_codes.get(uf)
    if state_code is None:
        raise ValueError(f"unknown state: {uf}")

    cf = climate_forecast[climate_forecast["geocode"] // 100000 == state_code].copy()
    if cf.empty:
        return pd.DataFrame()

    pop = population.copy()
    pop["state_code"] = pop["geocode"] // 100000
    pop = pop[pop["state_code"] == state_code]
    # population by geocode: latest year available per geocode
    pop = pop.sort_values("year").drop_duplicates(subset=["geocode"], keep="last")
    weights = pop.set_index("geocode")["population"]
    weights = weights.reindex(cf["geocode"].unique())
    weights = weights.fillna(weights.dropna().mean() if weights.notna().any() else 1.0)

    cf["weight"] = cf["geocode"].map(weights)
    cf["weight"] = cf["weight"].fillna(cf["weight"].mean())

    grouped = cf.groupby(["reference_month", "forecast_months_ahead"])
    out = pd.DataFrame(index=grouped.size().index)
    for col in CLIMATE_FORECAST_COLS:
        if col in cf.columns:
            out[col] = grouped.apply(
                lambda g, c=col: _weighted_mean(g, c, w_col="weight")
            )
    # pivot leads to wide format before collapsing the MultiIndex
    wide = out.unstack("forecast_months_ahead")
    wide.columns = [f"{c}_lead{int(l)}" for c, l in wide.columns]
    wide.index = pd.to_datetime(wide.index.get_level_values("reference_month"))
    return wide.sort_index()


def _weighted_mean(g: pd.DataFrame, col: str, w_col: str) -> float:
    weights = g[w_col].values.astype(float)
    vals = g[col].astype(float).values
    mask = np.isfinite(vals)
    if not mask.any():
        return np.nan
    return float(np.average(vals[mask], weights=weights[mask]))


def _monthly_to_weekly(monthly: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Expand a monthly series onto weekly dates (value of the containing month)."""
    if monthly.empty:
        return pd.DataFrame(index=dates)
    idx = monthly.index
    months = idx if isinstance(idx, pd.PeriodIndex) else idx.to_period("M")
    lookup = pd.DataFrame(monthly.values, index=months, columns=monthly.columns)
    target_months = pd.DatetimeIndex(dates).to_period("M")
    return lookup.reindex(target_months).set_index(pd.DatetimeIndex(dates))


def _climate_forecast_weekly(
    wide: pd.DataFrame,
    dates: pd.DatetimeIndex,
    origin_date: pd.Timestamp,
    max_lead: int = 6,
) -> pd.DataFrame:
    """Climate-forecast values for target weeks, leak-safe by issue time.

    For a target month ``M`` with forecast origin month ``O``: use the
    latest issue available at or before ``O`` (i.e. issue = ``min(M, O)``)
    and lead ``= M - issue`` (clipped to 1..max_lead). Weeks beyond
    ``max_lead`` months reuse the lead-``max_lead`` (stalest) forecast.
    """
    if wide.empty:
        return pd.DataFrame(index=dates)

    base_cols = sorted({c.rsplit("_lead", 1)[0] for c in wide.columns})
    origin_month = pd.Timestamp(origin_date).to_period("M")

    target_months = pd.DatetimeIndex(dates).to_period("M")
    rows = {}
    for m in target_months.unique():
        issue = min(m, origin_month)
        lead = (m.year - issue.year) * 12 + (m.month - issue.month)
        lead = int(max(1, min(max_lead, lead)))
        row = {}
        for col in base_cols:
            name = f"{col}_lead{lead}"
            ts = issue.to_timestamp()
            row[col] = wide.loc[wide.index == ts, name].iloc[0] if (
                name in wide.columns and (wide.index == ts).any()
            ) else np.nan
        rows[m] = row
    monthly = pd.DataFrame.from_dict(rows, orient="index")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return _monthly_to_weekly(monthly, dates)


def build_future_exog(
    loader,
    uf: str,
    dates: pd.DatetimeIndex,
    use_climate_forecast: bool = True,
    use_ocean: bool = True,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assemble future-known exogenous features for the given target dates.

    Args:
        loader: A :class:`CompetitionDataLoader` with cached data.
        uf: State abbreviation.
        dates: Weekly target dates (forecast horizon).
        use_climate_forecast: Include ECMWF monthly climate forecasts.
        use_ocean: Include ENSO/IOD/PDO indices (persistence beyond cutoff).
        train_end: Training cutoff; ocean values after it are replaced by
            persistence of the last observed value.

    Returns:
        DataFrame indexed by ``dates`` with exogenous columns; empty when
        no source is available.
    """
    from mosqlimate_ai.data.loader import STATE_CODES

    frames: list[pd.DataFrame] = []
    dates = pd.DatetimeIndex(dates)

    if use_climate_forecast:
        try:
            if hasattr(loader, "pop_weighted_climate_forecast"):
                # single streaming pass, cached per state on the loader
                wide = loader.pop_weighted_climate_forecast(uf)
            else:
                cf = loader.climate_forecast_df
                pop = loader.population_df
                wide = (
                    _pop_weighted_monthly_state(cf, pop, uf, STATE_CODES)
                    if not cf.empty
                    else pd.DataFrame()
                )
            origin = train_end if train_end is not None else dates.min()
            weekly = _climate_forecast_weekly(wide, dates, origin)
            if not weekly.empty and weekly.notna().any().any():
                frames.append(weekly.add_prefix("cf_"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("climate forecast features unavailable for %s: %s", uf, exc)

    if use_ocean:
        try:
            ocean = loader.ocean_df
            if not ocean.empty:
                ocean = ocean.sort_values("date").set_index("date")[OCEAN_COLS]
                if train_end is not None:
                    ocean = ocean[ocean.index <= pd.Timestamp(train_end)]
                if ocean.empty:
                    raise ValueError("no ocean observations at or before train_end")
                # persistence beyond the last observation
                union = ocean.index.union(dates)
                weekly = ocean.reindex(union, method="ffill").loc[dates]
                frames.append(weekly.add_prefix("oc_"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ocean features unavailable for %s: %s", uf, exc)

    if not frames:
        return pd.DataFrame(index=dates)

    out = pd.concat(frames, axis=1)
    return out.loc[~out.index.duplicated(keep="last")]


class ExogLookup:
    """Known-covariate provider for (forecast origin, target date) pairs.

    Supplies the values a forecaster would have known at ``origin_date``
    about ``target_date``:

    - **ECMWF climate forecasts**: issue/lead table (pop-weighted to the
      state); issue = ``min(target_month, origin_month)``, lead clipped
      to 1..6 (beyond 6 the stalest forecast is reused).
    - **Ocean indices** (ENSO/IOD/PDO): observed weekly values up to the
      data cutoff, persistence afterwards.

    Usage:
        lookup = ExogLookup(loader, "SP", train_end)
        feats = lookup.get(origin_date, target_date)  # dict of floats
    """

    OCEAN = ("enso", "iod", "pdo")
    CLIMATE = ("temp_med", "umid_med", "precip_tot")

    def __init__(self, loader, uf: str, train_end: Optional[pd.Timestamp] = None):
        from mosqlimate_ai.data.loader import STATE_CODES

        self.uf = uf
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None
        self.features_ = tuple(f"cf_{c}" for c in self.CLIMATE) + tuple(
            f"oc_{c}" for c in self.OCEAN
        )

        self.cf_wide_ = None
        try:
            if hasattr(loader, "pop_weighted_climate_forecast"):
                # single streaming pass, cached per state on the loader
                self.cf_wide_ = loader.pop_weighted_climate_forecast(uf)
            else:
                state_code = STATE_CODES.get(uf)
                if hasattr(loader, "load_climate_forecast_for_state"):
                    cf = loader.load_climate_forecast_for_state(state_code)
                else:
                    cf = loader.climate_forecast_df
                if cf is not None and not cf.empty:
                    self.cf_wide_ = _pop_weighted_monthly_state(
                        cf, loader.population_df, uf, STATE_CODES
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("climate forecast table unavailable for %s: %s", uf, exc)

        self.ocean_ = None
        try:
            ocean = loader.ocean_df
            if ocean is not None and not ocean.empty:
                ocean = ocean.sort_values("date").set_index("date")[list(self.OCEAN)]
                if self.train_end is not None:
                    ocean = ocean[ocean.index <= self.train_end]
                self.ocean_ = ocean[~ocean.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ocean table unavailable for %s: %s", uf, exc)

    # ------------------------------------------------------------------
    def _cf_row(self, origin_date: pd.Timestamp, target_date: pd.Timestamp) -> Dict[str, float]:
        out = {f"cf_{c}": np.nan for c in self.CLIMATE}
        if self.cf_wide_ is None or self.cf_wide_.empty:
            return out
        o_m = pd.Timestamp(origin_date).to_period("M")
        t_m = pd.Timestamp(target_date).to_period("M")
        issue = min(t_m, o_m)
        lead = (t_m.year - issue.year) * 12 + (t_m.month - issue.month)
        lead = int(max(1, min(6, lead)))
        ts = issue.to_timestamp()
        if ts not in self.cf_wide_.index:
            earlier = self.cf_wide_.index[self.cf_wide_.index <= ts]
            if not len(earlier):
                return out
            ts = earlier[-1]
        row = self.cf_wide_.loc[ts]
        for c in self.CLIMATE:
            out[f"cf_{c}"] = float(row.get(f"{c}_lead{lead}", np.nan))
        return out

    def _ocean_row(self, target_date: pd.Timestamp) -> Dict[str, float]:
        out = {f"oc_{c}": np.nan for c in self.OCEAN}
        if self.ocean_ is None or self.ocean_.empty:
            return out
        idx = self.ocean_.index
        if target_date in idx:
            row = self.ocean_.loc[target_date]
        else:
            prior = idx[idx <= target_date]
            if not len(prior):
                return out
            row = self.ocean_.loc[prior[-1]]
        for c in self.OCEAN:
            out[f"oc_{c}"] = float(row[c]) if pd.notna(row[c]) else np.nan
        return out

    def get(self, origin_date: pd.Timestamp, target_date: pd.Timestamp) -> Dict[str, float]:
        """Known covariates for ``target_date`` as of ``origin_date``."""
        row = self._cf_row(origin_date, target_date)
        row.update(self._ocean_row(target_date))
        return row
