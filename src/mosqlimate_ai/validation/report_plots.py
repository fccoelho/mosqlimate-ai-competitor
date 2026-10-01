"""Prediction-vs-observed plots for the validation report.

For every state and disease, produces a multi-panel figure (one panel
per validation test) showing:

- the observed training tail (context),
- the selected model's calibrated forecast: median line, 50% and 95%
  bands,
- the observed weekly cases over the target window,
- the training cutoff and the unobserved gap region.

Figures are written as PNGs under ``<backtest_dir>/plots/`` and
embedded into ``VALIDATION_REPORT.md`` via relative links.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - backend must be set before pyplot

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEST_LABELS = {
    "1": "Test 1 · 2022-23",
    "2": "Test 2 · 2023-24",
    "3": "Test 3 · 2024-25",
    "4": "Test 4 · 2025-26",
    "final": "Final · 2026-27",
}


def _load_forecast(fc_dir: Path, uf: str, disease: str, test_key: str, model: str):
    path = fc_dir / f"{uf}_{disease}_{test_key}_{model}.csv.gz"
    if not path.exists():
        # fall back to the robust ensemble when the selected model's
        # forecast is missing (e.g. selection chose a model that failed)
        alt = fc_dir / f"{uf}_{disease}_{test_key}_ens_qavg.csv.gz"
        if alt.exists():
            path = alt
        else:
            return None, None
    fc = pd.read_csv(path)
    fc["date"] = pd.to_datetime(fc["date"])
    return fc.set_index("date"), path.name


def plot_state_panels(
    uf: str,
    disease: str,
    state_result: Dict,
    selected_model: str,
    state_df: pd.DataFrame,
    out_path: Path,
    backtest_dir: Optional[Path] = None,
) -> Optional[Path]:
    """One figure with a panel per validation test for a single state."""
    backtest_dir = Path(backtest_dir) if backtest_dir else Path(out_path).parent.parent.parent
    fc_dir = backtest_dir / "forecasts"
    tests = dict(state_result.get("tests", {}))
    if state_result.get("final"):
        tests["final"] = state_result["final"]
    if not tests:
        return None

    state_df = state_df.copy()
    state_df["date"] = pd.to_datetime(state_df["date"])
    observed_all = state_df.set_index("date")["casos"].astype(float)

    n = len(tests)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 3.1 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for ax, (test_key, test_res) in zip(axes, sorted(tests.items(), key=lambda kv: str(kv[0]))):
        train_end = pd.Timestamp(test_res["train_end"])
        target_start = pd.Timestamp(test_res["target_start"])

        fc, _fname = _load_forecast(
            fc_dir, uf, disease, test_key, selected_model
        )
        tail = observed_all[(observed_all.index >= train_end - pd.Timedelta(weeks=26))]
        tail = tail[tail.index <= train_end]

        ax.fill_between(tail.index, 0, tail.values, color="#9ecae1", alpha=0.5,
                        label="observed (train tail)")

        if fc is not None:
            ax.fill_between(
                fc.index, fc["q025"], fc["q975"], color="#fc8d59", alpha=0.25,
                label="95% PI",
            )
            ax.fill_between(
                fc.index, fc["q250"], fc["q750"], color="#fc8d59", alpha=0.55,
                label="50% PI",
            )
            ax.plot(fc.index, fc["q500"], color="#d73027", lw=1.6, label="median")

        obs = observed_all[(observed_all.index >= target_start)]
        obs = obs[obs.index <= target_start + pd.Timedelta(weeks=52)]
        if len(obs):
            ax.plot(obs.index, obs.values, color="black", lw=1.4, marker="o",
                    ms=2.5, label="observed (target)")

        ax.axvline(train_end, color="gray", ls="--", lw=1)
        ax.axvspan(train_end, target_start, color="0.92", zorder=0)

        metrics = (test_res.get("models", {}).get(selected_model, {}) or {}).get("metrics") or {}
        wis = metrics.get("wis_total")
        title = TEST_LABELS.get(test_key, f"Split {test_key}")
        if wis is not None and np.isfinite(wis):
            title += f" · WIS {wis:,.0f}"
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(labelsize=8)
        ax.set_ylabel("cases", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(), loc="upper right", fontsize=8, ncol=1
    )
    model_label = selected_model
    fig.suptitle(f"{uf} · {dengue_label(disease)} · model: {model_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def dengue_label(disease: str) -> str:
    return disease.capitalize()


def generate_all_plots(
    backtest_dir: Path,
    loader,
    states: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, str]:
    """Generate per-state plots for every backtest result.

    Returns a mapping ``relative_path -> description`` for report embedding.
    """
    from mosqlimate_ai.validation.selection import select_model

    backtest_dir = Path(backtest_dir)
    frames_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    embedded: Dict[str, str] = {}

    for path in sorted(backtest_dir.glob("*_backtest.json")):
        result = json.loads(path.read_text())
        uf, disease = result["state"], result["disease"]

        selection = select_model(result)
        selected = selection.get("selected") or "ens_qavg"

        key = disease
        if key not in frames_cache:
            try:
                frames_cache[key] = loader.load_all_states(aggregate=True, disease=disease)
            except FileNotFoundError:
                logger.warning("no %s data for plots", disease)
                continue
        state_df = frames_cache[key].get(uf)
        if state_df is None:
            continue

        rel = f"plots/{disease}/{uf}.png"
        out_path = backtest_dir / rel
        try:
            if plot_state_panels(
                uf, disease, result, selected, state_df, out_path, backtest_dir
            ):
                embedded[f"{uf}/{disease}"] = rel
        except Exception:
            logger.exception("plot failed for %s/%s", uf, disease)

    return embedded
