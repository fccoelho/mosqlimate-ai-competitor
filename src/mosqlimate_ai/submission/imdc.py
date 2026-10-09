"""IMDC submission package builder.

Converts backtest/forecast outputs (canonical quantile frames per state
× disease × split) into Mosqlimate API submission payloads and validates
competition completeness:

- all 26 UFs (ES excluded) present,
- 52 weekly target dates per state/split,
- median + all four intervals (50/80/90/95%) present,
- monotone intervals containing the median,
- non-negative values.

The mandatory challenge is dengue state-level; the optional chikungunya
state-level challenge is handled identically (submissions carry the
disease in the description and are saved in disease-specific folders).
"""

from __future__ import annotations

from typing import Dict, List, Optional  # noqa: F401 - used in annotations

import json
import logging
from pathlib import Path

import pandas as pd

from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.validation.config import get_validation_config

logger = logging.getLogger(__name__)

REQUIRED_INTERVALS = [
    ("lower_50", "upper_50"),
    ("lower_80", "upper_80"),
    ("lower_90", "upper_90"),
    ("lower_95", "upper_95"),
]


def quantile_frame_to_forecast_df(quantile_frame: pd.DataFrame) -> pd.DataFrame:
    """Date-indexed quantile frame -> submission forecast DataFrame."""
    df = quantile_frame.reset_index(names="date")
    return quantiles_to_intervals(df).reset_index(drop=True)


class IMDCSubmissionBuilder:
    """Build and validate a complete IMDC submission package.

    Args:
        model_id: Mosqlimate model ID (per disease — register one model
            for dengue and one for chikungunya).
        description: Prediction description prefix.
        commit: Git commit of the generating code.
        predict_date: ISO date of the prediction run.
    """

    def __init__(
        self,
        model_id: int | None = None,
        description: str = "mosqlimate-ai-competitor",
        commit: str | None = None,
        predict_date: str | None = None,
    ):
        import datetime as _dt

        self.model_id = model_id
        self.description = description
        self.commit = commit
        self.predict_date = predict_date or _dt.date.today().isoformat()
        self.payloads: list[dict] = []

    # ------------------------------------------------------------------
    def add_forecast(
        self,
        uf: str,
        forecast_df: pd.DataFrame,
        split: str,
        disease: str = "dengue",
    ) -> Dict:
        """Add one state's 52-week forecast for a split (test1..4/final).

        Accepts either submission-style interval columns or canonical
        quantile columns (``q025``..``q975``).
        """
        df = forecast_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if "median" not in df.columns:
            from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals

            df = quantiles_to_intervals(df)
        df = df.sort_values("date").reset_index(drop=True)

        required = ["date", "median"] + [c for pair in REQUIRED_INTERVALS for c in pair]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{uf}/{disease}/{split}: missing columns {missing}")

        # enforce non-negativity and containment lower <= median <= upper
        num_cols = [c for c in required if c != "date"]
        df[num_cols] = df[num_cols].clip(lower=0.0)
        for lower, upper in REQUIRED_INTERVALS:
            df["median"] = df["median"].clip(lower=df[lower], upper=df[upper])

        prediction = {
            "dates": [d.strftime("%Y-%m-%d") for d in df["date"]],
            "preds": df["median"].round(2).tolist(),
        }
        for lower, upper in REQUIRED_INTERVALS:
            prediction[lower] = df[lower].round(2).tolist()
            prediction[upper] = df[upper].round(2).tolist()

        payload = {
            "model": self.model_id,
            "description": f"{self.description} {disease} {split}",
            "commit": self.commit,
            "predict_date": self.predict_date,
            "adm_0": "BRA",
            "adm_1": uf,
            "adm_2": None,
            "adm_3": None,
            "prediction": prediction,
        }
        self.payloads.append(payload)
        return payload

    # ------------------------------------------------------------------
    def save(self, out_dir: Path) -> list[Path]:
        """Save payloads grouped as <disease>/<split>/<UF>.json."""
        out_dir = Path(out_dir)
        saved = []
        for payload in self.payloads:
            disease = payload["description"].split()[1]
            split = payload["description"].split()[2]
            uf = payload["adm_1"] or "BRA"
            path = out_dir / disease / split / f"{uf}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
            saved.append(path)
        return saved

    # ------------------------------------------------------------------
    def validate_completeness(
        self,
        expected_states: list[str] | None = None,
        expected_splits: list[str] | None = None,
        n_weeks: int = 52,
    ) -> list[str]:
        """Return a list of completeness problems (empty = ready)."""
        cfg = get_validation_config()
        expected_states = expected_states or cfg.states
        expected_splits = expected_splits or [f"test{i}" for i in range(1, 5)] + ["final"]

        issues: list[str] = []
        seen: dict[tuple, dict] = {}
        for payload in self.payloads:
            disease = payload["description"].split()[1]
            split = payload["description"].split()[2]
            uf = payload["adm_1"]
            seen[(disease, split, uf)] = payload

            pred = payload["prediction"]
            if len(pred["dates"]) != n_weeks:
                issues.append(f"{disease}/{split}/{uf}: {len(pred['dates'])} weeks (expected {n_weeks})")
            if len(pred["dates"]) != len(pred["preds"]):
                issues.append(f"{disease}/{split}/{uf}: dates/preds length mismatch")
            for lower, upper in REQUIRED_INTERVALS:
                if lower not in pred or upper not in pred:
                    issues.append(f"{disease}/{split}/{uf}: missing interval {lower}/{upper}")
                else:
                    bad = [
                        j
                        for j in range(len(pred["preds"]))
                        if pred[lower][j] > pred["preds"][j] or pred["preds"][j] > pred[upper][j]
                    ]
                    if bad:
                        issues.append(
                            f"{disease}/{split}/{uf}: median outside {lower}-{upper} at weeks {bad[:5]}"
                        )
            neg = [j for j, v in enumerate(pred["preds"]) if v < 0]
            if neg:
                issues.append(f"{disease}/{split}/{uf}: negative preds at weeks {neg[:5]}")

        diseases = sorted({k[0] for k in seen})
        for disease in diseases:
            for split in expected_splits:
                for uf in expected_states:
                    if (disease, split, uf) not in seen:
                        issues.append(f"missing: {disease}/{split}/{uf}")
        return issues
