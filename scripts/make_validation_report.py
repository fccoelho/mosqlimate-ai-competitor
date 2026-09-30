"""Generate a markdown validation report from backtest results.

Summarizes per-state, per-model WIS across validation tests, skill vs
the seasonal-naive baseline, coverage quality, and the selected model
per state/disease.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mosqlimate_ai.validation.selection import BASELINE_MODEL, select_from_backtest_dir


def load_all(backtest_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(backtest_dir.glob("*_backtest.json")):
        result = json.loads(path.read_text())
        buckets = dict(result.get("tests", {}))
        if result.get("final"):
            buckets["final"] = result["final"]
        for test_key, test_res in buckets.items():
            for model_name, model_res in test_res.get("models", {}).items():
                metrics = model_res.get("metrics") or {}
                if not metrics:
                    continue
                frames.append(
                    {
                        "state": result["state"],
                        "disease": result["disease"],
                        "test": test_key,
                        "season": test_res.get("season"),
                        "model": model_name,
                        "wis": metrics.get("wis_total"),
                        "mae": metrics.get("mae"),
                        "coverage_50": metrics.get("coverage_50"),
                        "coverage_95": metrics.get("coverage_95"),
                        "n_eval_weeks": model_res.get("n_eval_weeks", 0),
                    }
                )
    return pd.DataFrame(frames)


def main() -> None:
    backtest_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "validation_results/backtest")
    out_path = backtest_dir / "VALIDATION_REPORT.md"

    df = load_all(backtest_dir)
    if df.empty:
        print(f"No backtest results in {backtest_dir}")
        sys.exit(1)

    # include every test with observed actuals (test 4 has a partial
    # 2025-26 season; the final split has none yet)
    evaluable = df[(df.n_eval_weeks > 0) & (~df.test.isin(["final", "5"]))]

    lines = [
        "# IMDC Validation Report",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M} · states: {df.state.nunique()} · "
        f"diseases: {', '.join(sorted(df.disease.unique()))}",
        "",
        "Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,",
        "computed weekly over the 52-week target season on conformally",
        "calibrated quantile forecasts.",
        "",
    ]

    for disease in sorted(df.disease.unique()):
        d = evaluable[evaluable.disease == disease]
        if d.empty:
            continue
        lines += [f"## {disease.capitalize()} — mean WIS per model (tests with actuals)", ""]
        pivot = d.pivot_table(index="model", values="wis", aggfunc="mean").sort_values("wis")
        skill = pivot.loc[BASELINE_MODEL, "wis"] if BASELINE_MODEL in pivot.index else np.nan
        pivot["skill_vs_naive"] = 1 - pivot["wis"] / skill
        cov = d.pivot_table(index="model", values="coverage_50", aggfunc="mean")
        pivot["coverage_50"] = cov["coverage_50"]
        lines += [pivot.round(2).to_markdown(), ""]

        per_state = d.pivot_table(index="state", columns="model", values="wis", aggfunc="mean")
        best = per_state.idxmin(axis=1)
        lines += [f"### Best model per state ({disease})", ""]
        counts = best.value_counts()
        lines += ["", ", ".join(f"{m}: {c}" for m, c in counts.items()), ""]

        lines += ["### Mean WIS per state and test", ""]
        pt = d.pivot_table(index="state", columns="test", values="wis", aggfunc="mean")
        lines += [pt.round(1).to_markdown(), ""]

    # selection
    selections = select_from_backtest_dir(backtest_dir)
    if not selections.empty:
        lines += ["## Selected model per state (skill-gated)", ""]
        for disease in sorted(selections.disease.dropna().unique()):
            sub = selections[selections.disease == disease]
            counts = sub.selected.value_counts()
            lines += [f"**{disease}**: " + ", ".join(f"{m}: {c}" for m, c in counts.items()), ""]
        lines += [selections[["state", "disease", "selected", "mean_wis", "fallback"]]
                  .round(1)
                  .to_markdown(index=False), ""]

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
