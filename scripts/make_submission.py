"""Build the IMDC submission package from backtest forecasts.

Steps:
1. Run model selection per state/disease from the backtest results.
2. Load the selected model's forecast parquet for every split
   (validation tests 1-4 and the final forecast).
3. Validate the complete package (26 UFs x 52 weeks x splits) and
   save submission JSONs under submissions/.

Note on the final forecast (2026-27): it requires data through EW25-2026.
When the local cache ends earlier (e.g., 2025-06), the model still
produces the extrapolation, but it should be re-run after downloading
refreshed data.
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.submission.imdc import IMDCSubmissionBuilder
from mosqlimate_ai.validation.selection import select_model

TARGET_WEEKS = 52


def slice_target(forecast: pd.DataFrame, target_start: str, target_end: str) -> pd.DataFrame:
    """52 weekly target dates from target_start (end of season)."""
    idx = pd.to_datetime(forecast["date"])
    mask = (idx >= pd.Timestamp(target_start)) & (
        idx < pd.Timestamp(target_start) + pd.Timedelta(weeks=52)
    )
    return forecast[mask].reset_index(drop=True)


def main() -> None:
    backtest_dir = Path("validation_results/backtest")
    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    from mosqlimate_ai.validation.config import get_validation_config

    cfg = get_validation_config()
    split_dates = {f"test{t.test_number}": (t.target_start, t.target_end) for t in cfg.validation_tests}
    split_dates["final"] = (cfg.final_forecast_target_start, cfg.final_forecast_target_end)

    builder = IMDCSubmissionBuilder(model_id=model_id, description="mosqlimate-ai-competitor")
    selections = []

    result_files = sorted(backtest_dir.glob("*_backtest.json"))
    if not result_files:
        print(f"No backtest results in {backtest_dir}; run scripts/run_backtests.py first.")
        sys.exit(1)

    fc_dir = backtest_dir / "forecasts"
    for path in result_files:
        result = json.loads(path.read_text())
        state, disease = result["state"], result["disease"]
        selection = select_model(result)
        selections.append(selection)
        chosen = selection["selected"]
        print(f"{state}/{disease}: {chosen} (mean WIS {selection['mean_wis']})")

        for split, (target_start, target_end) in split_dates.items():
            test_key = "final" if split == "final" else split.replace("test", "")
            fc_path = fc_dir / f"{state}_{disease}_{test_key}_{chosen}.csv.gz"
            if not fc_path.exists():
                print(f"  WARN missing forecast {fc_path.name}; skipping {split}")
                continue
            fc = pd.read_csv(fc_path)
            # forecasts carry interval columns already
            win = slice_target(fc, target_start, target_end)
            if len(win) < TARGET_WEEKS:
                print(f"  WARN {split}: only {len(win)} weeks")
            builder.add_forecast(state, win, split, disease=disease)

    pd.DataFrame(selections).to_csv(out_dir / "model_selection.csv", index=False)
    # out_dir already created
    saved = builder.save(out_dir)
    issues = builder.validate_completeness()

    print(f"\nSaved {len(saved)} payloads to {out_dir}/")
    if issues:
        print(f"COMPLETENESS ISSUES ({len(issues)}):")
        for issue in issues[:30]:
            print(" -", issue)
    else:
        print("Package complete: all states x splits present and valid.")


if __name__ == "__main__":
    main()
