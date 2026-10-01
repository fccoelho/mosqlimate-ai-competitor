"""Driver for the full IMDC backtest run (all states, both diseases).

Usage:
    python scripts/run_backtests.py [MAX_WORKERS] [STATES] [TUNE_TRIALS]

TUNE_TRIALS > 0 enables per-state GBM hyperparameter tuning (results
cached under validation_results/backtest/hyperparams/).
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from mosqlimate_ai.validation.backtest import run_full_pipeline


def main() -> None:
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    states = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    tune_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print(
        f"Running backtests: max_workers={max_workers}, states={states or 'ALL'}, "
        f"tune_trials={tune_trials}",
        flush=True,
    )
    combined = run_full_pipeline(
        states=states,
        diseases=("dengue", "chikungunya"),
        include_final=True,
        max_workers=max_workers,
        out_dir="validation_results/backtest",
        tune_trials=tune_trials,
    )
    print(
        combined.groupby(["disease", "model"])[["wis", "coverage_50", "coverage_95"]]
        .mean()
        .round(2)
        .to_string(),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
