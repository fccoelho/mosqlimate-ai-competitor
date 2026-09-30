"""Driver for the full IMDC backtest run (all states, both diseases)."""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from mosqlimate_ai.validation.backtest import run_full_pipeline


def main() -> None:
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    states = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    print(f"Running backtests with max_workers={max_workers}, states={states or 'ALL'}", flush=True)
    combined = run_full_pipeline(
        states=states,
        diseases=("dengue", "chikungunya"),
        include_final=True,
        max_workers=max_workers,
        out_dir="validation_results/backtest",
    )
    print(combined.groupby(["disease", "model"])[["wis", "coverage_50", "coverage_95"]]
          .mean()
          .round(2)
          .to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
