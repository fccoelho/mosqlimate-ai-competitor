"""Download the 3rd IMDC procc baseline predictions from Mosqlimate.

Fetches every state-level prediction of the reference model
``3rd_imdc_procc_bb_model`` (lsbastos/lsbastos repo, model id 85) from
api.mosqlimate.org and converts them into the project's forecast CSV
format (``date,q025..q975``) inside the backtest forecasts directory,
so the baseline can be scored and plotted alongside the local model
zoo under the name ``imdc_bb``.

Quantile mapping (mosqlimate -> project):

    lower_95 -> q025   lower_90 -> q050   lower_80 -> q100
    lower_50 -> q250   pred     -> q500   upper_50 -> q750
    upper_80 -> q900   upper_90 -> q950   upper_95 -> q975

Season mapping (prediction start date -> validation test):

    2022-10-09 -> 1     2023-10-08 -> 2     2024-10-06 -> 3
    2025-10-05 -> 4     2026-10-11 -> final (EW41 2026, 2026-2027)

Auth/API compatibility: the API requires the ``X-UID-Key`` header on
every request and only embeds the weekly ``data`` rows on the
per-prediction detail endpoint.

- With mosqlient >= 1.9 (2.x tested) the script uses the library's
  native authenticated ``get_predictions(api_key, ...)``.
- With mosqlient 1.8.x (which predates both API changes and cannot be
  locked alongside karldbot's numpy<2 / loguru<0.7 pins), the script
  monkeypatches mosqlient's sync GET helper to send the header and
  fetches the paginated list + per-id details through the library's
  HTTP layer.

Usage::

    MOSQLIMATE_API="username:key" \\
        python scripts/download_baseline_predictions.py

    python scripts/download_baseline_predictions.py \\
        --output-dir validation_results/backtest/forecasts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

MODEL_NAME = "3rd_imdc_procc_bb_model"
LOCAL_MODEL = "imdc_bb"

DISEASES = {"A90": "dengue", "A92.0": "chikungunya"}

QUANTILE_COLUMNS = {
    "lower_95": "q025",
    "lower_90": "q050",
    "lower_80": "q100",
    "lower_50": "q250",
    "pred": "q500",
    "upper_50": "q750",
    "upper_80": "q900",
    "upper_90": "q950",
    "upper_95": "q975",
}

TEST_BY_START = {
    "2022-10-09": "1",  # test 1: 2022-2023 season
    "2023-10-08": "2",  # test 2: 2023-2024 season
    "2024-10-06": "3",  # test 3: 2024-2025 season
    "2025-10-05": "4",  # test 4: 2025-2026 season
    "2026-10-11": "final",  # final forecast: 2026-2027 season (EW41 2026)
}

ROOT = Path(__file__).resolve().parents[1]


def _fetch_native(api_key: str, max_workers: int = 8) -> list[dict] | None:
    """mosqlient >= 1.9: authenticated GETs via the library itself."""
    import inspect

    from mosqlient import get_predictions

    if "api_key" not in inspect.signature(get_predictions).parameters:
        return None
    res = get_predictions(api_key, model_name=MODEL_NAME)
    preds = res if isinstance(res, list) else [res]

    # Prediction.data is a lazy per-prediction fetch (~1s each): parallelize
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        datas = list(pool.map(lambda p: p.data, preds))

    return [
        {
            "id": p.id,
            "disease": p.disease,
            "adm_1": p.adm_1,
            "adm_2": p.adm_2,
            "start": str(p.start),
            "commit": p.commit,
            "model": {"id": p.model.id, "repository": p.model.repository},
            "data": [row.model_dump() for row in data],
            "scores": p.scores,
        }
        for p, data in zip(preds, datas)
    ]


def _fetch_with_patched_1_8_x(api_key: str, max_workers: int = 8) -> list[dict]:
    """mosqlient 1.8.x: patched HTTP layer (auth header + detail fetches)."""
    import mosqlient.requests as mreq
    from mosqlient._config import get_api_url

    def get(app, endpoint, params, pagination=True, timeout=60, _retries=3):
        url = urljoin(get_api_url(), "/".join((str(app), str(endpoint)))) + "/?"
        for attempt in range(_retries):
            try:
                return requests.get(url, params, timeout=timeout, headers={"X-UID-Key": api_key})
            except requests.RequestException:
                if attempt == _retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    mreq.get = get

    items, page = [], 1
    while True:
        body = mreq.get(
            "registry",
            "predictions",
            {"model_name": MODEL_NAME, "page": page, "per_page": 100},
            pagination=True,
        ).json()
        items.extend(body["items"])
        if page >= body["pagination"]["total_pages"]:
            break
        page += 1

    # the list endpoint omits the weekly data rows: fetch details in parallel
    details = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(mreq.get, "registry", f"predictions/{item['id']}", {}, pagination=False): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            details[futures[future]] = future.result().json()
    return [d for d in details if d]


def fetch_model_predictions(api_key: str) -> list[dict]:
    """All predictions of MODEL_NAME (with data rows), via the mosqlient library."""
    native = _fetch_native(api_key)
    if native is not None:
        return native
    return _fetch_with_patched_1_8_x(api_key)


def uf_codes() -> dict[int, str]:
    df = pd.read_csv(ROOT / "data" / "map_regional_health.csv")
    return df.drop_duplicates("uf_code").set_index("uf_code")["uf"].to_dict()


def prediction_to_frame(pred: dict) -> pd.DataFrame:
    rows = [
        {"date": r["date"], **{dst: r.get(src) for src, dst in QUANTILE_COLUMNS.items()}}
        for r in pred["data"]
    ]
    return pd.DataFrame(rows, columns=["date", *QUANTILE_COLUMNS.values()]).sort_values("date")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 3rd IMDC procc baseline predictions as forecast CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "validation_results" / "backtest" / "forecasts",
        help=f"Directory for the {{uf}}_{{disease}}_{{test}}_{LOCAL_MODEL}.csv.gz files",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MOSQLIMATE_API") or os.environ.get("MOSQLIMATE_TOKEN")
    if not api_key:
        parser.error("set MOSQLIMATE_API (format 'username:key') to access the Mosqlimate API")

    from mosqlimate_ai.validation.config import DEFAULT_VALIDATION_CONFIG

    states = set(DEFAULT_VALIDATION_CONFIG.states)
    ufs = uf_codes()

    preds = fetch_model_predictions(api_key)
    print(f"fetched {len(preds)} predictions for {MODEL_NAME!r}")

    # keep the newest upload per (uf, disease, season)
    latest: dict[tuple, dict] = {}
    skipped: list[str] = []
    for pred in preds:
        disease = DISEASES.get(pred["disease"])
        test_key = TEST_BY_START.get(str(pred["start"]))
        adm_1 = pred.get("adm_1")
        uf = ufs.get(int(adm_1)) if adm_1 else None
        if pred.get("adm_2") is not None:
            skipped.append(f"id={pred['id']}: municipal-level prediction")
            continue
        if disease is None:
            skipped.append(f"id={pred['id']}: unmapped disease {pred['disease']}")
            continue
        if test_key is None:
            skipped.append(f"id={pred['id']}: unmapped season start {pred['start']}")
            continue
        if uf is None:
            skipped.append(f"id={pred['id']}: unknown adm_1 code {adm_1}")
            continue
        if uf not in states:
            skipped.append(f"id={pred['id']}: {uf} is not a competition state")
            continue
        key = (uf, disease, test_key)
        if key not in latest or pred["id"] > latest[key]["id"]:
            latest[key] = pred

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = next(iter(latest.values()))
    manifest = {
        "source_model": {
            "name": MODEL_NAME,
            "model_id": sample["model"]["id"],
            "repository": sample["model"]["repository"],
            "commit": sample["commit"],
        },
        "local_model": LOCAL_MODEL,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "quantile_mapping": QUANTILE_COLUMNS,
        "files": {},
    }

    for (uf, disease, test_key), pred in sorted(latest.items()):
        fc = prediction_to_frame(pred)
        name = f"{uf}_{disease}_{test_key}_{LOCAL_MODEL}.csv.gz"
        fc.to_csv(args.output_dir / name, index=False)
        manifest["files"][name] = {
            "prediction_id": pred["id"],
            "rows": len(fc),
            "first_date": str(fc.date.iloc[0]),
            "last_date": str(fc.date.iloc[-1]),
            "scores": pred.get("scores"),
        }

    manifest_path = args.output_dir.parent / f"{LOCAL_MODEL}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    by_test: dict[str, int] = {}
    for _, _, test_key in latest:
        by_test[test_key] = by_test.get(test_key, 0) + 1
    print(f"wrote {len(latest)} forecast files to {args.output_dir}")
    for test_key in sorted(by_test):
        print(f"  test {test_key}: {by_test[test_key]} files")
    print(f"manifest: {manifest_path}")
    if skipped:
        print(f"skipped {len(skipped)} predictions:")
        for reason in skipped:
            print(f"  {reason}")


if __name__ == "__main__":
    main()
