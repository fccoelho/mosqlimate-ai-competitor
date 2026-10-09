"""Submit IMDC predictions to the Mosqlimate platform.

Builds payloads from the deployment-recipe models and uploads them via
the authenticated registry API:

- tests 1-4: the validated backtest forecast CSVs
  (``validation_results/backtest/forecasts/{uf}_{disease}_{test}_<model>.csv.gz``)
- final: the freshly generated forecasts (``forecasts/final/<disease>/<UF>.csv``,
  from ``scripts/generate_forecast.py``)

Model choice follows ``validation_results/backtest/deployment_recipe.json``
(chikungunya -> univariate TimesFM, dengue -> ens_qavg). Payloads use
the current platform schema (repository + disease identify the model;
one registration per repository on the web UI, prediction rows carry
the nine interval bounds).

Usage::

    MOSQLIMATE_API="username:key" python scripts/submit_to_mosqlimate.py --dry-run
    MOSQLIMATE_API="username:key" python scripts/submit_to_mosqlimate.py
    python scripts/submit_to_mosqlimate.py --split final --states SP,RJ
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

from mosqlimate_ai.evaluation.quantiles import quantiles_to_intervals
from mosqlimate_ai.submission.imdc import REQUIRED_INTERVALS
from mosqlimate_ai.validation.config import get_validation_config

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.mosqlimate.org/api/registry/predictions/"
REPOSITORY = "fccoelho/mosqlimate-ai-competitor"
DISEASE_CODES = {"dengue": "A90", "chikungunya": "A92.0"}
SPLITS = ("test1", "test2", "test3", "test4", "final")


def uf_geocodes() -> dict[str, int]:
    df = pd.read_csv(ROOT / "data" / "map_regional_health.csv")
    return df.drop_duplicates("uf_code").set_index("uf")["uf_code"].to_dict()


def git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def load_rows(fc_path: Path) -> list[dict]:
    """Quantile CSV -> platform prediction rows (pred + 8 bounds)."""
    df = pd.read_csv(fc_path)
    df["date"] = pd.to_datetime(df["date"])
    if "median" not in df.columns:
        df = quantiles_to_intervals(df)
    df = df.sort_values("date").reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        row = {"date": r["date"].strftime("%Y-%m-%d"), "pred": round(float(r["median"]), 2)}
        for lower, upper in REQUIRED_INTERVALS:
            row[lower] = round(float(r[lower]), 2)
            row[upper] = round(float(r[upper]), 2)
        rows.append(row)
    return rows


def validate_rows(rows: list[dict], label: str) -> list[str]:
    issues = []
    # platform test-4 window is inclusive of both ends: 53 weekly dates
    expected = 53 if "test4" in label else 52
    if len(rows) != expected:
        issues.append(f"{label}: {len(rows)} weeks (expected {expected})")
    for i, r in enumerate(rows):
        if r["lower_95"] > r["lower_90"] or r["lower_90"] > r["lower_80"]:
            issues.append(f"{label}: non-monotone lower bounds at week {i}")
            break
        if not (r["lower_50"] <= r["pred"] <= r["upper_50"]):
            issues.append(f"{label}: median outside 50% interval at week {i}")
            break
        if r["pred"] < 0:
            issues.append(f"{label}: negative pred at week {i}")
            break
    return issues


def build_payloads(args, recipe: dict, geocodes: dict, commit: str, repository: str) -> list[dict]:
    cfg = get_validation_config()
    tests = {f"test{t.test_number}": t for t in cfg.validation_tests}
    states = (
        [s.strip().upper() for s in args.states.split(",")] if args.states else list(cfg.states)
    )
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else list(SPLITS)

    fc_dir = ROOT / "validation_results" / "backtest" / "forecasts"
    final_dir = ROOT / args.final_dir
    diseases = (
        tuple(d.strip() for d in args.diseases.split(",") if d.strip())
        if args.diseases
        else ("dengue", "chikungunya")
    )
    payloads, skipped = [], []
    for disease in diseases:
        model = (recipe.get(disease) or {}).get("model")
        if not model:
            skipped.append(f"{disease}: no recipe model")
            continue
        for split in splits:
            for uf in states:
                if split == "final":
                    fc_path = final_dir / disease / f"{uf}.csv"
                    predict_date = dt.date.today().isoformat()
                else:
                    test = tests.get(split)
                    if test is None:
                        skipped.append(f"unknown split {split}")
                        continue
                    test_key = split.replace("test", "")
                    fc_path = fc_dir / f"{uf}_{disease}_{test_key}_{model}.csv.gz"
                    predict_date = test.train_end
                if not fc_path.exists():
                    skipped.append(f"{disease}/{split}/{uf}: missing {fc_path.name}")
                    continue
                rows = load_rows(fc_path)
                payloads.append(
                    {
                        "repository": repository,
                        "disease": DISEASE_CODES[disease],
                        "description": f"{disease} {split} {uf} ({model})",
                        "commit": commit,
                        "case_definition": "probable",
                        "published": True,
                        "adm_level": 1,
                        "adm_0": "BRA",
                        "adm_1": int(geocodes[uf]),
                        "adm_2": None,
                        "adm_3": None,
                        "predict_date": predict_date,
                        "prediction": rows,
                        "_label": f"{disease}/{split}/{uf}",
                    }
                )
    return payloads, skipped


def existing_prediction_keys(session, headers) -> set[tuple]:
    """(disease, adm_1) pairs already on the platform for our repository."""
    keys = set()
    page = 1
    while True:
        resp = session.get(
            "https://api.mosqlimate.org/api/registry/predictions/",
            params={"repository": repository, "page": page, "per_page": 100},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        items = body["items"] if isinstance(body, dict) else body
        for it in items:
            keys.add((it.get("disease"), int(it["adm_1"])))
        total = body.get("pagination", {}).get("total_pages", 1) if isinstance(body, dict) else 1
        if page >= total:
            break
        page += 1
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit IMDC predictions to Mosqlimate.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate only")
    parser.add_argument("--splits", help="Comma-separated subset of test1..4,final")
    parser.add_argument(
        "--diseases",
        help="Comma-separated subset of dengue,chikungunya (default: both)",
    )
    parser.add_argument("--states", help="Comma-separated UF list (default: all 26)")
    parser.add_argument("--final-dir", default="forecasts/final")
    parser.add_argument(
        "--repository",
        default=REPOSITORY,
        help="Registered 'owner/name' repository on Mosqlimate (default: %(default)s)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing predictions for the repository before uploading",
    )
    args = parser.parse_args()
    repository = args.repository

    api_key = os.environ.get("MOSQLIMATE_API") or os.environ.get("MOSQLIMATE_TOKEN")
    if not api_key:
        sys.exit("set MOSQLIMATE_API (format 'username:key')")

    recipe_path = ROOT / "validation_results" / "backtest" / "deployment_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    geocodes = uf_geocodes()
    commit = git_commit()

    payloads, skipped = build_payloads(args, recipe, geocodes, commit, repository)
    print(f"built {len(payloads)} payloads (commit {commit[:8]})")
    for s in skipped[:20]:
        print("  skip:", s)

    issues = []
    for p in payloads:
        issues += validate_rows(p["prediction"], p["_label"])
    if issues:
        print(f"VALIDATION ISSUES ({len(issues)}):")
        for i in issues[:20]:
            print(" -", i)
        sys.exit(1)
    print("all payloads valid: 52 weeks, monotone bounds, non-negative")

    if args.dry_run:
        print("dry-run complete; no uploads performed")
        return

    session = requests.Session()
    headers = {"X-UID-Key": api_key, "Content-Type": "application/json"}

    if args.replace:
        resp = session.get(
            "https://api.mosqlimate.org/api/registry/predictions/",
            params={"repository": repository, "per_page": 100},
            headers=headers,
            timeout=60,
        )
        body = resp.json()
        for it in body["items"] if isinstance(body, dict) else body:
            session.delete(
                f"https://api.mosqlimate.org/api/registry/predictions/{it['id']}/",
                headers=headers,
                timeout=60,
            )
            print(f"  deleted existing prediction {it['id']} ({it.get('disease')}/{it['adm_1']})")

    done, failed = 0, []
    for p in payloads:
        body = {k: v for k, v in p.items() if not k.startswith("_")}
        try:
            resp = session.post(API_URL, json=body, headers=headers, timeout=120)
            if resp.status_code == 201:
                done += 1
                print(f"  ok {p['_label']} (id {resp.json().get('id')})")
            else:
                failed.append((p["_label"], f"{resp.status_code}: {resp.text[:200]}"))
                print(f"  FAIL {p['_label']}: {resp.status_code} {resp.text[:200]}")
        except requests.RequestException as exc:
            failed.append((p["_label"], str(exc)))
            print(f"  ERROR {p['_label']}: {exc}")
        time.sleep(0.5)

    print(f"\nuploaded {done}/{len(payloads)} predictions")
    if failed:
        print(f"{len(failed)} failures")
        sys.exit(1)


if __name__ == "__main__":
    main()
