#!/usr/bin/env python3
"""Compare every rating system elote ships, on one interface, on one split.

This script makes the comparison cheap enough to argue with: one dataset
generator, one training loop, one evaluation loop, one metrics block, and a
CSV you can regenerate. It does not crown a winner.

Usage:
    python scripts/rating_system_comparison.py
    python scripts/rating_system_comparison.py --repeats 3 --with-football
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import elote
from elote import LambdaArena
from elote.datasets import SyntheticDataset
from elote.datasets.utils import evaluate_arena_with_dataset, train_arena_with_dataset

# One seed for the whole run. Change it here and every number below moves together.
SEED = 20260809

# Every incremental system starts from the same rating so the comparison is not
# secretly a comparison of default starting points. Global-fit systems keep their
# own scale, because forcing an Elo-shaped start on them is meaningless.
INITIAL_RATING = 1500

TEST_RATIO = 0.3

# Fixed prediction thresholds. Threshold optimization is deliberately not used:
# it is not reproducible on every released version of the library.
LOWER_THRESHOLD = 0.45
UPPER_THRESHOLD = 0.55

# Wall-clock budget per (scenario, system) cell. A cell over budget is recorded
# as excluded rather than dropped silently.
BUDGET_SECONDS = 300.0

SCENARIOS: List[Dict[str, Any]] = [
    {
        "key": "balanced",
        "note": "default noise, few draws",
        "kwargs": {"num_competitors": 30, "num_matchups": 600, "seed": SEED},
    },
    {
        "key": "draw_heavy",
        "note": "high noise, many draws",
        "kwargs": {
            "num_competitors": 30,
            "num_matchups": 600,
            "seed": SEED,
            "draw_probability": 0.5,
            "noise_std": 200.0,
        },
    },
    {
        "key": "sparse",
        "note": "many competitors, few games each",
        "kwargs": {"num_competitors": 120, "num_matchups": 600, "seed": SEED},
    },
]

# The concrete competitors the library exports, in the order it exports them.
# BlendedCompetitor is excluded: it is an ensemble and needs an explicit child
# configuration, so it is not comparable to a single system out of the box.
# GlickoBoostCompetitor is excluded too: its update is defined over a rating
# period, and this harness trains bout by bout, so it would measure that
# system's one-game fallback rather than the system.
CANDIDATE_SYSTEMS: Tuple[str, ...] = (
    "EloCompetitor",
    "GlickoCompetitor",
    "Glicko2Competitor",
    "TrueSkillCompetitor",
    "ECFCompetitor",
    "DWZCompetitor",
    "ColleyMatrixCompetitor",
    "MasseyCompetitor",
    "KeenerCompetitor",
    "PythagoreanCompetitor",
    "BradleyTerryCompetitor",
    "WholeHistoryRatingCompetitor",
)


# The one real dataset the library can fetch without an API key. It is opt-in
# because it needs the `datasets` extra and a network round trip.
FOOTBALL_SEASON = 2021


def always_true(a: Any, b: Any, attributes: Optional[Dict[str, Any]] = None) -> bool:
    """Comparison function for the dataset helpers.

    The helpers read the outcome from the dataset row and encode it by argument
    order, so this function must simply agree every time. It is not a model.
    """
    return True


def actual_score(outcome: Any) -> Optional[float]:
    """Map a bout outcome onto the expected-score scale: win 1, draw 0.5, loss 0."""
    if isinstance(outcome, str):
        lowered = outcome.lower()
        if lowered in ("win", "won", "a", "1"):
            return 1.0
        if lowered in ("loss", "lost", "b", "0"):
            return 0.0
        if lowered in ("draw", "tie", "tied", "0.5"):
            return 0.5
        return None
    if isinstance(outcome, (int, float)):
        if outcome in (0.0, 0.5, 1.0):
            return float(outcome)
    return None


def is_global_fit(cls: type) -> bool:
    """Global-fit systems refit from the whole match graph and own their scale."""
    return hasattr(cls, "_recalculate_ratings")


def competitor_kwargs(cls: type) -> Dict[str, Any]:
    if is_global_fit(cls):
        return {}
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    if "initial_rating" in parameters:
        return {"initial_rating": INITIAL_RATING}
    return {}


def scoring(bouts: List[Any]) -> Dict[str, Any]:
    """Proper scores plus the diagnostics that say whether the scores mean anything."""
    squared_error = 0.0
    log_score = 0.0
    scored = 0
    out_of_range = 0
    probability_sum = 0.0
    for bout in bouts:
        predicted = bout.predicted_outcome
        observed = actual_score(bout.outcome)
        if predicted is None or observed is None:
            continue
        scored += 1
        probability_sum += predicted
        if predicted < 0.0 or predicted > 1.0:
            out_of_range += 1
        squared_error += (predicted - observed) ** 2
        clipped = min(max(predicted, 1e-12), 1.0 - 1e-12)
        log_score -= observed * math.log(clipped) + (1.0 - observed) * math.log(1.0 - clipped)
    if scored == 0:
        return {"scored_bouts": 0, "brier": "", "log_loss": "", "mean_prediction": "", "out_of_range": 0}
    return {
        "scored_bouts": scored,
        "brier": squared_error / scored,
        "log_loss": log_score / scored,
        "mean_prediction": probability_sum / scored,
        "out_of_range": out_of_range,
    }


def run_cell(cls: type, split: Any) -> Dict[str, Any]:
    arena = LambdaArena(always_true, base_competitor=cls, base_competitor_kwargs=competitor_kwargs(cls))
    started = time.perf_counter()
    train_arena_with_dataset(arena, split.train)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    history = evaluate_arena_with_dataset(arena, split.test)
    predict_seconds = time.perf_counter() - started

    metrics = history.calculate_metrics(lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD)
    row: Dict[str, Any] = {
        "status": "ok",
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "accuracy": metrics.get("accuracy", ""),
        "eval_bouts": len(history.bouts),
    }
    row.update(scoring(history.bouts))
    return row


def elote_version() -> str:
    try:
        from importlib.metadata import version

        return version("elote")
    except Exception:  # reporting the version must never break the run
        return "unknown"


def environment() -> Dict[str, str]:
    return {
        "elote_version": elote_version(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "seed": str(SEED),
        "initial_rating": str(INITIAL_RATING),
        "test_ratio": str(TEST_RATIO),
        "thresholds": f"{LOWER_THRESHOLD}/{UPPER_THRESHOLD}",
        "budget_seconds": str(BUDGET_SECONDS),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1, help="repeat the whole run this many times")
    parser.add_argument("--out", default="rating_system_comparison.csv", help="CSV output path")
    parser.add_argument("--manifest", default="rating_system_comparison_env.json", help="environment manifest path")
    parser.add_argument(
        "--with-football",
        action="store_true",
        help=f"also run the shipped college football adapter for the {FOOTBALL_SEASON} season",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated system names to leave out (recorded in the manifest)",
    )
    args = parser.parse_args(argv)

    skipped_by_request = [name for name in args.skip.split(",") if name]
    scenarios = list(SCENARIOS)
    if args.with_football:
        scenarios.append(
            {
                "key": "football_%d" % FOOTBALL_SEASON,
                "note": "shipped college football adapter, one season",
                "dataset": "football",
            }
        )

    available: List[Tuple[str, type]] = []
    missing: List[str] = []
    for name in CANDIDATE_SYSTEMS:
        cls = getattr(elote, name, None)
        if cls is None:
            missing.append(name)
        elif name in skipped_by_request:
            missing.append(name)
        else:
            available.append((name, cls))

    env = environment()
    print(json.dumps(env, indent=2))
    if missing:
        print(f"not run (absent from this version or skipped by request): {', '.join(missing)}")

    fieldnames = [
        "run",
        "scenario",
        "system",
        "status",
        "train_rows",
        "test_rows",
        "train_draws",
        "test_draws",
        "eval_bouts",
        "scored_bouts",
        "accuracy",
        "brier",
        "log_loss",
        "mean_prediction",
        "out_of_range",
        "train_seconds",
        "predict_seconds",
        "detail",
    ]
    rows: List[Dict[str, Any]] = []
    over_budget: Dict[Tuple[str, str], bool] = {}

    for run_index in range(1, args.repeats + 1):
        for scenario in scenarios:
            if scenario.get("dataset") == "football":
                from elote.datasets import CollegeFootballDataset

                dataset: Any = CollegeFootballDataset(start_year=FOOTBALL_SEASON, end_year=FOOTBALL_SEASON)
            else:
                dataset = SyntheticDataset(**scenario["kwargs"])
            split = dataset.time_split(test_ratio=TEST_RATIO)
            train_draws = sum(1 for row in split.train if row[2] == 0.5)
            test_draws = sum(1 for row in split.test if row[2] == 0.5)
            for name, cls in available:
                base = {
                    "run": run_index,
                    "scenario": scenario["key"],
                    "system": name,
                    "train_rows": len(split.train),
                    "test_rows": len(split.test),
                    "train_draws": train_draws,
                    "test_draws": test_draws,
                    "detail": "",
                }
                if over_budget.get((scenario["key"], name)):
                    base["status"] = "excluded_over_budget"
                    base["detail"] = f"exceeded {BUDGET_SECONDS:.0f}s on an earlier run"
                    rows.append(base)
                    print(f"{run_index} {scenario['key']:<11} {name:<24} excluded (over budget)")
                    continue
                try:
                    cell = run_cell(cls, split)
                except Exception as exc:  # a crash is a reportable result, not a reason to stop
                    base["status"] = "error"
                    base["detail"] = f"{type(exc).__name__}: {exc}"
                    rows.append(base)
                    print(f"{run_index} {scenario['key']:<11} {name:<24} ERROR {type(exc).__name__}: {exc}")
                    continue
                base.update(cell)
                elapsed = cell["train_seconds"] + cell["predict_seconds"]
                if elapsed > BUDGET_SECONDS:
                    over_budget[(scenario["key"], name)] = True
                    base["detail"] = f"took {elapsed:.1f}s, over the {BUDGET_SECONDS:.0f}s budget"
                rows.append(base)
                print(
                    f"{run_index} {scenario['key']:<11} {name:<24} "
                    f"acc={cell['accuracy']:.4f} brier={cell['brier']:.4f} "
                    f"logloss={cell['log_loss']:.4f} oor={cell['out_of_range']} "
                    f"train={cell['train_seconds']:.3f}s predict={cell['predict_seconds']:.3f}s"
                )

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(args.manifest, "w") as handle:
        json.dump({"environment": env, "scenarios": scenarios, "skipped_systems": missing}, handle, indent=2)

    print(f"\nwrote {len(rows)} rows to {args.out} and the environment to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
