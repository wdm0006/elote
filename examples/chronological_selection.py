"""Select on development periods, then evaluate once on later periods, offline."""

from datetime import datetime, timedelta, timezone

from elote import EloCompetitor, PythagoreanCompetitor, group_by_period, tune, walk_forward


INITIAL_WARMUP = 1
DEVELOPMENT_PERIODS = 4
SCORE_KEYS = ("home_points", "away_points")
SYSTEMS = {
    "Elo": (EloCompetitor, {"k_factor": [16, 32]}),
    "Pythagorean": (PythagoreanCompetitor, {"exponent": [1.5, 2.37]}),
}


def scored_schedule():
    """Six weekly periods; the last two are reserved before any search."""
    games = [
        [("A", "B", 24, 14), ("C", "D", 21, 17)],
        [("A", "C", 28, 20), ("B", "D", 17, 10)],
        [("A", "D", 21, 14), ("B", "C", 14, 14)],
        [("A", "B", 20, 17), ("C", "D", 24, 10)],
        [("A", "C", 17, 24), ("B", "D", 21, 21), ("E", "A", 10, 20)],
        [("A", "D", 28, 14), ("B", "C", 17, 20), ("E", "B", 14, 21)],
    ]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        (
            a,
            b,
            1.0 if home > away else 0.0 if home < away else 0.5,
            start + timedelta(weeks=week),
            {SCORE_KEYS[0]: home, SCORE_KEYS[1]: away},
        )
        for week, period in enumerate(games)
        for a, b, home, away in period
    ]


def select_and_evaluate(rows):
    """Return development candidates, frozen choice, and a separate final report."""
    periods = group_by_period(rows)
    development = periods[:DEVELOPMENT_PERIODS]
    candidates = []
    for name, (system, grid) in SYSTEMS.items():
        results = tune(
            system,
            grid,
            development,
            metric="log_loss",
            warmup=INITIAL_WARMUP,
            score_keys=SCORE_KEYS,
        )
        candidates.extend((name, result) for result in results)
    candidates.sort(key=lambda candidate: candidate[1].report.log_loss)
    name, selected = candidates[0]
    system = SYSTEMS[name][0]
    final_report = walk_forward(
        system,
        periods,
        competitor_params=dict(selected.params),
        warmup=len(development),
        score_keys=SCORE_KEYS,
    )
    return candidates, name, selected, final_report


def print_report(label, report, *, boundary, warmup):
    print(
        f"{label}: protocol=predict-then-learn; {boundary}; warmup={warmup}; "
        f"predictions={report.predictions}, skipped={report.skipped}, draws={report.draws}; "
        f"log_loss={report.log_loss:.6f}, brier={report.brier:.6f}, accuracy={report.accuracy:.6f}"
    )


def main():
    rows = scored_schedule()
    periods = group_by_period(rows)
    for index, period in enumerate(periods):
        print(f"Period {index}: {period[0][3].date()} through {period[-1][3].date()}, rows={len(period)}")
    candidates, name, selected, final_report = select_and_evaluate(rows)
    for candidate_name, candidate in candidates:
        print_report(
            f"Development selection {candidate_name} {candidate.params}",
            candidate.report,
            boundary="fit period 0; score periods 1-3",
            warmup=INITIAL_WARMUP,
        )
    print(f"Frozen choice: {name} {selected.params} (minimum development log loss)")
    print_report(
        "Final evaluation",
        final_report,
        boundary="replay periods 0-3; score untouched periods 4-5",
        warmup=DEVELOPMENT_PERIODS,
    )
    print("Parameters stay fixed; ratings learn each holdout period after all its predictions.")


if __name__ == "__main__":
    main()
