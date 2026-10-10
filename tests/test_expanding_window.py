"""Value-based expanding-window checks through the public API."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from elote import EloCompetitor, ExpandingWindowFold, GlickoBoostCompetitor, expanding_window_evaluate, walk_forward


@pytest.fixture
def periods():
    games = [
        [("A", "B", 1), ("C", "D", 0)],
        [("A", "B", 0), ("C", "D", 0.5), ("E", "A", 1)],
        [("A", "B", 1), ("C", "D", 1), ("E", "A", 0)],
        [("A", "B", 0), ("C", "D", 0.5)],
        [("A", "B", 1), ("E", "D", 0)],
        [("A", "B", 0), ("C", "D", 1)],
    ]
    return [
        [
            (
                a,
                b,
                outcome,
                datetime(2020, 1, 1) + timedelta(days=7 * i),
                {"home": 3 if outcome == 1 else 1, "away": 3 if outcome == 0 else 1},
            )
            for a, b, outcome in games_in_period
        ]
        for i, games_in_period in enumerate(games)
    ]


@pytest.mark.parametrize(
    "cls,params,kwargs",
    [
        (EloCompetitor, {"k_factor": 32}, {"initial_rating": 1200, "margin_of_victory": True}),
        (GlickoBoostCompetitor, {"eta": 30}, {"initial_rating": 1400}),
    ],
)
def test_prefix_reports_and_counts(cls, params, kwargs, periods):
    options = dict(
        competitor_params=params,
        base_competitor_kwargs=kwargs,
        score_keys=("home", "away"),
        calibration_bins=4,
        comparison_function=lambda a, b, attributes=None: True,
    )
    results = expanding_window_evaluate(cls, periods, [(1, 3), (3, 5)], **options)
    assert isinstance(results, tuple)
    for fold, bounds, counts in zip(results, [(1, 3), (3, 5)], [(4, 1, 1), (3, 0, 1)], strict=True):
        assert isinstance(fold, ExpandingWindowFold)
        assert (fold.train_end, fold.test_end) == bounds
        assert (fold.training_periods, fold.scoring_periods) == (bounds[0], 2)
        assert fold.report == walk_forward(cls, periods[: bounds[1]], warmup=bounds[0], **options)
        assert (fold.report.predictions, fold.report.skipped, fold.report.draws) == counts
        assert [row[0] for row in fold.report.by_period] == list(range(*bounds))
        assert sum(bin.count for bin in fold.report.reliability) == counts[0]
        with pytest.raises(FrozenInstanceError):
            fold.train_end = 9
    assert results[0] == expanding_window_evaluate(cls, periods, [(1, 3)], **options)[0]


@pytest.mark.parametrize("cls", [EloCompetitor, GlickoBoostCompetitor])
def test_temporal_dependence(cls, periods):
    original = expanding_window_evaluate(cls, periods, [(1, 3), (3, 5)])
    changed = [list(rows) for rows in periods]
    a, b, _, when, _ = changed[2][0]
    changed[2][0] = (a, b, 0, when, None)
    replay = expanding_window_evaluate(cls, changed, [(1, 3), (3, 5)])
    assert original[1].report.log_loss != replay[1].report.log_loss
    changed[4] = [("new", "opponent", 1, datetime(2021, 1, 1), None)]
    assert replay[0] == expanding_window_evaluate(cls, changed, [(1, 3), (3, 5)])[0]


@pytest.mark.parametrize(
    "folds",
    [
        [],
        [(0, 2)],
        [(1, 1)],
        [(2, 1)],
        [(1, 7)],
        [(-1, 2)],
        [(True, 2)],
        [(1, False)],
        [(1.0, 2)],
        [(1, "2")],
        [(1, None)],
        [(1,)],
        [None],
        [(1, 2), (2, 7)],
        [(1, 3), (2, 5)],
        [(3, 5), (1, 3)],
    ],
)
def test_all_boundaries_validated_before_replay(folds, periods):
    with patch("elote.evaluation.walk_forward") as evaluator:
        with pytest.raises(ValueError):
            expanding_window_evaluate(EloCompetitor, periods, folds)
        evaluator.assert_not_called()


def test_empty_schedule_and_reserved_warmup(periods):
    with pytest.raises(ValueError):
        expanding_window_evaluate(EloCompetitor, [], [(1, 2)])
    with pytest.raises(TypeError):
        expanding_window_evaluate(EloCompetitor, periods, [(1, 2)], warmup=0)


@pytest.mark.parametrize("cls,param,value", [(EloCompetitor, "k_factor", 32), (GlickoBoostCompetitor, "eta", 30)])
def test_class_configuration_restored_on_success_and_later_failure(cls, param, value, periods):
    original = getattr(cls, f"_{param}")
    expanding_window_evaluate(cls, periods, [(1, 3), (3, 5)], competitor_params={param: value})
    assert getattr(cls, f"_{param}") == original
    broken = [list(rows) for rows in periods]
    a, b, outcome, when, _ = broken[4][0]
    broken[4][0] = (a, b, outcome, when, {"home": 1, "away": 3})
    with pytest.raises(ValueError):
        expanding_window_evaluate(
            cls, broken, [(1, 3), (3, 5)], competitor_params={param: value}, score_keys=("home", "away")
        )
    assert getattr(cls, f"_{param}") == original


def test_gaps_are_training_history(periods):
    fold = expanding_window_evaluate(EloCompetitor, periods, [(1, 2), (4, 6)])[1]
    assert fold.report == walk_forward(EloCompetitor, periods, warmup=4)


@pytest.mark.parametrize("cls", [EloCompetitor, GlickoBoostCompetitor])
def test_folds_construct_distinct_empty_arenas(cls, periods):
    from elote import evaluation

    arenas = []
    arena_class = evaluation.LambdaArena

    def create_arena(*args, **kwargs):
        arena = arena_class(*args, **kwargs)
        assert arena.competitors == {}
        arenas.append(arena)
        return arena

    with patch.object(evaluation, "LambdaArena", side_effect=create_arena):
        expanding_window_evaluate(cls, periods, [(1, 3), (3, 5)])
    assert len(arenas) == 2
    assert arenas[0] is not arenas[1]
    assert arenas[0].competitors["A"] is not arenas[1].competitors["A"]


def test_elo_known_window_values(periods):
    import math

    report = expanding_window_evaluate(EloCompetitor, periods, [(1, 2)], competitor_params={"k_factor": 32})[0].report
    # The warmup win moves the fresh pair to 416 and 384 before this loss.
    probability = 1 / (1 + 10 ** (-32 / 400))
    assert (report.predictions, report.skipped, report.draws, report.accuracy) == (1, 1, 1, 0)
    assert report.log_loss == pytest.approx(-math.log(1 - probability))
    assert report.brier == pytest.approx(probability**2)
    assert report.reliability[5].count == 1
    assert report.reliability[5].mean_predicted == pytest.approx(probability)
    assert report.reliability[5].observed_rate == 0
