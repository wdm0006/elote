import datetime
from unittest.mock import patch

import pytest

from elote import (
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    DWZCompetitor,
    ECFCompetitor,
    EloCompetitor,
    Glicko2Competitor,
    GlickoCompetitor,
    KeenerCompetitor,
    LambdaArena,
    MasseyCompetitor,
    PythagoreanCompetitor,
    TrueSkillCompetitor,
    WholeHistoryRatingCompetitor,
)


COMPETITOR_CLASSES = (
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    ColleyMatrixCompetitor,
    MasseyCompetitor,
    KeenerCompetitor,
    PythagoreanCompetitor,
    BradleyTerryCompetitor,
    WholeHistoryRatingCompetitor,
)
PERIOD_END = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)


def _populations(competitor_class):
    return ({name: competitor_class() for name in "ABC"}, {name: competitor_class() for name in "ABC"})


def _apply_pairwise(results):
    for competitor_a, competitor_b, outcome, scores in results:
        supports_time = hasattr(competitor_a, "_last_activity")
        kwargs = {"scores": scores}
        if supports_time:
            kwargs["match_time"] = PERIOD_END
        if outcome == 1.0:
            competitor_a.beat(competitor_b, **kwargs)
        elif outcome == 0.0:
            reversed_scores = None if scores is None else (scores[1], scores[0])
            competitor_b.beat(competitor_a, **{**kwargs, "scores": reversed_scores})
        else:
            competitor_a.tied(competitor_b, **kwargs)


@pytest.mark.parametrize("competitor_class", COMPETITOR_CLASSES)
def test_default_rating_period_matches_pairwise_replay(competitor_class):
    period_population, pairwise_population = _populations(competitor_class)
    rows = (("A", "B", 1.0), ("B", "C", 0.5), ("A", "C", 0.0), ("A", "B", 1.0))
    period_results = [(period_population[a], period_population[b], outcome, None) for a, b, outcome in rows]
    pairwise_results = [(pairwise_population[a], pairwise_population[b], outcome, None) for a, b, outcome in rows]

    competitor_class.apply_rating_period(period_results, period_end=PERIOD_END)
    _apply_pairwise(pairwise_results)

    for name in period_population:
        if competitor_class is WholeHistoryRatingCompetitor:
            # WHR's lazy fit traverses object-hash-ordered graphs, so independently
            # built populations can differ slightly from floating-point summation order.
            assert period_population[name].rating == pytest.approx(pairwise_population[name].rating, abs=1e-4)
        else:
            assert period_population[name].rating == pairwise_population[name].rating


def test_rating_period_records_pre_period_predictions():
    arena = LambdaArena(lambda a, b: True)
    initial_prediction = EloCompetitor().expected_score(EloCompetitor())

    arena.rating_period([("A", "B", 1.0, None), ("A", "B", 1.0, None)])

    assert [bout.predicted_outcome for bout in arena.history.bouts] == [initial_prediction, initial_prediction]
    streaming = LambdaArena(lambda a, b: True)
    streaming.matchup("A", "B", outcome=1.0)
    assert streaming.expected_score("A", "B") != initial_prediction


def test_rating_period_applies_the_batch_once():
    arena = LambdaArena(lambda a, b: True)
    rows = [("A", "B", 1.0, None), ("B", "C", 0.5, None)]

    with patch.object(EloCompetitor, "apply_rating_period", wraps=EloCompetitor.apply_rating_period) as apply:
        arena.rating_period(rows, period_end=PERIOD_END)

    apply.assert_called_once()
    assert len(apply.call_args.args[0]) == len(rows)
    assert apply.call_args.kwargs == {"period_end": PERIOD_END}


@pytest.mark.parametrize(
    "matchups, message",
    [
        ([("A", "B", 1.0, None), ("C", "D", 0.25, None)], "outcome must be one of"),
        ([("A", "B", 1.0, None), ("C", "D", 1.0, (0, 1))], "do not describe a win"),
    ],
)
def test_invalid_rating_period_leaves_arena_unchanged(matchups, message):
    arena = LambdaArena(lambda a, b: True)

    with pytest.raises(ValueError, match=message):
        arena.rating_period(matchups)

    assert arena.competitors == {}
    assert arena.history.bouts == []


@pytest.mark.parametrize("competitor_class", [EloCompetitor, ColleyMatrixCompetitor])
def test_arena_rating_period_matches_streaming_final_ratings(competitor_class):
    rows = [("A", "B", 1.0, None), ("B", "C", 0.5, None), ("A", "C", 0.0, None)]
    period_arena = LambdaArena(lambda a, b: True, base_competitor=competitor_class)
    streaming_arena = LambdaArena(lambda a, b: True, base_competitor=competitor_class)

    period_arena.rating_period(rows)
    for a, b, outcome, scores in rows:
        streaming_arena.matchup(a, b, outcome=outcome, scores=scores)

    assert {name: competitor.rating for name, competitor in period_arena.competitors.items()} == {
        name: competitor.rating for name, competitor in streaming_arena.competitors.items()
    }
