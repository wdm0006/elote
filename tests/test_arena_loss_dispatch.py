"""The arena's loss branch dispatches through the loser's ``lost_to``.

Every result method reads its argument order as the caller-first order, and
:class:`~elote.GlickoBoostCompetitor` additionally reads it as the colour channel. A loss
dispatched as ``winner.beat(loser)`` therefore swaps the colours of that row relative to
the prediction the arena recorded for it. These tests pin the dispatch order.
"""

import datetime

import pytest

from elote import GlickoBoostCompetitor, LambdaArena
from elote.competitors.base import BaseCompetitor


MATCH_TIME = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)

# The canonical one-row rating-period answer for a fresh pair at eta = 30.0, taken from
# GlickoBoostCompetitor.apply_rating_period([(x, y, 0.0, None)]).
CANONICAL_LOSS_AT_ETA_30 = (1411.1086596902862, 1588.8913403097138)


@pytest.fixture
def boost_eta_30():
    """eta is class-level state, so it has to be restored or it leaks across the suite."""
    original = GlickoBoostCompetitor._eta
    GlickoBoostCompetitor.configure_class(eta=30.0)
    try:
        yield
    finally:
        GlickoBoostCompetitor.configure_class(eta=original)


def _construct_kwargs(competitor_class):
    """Constructor kwargs for the classes that cannot be built bare."""
    from elote import EloCompetitor

    if competitor_class.__name__ == "BlendedCompetitor":
        return {"competitors": [{"type": "EloCompetitor", "competitor_kwargs": {}}]}
    if competitor_class.__name__ == "TeamCompetitor":
        return {"members": [EloCompetitor(), EloCompetitor()]}
    return {}


def _fresh_pair(competitor_class):
    """Two independent competitors. The kwargs are rebuilt per side because a composite's
    roster is a list of live objects, and one shared list would make the two sides the
    same competitor."""
    return (
        competitor_class(**_construct_kwargs(competitor_class)),
        competitor_class(**_construct_kwargs(competitor_class)),
    )


def _arena_loss_ratings(competitor_class):
    """Ratings after one arena-driven loss row, ``a`` losing to ``b``."""
    a, b = _fresh_pair(competitor_class)
    arena = LambdaArena(
        None,
        base_competitor=competitor_class,
        initial_state={"a": a.export_state(), "b": b.export_state()},
    )
    arena.matchup("a", "b", match_time=MATCH_TIME, outcome=0.0)
    return arena.competitors["a"].rating, arena.competitors["b"].rating


def _direct_beat_ratings(competitor_class):
    """Ratings after the pre-change dispatch: the winner's ``beat``, arguments swapped."""
    a, b = _fresh_pair(competitor_class)
    if hasattr(a, "_last_activity"):
        b.beat(a, MATCH_TIME)
    else:
        b.beat(a)
    return a.rating, b.rating


class TestColourConventionOnLosingRows:
    """The behaviour this change carries: one bout, one colour assignment."""

    def test_arena_loss_matches_the_canonical_rating_period(self, boost_eta_30):
        """A streamed loss at eta = 30 lands on apply_rating_period's own answer."""
        arena_ratings = _arena_loss_ratings(GlickoBoostCompetitor)

        x = GlickoBoostCompetitor()
        y = GlickoBoostCompetitor()
        GlickoBoostCompetitor.apply_rating_period([(x, y, 0.0, None)], period_end=MATCH_TIME)

        assert arena_ratings == pytest.approx((x.rating, y.rating), abs=1e-9)
        assert arena_ratings == pytest.approx(CANONICAL_LOSS_AT_ETA_30, abs=1e-4)

    def test_streaming_and_one_row_periods_agree_over_a_sequence(self, boost_eta_30):
        """The prediction/update disagreement showed up as leaderboard drift; it is gone."""
        rows = [
            ("a", "b", 1.0),
            ("c", "a", 0.0),
            ("b", "d", 0.0),
            ("d", "c", 1.0),
            ("a", "d", 0.0),
            ("b", "c", 0.5),
            ("c", "d", 0.0),
            ("a", "c", 1.0),
        ]

        streamed = LambdaArena(None, base_competitor=GlickoBoostCompetitor)
        periodic = LambdaArena(None, base_competitor=GlickoBoostCompetitor)
        for left, right, outcome in rows:
            streamed.matchup(left, right, match_time=MATCH_TIME, outcome=outcome)
            periodic.rating_period([(left, right, outcome, None)], period_end=MATCH_TIME)

        streamed_board = {e["competitor"]: e["rating"] for e in streamed.leaderboard()}
        periodic_board = {e["competitor"]: e["rating"] for e in periodic.leaderboard()}

        assert streamed_board.keys() == periodic_board.keys()
        for name, rating in streamed_board.items():
            assert rating == pytest.approx(periodic_board[name], abs=1e-9)

    def test_recorded_prediction_matches_the_colour_that_was_applied(self, boost_eta_30):
        """The Bout's prediction is expected_score(a, b); the update must give a white too."""
        arena = LambdaArena(None, base_competitor=GlickoBoostCompetitor)
        arena.matchup("a", "b", match_time=MATCH_TIME, outcome=0.0)

        bout = arena.history.bouts[0]
        fresh_a = GlickoBoostCompetitor()
        fresh_b = GlickoBoostCompetitor()
        assert bout.predicted_outcome == pytest.approx(fresh_a.expected_score(fresh_b))

        # Applying the same row through the period solver -- the only colour assignment
        # the prediction could have been made under -- reproduces the arena exactly.
        GlickoBoostCompetitor.apply_rating_period([(fresh_a, fresh_b, 0.0, None)], period_end=MATCH_TIME)
        assert arena.competitors["a"].rating == pytest.approx(fresh_a.rating, abs=1e-9)


class TestLossDispatchIsBehaviourPreserving:
    """The regression net. This passes on the unfixed code too, by design: it exists to
    prove the swap changed nothing for the thirteen classes that are colour-blind."""

    @pytest.mark.parametrize("type_name", sorted(BaseCompetitor.list_competitor_types()))
    def test_arena_loss_matches_the_winner_side_beat(self, type_name):
        competitor_class = BaseCompetitor.get_competitor_class(type_name)
        assert _arena_loss_ratings(competitor_class) == pytest.approx(
            _direct_beat_ratings(competitor_class), abs=1e-9
        )


class TestScoreChannelStillReverses:
    """Glicko-1/Glicko-2 validate the score payload and then discard it, so a rating
    assertion cannot carry this. The validation asymmetry can."""

    @pytest.mark.parametrize("type_name", sorted(BaseCompetitor.list_competitor_types()))
    def test_lost_to_rejects_a_winning_payload(self, type_name):
        competitor_class = BaseCompetitor.get_competitor_class(type_name)
        a = competitor_class(**_construct_kwargs(competitor_class))
        b = competitor_class(**_construct_kwargs(competitor_class))
        with pytest.raises(ValueError):
            a.lost_to(b, scores=(3, 1))

    @pytest.mark.parametrize("type_name", sorted(BaseCompetitor.list_competitor_types()))
    def test_lost_to_accepts_a_losing_payload(self, type_name):
        competitor_class = BaseCompetitor.get_competitor_class(type_name)
        a = competitor_class(**_construct_kwargs(competitor_class))
        b = competitor_class(**_construct_kwargs(competitor_class))
        a.lost_to(b, scores=(1, 3))

    def test_arena_forwards_a_losing_payload_unreversed(self):
        """The arena hands lost_to the pair in its own (a, b) caller order."""
        arena = LambdaArena(None, base_competitor=GlickoBoostCompetitor)
        arena.matchup("a", "b", outcome=0.0, scores=(1, 3))

        with pytest.raises(ValueError):
            arena.matchup("c", "d", outcome=0.0, scores=(3, 1))
