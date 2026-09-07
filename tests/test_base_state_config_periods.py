"""configure_class and apply_rating_period contracts for BaseCompetitor.

Targets the ``configure_class`` (28 survivors) and ``apply_rating_period``
(16 survivors) clusters from the mutation campaign (research findings
art_dQPENpEX section 3). The library exposes no reset helper, so tests save
and restore touched class attributes themselves; the default period runner
replays results in order through the pairwise methods and mutates the given
competitors in place (returns ``None``).
"""

from datetime import datetime, timezone

import pytest

from elote import EloCompetitor, GlickoCompetitor, Glicko2Competitor, MasseyCompetitor, PythagoreanCompetitor
from elote.competitors.base import BaseCompetitor, InvalidParameterException

PERIOD = datetime(2026, 3, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _restore_elo_class_state():
    """Snapshot Elo's class dict and restore it, so configuration never leaks."""
    saved = dict(EloCompetitor.__dict__)
    yield
    for key, value in saved.items():
        setattr(EloCompetitor, key, value)
    for key in set(EloCompetitor.__dict__) - set(saved):
        delattr(EloCompetitor, key)


#
# configure_class
#


def test_configure_class_rejects_unknown_parameters():
    with pytest.raises(InvalidParameterException, match="Unknown class parameter"):
        EloCompetitor.configure_class(not_a_parameter=1)  # type: ignore[call-arg]


def test_configure_class_sets_class_attributes():
    EloCompetitor.configure_class(k_factor=32.0)

    assert EloCompetitor._k_factor == 32.0


def test_configured_k_factor_reaches_new_instances():
    EloCompetitor.configure_class(k_factor=32.0)

    fresh = EloCompetitor(initial_rating=1200)

    assert fresh._k_factor == 32.0


def test_configured_k_factor_changes_rating_math():
    """A configured K shows up in the update size, not just the attribute.

    Instances copy the class k_factor at construction, so configuration must
    precede construction to have any effect.
    """
    EloCompetitor.configure_class(k_factor=8.0)
    low = EloCompetitor(initial_rating=1200)
    low_opponent = EloCompetitor(initial_rating=1200)
    low.beat(low_opponent)

    EloCompetitor.configure_class(k_factor=64.0)
    high = EloCompetitor(initial_rating=1200)
    high_opponent = EloCompetitor(initial_rating=1200)
    high.beat(high_opponent)

    low_gain = low.rating - 1200
    high_gain = high.rating - 1200
    assert high_gain == pytest.approx(8 * low_gain)


def test_configure_class_sets_multiple_parameters_at_once():
    EloCompetitor.configure_class(k_factor=16.0, base_rating=900)

    assert EloCompetitor._k_factor == 16.0
    assert EloCompetitor._base_rating == 900


def test_instance_configure_does_not_touch_the_class():
    competitor = EloCompetitor(initial_rating=1200)

    competitor.configure(k_factor=64.0)

    assert competitor._k_factor == 64.0
    assert EloCompetitor._k_factor == 32.0


def test_instance_configure_rejects_unknown_parameters():
    competitor = EloCompetitor(initial_rating=1200)

    with pytest.raises(InvalidParameterException, match="Unknown instance parameter"):
        competitor.configure(not_a_parameter=1)  # type: ignore[call-arg]


#
# apply_rating_period
#


def _fresh_pair():
    return EloCompetitor(initial_rating=1200, k_factor=24), EloCompetitor(initial_rating=1200, k_factor=24)


def test_apply_rating_period_returns_none():
    a, b = _fresh_pair()

    result = EloCompetitor.apply_rating_period([(a, b, 1.0, None)], period_end=PERIOD)

    assert result is None


def test_apply_rating_period_replays_in_order():
    """Period replay equals playing the same games one by one."""
    season = [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 2.0, 1.0), (2.0, 0.0, 0.5), (1.0, 2.0, 0.0)]

    period_arena = {i: EloCompetitor(initial_rating=1200 + 15 * i, k_factor=24) for i in range(3)}
    manual_arena = {i: EloCompetitor(initial_rating=1200 + 15 * i, k_factor=24) for i in range(3)}

    EloCompetitor.apply_rating_period(
        [(period_arena[a], period_arena[b], outcome, None) for a, b, outcome in season],
        period_end=PERIOD,
    )
    for a, b, outcome in season:
        if outcome == 1.0:
            manual_arena[a].beat(manual_arena[b])
        elif outcome == 0.0:
            manual_arena[b].beat(manual_arena[a])
        else:
            manual_arena[a].tied(manual_arena[b])

    for i in range(3):
        assert period_arena[i].rating == manual_arena[i].rating


def test_apply_rating_period_supports_scores_and_reverses_them_for_b_wins():
    """A B-win result feeds the winner the reversed caller-order scores."""
    a = PythagoreanCompetitor()
    b = PythagoreanCompetitor()

    type(a).apply_rating_period([(a, b, 0.0, (10.0, 21.0))], period_end=PERIOD)

    assert b._points_for == 21.0
    assert b._points_against == 10.0
    assert a._points_for == 10.0
    assert a._points_against == 21.0


def test_apply_rating_period_draws():
    """A tie between unequally rated competitors moves them toward each other."""
    a = EloCompetitor(initial_rating=1300, k_factor=24)
    b = EloCompetitor(initial_rating=1200, k_factor=24)

    EloCompetitor.apply_rating_period([(a, b, 0.5, None)], period_end=PERIOD)

    assert a.rating < 1300.0
    assert b.rating > 1200.0


def test_apply_rating_period_rejects_bad_outcomes():
    a, b = _fresh_pair()

    with pytest.raises(ValueError, match="outcome must be one of"):
        EloCompetitor.apply_rating_period([(a, b, 1.5, None)], period_end=PERIOD)
    with pytest.raises(ValueError, match="outcome must be one of"):
        EloCompetitor.apply_rating_period([(a, b, "win", None)], period_end=PERIOD)  # type: ignore[list-item]


def test_apply_rating_period_propagates_invalid_scores():
    a, b = _fresh_pair()

    with pytest.raises(ValueError):
        EloCompetitor.apply_rating_period([(a, b, 1.0, (1.0,))], period_end=PERIOD)


def test_apply_rating_period_stamps_time_aware_competitors():
    """Time-aware systems receive period_end as the match time."""
    a = GlickoCompetitor(initial_rating=1500, initial_rd=200)
    b = GlickoCompetitor(initial_rating=1500, initial_rd=200)

    GlickoCompetitor.apply_rating_period([(a, b, 1.0, None)], period_end=PERIOD)

    assert a._last_activity == PERIOD
    assert a.rating != 1500.0


def test_glicko2_period_updates_rating():
    a = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.05)
    b = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.05)

    Glicko2Competitor.apply_rating_period([(a, b, 1.0, None)], period_end=PERIOD)

    assert a.rating != 1500.0


def test_apply_rating_period_forwards_margin_scores_to_the_winner_path():
    """A-win replay forwards the score payload so margin-sensitive systems see it.

    Massey consumes the margin (2.0 here); dropping or defaulting the scores
    would fall back to the unit margin and produce different ratings.
    """

    ref_a, ref_b = MasseyCompetitor(), MasseyCompetitor()
    ref_a.beat(ref_b, scores=(2.5, 0.5))

    a, b = MasseyCompetitor(), MasseyCompetitor()
    BaseCompetitor.apply_rating_period([(a, b, 1.0, (2.5, 0.5))])

    assert a.rating == ref_a.rating
    assert b.rating == ref_b.rating


def test_apply_rating_period_forwards_reversed_margin_scores_on_b_wins():
    """B-win replay reverses the payload before handing it to the winner."""

    ref_a, ref_b = MasseyCompetitor(), MasseyCompetitor()
    ref_b.beat(ref_a, scores=(2.5, 0.5))

    a, b = MasseyCompetitor(), MasseyCompetitor()
    BaseCompetitor.apply_rating_period([(a, b, 0.0, (0.5, 2.5))])

    assert a.rating == ref_a.rating
    assert b.rating == ref_b.rating

