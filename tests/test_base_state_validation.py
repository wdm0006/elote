"""Validation and error-path tests for the base.py state/config surface.

Targets the ``_validate_state_dict``, ``import_state``, ``from_state``,
``get_competitor_class``, ``verify_competitor_types`` and ``lost_to`` clusters
from the mutation campaign (research findings art_dQPENpEX section 3): malformed
state documents must raise the documented exceptions, never silently import.
"""

import operator
import re

import pytest

from elote import (
    EloCompetitor,
    GlickoCompetitor,
    PythagoreanCompetitor,
)
from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    InvalidStateException,
    MissMatchedCompetitorTypesException,
)

ALL_SYSTEM_NAMES = [
    "EloCompetitor",
    "GlickoCompetitor",
    "Glicko2Competitor",
    "GlickoBoostCompetitor",
    "TrueSkillCompetitor",
    "ECFCompetitor",
    "DWZCompetitor",
    "ColleyMatrixCompetitor",
    "MasseyCompetitor",
    "KeenerCompetitor",
    "PythagoreanCompetitor",
    "BradleyTerryCompetitor",
    "WholeHistoryRatingCompetitor",
    "BlendedCompetitor",
]


def _valid_elo_state():
    competitor = EloCompetitor(initial_rating=1200, k_factor=24)
    state = competitor.export_state()
    state["state"]["rating"] = 1265
    return state


@pytest.mark.parametrize("field", ["type", "version", "parameters", "state"])
def test_validate_state_dict_rejects_missing_fields(field):
    """Each required envelope field is mandatory, with the field named."""
    state = _valid_elo_state()
    del state[field]

    with pytest.raises(InvalidStateException, match=field):
        EloCompetitor(initial_rating=1200).import_state(state)


@pytest.mark.parametrize("bad_type", [123, None, ["EloCompetitor"]])
def test_validate_state_dict_rejects_non_string_type(bad_type):
    state = _valid_elo_state()
    state["type"] = bad_type

    with pytest.raises(InvalidStateException, match="type"):
        EloCompetitor(initial_rating=1200).import_state(state)


@pytest.mark.parametrize("bad_version", ["1", 1.5, None])
def test_validate_state_dict_rejects_non_integer_version(bad_version):
    state = _valid_elo_state()
    state["version"] = bad_version

    with pytest.raises(InvalidStateException, match="version"):
        EloCompetitor(initial_rating=1200).import_state(state)


@pytest.mark.parametrize("field", ["parameters", "state"])
@pytest.mark.parametrize("bad_value", [[], "x", None, 7])
def test_validate_state_dict_rejects_non_mapping_sections(field, bad_value):
    state = _valid_elo_state()
    state[field] = bad_value

    with pytest.raises(InvalidStateException, match=field):
        EloCompetitor(initial_rating=1200).import_state(state)


def test_validate_state_dict_accepts_valid_state():
    """A well-formed document imports: the validator is not rejection-only."""
    competitor = EloCompetitor(initial_rating=1200)
    competitor.import_state(_valid_elo_state())

    assert competitor.rating == 1265


def test_import_state_rejects_mismatched_competitor_type():
    """A glicko-1 document cannot be imported into an Elo competitor."""
    glicko_state = GlickoCompetitor(initial_rating=1500, initial_rd=200).export_state()

    with pytest.raises(InvalidStateException, match="Mismatched competitor types"):
        EloCompetitor(initial_rating=1500).import_state(glicko_state)


def test_import_state_rejects_invalid_parameters():
    """Parameter validation runs on import: below-minimum initial rating fails."""
    state = _valid_elo_state()
    state["parameters"]["initial_rating"] = 50

    with pytest.raises(InvalidParameterException):
        EloCompetitor(initial_rating=1200).import_state(state)


@pytest.mark.parametrize("field", ["type", "parameters", "state"])
def test_from_state_requires_core_fields(field):
    state = _valid_elo_state()
    del state[field]

    with pytest.raises(InvalidStateException, match=f"'{field}'"):
        BaseCompetitor.from_state(state)


def test_from_state_rejects_unknown_type():
    state = _valid_elo_state()
    state["type"] = "NoSuchCompetitor"

    with pytest.raises(InvalidParameterException, match="NoSuchCompetitor"):
        BaseCompetitor.from_state(state)


def test_from_state_dead_guard_raises_when_registry_returns_none(monkeypatch):
    """from_state's ``competitor_class is None`` guard still raises.

    get_competitor_class raises for unknown names, so the None branch is only
    reachable when the registry lookup is bypassed; pin the guard anyway so the
    campaign's mutants on it are killed rather than waived.
    """
    state = _valid_elo_state()

    monkeypatch.setattr(EloCompetitor, "get_competitor_class", classmethod(lambda cls, name: None))

    with pytest.raises(InvalidStateException, match="Unknown competitor type"):
        EloCompetitor.from_state(state)


def test_get_competitor_class_known_and_unknown():
    assert BaseCompetitor.get_competitor_class("EloCompetitor") is EloCompetitor

    with pytest.raises(InvalidParameterException, match="Unknown competitor type"):
        BaseCompetitor.get_competitor_class("DefinitelyNotACompetitor")


def test_list_competitor_types_contains_every_registered_system():
    listed = set(BaseCompetitor.list_competitor_types())

    for name in ALL_SYSTEM_NAMES:
        assert name in listed


def test_verify_competitor_types_rejects_cross_system_pairs():
    elo = EloCompetitor(initial_rating=1200)
    glicko = GlickoCompetitor(initial_rating=1200)

    with pytest.raises(MissMatchedCompetitorTypesException):
        elo.verify_competitor_types(glicko)


def test_cross_system_comparisons_raise():
    """Ordered comparisons and updates refuse mixed types; equality is False."""
    elo = EloCompetitor(initial_rating=1200)
    glicko = GlickoCompetitor(initial_rating=1200)

    for compare in (operator.lt, operator.gt, operator.le, operator.ge):
        with pytest.raises(MissMatchedCompetitorTypesException):
            compare(elo, glicko)
    with pytest.raises(MissMatchedCompetitorTypesException):
        elo.beat(glicko)
    with pytest.raises(MissMatchedCompetitorTypesException):
        elo.expected_score(glicko)

    assert (elo == glicko) is False


def test_lost_to_matches_beat_on_winner():
    """lost_to on the loser is exactly beat on the winner."""
    winner_a, loser_a = EloCompetitor(initial_rating=1300, k_factor=24), EloCompetitor(initial_rating=1100, k_factor=24)
    loser_a.lost_to(winner_a)

    winner_b, loser_b = EloCompetitor(initial_rating=1300, k_factor=24), EloCompetitor(initial_rating=1100, k_factor=24)
    winner_b.beat(loser_b)

    assert winner_a.rating == winner_b.rating
    assert loser_a.rating == loser_b.rating


def test_lost_to_reverses_score_payload():
    """Custom scores are handed to the winner in winner-first order.

    Kills the campaign's ``scores=None`` mutant on base.lost_to: the payload
    must be reversed and propagated, not dropped. Pythagorean consumes the
    margin, so a dropped payload changes points totals or raises.
    """
    loser = PythagoreanCompetitor()
    winner = PythagoreanCompetitor()

    loser.lost_to(winner, scores=(10, 21))

    assert winner._points_for == 21.0
    assert winner._points_against == 10.0
    assert loser._points_for == 10.0
    assert loser._points_against == 21.0


def test_lost_to_rejects_scores_contradicting_the_loss():
    """A score payload that describes a win for the caller is rejected."""
    loser = EloCompetitor(initial_rating=1200)
    winner = EloCompetitor(initial_rating=1200)

    with pytest.raises(ValueError, match="do not describe a loss"):
        loser.lost_to(winner, scores=(3, 1))


def test_validate_scores_rejects_malformed_payloads():
    """The shared score validator: length, type, negativity, finiteness."""
    a = EloCompetitor(initial_rating=1200)
    b = EloCompetitor(initial_rating=1200)

    with pytest.raises(ValueError, match="exactly two values"):
        a.beat(b, scores=(1,))
    with pytest.raises(ValueError, match="must contain only numbers"):
        a.beat(b, scores=("1", 2))
    with pytest.raises(ValueError, match="non-negative"):
        a.beat(b, scores=(-1, 2))
    with pytest.raises(ValueError, match="finite"):
        a.beat(b, scores=(float("inf"), 2))


def test_get_competitor_class_returns_the_registered_class_object():
    """Registration stores the class itself, not a stand-in value."""

    assert BaseCompetitor.get_competitor_class("EloCompetitor") is EloCompetitor
    assert BaseCompetitor.get_competitor_class("GlickoCompetitor") is GlickoCompetitor


def test_ordering_contract_for_equal_ratings():
    """Equal ratings compare strictly equal: no win either way, both draws hold."""

    a, b = EloCompetitor(), EloCompetitor()

    assert not operator.lt(a, b)
    assert not operator.gt(a, b)
    assert operator.le(a, b)
    assert operator.ge(a, b)
    assert a == b


def test_ordering_contract_for_unequal_ratings():
    """After a decisive game the ordering operators track the rating gap exactly."""

    a, b = EloCompetitor(), EloCompetitor()
    a.beat(b)

    assert a.rating > b.rating
    assert operator.lt(b, a)
    assert operator.le(b, a)
    assert operator.gt(a, b)
    assert operator.ge(a, b)
    assert not operator.lt(a, b)
    assert not operator.le(a, b)
    assert not operator.gt(b, a)
    assert not operator.ge(b, a)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("type", 123, "Field 'type' must be a string"),
        ("version", "1", "Field 'version' must be an integer"),
        ("parameters", [], "Field 'parameters' must be a dictionary"),
        ("state", [], "Field 'state' must be a dictionary"),
    ],
)
def test_validate_state_dict_error_messages_are_exact(field, bad_value, message):
    """Validation failures carry the exact documented message for the broken field."""

    state = EloCompetitor().export_state()
    state[field] = bad_value

    with pytest.raises(InvalidStateException, match="^" + re.escape(message) + "$"):
        EloCompetitor().import_state(state)


def test_verify_competitor_types_message_names_both_types():
    """The co-mingling error names both offending classes in caller order."""

    expected = re.escape(
        "Competitor types <class 'elote.competitors.glicko.GlickoCompetitor'> "
        "and <class 'elote.competitors.elo.EloCompetitor'> cannot be co-mingled"
    )

    with pytest.raises(MissMatchedCompetitorTypesException, match=expected):
        EloCompetitor().verify_competitor_types(GlickoCompetitor())
