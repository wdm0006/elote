"""Export/import round-trip contracts for every rating system.

Targets the base.py serialization cluster from the mutation campaign
(research findings art_dQPENpEX section 3): ``_export_parameters`` kwargs that
can be dropped silently, state keys that can be renamed silently (glicko-2's
``mu``, keener/massey's ``losses``), and the ``state.get(key, default)``
fallbacks in ``_import_current_state`` whose removal turns malformed imports
into ``KeyError`` instead of a documented default.

Every expected parameter dict and state key set below was read off a live
export on this tree; the round-trip equality assertions were verified to hold
for all 13 concrete systems.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from elote import (
    EloCompetitor,
    Glicko2Competitor,
    GlickoCompetitor,
)
from elote.competitors.base import InvalidParameterException

PERIOD_END = datetime(2026, 1, 15, tzinfo=timezone.utc)

# (system name, init kwargs, expected exported parameters, expected state key set).
# Parameter values are deliberately non-default so that dropping a kwarg from
# _export_parameters changes the exported dict instead of hiding in a default.
CASES = [
    (
        "EloCompetitor",
        {"initial_rating": 1234, "k_factor": 24},
        {"initial_rating": 1234, "k_factor": 24},
        {"rating"},
    ),
    (
        "GlickoCompetitor",
        {"initial_rating": 1450, "initial_rd": 180},
        {"initial_rating": 1450, "initial_rd": 180},
        {"rating", "rd", "last_activity"},
    ),
    (
        "Glicko2Competitor",
        {"initial_rating": 1450, "initial_rd": 180, "initial_volatility": 0.05},
        {"initial_rating": 1450, "initial_rd": 180, "initial_volatility": 0.05},
        {"rating", "rd", "volatility", "mu", "phi", "last_activity"},
    ),
    (
        "GlickoBoostCompetitor",
        {"initial_rating": 1450, "initial_rd": 200},
        {"initial_rating": 1450, "initial_rd": 200},
        {"rating", "rd", "last_activity"},
    ),
    (
        "TrueSkillCompetitor",
        {"initial_mu": 28.0, "initial_sigma": 3.0},
        {"initial_mu": 28.0, "initial_sigma": 3.0},
        {"mu", "sigma"},
    ),
    ("ECFCompetitor", {"initial_rating": 140}, {"initial_rating": 140}, {"scores"}),
    ("DWZCompetitor", {"initial_rating": 450}, {"initial_rating": 450}, {"rating", "count", "initial_count"}),
    ("ColleyMatrixCompetitor", {"initial_rating": 0.7}, {"initial_rating": 0.7}, {"rating", "wins", "losses", "ties"}),
    (
        "MasseyCompetitor",
        {"initial_rating": 0.25},
        {"initial_rating": 0.25},
        {"rating", "wins", "losses", "ties", "point_differential", "rating_scale"},
    ),
    (
        "KeenerCompetitor",
        {"initial_rating": 1.5},
        {"initial_rating": 1.5},
        {"rating", "wins", "losses", "ties", "points_for", "points_against"},
    ),
    ("PythagoreanCompetitor", {"exponent": 2.0}, {"exponent": 2.0}, {"points_for", "points_against", "num_games"}),
    (
        "BradleyTerryCompetitor",
        {"initial_rating": 1600},
        {"initial_rating": 1600},
        {"rating", "wins", "losses", "ties", "beta"},
    ),
    (
        "WholeHistoryRatingCompetitor",
        {"initial_rating": 1400},
        {"initial_rating": 1400.0, "w2": 300.0, "max_iterations": 20, "precision": 0.1},
        {"rating", "days", "ratings", "day_index", "last_activity"},
    ),
]

COUNTER_SYSTEMS = ["ColleyMatrixCompetitor", "MasseyCompetitor", "KeenerCompetitor", "BradleyTerryCompetitor"]


def _cls(name):
    """Resolve a competitor class by its registered name."""
    import elote

    return getattr(elote, name)


def _init_kwargs(name):
    return dict(next(kwargs for case_name, kwargs, _, _ in CASES if case_name == name))


def _opponent_kwargs(name):
    """Opponents use the same spec as the protagonist; only fresh instances matter."""
    return dict(_init_kwargs(name))


def _played(name):
    """A competitor of the requested system after one win, one draw, one loss."""
    cls = _cls(name)
    a = cls(**_init_kwargs(name))
    b = cls(**_opponent_kwargs(name))
    c = cls(**_opponent_kwargs(name))
    # Margin-style scores: margin-agnostic systems validate and ignore them,
    # margin-aware systems (Pythagorean, Massey, Keener, Colley) consume them.
    a.beat(b, scores=(21, 10))
    a.tied(c, scores=(7, 7))
    a.lost_to(c, scores=(10, 21))
    return a


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_export_parameters_contract(name):
    """Exported parameters carry every initialization kwarg with its value.

    Kills the mutants that drop a kwarg from a system's ``_export_parameters``
    dict (the constructor value then silently reverts to the class default on
    the next import).
    """
    played = _played(name)
    state = played.export_state()
    expected = next(params for case_name, _, params, _ in CASES if case_name == name)
    assert state["parameters"] == expected


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_export_state_key_contract(name):
    """The state payload exposes exactly the documented keys for this system.

    Kills the rename mutants (``"mu"`` -> ``"XXmuXX"``, ``"losses"`` ->
    ``"XXlossesXX"``, ...) that make a round-trip silently lose a field.
    """
    played = _played(name)
    state = played.export_state()
    expected_keys = next(keys for case_name, _, _, keys in CASES if case_name == name)
    assert set(state["state"]) == expected_keys


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_import_state_restores_rating_and_payload(name):
    """import_state into a fresh instance reproduces rating and state payload."""
    played = _played(name)
    state = played.export_state()

    fresh = _cls(name)(**_init_kwargs(name))
    fresh.import_state(state)

    assert fresh.rating == played.rating
    assert fresh.export_state()["state"] == state["state"]


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_from_state_rebuilds_competitor(name):
    """from_state rebuilds the same system with the same rating and parameters."""
    played = _played(name)
    state = played.export_state()

    rebuilt = _cls(name).from_state(state)

    assert type(rebuilt) is _cls(name)
    assert rebuilt.rating == played.rating
    expected = next(params for case_name, _, params, _ in CASES if case_name == name)
    assert rebuilt.export_state()["parameters"] == expected


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_missing_payload_keys_fall_back_to_initials(name):
    """A payload missing every state key imports to initial values, no KeyError.

    Pins the ``state.get(key, default)`` fallbacks in each system's
    ``_import_current_state``: the mutant that removes the default argument
    (``state.get("rating")``) raises ``KeyError`` on these imports and must
    fail here rather than surface as a crash on malformed documents.
    """
    cls = _cls(name)
    fresh = cls(**_init_kwargs(name))

    state = fresh.export_state()
    for key in list(state["state"]):
        del state["state"][key]

    importer = cls(**_init_kwargs(name))
    importer.import_state(state)  # must not raise KeyError

    assert importer.rating == fresh.rating


def test_glicko2_exports_internal_scale_values():
    """The glicko-2 payload carries mu/phi/volatility on the internal scale.

    Kills the ``"mu"`` rename mutant from the campaign's named survivor list:
    the key must be present with the competitor's actual internal value, not
    re-derivable defaults.
    """
    winner = Glicko2Competitor(initial_rating=1450, initial_rd=180, initial_volatility=0.05)
    loser = Glicko2Competitor(initial_rating=1300)
    winner.beat(loser, match_time=PERIOD_END)

    payload = winner.export_state()["state"]

    assert payload["mu"] == winner._mu
    assert payload["phi"] == winner._phi
    assert payload["volatility"] == winner._sigma
    assert payload["rating"] == winner.rating


@pytest.mark.parametrize("name", COUNTER_SYSTEMS)
def test_game_counters_round_trip_through_state(name):
    """Wins/losses/ties survive an export-import cycle with their values.

    Kills the keener/massey ``"losses"`` rename mutants: after one win, one
    draw and one loss the imported competitor must still carry all three
    counters.
    """
    a = _played(name)

    imported = _cls(name)(**_init_kwargs(name))
    imported.import_state(a.export_state())

    assert imported._wins == 1
    assert imported._ties == 1
    assert imported._losses == 1
    assert imported.rating == a.rating


def test_import_state_rejects_payload_below_minimum():
    """Payload validation still applies to imported values."""
    competitor = EloCompetitor(initial_rating=1200)
    state = competitor.export_state()
    state["state"]["rating"] = 50

    with pytest.raises(InvalidParameterException):
        EloCompetitor(initial_rating=1200).import_state(state)


def test_import_state_rejects_non_positive_rd():
    """Glicko rejects a non-positive RD in the imported payload."""
    competitor = GlickoCompetitor(initial_rating=1450, initial_rd=180)
    state = competitor.export_state()
    state["state"]["rd"] = 0

    with pytest.raises(InvalidParameterException):
        GlickoCompetitor(initial_rating=1450, initial_rd=180).import_state(state)


def test_export_envelope_shape():
    """The standardized envelope: fixed fields, version 1, uuid id, flat view."""
    competitor = EloCompetitor(initial_rating=1234, k_factor=24)
    state = competitor.export_state()

    assert state["type"] == "EloCompetitor"
    assert state["version"] == 1
    assert isinstance(state["created_at"], int)
    uuid.UUID(state["id"])  # raises ValueError if not a uuid
    assert set(state) >= {"type", "version", "created_at", "id", "parameters", "state", "class_vars"}

    # parameters and state are flattened into the top level for backwards compat
    assert state["initial_rating"] == 1234
    assert state["k_factor"] == 24
    assert state["rating"] == competitor.rating


def test_export_state_is_json_serializable():
    """export_state output survives a JSON dump/load unchanged."""
    competitor = Glicko2Competitor(initial_rating=1450, initial_rd=180, initial_volatility=0.05)
    opponent = Glicko2Competitor(initial_rating=1300)
    competitor.beat(opponent, match_time=PERIOD_END)

    loaded = json.loads(json.dumps(competitor.export_state()))

    assert loaded["type"] == "Glicko2Competitor"
    assert loaded["state"]["mu"] == competitor._mu


def test_export_state_carries_class_vars():
    """JSON-serializable class attributes are flattened into class_vars."""
    state = EloCompetitor(initial_rating=1200).export_state()

    assert state["class_vars"]["base_rating"] == EloCompetitor._base_rating
    assert state["class_vars"]["k_factor"] == EloCompetitor._k_factor
    assert state["class_vars"]["minimum_rating"] == EloCompetitor._minimum_rating


def test_export_state_class_vars_key_set_and_values():
    """class_vars carries exactly the JSON-safe class attributes, with live values.

    Guards the dunder/callable/_abc_ filter in export_state: any leak or a
    clobbered value shows up as a changed key set or a None entry.
    """

    class_vars = EloCompetitor().export_state()["class_vars"]

    assert set(class_vars) == {"k_factor", "base_rating", "minimum_rating"}
    assert class_vars["k_factor"] == 32
    assert not [k for k in class_vars if k.startswith("__") or k.startswith("_abc_")]

