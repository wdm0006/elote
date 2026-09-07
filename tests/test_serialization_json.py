"""to_json/from_json round-trips for every rating system, plus Blended.

Targets the ``to_json`` / ``from_json`` survivor cluster from the mutation
campaign (research findings art_dQPENpEX section 3). Verified contract of
``from_json``: it accepts a JSON *string* only (a dict raises ``TypeError``
from ``json.loads``), routes through ``from_state``, and systems overriding
``from_state`` (Elo, Glicko, Glicko-2, TrueSkill, ECF, DWZ, Colley, Blended)
delegate modern envelopes (with a ``type`` field) to the base dispatcher,
which builds the *document's* class regardless of the receiving class. A
flattened dict without ``type`` takes each override's legacy path, whose
``state.get(key, default)`` fallbacks are pinned here too.
"""

import json

import pytest

from elote import BlendedCompetitor, GlickoCompetitor
from elote.competitors.base import InvalidStateException

CONCRETE = [
    ("EloCompetitor", {"initial_rating": 1234, "k_factor": 24}),
    ("GlickoCompetitor", {"initial_rating": 1450, "initial_rd": 180}),
    ("Glicko2Competitor", {"initial_rating": 1450, "initial_rd": 180, "initial_volatility": 0.05}),
    ("GlickoBoostCompetitor", {"initial_rating": 1450, "initial_rd": 200}),
    ("TrueSkillCompetitor", {"initial_mu": 28.0, "initial_sigma": 3.0}),
    ("ECFCompetitor", {"initial_rating": 140}),
    ("DWZCompetitor", {"initial_rating": 450}),
    ("ColleyMatrixCompetitor", {"initial_rating": 0.7}),
    ("MasseyCompetitor", {"initial_rating": 0.25}),
    ("KeenerCompetitor", {"initial_rating": 1.5}),
    ("PythagoreanCompetitor", {"exponent": 2.0}),
    ("BradleyTerryCompetitor", {"initial_rating": 1600}),
    ("WholeHistoryRatingCompetitor", {"initial_rating": 1400}),
]


def _played(name, kwargs):
    """A competitor of the requested system whose rating diverged (one win)."""
    import elote

    cls = getattr(elote, name)
    a = cls(**kwargs)
    b = cls(**kwargs)
    a.beat(b)
    return a


@pytest.mark.parametrize("name,kwargs", CONCRETE)
def test_to_json_from_json_round_trip(name, kwargs):
    """to_json -> from_json rebuilds a competitor with the same state payload."""
    a = _played(name, kwargs)
    cls = type(a)

    rebuilt = cls.from_json(a.to_json())

    assert type(rebuilt) is cls
    assert rebuilt.rating == a.rating
    assert rebuilt.export_state()["state"] == a.export_state()["state"]
    assert rebuilt.export_state()["parameters"] == a.export_state()["parameters"]


def test_from_json_accepts_raw_json_string():
    """from_json takes the JSON text itself; the state payload survives."""
    a = _played("EloCompetitor", {"initial_rating": 1300, "k_factor": 20})

    rebuilt = type(a).from_json(a.to_json())

    assert rebuilt.rating == a.rating
    assert rebuilt.export_state()["state"] == a.export_state()["state"]


def test_from_json_rejects_non_string_input():
    """A parsed dict is not a valid from_json argument (string-only API)."""
    document = {"type": "EloCompetitor", "parameters": {}, "state": {}}

    with pytest.raises(TypeError):
        type(_played("EloCompetitor", {"initial_rating": 1200})).from_json(document)  # type: ignore[arg-type]


def test_to_json_is_a_json_string():
    """to_json emits text that json.loads accepts with the right type field."""
    a = _played("Glicko2Competitor", {"initial_rating": 1450, "initial_rd": 180, "initial_volatility": 0.05})

    document = json.loads(a.to_json())

    assert document["type"] == "Glicko2Competitor"
    assert document["state"]["mu"] == a._mu


@pytest.mark.parametrize("field", ["state", "parameters"])
def test_from_json_missing_required_section_raises(field):
    """A modern envelope missing a required section is rejected by name."""
    a = _played("EloCompetitor", {"initial_rating": 1200})
    document = json.loads(a.to_json())
    del document[field]

    with pytest.raises(InvalidStateException, match=f"'{field}'"):
        type(a).from_json(json.dumps(document))


def test_from_json_builds_the_documented_type():
    """The document's type wins: Elo.from_json(glicko_doc) -> GlickoCompetitor."""
    document = GlickoCompetitor(initial_rating=1500, initial_rd=200).to_json()

    rebuilt = type(_played("EloCompetitor", {"initial_rating": 1200})).from_json(document)

    assert type(rebuilt) is GlickoCompetitor
    assert rebuilt.rating == 1500


def test_from_json_legacy_flat_format_still_supported():
    """A flattened dict without 'type' takes the legacy path with get-fallbacks.

    Pins the legacy ``state.get(key, default)`` fallbacks (Elo: initial_rating
    defaults to 400 when absent; current_rating applied when present): the
    campaign's fallback-removal mutants raise KeyError here.
    """
    legacy = {"initial_rating": 1500, "k_factor": 20, "current_rating": 1550}

    rebuilt = type(_played("EloCompetitor", {"initial_rating": 1200})).from_json(json.dumps(legacy))

    assert type(rebuilt).__name__.endswith("EloCompetitor")
    assert rebuilt.rating == 1550


def test_from_json_legacy_defaults_initial_rating_to_400():
    """The legacy fallback for initial_rating is 400 (documented default)."""
    legacy = {"current_rating": 1550}

    rebuilt = type(_played("EloCompetitor", {"initial_rating": 1200})).from_json(json.dumps(legacy))

    assert rebuilt.rating == 1550
    assert rebuilt._initial_rating == 400


def test_blended_state_round_trip():
    """BlendedCompetitor round-trips through export_state/from_state."""
    blended = BlendedCompetitor(
        competitors=[
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1300, "k_factor": 24}},
            {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1450, "initial_rd": 180}},
        ],
        blend_mode="mean",
    )

    state = blended.export_state()
    rebuilt = BlendedCompetitor.from_state(state)

    assert rebuilt.rating == blended.rating
