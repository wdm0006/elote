"""Blend-weight and member-delegation contracts for BlendedCompetitor.

Targets the ensemble.py cluster from the mutation campaign (research findings
art_dQPENpEX section 3, 49 captured survivors): rating = sum of sub ratings,
mean-blended expected scores, per-sub delegation of beat/tied (strict zip),
composition verification, reset delegation, and the import error paths for
malformed sub-competitor state.
"""

import pytest

from elote import BlendedCompetitor, EloCompetitor
from elote.competitors.base import (
    InvalidParameterException,
    InvalidStateException,
    MissMatchedCompetitorTypesException,
)

ELO_ONLY = [
    {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200, "k_factor": 24}},
    {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1300, "k_factor": 24}},
]
MIXED = [
    {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200, "k_factor": 24}},
    {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1450, "initial_rd": 180}},
]


def _blend(specs=None, mode="mean"):
    return BlendedCompetitor(competitors=specs if specs is not None else MIXED, blend_mode=mode)


#
# Construction and configuration
#


def test_unknown_blend_mode_is_rejected():
    with pytest.raises(InvalidParameterException, match="Blend mode .* not supported"):
        BlendedCompetitor(competitors=ELO_ONLY, blend_mode="median")


def test_sub_type_defaults_to_elo_when_omitted():
    blended = BlendedCompetitor(competitors=[{"competitor_kwargs": {"initial_rating": 1100}}])

    assert [type(c).__name__ for c in blended.sub_competitors] == ["EloCompetitor"]
    assert blended.sub_competitors[0].rating == 1100


def test_sub_kwargs_default_to_empty():
    blended = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])

    assert blended.sub_competitors[0].rating == EloCompetitor().rating


def test_invalid_sub_specification_is_wrapped():
    with pytest.raises(InvalidParameterException, match="Failed to initialize sub-competitor EloCompetitor"):
        BlendedCompetitor(competitors=[{"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": -5}}])


#
# Combined rating and delegation
#


def test_rating_is_the_sum_of_sub_ratings():
    blended = _blend(MIXED)

    assert blended.rating == pytest.approx(1200 + 1450)


def test_rating_setter_is_not_supported():
    blended = _blend(MIXED)

    with pytest.raises(NotImplementedError):
        blended.rating = 2000.0


def test_expected_score_blends_sub_scores_by_mean():
    a = _blend(ELO_ONLY)
    b = _blend(ELO_ONLY)

    expected = sum(
        sub_a.expected_score(sub_b) for sub_a, sub_b in zip(a.sub_competitors, b.sub_competitors, strict=True)
    ) / len(a.sub_competitors)

    assert a.expected_score(b) == pytest.approx(expected)
    assert a.expected_score(b) == pytest.approx(0.5)  # equal ratings throughout


def test_expected_score_after_games_delegates_to_members():
    """The blend is the mean of member pairwise scores, computed on live members."""
    a, b = _blend(ELO_ONLY), _blend(ELO_ONLY)
    a.beat(b)

    manual = sum(
        sub_a.expected_score(sub_b) for sub_a, sub_b in zip(a.sub_competitors, b.sub_competitors, strict=True)
    ) / len(a.sub_competitors)

    assert a.expected_score(b) == pytest.approx(manual)
    assert a.expected_score(b) > 0.5  # the winner is now favored


def test_beat_updates_every_member():
    a, b = _blend(ELO_ONLY), _blend(ELO_ONLY)
    before = [c.rating for c in a.sub_competitors]

    a.beat(b)

    for sub, was in zip(a.sub_competitors, before, strict=True):
        assert sub.rating > was


def test_beat_forwards_scores_to_members_unchanged():
    """Margin scores reach each member: Pythagorean members record them."""
    specs = [
        {"type": "PythagoreanCompetitor", "competitor_kwargs": {}},
        {"type": "PythagoreanCompetitor", "competitor_kwargs": {}},
    ]
    a, b = _blend(specs), _blend(specs)

    a.beat(b, scores=(21.0, 10.0))

    for sub in a.sub_competitors:
        assert sub._points_for == 21.0
        assert sub._points_against == 10.0


def test_tied_delegates_to_member_ties():
    a = _blend(
        [
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1300, "k_factor": 24}},
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1300, "k_factor": 24}},
        ]
    )
    b = _blend(
        [
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200, "k_factor": 24}},
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200, "k_factor": 24}},
        ]
    )

    a.tied(b)

    for sub in a.sub_competitors:
        assert sub.rating < 1300.0
    for sub in b.sub_competitors:
        assert sub.rating > 1200.0


def test_reset_restores_every_member_to_initial():
    a, b = _blend(MIXED), _blend(MIXED)
    initial = [c.rating for c in a.sub_competitors]

    a.beat(b)
    a.reset()

    for sub, was in zip(a.sub_competitors, initial, strict=True):
        assert sub.rating == was


#
# Composition verification
#


def test_composition_mismatch_is_rejected():
    a = _blend(ELO_ONLY)
    b = _blend(MIXED)

    with pytest.raises(MissMatchedCompetitorTypesException, match="cannot be co-mingled"):
        a.verify_competitor_types(b)


def test_matching_composition_is_accepted():
    a, b = _blend(MIXED), _blend(MIXED)

    a.verify_competitor_types(b)  # must not raise
    assert a.expected_score(b) == pytest.approx(0.5)


def test_blend_versus_single_system_is_rejected():
    blended = _blend(ELO_ONLY)

    with pytest.raises(MissMatchedCompetitorTypesException):
        blended.verify_competitor_types(EloCompetitor(initial_rating=1200))


def test_members_of_same_composition_but_different_order_are_rejected():
    reordered = list(reversed(MIXED))
    a, b = _blend(MIXED), _blend(reordered)

    with pytest.raises(MissMatchedCompetitorTypesException, match="cannot be co-mingled"):
        a.verify_competitor_types(b)


#
# Serialization and import error paths
#


def test_state_round_trip_restores_every_member():
    a, b = _blend(MIXED), _blend(MIXED)
    a.beat(b)

    state = a.export_state()
    rebuilt = BlendedCompetitor.from_state(state)

    assert type(rebuilt) is BlendedCompetitor
    for sub, was in zip(rebuilt.sub_competitors, a.sub_competitors, strict=True):
        assert type(sub) is type(was)
        assert sub.rating == was.rating


def test_import_rejects_sub_state_missing_type():
    blended = _blend(MIXED)
    state = blended.export_state()
    del state["state"]["sub_competitors"][0]["type"]

    with pytest.raises(InvalidStateException, match="Missing competitor type in sub-competitor state"):
        BlendedCompetitor(competitors=MIXED).import_state(state)


def test_import_rejects_unknown_sub_type():
    """Unknown sub-competitor types surface from the registry, unwrapped.

    The class lookup in _import_current_state sits outside its try block, so
    the registry's InvalidParameterException propagates unchanged.
    """
    blended = _blend(MIXED)
    state = blended.export_state()
    state["state"]["sub_competitors"][0]["type"] = "NoSuchCompetitor"

    with pytest.raises(InvalidParameterException, match="Unknown competitor type"):
        BlendedCompetitor(competitors=MIXED).import_state(state)


def test_import_wraps_failing_sub_state_imports():
    """A sub-state that fails its own from_state is wrapped with its type name."""
    blended = _blend(MIXED)
    state = blended.export_state()
    state["state"]["sub_competitors"][1]["state"]["rd"] = -50

    with pytest.raises(InvalidStateException, match="Failed to import state for sub-competitor GlickoCompetitor"):
        BlendedCompetitor(competitors=MIXED).import_state(state)


def test_import_rejects_unsupported_blend_mode():
    blended = _blend(MIXED)
    state = blended.export_state()
    state["parameters"]["blend_mode"] = "median"

    with pytest.raises(InvalidParameterException, match="not supported"):
        BlendedCompetitor(competitors=MIXED).import_state(state)


def test_legacy_from_state_rebuilds_members_from_flat_specs():
    legacy = {
        "blend_mode": "mean",
        "competitors": [
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
            {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1450, "initial_rd": 180}},
        ],
    }

    rebuilt = BlendedCompetitor.from_state(legacy)

    assert rebuilt.sub_competitors[0].rating == 1500
    assert rebuilt.sub_competitors[1].rating == 1450


def test_blended_beat_rejects_scores_contradicting_a_win():
    """The blended win validator checks the payload against the declared win."""

    a = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    b = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])

    with pytest.raises(ValueError, match="do not describe a win"):
        a.beat(b, scores=(0.2, 0.7))


def test_blended_tied_rejects_scores_contradicting_a_draw():
    """The blended tie validator checks the payload against the declared draw."""

    a = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    b = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])

    with pytest.raises(ValueError, match="do not describe a draw"):
        a.tied(b, scores=(0.6, 0.4))


def test_modern_from_state_defaults_missing_parameter_keys():
    """Modern-format restore falls back to mean blending and an empty spec list."""

    state = {"type": "BlendedCompetitor", "version": 1, "parameters": {}, "state": {}}

    blended = BlendedCompetitor.from_state(state)

    assert blended.blend_mode == "mean"
    assert blended._initial_competitors == []
    assert blended.sub_competitors == []


def test_modern_from_state_restores_initial_competitor_specs():
    """Modern-format restore keeps the exact initial competitor specification."""

    spec = [{"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1700}}]
    state = {
        "type": "BlendedCompetitor",
        "version": 1,
        "parameters": {"blend_mode": "mean", "competitors": spec},
        "state": {},
    }

    blended = BlendedCompetitor.from_state(state)

    assert blended.blend_mode == "mean"
    assert blended._initial_competitors == spec


def test_import_state_defaults_missing_parameter_and_state_keys():
    """import_state falls back to mean blending and empty sections on sparse input."""

    blended = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    envelope = {"type": "BlendedCompetitor", "version": 1, "parameters": {}, "state": {}}

    blended.import_state(envelope)

    assert blended.blend_mode == "mean"
    assert blended._initial_competitors == []
    assert blended.sub_competitors == []


def test_import_state_restores_blend_mode_and_initial_specs():
    """import_state restores the blend mode and the initial specification list."""

    spec = [{"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1700}}]
    blended = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    envelope = {
        "type": "BlendedCompetitor",
        "version": 1,
        "parameters": {"blend_mode": "mean", "competitors": spec},
        "state": {},
    }

    blended.import_state(envelope)

    assert blended.blend_mode == "mean"
    assert blended._initial_competitors == spec


def test_legacy_from_state_defaults_for_missing_spec_keys():
    """Legacy entries without type or kwargs rebuild default Elo members."""

    legacy = {"competitors": [{"competitor_kwargs": {}}]}

    blended = BlendedCompetitor.from_state(legacy)

    assert [type(c).__name__ for c in blended.sub_competitors] == ["EloCompetitor"]
    assert blended.blend_mode == "mean"
    assert blended.sub_competitors[0].rating == 400


def test_legacy_from_state_without_competitors_constructs_empty():
    """Legacy state without a competitors section rebuilds an empty ensemble."""

    blended = BlendedCompetitor.from_state({"blend_mode": "mean"})

    assert blended.sub_competitors == []


def test_legacy_from_state_restores_explicit_member_specifications():
    """Legacy entries with explicit types and kwargs rebuild those exact members."""

    legacy = {
        "blend_mode": "mean",
        "competitors": [
            {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1600}},
            {"type": "EloCompetitor", "competitor_kwargs": {}},
        ],
    }

    blended = BlendedCompetitor.from_state(legacy)

    assert [type(c).__name__ for c in blended.sub_competitors] == ["GlickoCompetitor", "EloCompetitor"]
    assert blended.sub_competitors[0].rating == 1600
    assert blended.sub_competitors[1].rating == 400
    assert blended._initial_competitors == legacy["competitors"]


def test_legacy_from_state_wraps_sub_state_update_failures():
    """A member spec that builds but fails state import is wrapped and named."""

    legacy = {
        "competitors": [
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200, "k_factor": "oops"}}
        ]
    }

    with pytest.raises(InvalidStateException, match="Failed to update sub-competitor"):
        BlendedCompetitor.from_state(legacy)


def test_expected_score_rejects_unsupported_blend_mode():
    """The blend guard raises a named NotImplementedError with the mode in it."""

    a = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    b = BlendedCompetitor(competitors=[{"type": "EloCompetitor"}])
    a.blend_mode = "sum"

    with pytest.raises(NotImplementedError, match="not supported"):
        a.expected_score(b)

