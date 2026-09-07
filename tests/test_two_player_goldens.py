"""Golden fixtures proving the two-player surface is byte-identical.

The N-player bout work (``match_group``) must not change any two-player code
path. This module runs a fixed, deterministic battery of two-player flows --
``matchup`` wins/losses/draws with and without scores, ``rating_period``,
``tournament``, and ``process_history`` -- against every no-argument
constructible rating system and serializes the exported arena state to exact
JSON. The committed fixture at ``tests/data/two_player_golden_fixtures.json``
was captured from the pre-change tree; the test asserts byte equality against
it.

Regenerating the fixture re-baselines behavior: run
``scripts/generate_two_player_goldens.py`` only from a tree whose two-player
behavior is known-good (a fresh checkout of the base commit), never from a
tree with unreviewed rating changes.

``BlendedCompetitor`` and ``TeamCompetitor`` are excluded: their constructors
require per-instance children/rosters, and sharing those instances through
``base_competitor_kwargs`` across a whole arena population would alias member
state between competitors. Their two-player behavior is covered by the
serialization and team tests instead.

``WholeHistoryRatingCompetitor`` is excluded because its exported state is not
byte-stable across processes: the solver iterates the ``_opponents`` set while
aggregating win probabilities, and randomized string hashing reorders that
summation between runs, drifting the last digits of its ratings. Its behavior
is covered by its own known-values tests at the suite's usual tolerances.
"""

import datetime
import json
from pathlib import Path
from typing import Dict

import pytest

from elote import (
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    DWZCompetitor,
    ECFCompetitor,
    EloCompetitor,
    Glicko2Competitor,
    GlickoBoostCompetitor,
    GlickoCompetitor,
    KeenerCompetitor,
    LambdaArena,
    MasseyCompetitor,
    OpenSkillCompetitor,
    PythagoreanCompetitor,
    TrueSkillCompetitor,
)

FIXTURE_PATH = Path(__file__).parent / "data" / "two_player_golden_fixtures.json"
# Naive on purpose: the battery feeds the same timestamp to time-aware and
# non-time-aware systems, and Glicko compares it against a naive internal
# default when a competitor's first bout carries no match time.
FIXED_TIME = datetime.datetime(2026, 1, 15, 12, 0, 0)


def _is_time_aware(competitor_class) -> bool:
    """Whether bouts for this system consume wall-clock defaults when no match
    time is supplied (the ``_last_activity`` duck-type the arenas use)."""
    return hasattr(competitor_class(), "_last_activity")


def _strip_volatile_metadata(state: Dict) -> Dict:
    """Remove per-instance provenance fields from an exported state document.

    Exported documents carry fields that identify the specific competitor
    instance rather than its rating state: ``id`` (a random UUID, or a memory
    address for some systems) and ``created_at`` (a wall-clock export stamp).
    Both change on every run, so they are stripped before the byte comparison;
    otherwise the fixture would differ between runs for reasons unrelated to
    rating behavior.
    """
    if isinstance(state, dict):
        return {
            key: _strip_volatile_metadata(value)
            for key, value in state.items()
            if key not in ("id", "created_at")
        }
    if isinstance(state, list):
        return [_strip_volatile_metadata(item) for item in state]
    return state

# Every exported system whose exported state is byte-stable across processes.
# Composite systems that need per-instance members are excluded (see module
# docstring), as is WholeHistoryRatingCompetitor (hash-order float drift).
SYSTEMS = (
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    GlickoBoostCompetitor,
    TrueSkillCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    ColleyMatrixCompetitor,
    MasseyCompetitor,
    KeenerCompetitor,
    PythagoreanCompetitor,
    BradleyTerryCompetitor,
    OpenSkillCompetitor,
)


def _battery_state(competitor_class) -> str:
    """Run the deterministic two-player battery against one system.

    Covers every two-player arena entry point: matchup (win, loss, draw, with
    scores, with attributes and match time), rating_period, the tournament
    splat convention, the legacy process_history path, and a prediction read.

    Every bout that can carry a match time does (matchup, tournament tuples,
    ``period_end``), because time-aware systems default to ``datetime.now()``
    -- and inflate their uncertainty from it -- when no time is supplied, which
    would make the exported state depend on the wall clock. The legacy
    ``process_history`` path has no time channel at all, so it only runs for
    systems that are not time-aware.
    """
    arena = LambdaArena(lambda a, b: True, base_competitor=competitor_class)
    arena.matchup("alice", "bob", outcome=1.0, match_time=FIXED_TIME)
    arena.matchup("alice", "bob", outcome=0.0, match_time=FIXED_TIME)
    arena.matchup("alice", "bob", outcome=0.5, scores=(1.0, 1.0), match_time=FIXED_TIME)
    arena.matchup("bob", "carol", outcome=0.0, attributes={"surface": "clay"}, match_time=FIXED_TIME)
    arena.matchup("carol", "dave", outcome=1.0, scores=(3.0, 2.0), match_time=FIXED_TIME)
    arena.rating_period(
        [
            ("alice", "carol", 0.5, None),
            ("bob", "dave", 1.0, (2.0, 1.0)),
        ],
        period_end=FIXED_TIME,
    )
    arena.tournament(
        [
            ("alice", "bob", None, FIXED_TIME),
            ("dave", "carol", None, FIXED_TIME, 1.0),
        ]
    )
    if not _is_time_aware(competitor_class):
        arena.process_history([("alice", "bob", 1.0), ("bob", "carol", 0.5)])
    assert arena.expected_score("alice", "bob") is not None
    state = _strip_volatile_metadata(arena.export_state())
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def build_golden_states() -> Dict[str, str]:
    """Run the battery for every system, keyed by class name."""
    return {competitor_class.__name__: _battery_state(competitor_class) for competitor_class in SYSTEMS}


@pytest.fixture(scope="module")
def golden_fixture() -> Dict:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            "Golden fixture is missing. Regenerate it from a known-good (pre-change) tree with "
            "scripts/generate_two_player_goldens.py -- never from a tree with rating changes."
        )
    return json.loads(FIXTURE_PATH.read_text())


@pytest.mark.parametrize("competitor_class", SYSTEMS, ids=lambda c: c.__name__)
def test_two_player_paths_byte_identical(competitor_class, golden_fixture):
    """The full two-player battery reproduces the pre-change bytes exactly."""
    name = competitor_class.__name__
    assert name in golden_fixture["systems"], f"{name} missing from the golden fixture"
    assert _battery_state(competitor_class) == golden_fixture["systems"][name], (
        f"{name}'s two-player state diverged from the golden fixture -- the N-player work "
        "changed a two-player code path. Fix the regression; do not regenerate the fixture."
    )


def test_golden_fixture_covers_all_systems(golden_fixture):
    """The fixture records its provenance and one entry per shipped system."""
    assert golden_fixture["generated_from_commit"]
    assert set(golden_fixture["systems"]) == {c.__name__ for c in SYSTEMS}
