"""Tests for replaying historical, dated results through the Glicko rating systems.

Glicko-1 and Glicko-2 are the two systems whose rating deviation inflates over elapsed
time, so they are the two that accept ``match_time``. A competitor that has never played
has no activity to compare a match time against: it adopts the time of its first match.
That is what makes ``LambdaArena.matchup(..., match_time=...)`` usable on historical data,
where competitors are created lazily and every timestamp is in the past.
"""

import unittest
from datetime import datetime, timedelta

from elote import Glicko2Competitor, GlickoCompetitor, LambdaArena
from elote.competitors.base import InvalidParameterException


def _always_true(a, b, attributes=None):
    """Comparison function that always awards the win to the first competitor."""
    return True


COMPETITOR_CLASSES = (GlickoCompetitor, Glicko2Competitor)

# A dated schedule far enough in the past that it can never be mistaken for wall-clock now,
# spanning months and given in increasing order.
START = datetime(2024, 1, 1)
SCHEDULE = (
    ("a", "b", START),
    ("b", "c", START + timedelta(days=45)),
    ("a", "c", START + timedelta(days=120)),
    ("c", "b", START + timedelta(days=200)),
    ("a", "b", START + timedelta(days=280)),
)


def _replay(competitor_class, schedule):
    """Run a dated schedule through a fresh arena and return it."""
    arena = LambdaArena(_always_true, base_competitor=competitor_class)
    for a, b, match_time in schedule:
        arena.matchup(a, b, match_time=match_time)
    return arena


class TestHistoricalMatchups(unittest.TestCase):
    def test_first_matchup_accepts_a_past_timestamp(self):
        """A lazily created competitor takes its first match's time, however old."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                arena = LambdaArena(_always_true, base_competitor=competitor_class)
                arena.matchup("x", "y", match_time=datetime(2024, 1, 1))

                self.assertEqual(arena.competitors["x"]._last_activity, datetime(2024, 1, 1))
                self.assertEqual(arena.competitors["y"]._last_activity, datetime(2024, 1, 1))

    def test_dated_replay_inflates_rd_relative_to_a_gapless_replay(self):
        """The elapsed months between historical matches actually inflate RD."""
        gapless = tuple((a, b, START) for a, b, _ in SCHEDULE)

        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                spread = _replay(competitor_class, SCHEDULE)
                simultaneous = _replay(competitor_class, gapless)

                for name in ("a", "b", "c"):
                    self.assertGreater(
                        spread.competitors[name].rd,
                        simultaneous.competitors[name].rd,
                        f"{name} should be less certain after months of gaps",
                    )

    def test_out_of_order_match_still_raises(self):
        """A second match dated before the first is still rejected."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                arena = LambdaArena(_always_true, base_competitor=competitor_class)
                arena.matchup("x", "y", match_time=datetime(2024, 6, 1))

                with self.assertRaises(InvalidParameterException):
                    arena.matchup("x", "y", match_time=datetime(2024, 5, 31))

    def test_out_of_order_match_raises_for_the_opponent_too(self):
        """The guard covers a competitor that is stale only via the opponent."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                arena = LambdaArena(_always_true, base_competitor=competitor_class)
                arena.matchup("x", "y", match_time=datetime(2024, 6, 1))

                # "z" has never played, so only "y" carries a last-activity time here.
                with self.assertRaises(InvalidParameterException):
                    arena.matchup("z", "y", match_time=datetime(2024, 5, 31))

    def test_match_time_none_is_unaffected(self):
        """Omitting match_time keeps working and stamps wall-clock activity."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                before = datetime.now()
                arena = LambdaArena(_always_true, base_competitor=competitor_class)
                arena.matchup("x", "y")
                after = datetime.now()

                for name in ("x", "y"):
                    self.assertIsNotNone(arena.competitors[name]._last_activity)
                    self.assertGreaterEqual(arena.competitors[name]._last_activity, before)
                    self.assertLessEqual(arena.competitors[name]._last_activity, after)

    def test_explicit_initial_time_still_bounds_matches(self):
        """A competitor constructed with initial_time keeps rejecting earlier matches."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                first = competitor_class(initial_time=datetime(2024, 6, 1))
                second = competitor_class(initial_time=datetime(2024, 6, 1))

                with self.assertRaises(InvalidParameterException):
                    first.beat(second, match_time=datetime(2024, 5, 1))


class TestHistoricalStateRoundTrip(unittest.TestCase):
    """A never-played competitor and a played one must both survive serialization."""

    def test_never_played_competitor_round_trips_and_accepts_history(self):
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                fresh = competitor_class()
                state = fresh.export_state()
                self.assertIsNone(state["state"]["last_activity"])

                restored = competitor_class.from_state(state)
                self.assertIsNone(restored._last_activity)

                # The restored competitor can still start its history in the past.
                restored.beat(competitor_class.from_state(state), match_time=datetime(2024, 1, 1))
                self.assertEqual(restored._last_activity, datetime(2024, 1, 1))

    def test_played_competitor_round_trips_and_continues_identically(self):
        follow_up = START + timedelta(days=365)

        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                original = _replay(competitor_class, SCHEDULE)
                restored = LambdaArena(
                    _always_true,
                    base_competitor=competitor_class,
                    initial_state=original.export_state(),
                )

                for name in ("a", "b", "c"):
                    self.assertAlmostEqual(restored.competitors[name].rating, original.competitors[name].rating)
                    self.assertAlmostEqual(restored.competitors[name].rd, original.competitors[name].rd)

                # Continue play on both with an explicit match time, so wall-clock drift
                # between the two arenas cannot enter the comparison.
                original.matchup("a", "c", match_time=follow_up)
                restored.matchup("a", "c", match_time=follow_up)

                for name in ("a", "c"):
                    self.assertAlmostEqual(restored.competitors[name].rating, original.competitors[name].rating)
                    self.assertAlmostEqual(restored.competitors[name].rd, original.competitors[name].rd)

    def test_legacy_state_without_last_activity_still_imports(self):
        """A state document written before null activity was representable is unchanged."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                state = competitor_class(initial_time=datetime(2024, 1, 1)).export_state()
                del state["state"]["last_activity"]
                del state["last_activity"]

                before = datetime.now()
                restored = competitor_class.from_state(state)
                after = datetime.now()

                self.assertIsNotNone(restored._last_activity)
                self.assertGreaterEqual(restored._last_activity, before)
                self.assertLessEqual(restored._last_activity, after)


if __name__ == "__main__":
    unittest.main()
