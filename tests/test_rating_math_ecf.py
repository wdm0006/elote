"""Rating-math tests for ECFCompetitor.

The kill list's "±40 band" is the ±_delta clamp in the current implementation
(_delta defaults to 50, verified by reading ecf.py on this branch); the tests
pin the real boundary at own_rating ± _delta. Ratings are the mean of a rolling
deque of performance ratings, so every pin below is a closed-form deque mean.

The ECF floor is 100, so boundary fixtures live at 250-351 rather than 49-151.
"""

import unittest
from collections import deque

from elote.competitors.ecf import ECFCompetitor
from elote.competitors.base import InvalidRatingValueException


def clamp(own, opp, delta=50.0):
    return max(own - delta, min(own + delta, opp))


def mean(values):
    return sum(values) / len(values)


class ECFBeatBandBoundaryTest(unittest.TestCase):
    """The clamp boundary between opponent ratings one point apart."""

    def test_even_game_pin(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=300)
        a.beat(b)
        # performance ratings 350 and 250 -> deques [300, 350] and [300, 250].
        self.assertAlmostEqual(a.rating, 325.0, places=12)
        self.assertAlmostEqual(b.rating, 275.0, places=12)

    def test_upper_clamp_boundary_below(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=349)
        a.beat(b)
        # 349 inside window [250, 350]: perf_self = 399, perf_opp = 299.
        self.assertAlmostEqual(a.rating, 349.5, places=12)
        self.assertAlmostEqual(b.rating, 299.5, places=12)

    def test_upper_clamp_boundary_at(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=350)
        a.beat(b)
        # 350 exactly at own + delta: perf_self = 400, perf_opp = 250.
        self.assertAlmostEqual(a.rating, 350.0, places=12)
        self.assertAlmostEqual(b.rating, 300.0, places=12)

    def test_upper_clamp_boundary_above(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=351)
        a.beat(b)
        # 351 clipped to 350: perf_self = 400, perf_opp = 251.
        self.assertAlmostEqual(a.rating, 350.0, places=12)
        self.assertAlmostEqual(b.rating, 301.0, places=12)

    def test_lower_clamp_boundary_at(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=250)
        a.beat(b)
        # 250 exactly at own - delta: perf_self = 300 (no gain), perf_opp = 200.
        self.assertAlmostEqual(a.rating, 300.0, places=12)
        self.assertAlmostEqual(b.rating, 250.0, places=12)

    def test_lower_clamp_boundary_above(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=251)
        a.beat(b)
        # 251 just inside: perf_self = 301, perf_opp = 201.
        self.assertAlmostEqual(a.rating, 300.5, places=12)
        self.assertAlmostEqual(b.rating, 250.5, places=12)

    def test_lower_clamp_boundary_below(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=249)
        a.beat(b)
        # 249 clipped to 250 on both sides: no change for either player.
        self.assertAlmostEqual(a.rating, 300.0, places=12)
        self.assertAlmostEqual(b.rating, 249.0, places=12)


class ECFTieTest(unittest.TestCase):
    def test_tie_pulls_both_toward_each_other(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=251)
        a.tied(b)
        # Entries: clamp(300, 251) = 251 for a; clamp(251, 300) = 300 for b.
        self.assertAlmostEqual(a.rating, 275.5, places=12)
        self.assertAlmostEqual(b.rating, 275.5, places=12)

    def test_tie_between_equals_is_noop(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=300)
        a.tied(b)
        self.assertAlmostEqual(a.rating, 300.0, places=12)
        self.assertAlmostEqual(b.rating, 300.0, places=12)

    def test_lost_to_matches_beat(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=251)
        b.lost_to(a)
        c, d = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=251)
        c.beat(d)
        self.assertAlmostEqual(a.rating, c.rating, places=12)
        self.assertAlmostEqual(b.rating, d.rating, places=12)


class ECFDequeTest(unittest.TestCase):
    def test_rating_is_mean_of_deque(self):
        a = ECFCompetitor(initial_rating=300)
        self.assertEqual(a.rating, 300.0)

    def test_oldest_score_evicted_after_n_periods(self):
        """After 31 games the initial rating must be gone from the deque."""
        a = ECFCompetitor(initial_rating=300)
        opponent = ECFCompetitor(initial_rating=1000)
        for _ in range(31):
            a.beat(opponent)
        # Independent two-sided simulation of the same deque rule.
        own_entries = deque([300.0], maxlen=30)
        opp_entries = deque([1000.0], maxlen=30)
        for _ in range(31):
            own_rating = mean(own_entries)
            opp_rating = mean(opp_entries)
            own_entries.append(clamp(own_rating, opp_rating) + 50.0)
            opp_entries.append(clamp(opp_rating, own_rating) - 50.0)
        self.assertAlmostEqual(a.rating, mean(own_entries), places=6)
        self.assertAlmostEqual(opponent.rating, mean(opp_entries), places=6)
        # The initial rating must have been evicted after 31 updates.
        self.assertEqual(len(own_entries), 30)
        self.assertNotIn(300.0, list(own_entries))

    def test_initial_rating_below_minimum_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            ECFCompetitor(initial_rating=-10)

    def test_initial_rating_just_below_floor_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            ECFCompetitor(initial_rating=99)


class ECFConversionsTest(unittest.TestCase):
    def test_elo_conversion_pin(self):
        a = ECFCompetitor(initial_rating=300)
        self.assertAlmostEqual(a.elo_conversion, 2950.0, places=9)

    def test_expected_score_equal_is_half(self):
        a, b = ECFCompetitor(initial_rating=300), ECFCompetitor(initial_rating=300)
        self.assertAlmostEqual(a.expected_score(b), 0.5, places=12)

    def test_expected_score_higher_rating_wins(self):
        a, b = ECFCompetitor(initial_rating=350), ECFCompetitor(initial_rating=260)
        self.assertGreater(a.expected_score(b), 0.5)
        self.assertLess(b.expected_score(a), 0.5)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
