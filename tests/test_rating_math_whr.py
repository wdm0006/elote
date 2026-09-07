"""Rating-math and edge tests for the WholeHistoryRating system.

Covers the parameter-validation boundaries, chronological day handling,
history lookup edges, strict expected-score clamping, score forwarding,
serialization restart, and repeated-fit stability flagged by the mutation
campaign.
"""

import unittest
from datetime import date, datetime

from elote.competitors.base import InvalidParameterException
from elote.competitors.whr import WholeHistoryRatingCompetitor


class WHRParameterValidationTest(unittest.TestCase):
    def test_nonpositive_w2_rejected(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(InvalidParameterException):
                WholeHistoryRatingCompetitor(w2=bad)

    def test_nonpositive_precision_rejected(self):
        for bad in (0.0, -1e-3, float("inf"), float("nan")):
            with self.assertRaises(InvalidParameterException):
                WholeHistoryRatingCompetitor(precision=bad)

    def test_bad_max_iterations_rejected(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(InvalidParameterException):
                WholeHistoryRatingCompetitor(max_iterations=bad)

    def test_nonfinite_initial_rating_rejected(self):
        for bad in (float("inf"), float("nan")):
            with self.assertRaises(InvalidParameterException):
                WholeHistoryRatingCompetitor(initial_rating=bad)


class WHRChronologyTest(unittest.TestCase):
    def test_out_of_order_days_are_sorted(self):
        """Games logged out of order must land on a sorted day axis."""
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, match_time=datetime(2020, 1, 5))
        a.beat(b, match_time=datetime(2020, 1, 1))

        self.assertEqual(a._days, sorted(a._days))
        self.assertEqual(len(a._days), 2)

    def test_rating_at_before_first_game_returns_initial(self):
        a = WholeHistoryRatingCompetitor(initial_rating=1234.0)
        b = WholeHistoryRatingCompetitor()
        a.beat(b, match_time=datetime(2020, 6, 1))

        self.assertEqual(a.rating_at(date(2019, 1, 1)), 1234.0)

    def test_rating_at_between_days_returns_last_fit(self):
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, match_time=datetime(2020, 1, 1))
        a.beat(b, match_time=datetime(2020, 3, 1))

        history = a.rating_history()
        self.assertEqual(a.rating_at(date(2020, 2, 1)), history[0][1])
        self.assertEqual(a.rating_at(datetime(2020, 4, 1)), history[-1][1])

    def test_repeated_fit_is_stable(self):
        """The dirty-flag refit must be idempotent for identical game sets."""
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, match_time=datetime(2020, 1, 1))

        first = a.rating
        second = a.rating
        self.assertEqual(first, second)


class WHRExpectedScoreTest(unittest.TestCase):
    def test_extreme_ratings_clamp_strictly(self):
        """The probability must approach but never reach 1.0."""
        big = WholeHistoryRatingCompetitor(initial_rating=10**6)
        tiny = WholeHistoryRatingCompetitor(initial_rating=-(10**6))

        self.assertGreater(big.expected_score(tiny), 0.5)
        self.assertLess(big.expected_score(tiny), 1.0)
        self.assertAlmostEqual(tiny.expected_score(big), 1.0 - big.expected_score(tiny), places=9)

    def test_fresh_players_are_near_even(self):
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        self.assertAlmostEqual(a.expected_score(b), 0.5, places=1)


class WHRScorePropagationTest(unittest.TestCase):
    def test_lost_to_forwards_custom_scores(self):
        winner = WholeHistoryRatingCompetitor()
        loser = WholeHistoryRatingCompetitor()
        winner.beat(loser, match_time=datetime(2020, 2, 1), scores=(6.0, 1.0))

        winner2 = WholeHistoryRatingCompetitor()
        loser2 = WholeHistoryRatingCompetitor()
        loser2.lost_to(winner2, match_time=datetime(2020, 2, 1), scores=(1.0, 6.0))

        self.assertAlmostEqual(winner.rating, winner2.rating, places=9)
        self.assertAlmostEqual(loser.rating, loser2.rating, places=9)


class WHRSerializationTest(unittest.TestCase):
    def test_state_round_trip_and_restart(self):
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, match_time=datetime(2020, 1, 1))

        restored = WholeHistoryRatingCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=6)

        # The restored competitor must keep fitting on new games; the restart
        # seeds the curve from the restored last-known rating.
        rating_before = restored.rating
        c = WholeHistoryRatingCompetitor()
        restored.beat(c, match_time=datetime(2020, 2, 1))
        self.assertTrue(restored.rating_history())
        self.assertNotAlmostEqual(restored.rating, rating_before, places=3)


if __name__ == "__main__":
    unittest.main()
