"""Known-value and defining-behaviour tests for Whole-History Rating."""

import datetime
import unittest

from elote import WholeHistoryRatingCompetitor
from elote.competitors.base import BaseCompetitor


class TestWholeHistoryRating(unittest.TestCase):
    def test_known_value_single_win(self):
        """Reference: direct root of Coulom (2008) Eq. 4 likelihood plus Gaussian prior."""
        winner = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        loser = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        winner.beat(loser, datetime.datetime(2020, 1, 1))

        self.assertAlmostEqual(winner.rating, 1500.8591987719, places=7)
        self.assertAlmostEqual(loser.rating, 1499.1408012281, places=7)

    def test_known_value_two_wins_thirty_days_apart(self):
        """Reference: independent dense-Hessian Newton solve of Coulom (2008) Eqs. 4-6."""
        winner = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        loser = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        start = datetime.datetime(2020, 1, 1)
        winner.beat(loser, start)
        winner.beat(loser, start + datetime.timedelta(days=30))

        self.assertAlmostEqual(winner.rating_at(start), 1501.6006622749, places=7)
        self.assertAlmostEqual(winner.rating, 1523.9551256433, places=7)

    def test_later_results_revise_past_rating(self):
        start = datetime.datetime(2020, 1, 1)
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, start)
        before = a.rating_at(start)
        for offset in range(1, 11):
            a.beat(b, start + datetime.timedelta(days=offset * 10))

        self.assertGreater(a.rating_at(start), before)

    def test_long_lopsided_run_has_strict_exactly_complementary_probabilities(self):
        a = WholeHistoryRatingCompetitor(w2=1e9)
        b = WholeHistoryRatingCompetitor(w2=1e9)
        for _ in range(60):
            a.beat(b)
        probability = a.expected_score(b)
        reverse = b.expected_score(a)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)
        self.assertGreater(reverse, 0.0)
        self.assertLess(reverse, 1.0)
        self.assertEqual(probability + reverse, 1.0)

    def test_state_preserves_curve_and_registry(self):
        a = WholeHistoryRatingCompetitor()
        b = WholeHistoryRatingCompetitor()
        a.beat(b, datetime.datetime(2020, 1, 1))
        restored = BaseCompetitor.from_state(a.export_state())
        self.assertEqual(restored.rating_history(), a.rating_history())
        self.assertIn("WholeHistoryRatingCompetitor", BaseCompetitor.list_competitor_types())


if __name__ == "__main__":
    unittest.main()
