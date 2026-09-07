"""Rating-math tests for PythagoreanExpectationCompetitor.

The shipped class is ``PythagoreanCompetitor`` (verified by reading
pythagorean.py on this branch). Pins come from an independent log5 reference:
ratings are prior-smoothed win expectations 1/(1 + ((PA+p)/(PF+p))^k), combined
with Bill James' log5 formula in its "sum of edges" form.
"""

import unittest

from elote.competitors.pythagorean import PythagoreanCompetitor
from elote.competitors.base import InvalidParameterException


def pythagorean_rating(points_for, points_against, k=2.37, prior=1.0):
    """Independent rating reference: prior-smoothed Pythagorean expectation."""
    scored = points_for + prior
    allowed = points_against + prior
    return 1.0 / (1.0 + (allowed / scored) ** k)


def log5(higher, lower):
    """Independent log5 in the sum-of-edges form used by the documentation."""
    product = higher * lower
    denominator = (higher - product) + (lower - product)
    if denominator <= 0.0:
        return 0.5
    return (higher - product) / denominator


def unit_record(wins, losses, k=2.37):
    return pythagorean_rating(float(wins), float(losses), k)


class PythagoreanRatingTest(unittest.TestCase):
    def test_fresh_rating_is_exactly_half(self):
        a = PythagoreanCompetitor()
        self.assertEqual(a.rating, 0.5)

    def test_one_unit_win_pin(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        # PF=1, PA=0 with prior 1: 1/(1 + (1/2)^k).
        self.assertAlmostEqual(a.rating, unit_record(1, 0), places=12)
        self.assertAlmostEqual(b.rating, unit_record(0, 1), places=12)
        self.assertAlmostEqual(a.rating, 0.8379, places=4)

    def test_unbeaten_stays_below_one(self):
        """The symmetric prior keeps every rating strictly inside (0, 1)."""
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        for _ in range(10):
            a.beat(b)
        # 10-0 with the prior: 1/(1 + (1/11)^2.37) ~= 0.9966.
        self.assertGreater(a.rating, 0.99)
        self.assertLess(a.rating, 1.0)
        self.assertAlmostEqual(a.rating, unit_record(10, 0), places=12)

    def test_winless_stays_above_zero(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        for _ in range(10):
            b.beat(a)
        self.assertLess(a.rating, 0.01)
        self.assertGreater(a.rating, 0.0)
        self.assertAlmostEqual(a.rating, unit_record(0, 10), places=12)

    def test_symmetric_records_tie_at_half(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        b.beat(a)
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, 0.5, places=12)

    def test_rating_read_only(self):
        a = PythagoreanCompetitor()
        with self.assertRaises(NotImplementedError):
            a.rating = 0.9

    def test_num_games_counts_both_sides(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        a.tied(b)
        self.assertEqual(a.num_games, 2)
        self.assertEqual(b.num_games, 2)


class PythagoreanLog5Test(unittest.TestCase):
    def test_equal_ratings_is_exactly_half(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        self.assertEqual(a.expected_score(b), 0.5)

    def test_fresh_opponent_returns_own_rating(self):
        """log5 against w_b = 0.5 reduces exactly to w_a."""
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        fresh = PythagoreanCompetitor()
        self.assertAlmostEqual(a.expected_score(fresh), a.rating, places=12)

    def test_two_player_league_pin(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        for _ in range(6):
            a.beat(b)
        for _ in range(2):
            b.beat(a)
        mine = log5(unit_record(6, 2), unit_record(2, 6))
        self.assertAlmostEqual(a.expected_score(b), mine, places=12)
        self.assertAlmostEqual(b.expected_score(a), 1.0 - mine, places=12)
        self.assertAlmostEqual(a.expected_score(b), 0.9823, places=4)

    def test_argument_orders_sum_to_exactly_one(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        total = a.expected_score(b) + b.expected_score(a)
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_heavily_lopsided_pair_stays_at_most_one(self):
        """The sum-of-edges denominator keeps the quotient inside [0.5, 1]."""
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        for _ in range(30):
            a.beat(b)
        self.assertLessEqual(a.expected_score(b), 1.0)
        self.assertGreater(a.expected_score(b), 0.99)

    def test_lost_to_matches_beat(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        b.lost_to(a)
        c = PythagoreanCompetitor()
        d = PythagoreanCompetitor()
        c.beat(d)
        self.assertAlmostEqual(a.rating, c.rating, places=12)
        self.assertAlmostEqual(b.rating, d.rating, places=12)


class PythagoreanExponentTest(unittest.TestCase):
    def _played_pair(self, exponent):
        a = PythagoreanCompetitor(exponent=exponent)
        b = PythagoreanCompetitor(exponent=exponent)
        for _ in range(6):
            a.beat(b)
        for _ in range(2):
            b.beat(a)
        return a, b

    def test_larger_exponent_sharpens_log5(self):
        flat_a, _ = self._played_pair(2.37)
        sharp_a, _ = self._played_pair(3.0)
        self.assertGreater(sharp_a.rating, flat_a.rating)
        b_ref = PythagoreanCompetitor(exponent=3.0)
        # The winner's log5 edge grows with k and stays strictly below 1.
        flat = flat_a.expected_score(PythagoreanCompetitor())
        sharp = sharp_a.expected_score(b_ref)
        self.assertGreater(sharp, flat)
        self.assertLess(sharp, 1.0)

    def test_zero_exponent_rejected(self):
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor(exponent=0)

    def test_negative_exponent_rejected(self):
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor(exponent=-1.0)

    def test_class_config_rejects_bad_exponent(self):
        original = PythagoreanCompetitor._exponent
        try:
            with self.assertRaises(InvalidParameterException):
                PythagoreanCompetitor.configure_class(exponent=0)
        finally:
            PythagoreanCompetitor._exponent = original

    def test_class_config_rejects_nonpositive_prior(self):
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor.configure_class(prior_points=0)


class PythagoreanStateTest(unittest.TestCase):
    def test_state_round_trip_restores_totals(self):
        a = PythagoreanCompetitor(exponent=2.0)
        b = PythagoreanCompetitor(exponent=2.0)
        a.beat(b, scores=(24.0, 10.0))
        a.tied(b)
        restored = PythagoreanCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=12)
        self.assertEqual(restored.num_games, 2)

    def test_reset_clears_totals(self):
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b)
        a.reset()
        self.assertEqual(a.rating, 0.5)
        self.assertEqual(a.num_games, 0)

    def test_real_scores_move_the_rating(self):
        """Real point totals must differ from the unit-score smoothing."""
        a = PythagoreanCompetitor()
        b = PythagoreanCompetitor()
        a.beat(b, scores=(30.0, 3.0))
        unit = PythagoreanCompetitor()
        other = PythagoreanCompetitor()
        unit.beat(other)
        self.assertGreater(a.rating, unit.rating)
        self.assertAlmostEqual(a.rating, pythagorean_rating(30.0, 3.0), places=12)


if __name__ == "__main__":
    unittest.main()
