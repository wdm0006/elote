"""Rating-math tests for EloCompetitor.

Pins come from the classic Elo expectation (10**(r/400) ratio) and the
K-factor update rule, computed by hand. Note: the kill list mentions
"escalation multipliers", but no escalation branch exists in the current
implementation (verified by reading elo.py on this branch); the clamps and
K-factor paths below are the mutation targets that do exist.
"""

import unittest

from elote.competitors.elo import EloCompetitor
from elote.competitors.base import InvalidParameterException, InvalidRatingValueException


def elo_expected(rating, opponent_rating, base=400.0):
    return 10 ** (rating / base) / (10 ** (rating / base) + 10 ** (opponent_rating / base))


class EloExpectedScoreTest(unittest.TestCase):
    def test_equal_ratings_is_half(self):
        a, b = EloCompetitor(), EloCompetitor()
        self.assertAlmostEqual(a.expected_score(b), 0.5, places=12)

    def test_two_hundred_point_gap_pin(self):
        a, b = EloCompetitor(initial_rating=1400), EloCompetitor(initial_rating=1200)
        self.assertAlmostEqual(a.expected_score(b), 0.7597469266479577, places=10)
        self.assertAlmostEqual(a.expected_score(b), elo_expected(1400, 1200), places=10)

    def test_four_hundred_point_gap_pin(self):
        a, b = EloCompetitor(initial_rating=1400), EloCompetitor(initial_rating=1000)
        self.assertAlmostEqual(a.expected_score(b), 10.0 / 11.0, places=12)

    def test_argument_orders_sum_to_one(self):
        a, b = EloCompetitor(initial_rating=1500), EloCompetitor(initial_rating=1300)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)

    def test_transformed_rating_is_ten_power(self):
        a = EloCompetitor(initial_rating=400)
        self.assertAlmostEqual(a.transformed_rating, 10.0, places=9)


class EloUpdateTest(unittest.TestCase):
    def test_equal_players_win_pin(self):
        a, b = EloCompetitor(), EloCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.rating, 416.0, places=12)
        self.assertAlmostEqual(b.rating, 384.0, places=12)

    def test_tied_equal_players_unchanged(self):
        a, b = EloCompetitor(), EloCompetitor()
        a.tied(b)
        self.assertAlmostEqual(a.rating, 400.0, places=12)
        self.assertAlmostEqual(b.rating, 400.0, places=12)

    def test_tied_asymmetric_pin(self):
        a, b = EloCompetitor(initial_rating=1200), EloCompetitor(initial_rating=800)
        a.tied(b)
        gain = 32 * (0.5 - elo_expected(1200, 800))
        loss = 32 * (0.5 - elo_expected(800, 1200))
        self.assertAlmostEqual(a.rating, 1200 + gain, places=9)
        self.assertAlmostEqual(b.rating, 800 + loss, places=9)
        # A draw pulls the stronger player DOWN: actual 0.5 < expected 0.909.
        self.assertAlmostEqual(a.rating, 1186.909090909091, places=9)
        self.assertAlmostEqual(b.rating, 813.0909090909091, places=9)

    def test_blowout_gain_is_tiny(self):
        a, b = EloCompetitor(initial_rating=2800), EloCompetitor(initial_rating=1200)
        a.beat(b)
        change = a.rating - 2800.0
        self.assertGreater(change, 0.0)
        self.assertLess(change, 0.01)

    def test_ratings_are_zero_sum_per_game(self):
        a, b = EloCompetitor(initial_rating=1500), EloCompetitor(initial_rating=1300)
        a.beat(b)
        self.assertAlmostEqual((a.rating - 1500.0) + (b.rating - 1300.0), 0.0, places=9)


class EloClampTest(unittest.TestCase):
    """The kill list's "clamps": the rating floor and K-factor validation."""

    def test_rating_floored_at_minimum_on_loss(self):
        a = EloCompetitor(initial_rating=100)
        b = EloCompetitor(initial_rating=2000)
        b.beat(a)
        self.assertEqual(a.rating, 100.0)

    def test_unclamped_loss_would_dip_below(self):
        """Sanity: the same loss at a higher rating does move the rating."""
        a = EloCompetitor(initial_rating=1200)
        b = EloCompetitor(initial_rating=2000)
        b.beat(a)
        self.assertLess(a.rating, 1200.0)

    def test_rating_setter_rejects_negative(self):
        a = EloCompetitor()
        with self.assertRaises(InvalidRatingValueException):
            a.rating = -5.0

    def test_initial_rating_below_minimum_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            EloCompetitor(initial_rating=-100)

    def test_zero_k_factor_rejected(self):
        with self.assertRaises(InvalidParameterException):
            EloCompetitor(k_factor=0)

    def test_negative_k_factor_rejected(self):
        with self.assertRaises(InvalidParameterException):
            EloCompetitor(k_factor=-5.0)


class EloKFactorTest(unittest.TestCase):
    def test_custom_k_factor_pin(self):
        a, b = EloCompetitor(), EloCompetitor(k_factor=16)
        a.beat(b)
        # Each side updates with its own k_factor: a uses 32, b uses 16.
        self.assertAlmostEqual(a.rating, 416.0, places=12)
        self.assertAlmostEqual(b.rating, 392.0, places=12)
        c, d = EloCompetitor(k_factor=16), EloCompetitor()
        c.beat(d)
        self.assertAlmostEqual(c.rating, 408.0, places=12)
        self.assertAlmostEqual(d.rating, 384.0, places=12)

    def test_state_round_trip_preserves_k_factor(self):
        a = EloCompetitor(k_factor=16)
        b = EloCompetitor()
        a.beat(b)
        restored = EloCompetitor.from_state(a.export_state())
        self.assertEqual(restored._k_factor, 16.0)

    def test_default_k_factor_not_serialized(self):
        a = EloCompetitor()
        state = a.export_state()
        self.assertIsNone(state["parameters"]["k_factor"])

    def test_reset_restores_initial_rating(self):
        a = EloCompetitor(initial_rating=900)
        b = EloCompetitor(initial_rating=1100)
        a.beat(b)
        a.reset()
        self.assertEqual(a.rating, 900.0)


if __name__ == "__main__":
    unittest.main()
