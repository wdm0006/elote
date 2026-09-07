"""Rating-math tests for MasseyCompetitor.

Pins come from closed-form least-squares solutions of the documented Massey system
(M = D - A with the last row replaced by the zero-mean constraint) computed by hand
and with numpy, independently of the implementation under test.
"""

import unittest

import numpy as np

from elote.competitors.massey import MasseyCompetitor
from elote.competitors.base import InvalidParameterException


def independent_massey(games_, n):
    """Solve the documented Massey system; games_ is one (i, j, margin) per game."""
    games = [dict() for _ in range(n)]
    for i, j, _ in games_:
        games[i][j] = games[i].get(j, 0) + 1
        games[j][i] = games[j].get(i, 0) + 1
    matrix = np.zeros((n, n))
    p = np.zeros(n)
    for i in range(n):
        matrix[i, i] = sum(games[i].values())
        p[i] = sum(m for (a, b, m) in games_ if a == i) - sum(m for (a, b, m) in games_ if b == i)
        for j, count in games[i].items():
            if j != i:
                matrix[i, j] = -count
    matrix[n - 1, :] = 1.0
    p[n - 1] = 0.0
    return np.linalg.solve(matrix, p).tolist()


class MasseyUnitMarginTest(unittest.TestCase):
    def test_one_game_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, -0.5, places=12)

    def test_two_wins_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.beat(b)
        reference = independent_massey([(0, 1, 1.0), (0, 1, 1.0)], 2)
        self.assertAlmostEqual(a.rating, reference[0], places=12)
        self.assertAlmostEqual(b.rating, reference[1], places=12)
        # Cumulative margins [2, -2] with a singular M (two meetings) resolve through the
        # zero-mean constraint to a unit-margin split, same as a single win.
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, -0.5, places=12)

    def test_margin_scores_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(6.0, 1.0))
        self.assertAlmostEqual(a.rating, 2.5, places=12)
        self.assertAlmostEqual(b.rating, -2.5, places=12)

    def test_unit_and_margin_paths_differ(self):
        """Score-sensitive margins must not collapse to unit margins."""
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(6.0, 1.0))
        c, d = MasseyCompetitor(), MasseyCompetitor()
        c.beat(d)
        self.assertNotAlmostEqual(a.rating, c.rating, places=6)

    def test_tie_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.tied(b)
        self.assertAlmostEqual(a.rating, 0.0, places=12)
        self.assertAlmostEqual(b.rating, 0.0, places=12)
        self.assertEqual(a.num_games, 1)
        self.assertEqual(b.num_games, 1)

    def test_three_player_chain_pin(self):
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        b.beat(c)
        reference = independent_massey([(0, 1, 1.0), (1, 2, 1.0)], 3)
        for player, expected in zip((a, b, c), reference, strict=True):
            self.assertAlmostEqual(player.rating, expected, places=12)
        # Chain margins [1, 0, -1]: the middle player nets zero.
        self.assertAlmostEqual(a.rating, 1.0, places=12)
        self.assertAlmostEqual(b.rating, 0.0, places=12)
        self.assertAlmostEqual(c.rating, -1.0, places=12)

    def test_order_independence(self):
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        c.lost_to(a)
        b.lost_to(a)
        c.lost_to(b)
        d, e, f = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        d.beat(e)
        d.beat(f)
        e.beat(f)
        for first, second in zip((a, b, c), (d, e, f), strict=True):
            self.assertAlmostEqual(first.rating, second.rating, places=9)


class MasseyLostToTest(unittest.TestCase):
    def test_lost_to_forwards_reversed_scores(self):
        """lost_to must reverse the caller-order score pair before beat validates it."""
        a, b = MasseyCompetitor(), MasseyCompetitor()
        b.lost_to(a, scores=(1.0, 6.0))
        c, d = MasseyCompetitor(), MasseyCompetitor()
        c.beat(d, scores=(6.0, 1.0))
        self.assertAlmostEqual(a.rating, c.rating, places=12)
        self.assertAlmostEqual(b.rating, d.rating, places=12)

    def test_lost_to_without_scores_uses_unit_margins(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        b.lost_to(a)
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, -0.5, places=12)


class MasseyExpectedScoreTest(unittest.TestCase):
    def test_fresh_unit_difference_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a._rating = 1.0
        b._rating = 0.0
        # Unfitted competitors keep rating scale 1.0: tanh(0.5 * 2 * 1).
        expected = 0.5 * (1.0 + np.tanh(1.0))
        self.assertAlmostEqual(a.expected_score(b), float(expected), places=12)

    def test_fitted_one_game_pin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        # Ratings 0.5/-0.5, spread sqrt(2) * std = 0.7071; tanh(0.5 * 2 * 1 / 0.7071).
        self.assertAlmostEqual(a.expected_score(b), 0.944192780793, places=9)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)

    def test_tied_pair_is_half(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.tied(b)
        self.assertAlmostEqual(a.expected_score(b), 0.5, places=12)
        self.assertAlmostEqual(b.expected_score(a), 0.5, places=12)

    def test_rating_scale_floored_for_degenerate_groups(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.tied(b)
        # The fit is lazy: recording the tie only marks the group stale, and the floored
        # scale is recorded by that deferred re-fit. Reading the ratings forces it; a tied
        # pair fits to equal (zero) ratings, so the spread floors at _minimum_rating_scale.
        self.assertAlmostEqual(a.rating, 0.0, places=12)
        self.assertAlmostEqual(b.rating, 0.0, places=12)
        self.assertEqual(a._rating_scale, 1e-9)
        self.assertEqual(b._rating_scale, 1e-9)


class MasseyFallbackTest(unittest.TestCase):
    """The LinAlgError fallback assigns average-margin ratings."""

    def test_fallback_math_pin(self):
        a = MasseyCompetitor()
        b = MasseyCompetitor()
        # Simulate a two-game history for a and an empty one for b.
        a._point_differential = 2.0
        a._wins = 2
        a._losses = 0
        a._ties = 0
        b._point_differential = 0.0
        a._fallback_rating_calculation([a, b])
        # Average margins: a = 1.0, b = 0.0; mean 0.5; centered [0.5, -0.5].
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, -0.5, places=12)

    def test_fallback_keeps_zero_game_competitor_at_zero_average(self):
        a = MasseyCompetitor()
        b = MasseyCompetitor()
        a._point_differential = 2.0
        a._wins = 2
        b._fallback_rating_calculation([a, b])
        self.assertAlmostEqual(b.rating, -0.5, places=12)

    def test_fallback_centers_around_mean(self):
        comps = [MasseyCompetitor() for _ in range(3)]
        comps[0]._point_differential = 3.0
        comps[0]._wins = 3
        comps[0]._ties = 0
        comps[0]._losses = 0
        comps[1]._point_differential = -3.0
        comps[1]._wins = 0
        comps[1]._losses = 3
        comps[1]._ties = 0
        comps[2]._point_differential = 0.0
        comps[2]._wins = 0
        comps[2]._losses = 0
        comps[2]._ties = 0
        comps[0]._fallback_rating_calculation(comps)
        ratings = [c.rating for c in comps]
        self.assertAlmostEqual(sum(ratings), 0.0, places=12)
        self.assertAlmostEqual(ratings[0], 1.0, places=12)
        self.assertAlmostEqual(ratings[1], -1.0, places=12)
        self.assertAlmostEqual(ratings[2], 0.0, places=12)


class MasseyDegenerateAndConfigTest(unittest.TestCase):
    def test_single_competitor_recalculation_is_a_no_op(self):
        a = MasseyCompetitor(initial_rating=2.0)
        a._recalculate_ratings()
        self.assertEqual(a.rating, 2.0)

    def test_rating_scale_survives_state_round_trip(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        restored = MasseyCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=12)
        self.assertAlmostEqual(restored._rating_scale, a._rating_scale, places=12)
        self.assertEqual(restored._wins, 1)
        self.assertEqual(restored._point_differential, 1.0)

    def test_scale_config_rejected_at_zero(self):
        original = MasseyCompetitor._expected_score_scale
        try:
            with self.assertRaises(InvalidParameterException):
                MasseyCompetitor.configure_class(expected_score_scale=0)
        finally:
            MasseyCompetitor._expected_score_scale = original

    def test_negative_scale_config_rejected(self):
        original = MasseyCompetitor._expected_score_scale
        try:
            with self.assertRaises(InvalidParameterException):
                MasseyCompetitor.configure_class(expected_score_scale=-0.5)
        finally:
            MasseyCompetitor._expected_score_scale = original


if __name__ == "__main__":
    unittest.main()
