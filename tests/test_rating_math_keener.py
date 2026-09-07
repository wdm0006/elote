"""Rating-math tests for KeenerCompetitor.

Every hard number below was computed with an independent reference implementation
of Keener's method (Laplace-smoothed score shares, the h(x) transform, a symmetric
perturbation, division by games + prior, dominant eigenvector rescaled to sum to n)
and cross-checked against the Glicko-style audit anchors before being pinned here.
"""

import math
import unittest

from elote.competitors.keener import KeenerCompetitor
from elote.competitors.base import InvalidParameterException


def keener_h(x: float) -> float:
    """The nonlinear Keener score transform: h on the preference share."""
    return 0.5 + 0.5 * math.copysign(math.sqrt(abs(2 * x - 1)), x - 0.5)


def independent_keener_ratings(scores, pair_games, games, games_prior=2.0, eps=1e-4):
    """Dominant eigenvector of the documented Keener preference matrix."""
    import numpy as np

    n = len(games)
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if pair_games[i][j] <= 0:
                continue
            total = scores[i][j] + scores[j][i]
            share = (scores[i][j] + 1.0) / (total + 2.0)
            a[i][j] = keener_h(share) / (games[i] + games_prior)
    # The implementation perturbs every entry (eps * E), not just the diagonal.
    matrix = [[a[i][j] + eps for j in range(n)] for i in range(n)]
    values, vectors = np.linalg.eig(np.array(matrix))
    dom = int(np.argmax(values.real))
    vec = np.real(vectors[:, dom])
    if vec.sum() < 0:
        vec = -vec
    return [float(v * n / vec.sum()) for v in vec]


class KeenerScaleConfigTest(unittest.TestCase):
    def setUp(self):
        self._original = KeenerCompetitor._expected_score_scale

    def tearDown(self):
        KeenerCompetitor._expected_score_scale = self._original

    def test_zero_scale_rejected(self):
        with self.assertRaises(InvalidParameterException):
            KeenerCompetitor.configure_class(expected_score_scale=0)

    def test_negative_scale_rejected(self):
        with self.assertRaises(InvalidParameterException):
            KeenerCompetitor.configure_class(expected_score_scale=-1.5)

    def test_positive_scale_accepted(self):
        KeenerCompetitor.configure_class(expected_score_scale=2.0)
        self.assertEqual(KeenerCompetitor._expected_score_scale, 2.0)

    def test_positive_scale_changes_expected_score(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        baseline = a.expected_score(b)
        KeenerCompetitor.configure_class(expected_score_scale=2.0)
        # Sharpening the logistic moves the same rating gap to a more extreme probability.
        self.assertGreater(a.expected_score(b), baseline)
        self.assertLess(a.expected_score(b), 1.0)


class KeenerOneGameTest(unittest.TestCase):
    """One unit-score game: the canonical audit anchor."""

    def test_ratings_pin(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        reference = independent_keener_ratings([[0.0, 1.0], [0.0, 0.0]], [[0, 1], [1, 0]], [1.0, 1.0])
        self.assertAlmostEqual(a.rating, reference[0], places=9)
        self.assertAlmostEqual(b.rating, reference[1], places=9)
        self.assertAlmostEqual(a.rating, 1.317603874, places=6)
        self.assertAlmostEqual(b.rating, 0.682396126, places=6)

    def test_ratings_sum_to_n(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.rating + b.rating, 2.0, places=9)

    def test_ratings_are_scale_sensitive_to_scores(self):
        """A 6-1 win must not produce the ratings of a 1-0 win."""
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(6.0, 1.0))
        reference = independent_keener_ratings([[0.0, 6.0], [1.0, 0.0]], [[0, 1], [1, 0]], [1.0, 1.0])
        self.assertAlmostEqual(a.rating, reference[0], places=9)
        self.assertAlmostEqual(b.rating, reference[1], places=9)
        self.assertAlmostEqual(a.rating, 1.446811556, places=6)
        self.assertAlmostEqual(b.rating, 0.553188444, places=6)

    def test_scores_reversed_between_sides(self):
        """The opponent must be credited the score it conceded, not the winner's."""
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(6.0, 1.0))
        # Same result recorded the other way round must agree exactly.
        c, d = KeenerCompetitor(), KeenerCompetitor()
        d.lost_to(c, scores=(1.0, 6.0))
        self.assertAlmostEqual(a.rating, c.rating, places=9)
        self.assertAlmostEqual(b.rating, d.rating, places=9)

    def test_games_count_recorded_once(self):
        """_record_game must count one game per side, not two."""
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        self.assertEqual(a._opponents[b], 1)
        self.assertEqual(b._opponents[a], 1)
        self.assertAlmostEqual(a._scores_for[b], 1.0, places=12)
        self.assertAlmostEqual(b._scores_for[a], 0.0, places=12)


class KeenerChainTest(unittest.TestCase):
    """Three-player chain: connectivity must reach every member of the group."""

    def test_chain_ratings_pin(self):
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        b.beat(c)
        a.beat(c)
        reference = independent_keener_ratings(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [2.0, 2.0, 2.0],
        )
        for player, expected in zip((a, b, c), reference, strict=True):
            self.assertAlmostEqual(player.rating, expected, places=9)
        self.assertAlmostEqual(a.rating, 1.455606945, places=6)
        self.assertAlmostEqual(b.rating, 0.938848006, places=6)
        self.assertAlmostEqual(c.rating, 0.605545049, places=6)

    def test_order_independence(self):
        """Same results in a different order give the same fit."""
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        c.lost_to(b)
        c.lost_to(a)
        b.lost_to(a)
        d, e, f = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        d.beat(e)
        d.beat(f)
        e.beat(f)
        for first, second in zip((a, b, c), (d, e, f), strict=True):
            self.assertAlmostEqual(first.rating, second.rating, places=9)

    def test_unplayed_players_are_not_isolated_from_the_fit(self):
        """A player only connected through one opponent still gets fitted values."""
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        b.beat(c)
        # c is only connected to b; if traversal stopped early its rating would stay flat.
        self.assertNotEqual(c.rating, c._initial_rating)


class KeenerExpectedScoreTest(unittest.TestCase):
    def test_fresh_equal_is_half(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        self.assertEqual(a.expected_score(b), 0.5)

    def test_one_game_pin(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        log_ratio = math.log(a.rating) - math.log(b.rating)
        expected = 0.5 * (1.0 + math.tanh(0.5 * 1.0 * log_ratio))
        self.assertAlmostEqual(a.expected_score(b), expected, places=12)
        self.assertAlmostEqual(a.expected_score(b), 0.658801937, places=9)

    def test_argument_orders_are_exactly_complementary(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(6.0, 1.0))
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)

    def test_probability_floor_keeps_extremes_below_one(self):
        a, b = KeenerCompetitor(initial_rating=1000.0), KeenerCompetitor()
        a.beat(b)
        self.assertLess(a.expected_score(b), 1.0)
        self.assertGreater(b.expected_score(a), 0.0)


class KeenerDegeneratePathsTest(unittest.TestCase):
    def test_single_competitor_recalculation_is_a_no_op(self):
        """The n <= 1 guard must leave a lone competitor's rating untouched."""
        a = KeenerCompetitor(initial_rating=2.0)
        a._recalculate_ratings()
        self.assertEqual(a.rating, 2.0)

    def test_unplayed_competitor_keeps_initial_rating(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        c = KeenerCompetitor(initial_rating=3.0)
        self.assertEqual(c.rating, 3.0)

    def test_h_transform_bounds(self):
        self.assertAlmostEqual(keener_h(0.5), 0.5, places=12)
        self.assertLess(keener_h(0.1), 0.5)
        self.assertGreater(keener_h(0.9), 0.5)


if __name__ == "__main__":
    unittest.main()
