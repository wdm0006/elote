"""Rating-math tests for ColleyMatrixCompetitor.

Pins come from closed-form solutions of the Colley system
(C = 2I + D - A, b = 1 + (2*wins - games)/2) computed independently with numpy
and by hand, per Colley's 2002 paper.
"""

import unittest

import numpy as np

from elote.competitors.colley import ColleyMatrixCompetitor
from elote.competitors.base import InvalidRatingValueException


def independent_colley(results, n):
    """Solve C r = b; results are (i, j, s_i, s_j) with s_i + s_j == 1.

    A tie (0.5, 0.5) counts one game towards C but a half-win to neither side,
    so b_i = 1 + sum(s_i - 0.5) over games.
    """
    matrix = np.zeros((n, n))
    b = np.zeros(n)
    for i, j, si, sj in results:
        matrix[i, i] += 1
        matrix[j, j] += 1
        matrix[i, j] -= 1
        matrix[j, i] -= 1
        b[i] += si - 0.5
        b[j] += sj - 0.5
    for i in range(n):
        matrix[i, i] += 2
        b[i] += 1
    return np.linalg.solve(matrix, b).tolist()


class ColleyAssemblyTest(unittest.TestCase):
    """_recalculate_ratings must assemble C and b exactly as documented."""

    def test_one_win_pin(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.rating, 0.625, places=12)
        self.assertAlmostEqual(b.rating, 0.375, places=12)

    def test_two_wins_pin(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        a.beat(b)
        self.assertAlmostEqual(a.rating, 2.0 / 3.0, places=12)
        self.assertAlmostEqual(b.rating, 1.0 / 3.0, places=12)

    def test_win_plus_tie_pin(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        a.tied(b)
        reference = independent_colley([(0, 1, 1.0, 0.0), (0, 1, 0.5, 0.5)], 2)
        self.assertAlmostEqual(a.rating, reference[0], places=12)
        self.assertAlmostEqual(b.rating, reference[1], places=12)
        self.assertAlmostEqual(a.rating, 7.0 / 12.0, places=12)
        self.assertAlmostEqual(b.rating, 5.0 / 12.0, places=12)

    def test_tie_only_pin(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.tied(b)
        self.assertAlmostEqual(a.rating, 0.5, places=12)
        self.assertAlmostEqual(b.rating, 0.5, places=12)

    def test_three_player_connectivity_pin(self):
        """Both losers connect only through the winner; all three must be re-fitted."""
        a, b, c = ColleyMatrixCompetitor(), ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        a.beat(c)
        reference = independent_colley([(0, 1, 1.0, 0.0), (0, 2, 1.0, 0.0)], 3)
        for player, expected in zip((a, b, c), reference, strict=True):
            self.assertAlmostEqual(player.rating, expected, places=12)
        self.assertAlmostEqual(a.rating, 0.7, places=12)
        self.assertAlmostEqual(b.rating, 0.4, places=12)
        self.assertAlmostEqual(c.rating, 0.4, places=12)

    def test_four_player_chain_pin(self):
        a, b, c, d = (
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
        )
        a.beat(b)
        b.beat(c)
        c.beat(d)
        reference = independent_colley([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0), (2, 3, 1.0, 0.0)], 4)
        for player, expected in zip((a, b, c, d), reference, strict=True):
            self.assertAlmostEqual(player.rating, expected, places=12)
        self.assertAlmostEqual(a.rating, 19.0 / 28.0, places=12)
        self.assertAlmostEqual(b.rating, 15.0 / 28.0, places=12)
        self.assertAlmostEqual(c.rating, 13.0 / 28.0, places=12)
        self.assertAlmostEqual(d.rating, 9.0 / 28.0, places=12)

    def test_cycle_network_connectivity_pin(self):
        """Cycles in the match graph must not truncate the connected set."""
        a, b, c, d = (
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
            ColleyMatrixCompetitor(),
        )
        a.beat(b)
        a.beat(c)
        a.beat(d)
        c.beat(d)  # cross edge: duplicate pop must not stop the traversal
        self.assertEqual(len(a._get_connected_competitors()), 4)
        for player in (a, b, c, d):
            self.assertNotEqual(player.rating, player._initial_rating)

    def test_ratings_sum_to_n_over_two(self):
        a, b, c = ColleyMatrixCompetitor(), ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        b.beat(c)
        total = a.rating + b.rating + c.rating
        self.assertAlmostEqual(total, 1.5, places=12)

    def test_order_independence(self):
        a, b, c = ColleyMatrixCompetitor(), ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        c.lost_to(a)
        b.lost_to(a)
        c.lost_to(b)
        d, e, f = ColleyMatrixCompetitor(), ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        d.beat(e)
        d.beat(f)
        e.beat(f)
        for first, second in zip((a, b, c), (d, e, f), strict=True):
            self.assertAlmostEqual(first.rating, second.rating, places=9)

    def test_tied_counts_games_and_half_wins(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.tied(b)
        self.assertEqual(a._ties, 1)
        self.assertEqual(b._ties, 1)
        self.assertEqual(a.num_games, 1)
        self.assertAlmostEqual(a._head_to_head[b], 0.5, places=12)
        self.assertAlmostEqual(b._head_to_head[a], 0.5, places=12)


class ColleyExpectedScoreTest(unittest.TestCase):
    def test_equal_ratings_is_half(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        self.assertEqual(a.expected_score(b), 0.5)

    def test_one_win_difference_pin(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        # ratings differ by 0.25: logistic with sharpness 4.
        expected = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(a.expected_score(b), float(expected), places=12)
        self.assertAlmostEqual(a.expected_score(b), 0.7310585786300049, places=12)

    def test_argument_orders_sum_to_one(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)


class ColleyStateTest(unittest.TestCase):
    def test_state_round_trip_preserves_record(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        a.tied(b)
        restored = ColleyMatrixCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=12)
        self.assertEqual(restored._wins, 1)
        self.assertEqual(restored._ties, 1)
        self.assertEqual(restored._losses, 0)
        self.assertEqual(restored._initial_rating, 0.5)

    def test_rebuilt_competitor_can_play_again(self):
        a, b = ColleyMatrixCompetitor(), ColleyMatrixCompetitor()
        a.beat(b)
        restored = ColleyMatrixCompetitor.from_state(a.export_state())
        # The restored graph is empty; a fresh game must still produce valid ratings.
        other = ColleyMatrixCompetitor()
        restored.beat(other)
        self.assertGreater(restored.rating, 0.5)
        self.assertLess(other.rating, 0.5)

    def test_rating_below_minimum_rejected(self):
        a = ColleyMatrixCompetitor()
        with self.assertRaises(InvalidRatingValueException):
            a.rating = -0.1

    def test_initial_rating_below_minimum_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            ColleyMatrixCompetitor(initial_rating=-1.0)


class ColleyDegeneratePathsTest(unittest.TestCase):
    def test_single_competitor_recalculation_is_a_no_op(self):
        a = ColleyMatrixCompetitor(initial_rating=2.0)
        a._recalculate_ratings()
        self.assertEqual(a.rating, 2.0)

    def test_fallback_win_percentage_math(self):
        """White-box: the singular-matrix fallback scales win percentage around 0.5."""
        a = ColleyMatrixCompetitor()
        b = ColleyMatrixCompetitor()
        a._wins = 2
        a._losses = 0
        a._ties = 0
        a._opponents = {b: 2}
        b._wins = 0
        b._losses = 1
        b._ties = 0
        b._opponents = {a: 1}
        a._fallback_rating_calculation([a, b])
        # a: win_pct 1.0 -> 0.75; b: win_pct 0.0 -> 0.25; sums to n/2, no rescale.
        self.assertAlmostEqual(a.rating, 0.75, places=12)
        self.assertAlmostEqual(b.rating, 0.25, places=12)

    def test_fallback_keeps_unplayed_competitor_at_initial(self):
        a = ColleyMatrixCompetitor()
        fresh = ColleyMatrixCompetitor(initial_rating=0.9)
        a._wins = 1
        a._fallback_rating_calculation([a, fresh])
        # Win pct 1.0 -> 0.75; the unplayed competitor keeps 0.9. The sum is then
        # normalized to n/2 (scale 1/1.65), and the winner is bumped to initial + 0.001
        # because normalization alone would have dropped it below its initial rating.
        self.assertAlmostEqual(a.rating, 0.501, places=9)
        self.assertAlmostEqual(fresh.rating, 0.5454545454545455, places=9)


if __name__ == "__main__":
    unittest.main()
