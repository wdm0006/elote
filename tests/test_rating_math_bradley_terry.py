"""Rating-math tests for the Bradley-Terry system (Hunter 2004 MM updates).

Anchors come from an independent MM iteration implemented in this module over
the paired-comparison counts; guards target the iteration/regularization
bounds the mutation campaign flagged.
"""

import math
import unittest

from elote.competitors.bradley_terry import BradleyTerryCompetitor

ANCHOR = 1500.0  # the mean log-strength maps back to 1500, not 1000
SCALE = 400.0 / math.log(10.0)
REG = 0.1


def independent_mm_ratings(games, iterations=10000, reg=REG, tol=1e-8):
    """BT MM ratings for ``games`` = [(winner_idx, loser_idx, win_score)].

    ``win_score`` is 1.0 for a win or 0.5 each side for a tie.  The update is
    p_i <- (W_i + reg) / (sum_j N_ij / (p_i + p_j) + 2*reg / (p_i + 1)), with
    a unit-strength phantom opponent providing the virtual results; strengths
    are geometric-mean normalized and mapped with mean-centered betas.
    """
    n = 1 + max(max(winner, loser) for winner, loser, _ in games)
    wins = [0.0] * n
    counts = [[0.0] * n for _ in range(n)]
    for winner, loser, score in games:
        wins[winner] += score
        wins[loser] += 1.0 - score
        counts[winner][loser] += 1.0
        counts[loser][winner] += 1.0

    p = [1.0] * n
    for _ in range(iterations):
        new_p = []
        for i in range(n):
            denom = 2.0 * reg / (p[i] + 1.0)
            for j in range(n):
                if j != i and counts[i][j] > 0:
                    denom += counts[i][j] / (p[i] + p[j])
            new_p.append((wins[i] + reg) / denom)
        geo_mean = math.exp(sum(math.log(x) for x in new_p) / n)
        p = [x / geo_mean for x in new_p]
        if max(abs(math.log(p[i]) - math.log(new_p[i])) for i in range(n)) < tol:
            break

    betas = [math.log(x) for x in p]
    mean_beta = sum(betas) / n
    return [ANCHOR + SCALE * (b - mean_beta) for b in betas]


class BradleyTerryAnchorsTest(unittest.TestCase):
    def test_single_game_pin(self):
        """One decisive game must land on the documented Elo-scale anchor."""
        winner = BradleyTerryCompetitor()
        loser = BradleyTerryCompetitor()
        winner.beat(loser)

        self.assertAlmostEqual(winner.rating, 1739.400869, places=6)
        self.assertAlmostEqual(loser.rating, 1260.599131, places=6)

        ref = independent_mm_ratings([(0, 1, 1.0)])
        self.assertAlmostEqual(winner.rating, ref[0], places=3)
        self.assertAlmostEqual(loser.rating, ref[1], places=3)

    def test_win_plus_tie_pin(self):
        """A win followed by a tie must match the MM solution for 1.5/0.5."""
        winner = BradleyTerryCompetitor()
        loser = BradleyTerryCompetitor()
        winner.beat(loser)
        winner.tied(loser)

        self.assertAlmostEqual(winner.rating, 1589.670249, places=6)
        self.assertAlmostEqual(loser.rating, 1410.329751, places=6)

    def test_three_player_chain_pin(self):
        """a>b, a>c, b>c: the middle player nets zero and stays at anchor."""
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        c = BradleyTerryCompetitor()
        a.beat(b)
        a.beat(c)
        b.beat(c)

        # places=4: the anchored mean leaves O(1e-6) float residue on 1500.
        self.assertAlmostEqual(a.rating, 1930.280418, places=4)
        self.assertAlmostEqual(b.rating, 1500.0, places=4)
        self.assertAlmostEqual(c.rating, 1069.719575, places=4)

        ref = independent_mm_ratings([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)])
        self.assertAlmostEqual(a.rating, ref[0], places=3)
        self.assertAlmostEqual(b.rating, ref[1], places=3)
        self.assertAlmostEqual(c.rating, ref[2], places=3)

    def test_order_independence(self):
        """MM is order-independent: any play order must give one solution."""
        forward = (BradleyTerryCompetitor(), BradleyTerryCompetitor())
        forward[0].beat(forward[1])
        forward[0].tied(forward[1])

        reverse = (BradleyTerryCompetitor(), BradleyTerryCompetitor())
        reverse[0].tied(reverse[1])
        reverse[0].beat(reverse[1])

        self.assertAlmostEqual(forward[0].rating, reverse[0].rating, places=6)
        self.assertAlmostEqual(forward[1].rating, reverse[1].rating, places=6)


class BradleyTerryGuardsTest(unittest.TestCase):
    def test_iteration_cap_keeps_ratings_finite(self):
        """A single MM step must still produce finite, ordered ratings."""
        original = BradleyTerryCompetitor._max_iter
        try:
            BradleyTerryCompetitor._max_iter = 1
            winner = BradleyTerryCompetitor()
            loser = BradleyTerryCompetitor()
            winner.beat(loser)

            self.assertTrue(math.isfinite(winner.rating))
            self.assertTrue(math.isfinite(loser.rating))
            self.assertGreater(winner.rating, loser.rating)
        finally:
            BradleyTerryCompetitor._max_iter = original

    def test_zero_game_player_is_updated_by_group_fit(self):
        """Idle players in a connected group still get fitted (not frozen)."""
        played = BradleyTerryCompetitor()
        idle = BradleyTerryCompetitor()
        played.beat(idle)

        self.assertAlmostEqual(played.rating, 1739.400869, places=6)
        self.assertAlmostEqual(idle.rating, 1260.599131, places=6)

    def test_lone_competitor_direct_recalc_is_noop(self):
        """A one-node network refit (white-box) must not move the rating.

        The n <= 1 guard keeps a vacuous MM fit from pulling a lone
        competitor's custom initial rating back to the 1500 anchor.
        """
        lone = BradleyTerryCompetitor(initial_rating=1600)
        lone._recalculate_ratings()
        self.assertAlmostEqual(lone.rating, 1600.0, places=9)


class BradleyTerryStateTest(unittest.TestCase):
    def test_state_round_trip_preserves_math(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.beat(b)

        restored = BradleyTerryCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=9)

        # A win plus a tie against a fresh opponent reuses the documented
        # two-game anchor -- proof the restored competitor keeps fitting.
        fresh = BradleyTerryCompetitor()
        restored.beat(fresh)
        restored.tied(fresh)
        self.assertAlmostEqual(restored.rating, 1589.670249, places=4)


if __name__ == "__main__":
    unittest.main()
