"""Numeric reference values for KeenerCompetitor.

Every value below was calculated independently of ``elote``: the two-competitor cases have a
closed form (recomputed here from the documented formula), and the three-competitor cases
were produced by a 60-digit ``decimal`` power iteration written separately from the shipped
implementation, which uses ``numpy.linalg.eig``.

Recall the construction for a connected group, with ``S_ij`` the points ``i`` scored on ``j``:

    a_ij = (S_ij + 1) / (S_ij + S_ji + 2)                  Laplace-smoothed score share
    h(x) = 1/2 + 1/2 * sgn(x - 1/2) * sqrt(|2x - 1|)       Keener's skew transform
    H_ij = h(a_ij) / games_i + eps                         row-normalized, eps = 1e-4
    r    = dominant eigenvector of H, scaled to mean 1.0

Ratings are canonicalized to ``_round_decimals`` (10) places, so assertions use 9 places.
"""

import math
import unittest

from elote import KeenerCompetitor


PLACES = 9
EPS = KeenerCompetitor._perturbation


def _skew(share):
    """Keener's skew transform, written out independently of the implementation."""
    centered = 2.0 * share - 1.0
    return 0.5 + 0.5 * math.copysign(math.sqrt(abs(centered)), centered)


class TestKeenerKnownValues(unittest.TestCase):
    def test_single_scored_game_matches_the_two_by_two_closed_form(self):
        """One 35-3 game between two competitors.

        S = [[0, 35], [3, 0]], so a_01 = 36/40 = 0.9 and a_10 = 4/40 = 0.1, and

            h(0.9) = 1/2 + 1/2*sqrt(0.8),   h(0.1) = 1/2 - 1/2*sqrt(0.8).

        Both played one game, so the row normalization is a no-op and

            H = [[eps, p], [q, eps]]  with  p = h(0.9) + eps, q = h(0.1) + eps.

        For that shape the eigenvalues are eps +/- sqrt(p*q) and the dominant eigenvector is
        proportional to (sqrt(p), sqrt(q)); scaling to mean 1.0 over n = 2 gives

            r = (2*sqrt(p) / (sqrt(p) + sqrt(q)),  2*sqrt(q) / (sqrt(p) + sqrt(q))).
        """
        p = _skew(0.9) + EPS
        q = _skew(0.1) + EPS
        root_p, root_q = math.sqrt(p), math.sqrt(q)
        expected_a = 2.0 * root_p / (root_p + root_q)
        expected_b = 2.0 * root_q / (root_p + root_q)
        # Guards the arithmetic above against a silent change in the closed form.
        self.assertAlmostEqual(expected_a, 1.617757795347885, places=12)
        self.assertAlmostEqual(expected_b, 0.382242204652116, places=12)

        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        self.assertAlmostEqual(a.rating, expected_a, places=PLACES)
        self.assertAlmostEqual(b.rating, expected_b, places=PLACES)

    def test_single_drawn_game_leaves_both_at_the_population_mean(self):
        """A drawn game gives a_01 = a_10 = 1/2, so h = 1/2 on both sides and H is symmetric.

        The dominant eigenvector of a symmetric 2x2 with equal off-diagonals is (1, 1),
        which scales to (1.0, 1.0).
        """
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.tied(b, scores=(14.0, 14.0))

        self.assertAlmostEqual(a.rating, 1.0, places=PLACES)
        self.assertAlmostEqual(b.rating, 1.0, places=PLACES)

    def test_asymmetric_scored_schedule(self):
        """A beat B 28-7, B beat C 14-7, A beat C 35-3.

        Score matrix (rows A, B, C):

            S = [[ 0, 28, 35],
                 [ 7,  0, 14],
                 [ 3,  7,  0]]

        so a_AB = 29/37, a_BA = 8/37, a_AC = 36/40, a_CA = 4/40, a_BC = 15/23, a_CB = 8/23.
        Games played: A two, B two, C two.  Reference ratings from a 60-digit power
        iteration on the resulting H:

            A = 1.73792467527929154, B = 0.83492367232361227, C = 0.42715165239709619
        """
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(28.0, 7.0))
        b.beat(c, scores=(14.0, 7.0))
        a.beat(c, scores=(35.0, 3.0))

        self.assertAlmostEqual(a.rating, 1.73792467527929154, places=PLACES)
        self.assertAlmostEqual(b.rating, 0.83492367232361227, places=PLACES)
        self.assertAlmostEqual(c.rating, 0.42715165239709619, places=PLACES)

    def test_scored_schedule_containing_a_draw(self):
        """A beat B 30-10, B drew C 14-14, A beat C 21-14.

        The drawn game contributes a_BC = a_CB = 1/2, i.e. h = 1/2 in both directions, and
        still counts as a game played for both -- which is what pulls C above B here despite
        C never winning.  Reference ratings from the same 60-digit power iteration:

            A = 1.40226342754892460, B = 0.73428336424082162, C = 0.86345320821025379
        """
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(30.0, 10.0))
        b.tied(c, scores=(14.0, 14.0))
        a.beat(c, scores=(21.0, 14.0))

        self.assertAlmostEqual(a.rating, 1.40226342754892460, places=PLACES)
        self.assertAlmostEqual(b.rating, 0.73428336424082162, places=PLACES)
        self.assertAlmostEqual(c.rating, 0.86345320821025379, places=PLACES)

    def test_unit_score_fallback_schedule(self):
        """A beat B, B beat C, with no score payload at all.

        The unit fallback records 1-0 for each win, so a_AB = 2/3 and h(2/3) = 1/2 +
        sqrt(1/3)/2 = 0.7886751345948129, with h(1/3) the complement.  A and C never met, so
        their entries are 0 before the eps perturbation, and B's row is halved because B
        played twice:

            H = [[eps,      0.7886751 + eps,  eps            ],
                 [0.1056624 + eps, eps,       0.3943376 + eps],
                 [eps,      0.2114249 + eps,  eps            ]]

        Reference ratings from the same independent power iteration over that matrix:

            A = 1.67957399744031952, B = 0.86984806605940995, C = 0.45057793650027053
        """
        a, b, c = KeenerCompetitor(), KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        b.beat(c)

        self.assertAlmostEqual(a.rating, 1.67957399744031952, places=PLACES)
        self.assertAlmostEqual(b.rating, 0.86984806605940995, places=PLACES)
        self.assertAlmostEqual(c.rating, 0.45057793650027053, places=PLACES)


if __name__ == "__main__":
    unittest.main()
