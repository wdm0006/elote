import unittest

from elote import MasseyCompetitor


class TestMasseyKnownValues(unittest.TestCase):
    """Numeric references for MasseyCompetitor, hand-solved from the linear system."""

    def test_initial_rating(self):
        """A fresh competitor sits at the zero-mean origin unless told otherwise."""
        self.assertEqual(MasseyCompetitor().rating, 0.0)
        self.assertEqual(MasseyCompetitor(initial_rating=-2.5).rating, -2.5)

    def test_expected_score(self):
        """expected_score is the documented logistic on the rating difference."""
        # scale = 2.0, so a rating difference d gives 1 / (1 + exp(-2 d)).
        # d = 0.5  ->  1 / (1 + exp(-1)) = 0.7310585786300049
        strong = MasseyCompetitor(initial_rating=0.25)
        weak = MasseyCompetitor(initial_rating=-0.25)

        self.assertAlmostEqual(strong.expected_score(weak), 0.7310585786300049, places=12)
        self.assertAlmostEqual(weak.expected_score(strong), 0.2689414213699951, places=12)

    def test_three_competitor_transitive_schedule(self):
        """A > B, A > C, B > C solves to (2/3, 0, -2/3).

        Every competitor plays two games, so D = diag(2, 2, 2), and each pair meets once,
        giving

            M = D - A = [[ 2, -1, -1],
                         [-1,  2, -1],
                         [-1, -1,  2]]

        Unit margins give p = (+2, 0, -2): A wins twice, B splits, C loses twice.
        Replacing the last row of M with ones and the last entry of p with 0 pins the
        ratings to zero mean, so with a + b + c = 0:

            row 1:  2a - b - c = 2  ->  2a + a = 3a = 2  ->  a =  2/3
            row 2: -a + 2b - c = 0  -> -a + 2b + a + b = 3b = 0  ->  b =  0
            zero mean:                                             c = -2/3
        """
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.beat(c)
        b.beat(c)

        self.assertAlmostEqual(a.rating, 2.0 / 3.0, places=9)
        self.assertAlmostEqual(b.rating, 0.0, places=9)
        self.assertAlmostEqual(c.rating, -2.0 / 3.0, places=9)

    def test_four_competitor_schedule_with_a_draw(self):
        """A > B, A > C, B ~ C, C > D, D > A solves to (1/4, -3/8, 0, 1/8).

        Games played: A three (B, C, D), B two (A, C), C three (A, B, D), D two (C, A),
        so D = diag(3, 2, 3, 2). Every listed pair meets once and B never plays D:

            M = D - A = [[ 3, -1, -1, -1],
                         [-1,  2, -1,  0],
                         [-1, -1,  3, -1],
                         [-1,  0, -1,  2]]

        Unit margins, with the draw contributing 0 to both sides but still counting as a
        game played, give p = (+1, -1, 0, 0).  Replacing the last row with ones and the
        last entry of p with 0 gives, with a + b + c + d = 0:

            row 1:  3a - b - c - d = 1  ->  3a + a = 4a = 1        ->  a =  1/4
            row 3: -a - b + 3c - d = 0  -> -a - b + 3c + a + b + c
                                        =  4c = 0                 ->  c =  0
            row 2:   -a + 2b - c   = -1 ->  -1/4 + 2b = -1         ->  b = -3/8
            zero mean:                                               d =  1/8
        """
        a, b, c, d = (MasseyCompetitor() for _ in range(4))
        a.beat(b)
        a.beat(c)
        b.tied(c)
        c.beat(d)
        d.beat(a)

        self.assertAlmostEqual(a.rating, 0.25, places=9)
        self.assertAlmostEqual(b.rating, -0.375, places=9)
        self.assertAlmostEqual(c.rating, 0.0, places=9)
        self.assertAlmostEqual(d.rating, 0.125, places=9)

    def test_a_draw_counts_as_a_game_played(self):
        """A > B, A ~ C solves to (1/3, -2/3, 1/3).

        This is the case that pins the draw semantics: a draw adds nothing to either
        cumulative margin but still counts as a game played, so it appears in D and A but
        not in p.  Games played: A two, B one, C one; A meets each of B and C once:

            M = D - A = [[ 2, -1, -1],
                         [-1,  1,  0],
                         [-1,  0,  1]]

        and p = (+1, -1, 0).  With a + b + c = 0:

            row 1:  2a - b - c = 1  ->  2a + a = 3a = 1  ->  a =  1/3
            row 2:      -a + b  = -1                     ->  b = -2/3
            zero mean:                                      c =  1/3

        Note C ends level with A rather than at zero: drawing the eventual leader is worth
        more than beating nobody, which is exactly what the least-squares fit encodes.
        """
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.tied(c)

        self.assertAlmostEqual(a.rating, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(b.rating, -2.0 / 3.0, places=9)
        self.assertAlmostEqual(c.rating, 1.0 / 3.0, places=9)

    def test_single_game_splits_the_unit_margin(self):
        """One game between two competitors gives ratings of +1/2 and -1/2.

        M = [[1, -1], [-1, 1]], p = (+1, -1).  The constraint row makes this
        [[1, -1], [1, 1]] r = (1, 0), so a - b = 1 and a + b = 0, i.e. a = 1/2, b = -1/2.
        """
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)

        self.assertAlmostEqual(a.rating, 0.5, places=9)
        self.assertAlmostEqual(b.rating, -0.5, places=9)

    def test_single_game_splits_a_real_margin(self):
        """With scores supplied, one 35-3 game gives ratings of +16 and -16.

        The structure is identical to the unit-margin case above, only p = (+32, -32)
        instead of (+1, -1), so a - b = 32 and a + b = 0.
        """
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        self.assertAlmostEqual(a.rating, 16.0, places=9)
        self.assertAlmostEqual(b.rating, -16.0, places=9)

    def test_three_team_schedule_with_real_margins(self):
        """A beats B by 21, B beats C by 7.

        Games played: A one, B two, C one; A meets B once and B meets C once:

            M = D - A = [[ 1, -1,  0],
                         [-1,  2, -1],
                         [ 0, -1,  1]]

        and p = (+21, -21 + 7, -7) = (+21, -14, -7).  With a + b + c = 0:

            row 1:  a - b = 21
            row 2:  -a + 2b - c = -14, and c = -a - b, so -a + 2b + a + b = 3b = -14
                    ->  b = -14/3, a = 21 - 14/3 = 49/3, c = -a - b = -35/3
        """
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(28.0, 7.0))
        b.beat(c, scores=(14.0, 7.0))

        self.assertAlmostEqual(a.rating, 49.0 / 3.0, places=9)
        self.assertAlmostEqual(b.rating, -14.0 / 3.0, places=9)
        self.assertAlmostEqual(c.rating, -35.0 / 3.0, places=9)


if __name__ == "__main__":
    unittest.main()
