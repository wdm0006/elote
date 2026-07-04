import math
import unittest

from elote import BradleyTerryCompetitor, EloCompetitor


class TestBradleyTerryKnownValues(unittest.TestCase):
    """Tests for BradleyTerryCompetitor against hand-computed values."""

    def test_initial_rating(self):
        self.assertAlmostEqual(BradleyTerryCompetitor().rating, 1500.0)
        self.assertAlmostEqual(BradleyTerryCompetitor(initial_rating=1234).rating, 1234.0)

    def test_expected_score_known(self):
        # Equal strengths -> 0.5.
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        self.assertAlmostEqual(a.expected_score(b), 0.5)

        # 200-point gap on the Elo scale -> the classic Elo win probability.
        a = BradleyTerryCompetitor(initial_rating=1600)
        b = BradleyTerryCompetitor(initial_rating=1400)
        expected = 1.0 / (1.0 + 10 ** (-200 / 400))
        self.assertAlmostEqual(a.expected_score(b), expected)

    def test_matches_elo_expected_score(self):
        for ra, rb in [(1500, 1500), (1600, 1400), (1200, 1800), (1723, 1601)]:
            bt = BradleyTerryCompetitor(initial_rating=ra).expected_score(
                BradleyTerryCompetitor(initial_rating=rb)
            )
            elo = EloCompetitor(initial_rating=ra).expected_score(EloCompetitor(initial_rating=rb))
            self.assertAlmostEqual(bt, elo)

    def test_two_player_mle_ratio(self):
        # A beats B twice, B beats A once. The unregularized MLE gives p_A / p_B = 2,
        # i.e. beta_A - beta_B = ln(2). With light regularization the gap is very close.
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.beat(b)
        a.beat(b)
        b.beat(a)
        beta_gap = a._beta - b._beta
        self.assertAlmostEqual(beta_gap, math.log(2), places=1)
        self.assertGreater(a.rating, b.rating)

    def test_recentering(self):
        # After a fit the mean log-strength of a component is zero, so equal records keep
        # the two competitors centered on the anchor rating.
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.beat(b)
        b.beat(a)
        self.assertAlmostEqual((a.rating + b.rating) / 2, 1500.0, places=3)

    def test_transitive_ordering(self):
        # A beats B, B beats C -> A > B > C.
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        c = BradleyTerryCompetitor()
        a.beat(b)
        b.beat(c)
        self.assertGreater(a.rating, b.rating)
        self.assertGreater(b.rating, c.rating)


if __name__ == "__main__":
    unittest.main()
