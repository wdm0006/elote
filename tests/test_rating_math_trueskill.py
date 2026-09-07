"""Rating-math tests for the TrueSkill system (Herbrich et al. 2005).

Draw-margin assertions recompute the published ``sqrt(2) * beta *
Phi^-1((p + 1) / 2)`` formula in-module; update anchors were derived from an
independent two-player factor-graph reference, never from elote code.
"""

import math
import unittest
from statistics import NormalDist

from elote.competitors.base import InvalidParameterException
from elote.competitors.trueskill import TrueSkillCompetitor

BETA = 25.0 / 6.0


def published_draw_margin(draw_probability: float, beta: float = BETA) -> float:
    """Herbrich 2005 draw margin: sqrt(2) * beta * Phi^-1((p + 1) / 2)."""
    return math.sqrt(2.0) * beta * NormalDist().inv_cdf((draw_probability + 1.0) / 2.0)


class TrueSkillDrawMarginTest(unittest.TestCase):
    def test_margin_matches_published_formula(self):
        for p in (0.0, 0.1, 0.5, 0.9):
            self.assertAlmostEqual(
                TrueSkillCompetitor._calculate_draw_margin(BETA, p),
                published_draw_margin(p),
                places=9,
            )

    def test_margin_extremes(self):
        """p=0 must collapse the margin to zero; p=0.9 reaches ~9.69 points."""
        self.assertEqual(TrueSkillCompetitor._calculate_draw_margin(BETA, 0.0), 0.0)
        self.assertAlmostEqual(TrueSkillCompetitor._calculate_draw_margin(BETA, 0.1), 0.740467, places=6)
        self.assertAlmostEqual(TrueSkillCompetitor._calculate_draw_margin(BETA, 0.9), 9.692393, places=6)

    def test_draw_probability_of_one_rejected(self):
        """p=1 is outside the documented half-open [0, 1) domain."""
        with self.assertRaises(InvalidParameterException):
            TrueSkillCompetitor._calculate_draw_margin(BETA, 1.0)

    def test_negative_draw_probability_rejected(self):
        with self.assertRaises(InvalidParameterException):
            TrueSkillCompetitor._calculate_draw_margin(BETA, -0.1)


class TrueSkillUpdateAnchorsTest(unittest.TestCase):
    def test_equal_players_win_pin(self):
        """Herbrich two-player anchor for a 25-vs-25 default-prior win."""
        winner = TrueSkillCompetitor()
        loser = TrueSkillCompetitor()
        winner.beat(loser)

        self.assertAlmostEqual(winner.mu, 29.395741, places=6)
        self.assertAlmostEqual(winner.sigma, 7.171128, places=6)
        self.assertAlmostEqual(loser.mu, 20.604259, places=6)
        self.assertAlmostEqual(loser.sigma, 7.171128, places=6)
        self.assertAlmostEqual(winner.mu + loser.mu, 50.0, places=6)

    def test_scores_do_not_change_unit_result(self):
        """A 2-1 win and a 1-0 win carry the same update in TrueSkill."""
        with_scores = TrueSkillCompetitor()
        opponent = TrueSkillCompetitor()
        with_scores.beat(opponent, scores=(2.0, 1.0))

        self.assertAlmostEqual(with_scores.mu, 29.395741, places=6)

    def test_tie_pin(self):
        a = TrueSkillCompetitor()
        b = TrueSkillCompetitor()
        a.tied(b)

        self.assertAlmostEqual(a.mu, 25.0, places=6)
        self.assertAlmostEqual(a.sigma, 6.457152, places=6)
        self.assertAlmostEqual(b.mu, 25.0, places=6)

    def test_upset_resistant_expected_win(self):
        """A heavy favourite who wins must move less than the equal pair."""
        favourite = TrueSkillCompetitor(initial_mu=30.0)
        underdog = TrueSkillCompetitor(initial_mu=5.0)
        favourite.beat(underdog)

        self.assertAlmostEqual(favourite.mu, 30.399124, places=6)
        self.assertAlmostEqual(underdog.mu, 4.600876, places=6)
        self.assertAlmostEqual(favourite.mu + underdog.mu, 35.0, places=6)


class TrueSkillTeamTest(unittest.TestCase):
    def test_create_team_sums_mu_and_variances(self):
        p1 = TrueSkillCompetitor()
        p2 = TrueSkillCompetitor()
        team_mu, team_sigma = TrueSkillCompetitor.create_team([p1, p2])

        self.assertAlmostEqual(team_mu, p1.mu + p2.mu, places=9)
        expected_sigma = math.sqrt(p1.sigma**2 + p2.sigma**2)
        self.assertAlmostEqual(team_sigma, expected_sigma, places=9)
        self.assertAlmostEqual(team_sigma, 11.784641615255001, places=9)

    def test_update_team_moves_every_player(self):
        """update_team must push each member's mu by sigma^2 / v^2 * v_value."""
        p1 = TrueSkillCompetitor()
        p2 = TrueSkillCompetitor()
        mu_before = (p1.mu, p2.mu)
        sigma_before = (p1.sigma, p2.sigma)
        v = 20.0

        def result_func(_t, derivative: bool = False):
            return 0.5 if derivative else 1.0

        TrueSkillCompetitor.update_team([p1, p2], 1.0, 1.0, v, result_func)

        for player, mu0, sigma0 in zip((p1, p2), mu_before, sigma_before, strict=True):
            self.assertAlmostEqual(player.mu - mu0, sigma0**2 / v**2 * 1.0, places=9)
            self.assertLess(player.sigma, sigma0)


class TrueSkillConfigurationTest(unittest.TestCase):
    def test_configured_draw_probability_changes_margin(self):
        original = TrueSkillCompetitor._draw_probability
        try:
            TrueSkillCompetitor.configure_class(draw_probability=0.5)
            self.assertAlmostEqual(
                TrueSkillCompetitor._calculate_draw_margin(BETA, 0.5),
                published_draw_margin(0.5),
                places=9,
            )
        finally:
            TrueSkillCompetitor._draw_probability = original


class TrueSkillStateRoundTripTest(unittest.TestCase):
    def test_export_import_preserves_math(self):
        a = TrueSkillCompetitor()
        b = TrueSkillCompetitor()
        a.beat(b)

        restored = TrueSkillCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.mu, a.mu, places=9)
        self.assertAlmostEqual(restored.sigma, a.sigma, places=9)

        fresh = TrueSkillCompetitor(initial_mu=10.0)
        restored.beat(fresh)
        self.assertNotAlmostEqual(restored.mu, a.mu, places=3)


if __name__ == "__main__":
    unittest.main()
