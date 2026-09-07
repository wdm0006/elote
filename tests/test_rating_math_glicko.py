"""Rating-math tests for the Glicko system (Glickman 1999).

Every update assertion is checked against an independent implementation of the
published formulae -- g(RD), E, d^2, and the precision combination
``1/rd^2 + 1/d^2`` -- computed in this module, never from elote code.
"""

import math
import unittest
from datetime import datetime, timedelta

from elote.competitors.base import InvalidParameterException
from elote.competitors.glicko import GlickoCompetitor

Q = 0.0057565


def g_factor(rd: float) -> float:
    """Glickman's g(RD): down-weights a game by the opponent's uncertainty."""
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi**2))


def expected_score(rating: float, opp_rating: float, opp_rd: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-g_factor(opp_rd) * (rating - opp_rating) / 400.0))


def independent_update(rating: float, rd: float, opp_rating: float, opp_rd: float, score: float):
    """One Glickman 1999 update; returns (rating', rd').

    The 1/d^2 term enters through ``denom = 1/rd^2 + 1/d^2`` -- the precision
    combination the mutation campaign flagged (1/d^2 weight survivors).
    """
    gj = g_factor(opp_rd)
    e = expected_score(rating, opp_rating, opp_rd)
    d_squared = 1.0 / (Q * Q * gj * gj * e * (1.0 - e))
    denom = 1.0 / (rd * rd) + 1.0 / d_squared
    return rating + (Q / denom) * gj * (score - e), math.sqrt(1.0 / denom)


def inflate_rd(rd: float, periods: float, c: float) -> float:
    """RD growth over inactivity: min(350, sqrt(rd^2 + c^2 * periods))."""
    if periods <= 0:
        return rd
    return min(350.0, math.sqrt(rd * rd + c * c * periods))


class GlickoPaperExampleTest(unittest.TestCase):
    def test_1999_example_pin(self):
        """The worked example from Glickman's 1999 paper, both sides."""
        winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)
        winner.beat(loser)

        self.assertAlmostEqual(winner.rating, 1563.432268, places=6)
        self.assertAlmostEqual(winner.rd, 175.219972, places=6)
        self.assertAlmostEqual(loser.rating, 1398.342504, places=6)
        self.assertAlmostEqual(loser.rd, 29.925090, places=6)

        ref_winner = independent_update(1500.0, 200.0, 1400.0, 30.0, 1.0)
        ref_loser = independent_update(1400.0, 30.0, 1500.0, 200.0, 0.0)
        self.assertAlmostEqual(winner.rating, ref_winner[0], places=9)
        self.assertAlmostEqual(winner.rd, ref_winner[1], places=9)
        self.assertAlmostEqual(loser.rating, ref_loser[0], places=9)
        self.assertAlmostEqual(loser.rd, ref_loser[1], places=9)


class GlickoPrecisionWeightTest(unittest.TestCase):
    def test_precision_combination_matches_reference(self):
        """The update must combine 1/rd^2 with 1/d^2 (not 1/rd^2 + d^2)."""
        winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)
        winner.beat(loser)

        ref = independent_update(1500.0, 200.0, 1400.0, 30.0, 1.0)
        self.assertAlmostEqual(winner.rd, ref[1], places=9)
        self.assertLess(winner.rd, 200.0, "playing must reduce the winner's RD")

    def test_expected_score_matches_closed_form(self):
        player = GlickoCompetitor(initial_rating=1600, initial_rd=120)
        opponent = GlickoCompetitor(initial_rating=1400, initial_rd=120)
        observed = player.expected_score(opponent)
        self.assertAlmostEqual(observed, expected_score(1600.0, 1400.0, 120.0), places=9)


class GlickoValidationTest(unittest.TestCase):
    def test_nonpositive_initial_rd_rejected(self):
        with self.assertRaises(InvalidParameterException):
            GlickoCompetitor(initial_rating=1500, initial_rd=0)
        with self.assertRaises(InvalidParameterException):
            GlickoCompetitor(initial_rating=1500, initial_rd=-50)


class GlickoInactivityTest(unittest.TestCase):
    def test_multi_period_gap_inflates_then_updates(self):
        """A 30-day gap inflates both RDs before the second update."""
        t0 = datetime(2020, 1, 1)
        t30 = t0 + timedelta(days=30)
        winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)
        winner.beat(loser, match_time=t0)

        rd_after_first = winner.rd
        rating_after_first = winner.rating
        loser_rd_after_first = loser.rd
        loser_rating_after_first = loser.rating
        self.assertAlmostEqual(rd_after_first, 175.219972, places=6)

        winner.beat(loser, match_time=t30)

        rd_w = inflate_rd(rd_after_first, 30.0, GlickoCompetitor._c)
        rd_l = inflate_rd(loser_rd_after_first, 30.0, GlickoCompetitor._c)
        ref_winner = independent_update(rating_after_first, rd_w, loser_rating_after_first, rd_l, 1.0)
        self.assertAlmostEqual(winner.rating, ref_winner[0], places=9)
        self.assertAlmostEqual(winner.rd, ref_winner[1], places=9)

    def test_longer_gap_inflates_more(self):
        """Off-by-one in the period count shows up as a monotonicity break."""
        results = []
        for days in (1, 30):
            winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
            loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)
            t0 = datetime(2020, 1, 1)
            winner.beat(loser, match_time=t0)
            winner.beat(loser, match_time=t0 + timedelta(days=days))
            results.append(winner.rd)
        self.assertGreater(results[1], results[0])

    def test_rd_cap_at_350(self):
        """Long inactivity saturates at the documented 350 ceiling."""
        self.assertEqual(inflate_rd(340.0, 1000.0, GlickoCompetitor._c), 350.0)


class GlickoScorePropagationTest(unittest.TestCase):
    def test_lost_to_reverses_custom_scores(self):
        """lost_to on the loser equals beat on the winner with reversed scores."""
        via_beat = (
            GlickoCompetitor(initial_rating=1500, initial_rd=200),
            GlickoCompetitor(initial_rating=1400, initial_rd=30),
        )
        via_beat[0].beat(via_beat[1], scores=(6.0, 1.0))

        via_lost = (
            GlickoCompetitor(initial_rating=1500, initial_rd=200),
            GlickoCompetitor(initial_rating=1400, initial_rd=30),
        )
        via_lost[1].lost_to(via_lost[0], scores=(1.0, 6.0))

        self.assertAlmostEqual(via_beat[0].rating, via_lost[0].rating, places=9)
        self.assertAlmostEqual(via_beat[1].rating, via_lost[1].rating, places=9)
        self.assertAlmostEqual(via_beat[0].rating, 1563.432268, places=6)


class GlickoTieTest(unittest.TestCase):
    def test_equal_players_tie_changes_ratings_not(self):
        """E=0.5 and s=0.5 leave the mean rating untouched; RD still shrinks."""
        a = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        b = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        a.tied(b)

        self.assertAlmostEqual(a.rating, 1500.0, places=9)
        self.assertAlmostEqual(b.rating, 1500.0, places=9)
        self.assertLess(a.rd, 200.0)


class GlickoRatingPeriodTest(unittest.TestCase):
    def test_apply_rating_period_outcome_zero_reverses_call(self):
        """apply_rating_period with outcome 0.0 must let B beat A."""
        via_direct = (
            GlickoCompetitor(initial_rating=1500, initial_rd=200),
            GlickoCompetitor(initial_rating=1400, initial_rd=30),
        )
        via_direct[1].beat(via_direct[0])

        via_period = (
            GlickoCompetitor(initial_rating=1500, initial_rd=200),
            GlickoCompetitor(initial_rating=1400, initial_rd=30),
        )
        from elote.competitors.base import BaseCompetitor

        BaseCompetitor.apply_rating_period([(via_period[0], via_period[1], 0.0, None)])

        self.assertAlmostEqual(via_period[0].rating, via_direct[0].rating, places=9)
        self.assertAlmostEqual(via_period[1].rating, via_direct[1].rating, places=9)


class GlickoStateRoundTripTest(unittest.TestCase):
    def test_export_import_preserves_math(self):
        a = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        b = GlickoCompetitor(initial_rating=1400, initial_rd=30)
        a.beat(b)

        restored = GlickoCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=9)
        self.assertAlmostEqual(restored.rd, a.rd, places=9)

        fresh = GlickoCompetitor(initial_rating=1200, initial_rd=200)
        restored.beat(fresh)
        self.assertNotAlmostEqual(restored.rating, a.rating, places=3)


if __name__ == "__main__":
    unittest.main()
