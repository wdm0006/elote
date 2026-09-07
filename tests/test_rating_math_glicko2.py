"""Rating-math tests for the Glicko-2 system (Glickman 2012a).

Assertions are checked against the paper's worked example and an independent
implementation of the full update (v, delta, Illinois-algorithm volatility
solve) computed in this module -- never from elote code.
"""

import math
import unittest
from datetime import datetime, timedelta

from elote.competitors.base import InvalidRatingValueException
from elote.competitors.glicko2 import Glicko2Competitor

GL2_SCALE = 173.7178
TAU = 0.5


def to_mu(rating: float) -> float:
    return (rating - 1500.0) / GL2_SCALE


def to_phi(rd: float) -> float:
    return rd / GL2_SCALE


def g_of(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi**2))


def e_of(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-g_of(phi_j) * (mu - mu_j)))


def independent_g2_update(mu, phi, sigma, results, tau=TAU, eps=1e-6):
    """Glickman 2012a update over ``results`` = [(score, mu_j, phi_j)].

    Returns (mu', phi', sigma').  v and delta aggregation and the Illinois
    volatility solve mirror the paper's sections 5 and 6.
    """
    v_inv = 0.0
    delta_sum = 0.0
    for score, mu_j, phi_j in results:
        gj = g_of(phi_j)
        ej = e_of(mu, mu_j, phi_j)
        v_inv += gj * gj * ej * (1.0 - ej)
        delta_sum += gj * (score - ej)
    v = 1.0 / v_inv
    delta = v * delta_sum

    def f(x):
        ex = math.exp(x)
        a = ex * (delta * delta - phi * phi - v - ex)
        b = 2.0 * ((phi * phi + v + ex) ** 2)
        return a / b - (x - math.log(sigma * sigma)) / (tau * tau)

    big_a = math.log(sigma * sigma)
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(big_a - k * tau) < 0:
            k += 1
        big_b = big_a - k * tau

    fa = f(big_a)
    fb = f(big_b)
    while abs(big_b - big_a) > eps:
        cc = big_a + (big_a - big_b) * fa / (fb - fa)
        fc = f(cc)
        if fc * fb < 0:
            big_a, fa = big_b, fb
        else:
            fa = fa / 2.0
        big_b, fb = cc, fc

    sigma_prime = math.exp(big_a / 2.0)
    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * delta_sum
    return mu_prime, phi_prime, sigma_prime


class Glicko2PaperExampleTest(unittest.TestCase):
    def test_2012_example_pin(self):
        """The worked example from Glickman's 2012a paper, both sides."""
        winner = Glicko2Competitor(initial_rating=1500, initial_rd=200)
        loser = Glicko2Competitor(initial_rating=1400, initial_rd=30)
        winner.beat(loser)

        self.assertAlmostEqual(winner.rating, 1563.564194, places=6)
        self.assertAlmostEqual(winner.rd, 175.402656, places=6)
        self.assertAlmostEqual(loser.rating, 1398.143558, places=6)
        self.assertAlmostEqual(loser.rd, 31.670215, places=6)

        ref_winner = independent_g2_update(to_mu(1500.0), to_phi(200.0), 0.06, [(1.0, to_mu(1400.0), to_phi(30.0))])
        self.assertAlmostEqual(winner.rating, ref_winner[0] * GL2_SCALE + 1500.0, places=6)
        self.assertAlmostEqual(winner.rd, ref_winner[1] * GL2_SCALE, places=6)


class Glicko2AggregationTest(unittest.TestCase):
    def test_sequential_games_match_reference(self):
        """Two games must apply v/delta aggregation exactly as published."""
        x = Glicko2Competitor(initial_rating=1500, initial_rd=200)
        a = Glicko2Competitor(initial_rating=1400, initial_rd=30)
        b = Glicko2Competitor(initial_rating=1550, initial_rd=100)

        x.beat(a)
        x.tied(b)

        # Sequential single-game updates through the independent reference.
        mu, phi, sigma = to_mu(1500.0), to_phi(200.0), 0.06
        mu, phi, sigma = independent_g2_update(mu, phi, sigma, [(1.0, to_mu(1400.0), to_phi(30.0))])
        mu, phi, sigma = independent_g2_update(mu, phi, sigma, [(0.5, to_mu(1550.0), to_phi(100.0))])

        self.assertAlmostEqual(x.rating, mu * GL2_SCALE + 1500.0, places=6)
        self.assertAlmostEqual(x.rd, phi * GL2_SCALE, places=6)
        self.assertGreater(sigma, 0.0)


class Glicko2ValidationTest(unittest.TestCase):
    def test_below_minimum_rating_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            Glicko2Competitor(initial_rating=-100)


class Glicko2InactivityTest(unittest.TestCase):
    def test_multi_period_gap_inflates_then_updates(self):
        """A 10-day gap inflates phi by sqrt(phi^2 + t * sigma^2) first."""
        t0 = datetime(2020, 1, 1)
        t10 = t0 + timedelta(days=10)
        winner = Glicko2Competitor(initial_rating=1500, initial_rd=50)
        loser = Glicko2Competitor(initial_rating=1500, initial_rd=50)
        winner.beat(loser, match_time=t0)

        rd_after_first = winner.rd
        loser_rd_after_first = loser.rd
        rating_after_first = winner.rating
        loser_rating_after_first = loser.rating
        sigma_w = winner.volatility
        sigma_l = loser.volatility

        winner.beat(loser, match_time=t10)

        phi_w = math.sqrt(to_phi(rd_after_first) ** 2 + 10 * sigma_w**2)
        phi_l = math.sqrt(to_phi(loser_rd_after_first) ** 2 + 10 * sigma_l**2)
        ref_winner = independent_g2_update(
            to_mu(rating_after_first),
            phi_w,
            sigma_w,
            [(1.0, to_mu(loser_rating_after_first), phi_l)],
        )
        self.assertAlmostEqual(winner.rating, ref_winner[0] * GL2_SCALE + 1500.0, places=6)
        self.assertAlmostEqual(winner.rd, ref_winner[1] * GL2_SCALE, places=6)

    def test_longer_gap_inflates_more(self):
        results = []
        for days in (1, 10):
            winner = Glicko2Competitor(initial_rating=1500, initial_rd=50)
            loser = Glicko2Competitor(initial_rating=1500, initial_rd=50)
            t0 = datetime(2020, 1, 1)
            winner.beat(loser, match_time=t0)
            winner.beat(loser, match_time=t0 + timedelta(days=days))
            results.append(winner.rd)
        self.assertGreater(results[1], results[0])


class Glicko2VolatilityTest(unittest.TestCase):
    def test_volatility_stays_bounded_and_drifts_down(self):
        """Equal-play volatility must stay in (0, sigma0] and converge down."""
        a = Glicko2Competitor(initial_rating=1500, initial_rd=50)
        b = Glicko2Competitor(initial_rating=1500, initial_rd=50)
        for _ in range(5):
            a.beat(b)

        self.assertGreater(a.volatility, 0.0)
        self.assertLess(a.volatility, 0.06)
        self.assertAlmostEqual(a.volatility, 0.0599974, places=6)

        for _ in range(25):
            a.beat(b)
        self.assertGreater(a.volatility, 0.0)
        self.assertLessEqual(a.volatility, 0.06)


class Glicko2ScorePropagationTest(unittest.TestCase):
    def test_lost_to_reverses_custom_scores(self):
        via_beat = (
            Glicko2Competitor(initial_rating=1500, initial_rd=200),
            Glicko2Competitor(initial_rating=1400, initial_rd=30),
        )
        via_beat[0].beat(via_beat[1], scores=(6.0, 1.0))

        via_lost = (
            Glicko2Competitor(initial_rating=1500, initial_rd=200),
            Glicko2Competitor(initial_rating=1400, initial_rd=30),
        )
        via_lost[1].lost_to(via_lost[0], scores=(1.0, 6.0))

        self.assertAlmostEqual(via_beat[0].rating, via_lost[0].rating, places=9)
        self.assertAlmostEqual(via_beat[1].rating, via_lost[1].rating, places=9)
        self.assertAlmostEqual(via_beat[0].rating, 1563.564194, places=6)


class Glicko2StateRoundTripTest(unittest.TestCase):
    def test_export_import_preserves_math(self):
        a = Glicko2Competitor(initial_rating=1500, initial_rd=200)
        b = Glicko2Competitor(initial_rating=1400, initial_rd=30)
        a.beat(b)

        restored = Glicko2Competitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=9)
        self.assertAlmostEqual(restored.rd, a.rd, places=9)
        self.assertAlmostEqual(restored.volatility, a.volatility, places=9)

        fresh = Glicko2Competitor(initial_rating=1200, initial_rd=200)
        restored.beat(fresh)
        self.assertNotAlmostEqual(restored.rating, a.rating, places=3)


if __name__ == "__main__":
    unittest.main()
