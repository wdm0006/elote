"""Numeric reference values for PythagoreanCompetitor.

Every value below is recomputed here from the documented formulas, written out
independently of the implementation:

    w = (PF + c)^k / ((PF + c)^k + (PA + c)^k)          Pythagorean win expectation
    E_a = (w_a - w_a*w_b) / (w_a + w_b - 2*w_a*w_b)     log5 combination

with ``k = _exponent`` (2.37 by default, the standard American-football fit) and ``c =
_prior_points`` (1.0), the symmetric prior that keeps a fresh competitor at exactly 0.5 and
an unbeaten one strictly below 1.0.

The implementation evaluates the rating in the algebraically identical ratio form
``1 / (1 + ((PA + c) / (PF + c))^k)``, which cannot overflow on large totals, so the two
agree to within a few ulps rather than bit for bit; the assertions below allow for that.
"""

import unittest

from elote import PythagoreanCompetitor


PLACES = 12
K = PythagoreanCompetitor._exponent
PRIOR = PythagoreanCompetitor._prior_points


def _expectation(points_for, points_against, exponent=K):
    """The Pythagorean win expectation, written out from the documented formula."""
    scored = (points_for + PRIOR) ** exponent
    allowed = (points_against + PRIOR) ** exponent
    return scored / (scored + allowed)


def _log5(first, second):
    """The log5 combination of two win expectations, written out from the formula."""
    return (first - first * second) / (first + second - 2.0 * first * second)


class TestPythagoreanKnownValues(unittest.TestCase):
    def test_default_exponent_is_the_documented_football_fit(self):
        self.assertEqual(K, 2.37)
        self.assertEqual(PRIOR, 1.0)

    def test_fresh_competitor_is_exactly_one_half(self):
        """PF = PA = 0 is a 0/0 without the prior; with it, (0+1)^k / (2*(0+1)^k) = 1/2."""
        self.assertEqual(_expectation(0.0, 0.0), 0.5)
        self.assertEqual(PythagoreanCompetitor().rating, 0.5)

    def test_single_game_totals(self):
        """One 28-14 game: PF = 28, PA = 14, so the prior-adjusted totals are 29 and 15.

            w = 29^2.37 / (29^2.37 + 15^2.37) = 0.826699207003...

        The loser is the mirror image, 15 against 29.
        """
        expected_winner = _expectation(28.0, 14.0)
        expected_loser = _expectation(14.0, 28.0)
        # Guards the arithmetic above against a silent change in the closed form.
        self.assertAlmostEqual(expected_winner, 0.826699207003455, places=PLACES)
        self.assertAlmostEqual(expected_loser, 0.173300792996545, places=PLACES)

        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(28.0, 14.0))

        self.assertAlmostEqual(a.rating, expected_winner, places=PLACES)
        self.assertAlmostEqual(b.rating, expected_loser, places=PLACES)

    def test_two_competitors_with_different_records(self):
        """A season of three games each, against opponents that do not matter to the fit.

        Competitor A: 31-17, 24-21, 10-27  ->  PF = 65, PA = 65, so exactly 0.5 despite the
        two wins, since Pythagorean sees only points.
        Competitor B: 20-14, 35-3, 28-24   ->  PF = 83, PA = 41.
        """
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        foils = [PythagoreanCompetitor() for _ in range(6)]

        a.beat(foils[0], scores=(31.0, 17.0))
        a.beat(foils[1], scores=(24.0, 21.0))
        a.lost_to(foils[2], scores=(10.0, 27.0))

        b.beat(foils[3], scores=(20.0, 14.0))
        b.beat(foils[4], scores=(35.0, 3.0))
        b.beat(foils[5], scores=(28.0, 24.0))

        self.assertEqual((a._points_for, a._points_against), (65.0, 65.0))
        self.assertEqual((b._points_for, b._points_against), (83.0, 41.0))

        self.assertEqual(a.rating, 0.5)
        expected_b = _expectation(83.0, 41.0)
        self.assertAlmostEqual(expected_b, 0.837909980755396, places=PLACES)
        self.assertAlmostEqual(b.rating, expected_b, places=PLACES)

    def test_expected_score_is_the_log5_of_the_two_expectations(self):
        """A hand-computed log5 pair: w_a from 83-41, w_b from 65-65 (exactly 0.5).

        With w_b = 1/2 log5 reduces to w_a itself, so the pair below deliberately uses a
        third record as well, where neither side is one half.
        """
        strong = _expectation(83.0, 41.0)
        even = _expectation(65.0, 65.0)
        middling = _expectation(48.0, 40.0)

        self.assertAlmostEqual(_log5(strong, even), strong, places=PLACES)
        self.assertAlmostEqual(_log5(strong, middling), 0.772118188713780, places=PLACES)
        self.assertAlmostEqual(_log5(middling, strong), 0.227881811286220, places=PLACES)

        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        foil_a, foil_b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(foil_a, scores=(83.0, 41.0))
        b.beat(foil_b, scores=(48.0, 40.0))

        self.assertAlmostEqual(a.rating, strong, places=PLACES)
        self.assertAlmostEqual(b.rating, middling, places=PLACES)
        self.assertAlmostEqual(a.expected_score(b), _log5(strong, middling), places=PLACES)
        self.assertAlmostEqual(b.expected_score(a), _log5(middling, strong), places=PLACES)

    def test_baseball_exponent_reproduces_the_classic_two(self):
        """k = 2 is Bill James' original: a 700-600 run record gives 701^2/(701^2 + 601^2)."""
        expected = _expectation(700.0, 600.0, exponent=2.0)
        self.assertAlmostEqual(expected, 0.576354500693172, places=PLACES)

        a, b = PythagoreanCompetitor(exponent=2.0), PythagoreanCompetitor(exponent=2.0)
        a.beat(b, scores=(700.0, 600.0))
        self.assertAlmostEqual(a.rating, expected, places=PLACES)


if __name__ == "__main__":
    unittest.main()
