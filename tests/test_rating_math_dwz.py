"""Rating-math tests for DWZCompetitor.

Pins come from an independent implementation of the published DWZ rules
(E = a * ((r/1000)^4 + J) + B with age-bracketed J, acceleration and braking
terms, the braking upper bound of 150, and the per-game-count upper bounds),
hand-computed and cross-checked before being pinned here.
"""

import unittest

from elote.competitors.dwz import DWZCompetitor
from elote.competitors.base import InvalidRatingValueException


def independent_dwz_new(rating, count, opp_rating, score, age=None):
    """The published DWZ update, independent of the implementation."""
    current_age = age if age is not None else 26
    if current_age <= 20:
        j_value = 5
    elif current_age <= 25:
        j_value = 10
    else:
        j_value = 15
    e0 = (rating / 1000.0) ** 4 + j_value
    acceleration = 1.0
    if current_age <= 20 and score > 0.5:
        acceleration = max(0.5, min(1.0, rating / 2000.0))
    braking = 0.0
    if rating < 1300 and score <= 0.5:
        braking = 2.718281828459045 ** ((1300.0 - rating) / 150.0) - 1.0
    expected = acceleration * e0 + braking
    upper = min(30.0, 5.0 * (count + 1)) if braking == 0.0 else 150.0
    development = max(5.0, min(expected, upper))
    w_expected = 1.0 / (1.0 + 10 ** ((opp_rating - rating) / 400.0))
    return rating + (800.0 / (development + 1.0)) * (score - w_expected)


class DWZFirstGameTest(unittest.TestCase):
    def test_equal_first_game_pin(self):
        a, b = DWZCompetitor(), DWZCompetitor()
        a.beat(b)
        self.assertAlmostEqual(a.rating, 466.666666666667, places=9)
        self.assertAlmostEqual(b.rating, 397.350993377483, places=9)
        self.assertAlmostEqual(a.rating, independent_dwz_new(400, 0, 400, 1.0), places=9)

    def test_independent_reference_agrees(self):
        a, b = DWZCompetitor(initial_rating=1000), DWZCompetitor(initial_rating=1000)
        a.beat(b)
        self.assertAlmostEqual(a.rating, independent_dwz_new(1000, 0, 1000, 1.0), places=9)
        self.assertAlmostEqual(b.rating, independent_dwz_new(1000, 0, 1000, 0.0), places=9)


class DWZDevelopmentCoefficientTest(unittest.TestCase):
    """The E development coefficient: braking boundaries and game-count caps."""

    def test_three_game_history_pin(self):
        a = DWZCompetitor(initial_rating=1000)
        b = DWZCompetitor(initial_rating=1000)
        for _ in range(3):
            a.beat(b)
        self.assertAlmostEqual(a.rating, 1110.857785, places=6)
        self.assertAlmostEqual(b.rating, 959.727258, places=6)

    def test_braking_kicks_in_below_1300_on_loss(self):
        """rating < 1300 with a loss adds the braking term B (upper bound 150)."""
        a, b = DWZCompetitor(initial_rating=1290), DWZCompetitor(initial_rating=1290)
        b.beat(a)  # a loses
        self.assertAlmostEqual(a.rating, independent_dwz_new(1290, 0, 1290, 0.0), places=9)
        self.assertAlmostEqual(a.rating, 1268.766513, places=6)

    def test_no_braking_at_1300_on_loss(self):
        """rating == 1300 is not below 1300, so B == 0 and the 5-point cap applies."""
        a, b = DWZCompetitor(initial_rating=1300), DWZCompetitor(initial_rating=1300)
        b.beat(a)
        self.assertAlmostEqual(a.rating, 1233.333333333333, places=9)

    def test_braking_boundary_is_strictly_below_1300(self):
        a, b = DWZCompetitor(initial_rating=1299.99), DWZCompetitor(initial_rating=1299.99)
        c, d = DWZCompetitor(initial_rating=1300.0), DWZCompetitor(initial_rating=1300.0)
        b.beat(a)
        d.beat(c)
        # The braking term and the 5-point cap give visibly different outcomes.
        self.assertNotAlmostEqual(a.rating, c.rating, places=3)

    def test_braking_upper_bound_is_generous(self):
        """A braking loss can move far more than the 5 * (count + 1) cap."""
        a, b = DWZCompetitor(initial_rating=1000), DWZCompetitor(initial_rating=1000)
        b.beat(a)
        self.assertAlmostEqual(a.rating, independent_dwz_new(1000, 0, 1000, 0.0), places=9)
        # B = exp(2) - 1 ~= 6.389 and E0 = 16 (default age 26, J = 15), so
        # E ~= 22.4: far above the 5-point first-game cap, but well under 150.
        self.assertGreater(a.rating, 970.0)
        self.assertLess(a.rating, 990.0)

    def test_age_brackets_change_development(self):
        """J = 5 / 10 / 15 across the age boundaries 20/21 and 25/26.

        A first game cannot expose the brackets (count = 0 caps E at 5 for every
        age), so the fixture plays three games first and clones the settled
        state for each age's deciding game.
        """
        base = DWZCompetitor(initial_rating=1000)
        opponent = DWZCompetitor(initial_rating=1000)
        for _ in range(3):
            base.beat(opponent)
        pre_rating, pre_opp = base.rating, opponent.rating
        ratings = {}
        for age in (20, 21, 25, 26):
            winner = DWZCompetitor.from_state(base.export_state())
            loser = DWZCompetitor.from_state(opponent.export_state())
            winner.beat(loser, age)
            ratings[age] = winner.rating
            self.assertAlmostEqual(
                ratings[age],
                independent_dwz_new(pre_rating, 3, pre_opp, 1.0, age=age),
                places=9,
            )
        # count = 3 caps E at 20, so the J brackets show: age <= 20 clamps E to 5
        # (max gain), 21-25 use J = 10, 26+ use J = 15 (smallest gain).
        self.assertGreater(ratings[20], ratings[21])
        self.assertAlmostEqual(ratings[21], ratings[25], places=12)
        self.assertGreater(ratings[21], ratings[26])

    def test_acceleration_young_winner(self):
        """age <= 20 winners get the acceleration factor a = clamp(r/2000)."""
        a, b = DWZCompetitor(initial_rating=1000), DWZCompetitor(initial_rating=1000)
        a.beat(b, 15)
        self.assertAlmostEqual(a.rating, 1066.666666666667, places=9)

    def test_elo_style_expected_score_identity(self):
        a = DWZCompetitor(initial_rating=1500)
        b = DWZCompetitor(initial_rating=1300)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)


class DWZValidationTest(unittest.TestCase):
    def test_initial_rating_below_minimum_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            DWZCompetitor(initial_rating=-1)

    def test_rating_setter_rejects_below_minimum(self):
        a = DWZCompetitor()
        with self.assertRaises(InvalidRatingValueException):
            a.rating = -10.0

    def test_state_round_trip(self):
        a, b = DWZCompetitor(initial_rating=1000), DWZCompetitor(initial_rating=1000)
        a.beat(b)
        restored = DWZCompetitor.from_state(a.export_state())
        self.assertAlmostEqual(restored.rating, a.rating, places=9)


if __name__ == "__main__":
    unittest.main()
