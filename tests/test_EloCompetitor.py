import unittest
from elote import EloCompetitor, GlickoCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException


class TestElo(unittest.TestCase):
    def test_Improvement(self):
        initial_rating = 100
        player1 = EloCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = EloCompetitor(initial_rating=800)
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Decay(self):
        initial_rating = 800
        player1 = EloCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = EloCompetitor(initial_rating=100)
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Expectation(self):
        player1 = EloCompetitor(initial_rating=1000)
        player2 = EloCompetitor(initial_rating=100)
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))

    def test_Exceptions(self):
        player1 = EloCompetitor(initial_rating=1000)
        player2 = GlickoCompetitor(initial_rating=100)

        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.verify_competitor_types(player2)
