import unittest
from elote import BlendedCompetitor, GlickoCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException


class TestBlendedCompetitor(unittest.TestCase):
    def test_Improvement(self):
        player1 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                {"type": "DWZCompetitor", "competitor_kwargs": {}},
                {"type": "ECFCompetitor", "competitor_kwargs": {}},
            ]
        )
        initial_rating = player1.rating
        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = BlendedCompetitor(
                competitors=[
                    {
                        "type": "EloCompetitor",
                        "competitor_kwargs": {"initial_rating": 1000},
                    },
                    {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                    {"type": "DWZCompetitor", "competitor_kwargs": {}},
                    {"type": "ECFCompetitor", "competitor_kwargs": {}},
                ]
            )
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Decay(self):
        player1 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                {"type": "DWZCompetitor", "competitor_kwargs": {}},
                {"type": "ECFCompetitor", "competitor_kwargs": {}},
            ]
        )
        initial_rating = player1.rating
        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = BlendedCompetitor(
                competitors=[
                    {
                        "type": "EloCompetitor",
                        "competitor_kwargs": {"initial_rating": 1000},
                    },
                    {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                    {"type": "DWZCompetitor", "competitor_kwargs": {}},
                    {"type": "ECFCompetitor", "competitor_kwargs": {}},
                ]
            )
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Expectation(self):
        player1 = BlendedCompetitor(
            competitors=[
                {
                    "type": "EloCompetitor",
                    "competitor_kwargs": {"initial_rating": 1000},
                },
                {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                {"type": "DWZCompetitor", "competitor_kwargs": {}},
                {"type": "ECFCompetitor", "competitor_kwargs": {}},
            ]
        )
        player2 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 100}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                {"type": "DWZCompetitor", "competitor_kwargs": {}},
                {"type": "ECFCompetitor", "competitor_kwargs": {}},
            ]
        )
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))

    def test_Exceptions(self):
        player1 = BlendedCompetitor(
            competitors=[
                {
                    "type": "EloCompetitor",
                    "competitor_kwargs": {"initial_rating": 1000},
                },
                {"type": "GlickoCompetitor", "competitor_kwargs": {}},
                {"type": "DWZCompetitor", "competitor_kwargs": {}},
                {"type": "ECFCompetitor", "competitor_kwargs": {}},
            ]
        )
        player2 = GlickoCompetitor(initial_rating=100)

        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.verify_competitor_types(player2)
