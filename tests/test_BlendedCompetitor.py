import unittest
from elote import BlendedCompetitor, GlickoCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException


class TestBlendedCompetitor(unittest.TestCase):
    @staticmethod
    def _blended(*competitor_types):
        return BlendedCompetitor(competitors=[{"type": competitor_type} for competitor_type in competitor_types])

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

    def test_rejects_different_composition_lengths_before_mutation(self):
        for operation in ("expected_score", "beat", "tied"):
            with self.subTest(operation=operation):
                player1 = self._blended("EloCompetitor", "GlickoCompetitor")
                player2 = self._blended("EloCompetitor")
                sub_competitors = player1.sub_competitors + player2.sub_competitors
                initial_ratings = [c.rating for c in sub_competitors]

                with self.assertRaisesRegex(
                    MissMatchedCompetitorTypesException,
                    r"\['EloCompetitor', 'GlickoCompetitor'\].*\['EloCompetitor'\]",
                ):
                    getattr(player1, operation)(player2)

                self.assertEqual([c.rating for c in sub_competitors], initial_ratings)

    def test_rejects_different_ordered_compositions(self):
        for operation in ("expected_score", "beat", "tied"):
            with self.subTest(operation=operation):
                player1 = self._blended("EloCompetitor", "GlickoCompetitor")
                player2 = self._blended("GlickoCompetitor", "EloCompetitor")

                with self.assertRaisesRegex(
                    MissMatchedCompetitorTypesException,
                    r"BlendedCompetitor compositions .* cannot be co-mingled",
                ):
                    getattr(player1, operation)(player2)
