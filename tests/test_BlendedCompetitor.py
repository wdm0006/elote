import inspect
import unittest

from elote import BlendedCompetitor, BradleyTerryCompetitor, EloCompetitor, GlickoCompetitor
from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    MissMatchedCompetitorTypesException,
)


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


class TestBlendedCompetitorLegacyState(unittest.TestCase):
    """The legacy ``from_state`` shape resolves sub-competitors through the class registry."""

    @staticmethod
    def _legacy_state(*sub_states):
        return {
            "blend_mode": "mean",
            "competitors": [{"type": type(c).__name__, "competitor_kwargs": c.export_state()} for c in sub_states],
        }

    @staticmethod
    def _new_competitor(name):
        competitor_class = BaseCompetitor.get_competitor_class(name)
        if "initial_rating" in inspect.signature(competitor_class.__init__).parameters:
            return competitor_class(initial_rating=1234)
        return competitor_class()

    def test_restores_bradley_terry_sub_competitor(self):
        elo = EloCompetitor(initial_rating=400)
        bradley_terry = BradleyTerryCompetitor(initial_rating=400)
        bradley_terry.beat(BradleyTerryCompetitor(initial_rating=400))
        expected_rating = bradley_terry.rating

        restored = BlendedCompetitor.from_state(self._legacy_state(elo, bradley_terry))

        self.assertEqual(
            [type(c).__name__ for c in restored.sub_competitors], ["EloCompetitor", "BradleyTerryCompetitor"]
        )
        self.assertEqual(restored.sub_competitors[0].rating, 400)
        self.assertAlmostEqual(restored.sub_competitors[1].rating, expected_rating)
        self.assertAlmostEqual(restored.expected_score(restored), 0.5)

    def test_accepts_every_registered_competitor_type(self):
        names = [n for n in BaseCompetitor.list_competitor_types() if n != "BlendedCompetitor"]
        self.assertIn("BradleyTerryCompetitor", names)

        for name in names:
            with self.subTest(competitor_type=name):
                sub_competitor = self._new_competitor(name)
                restored = BlendedCompetitor.from_state(self._legacy_state(sub_competitor))
                self.assertEqual(len(restored.sub_competitors), 1)
                self.assertIsInstance(restored.sub_competitors[0], type(sub_competitor))
                self.assertAlmostEqual(restored.sub_competitors[0].rating, sub_competitor.rating)

    def test_unknown_competitor_type_raises(self):
        state = {"blend_mode": "mean", "competitors": [{"type": "NotACompetitor", "competitor_kwargs": {}}]}

        with self.assertRaisesRegex(InvalidParameterException, "Unknown competitor type: NotACompetitor"):
            BlendedCompetitor.from_state(state)
