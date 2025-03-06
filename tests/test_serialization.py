import unittest
import json

from elote import (
    EloCompetitor,
    GlickoCompetitor,
    BlendedCompetitor,
    ColleyMatrixCompetitor,
)
from elote.competitors.base import BaseCompetitor, InvalidStateException


class TestStandardizedSerialization(unittest.TestCase):
    """Test the standardized serialization functionality."""

    def test_elo_serialization(self):
        """Test serialization and deserialization of EloCompetitor."""
        # Create a competitor
        competitor = EloCompetitor(initial_rating=1200, k_factor=24)

        # Export the state
        state = competitor.export_state()

        # Check that the state has the required fields
        self.assertIn("type", state)
        self.assertIn("version", state)
        self.assertIn("created_at", state)
        self.assertIn("id", state)
        self.assertIn("parameters", state)
        self.assertIn("state", state)

        # Check the type and version
        self.assertEqual(state["type"], "EloCompetitor")
        self.assertEqual(state["version"], 1)

        # Check the parameters
        self.assertIn("initial_rating", state["parameters"])
        self.assertEqual(state["parameters"]["initial_rating"], 1200)
        self.assertIn("k_factor", state["parameters"])
        self.assertEqual(state["parameters"]["k_factor"], 24)

        # Check the state
        self.assertIn("rating", state["state"])
        self.assertEqual(state["state"]["rating"], 1200)

        # Create a new competitor from the state
        new_competitor = BaseCompetitor.from_state(state)

        # Check that the new competitor has the same properties
        self.assertIsInstance(new_competitor, EloCompetitor)
        self.assertEqual(new_competitor.rating, 1200)
        self.assertEqual(new_competitor._k_factor, 24)

        # Update the competitor's rating
        competitor.rating = 1300

        # Export the state again
        state = competitor.export_state()

        # Check that the state reflects the updated rating
        self.assertEqual(state["state"]["rating"], 1300)

        # Create a new competitor from the state
        new_competitor = BaseCompetitor.from_state(state)

        # Check that the new competitor has the updated rating
        self.assertEqual(new_competitor.rating, 1300)

        # Test import_state
        competitor2 = EloCompetitor(initial_rating=1000)
        competitor2.import_state(state)

        # Check that competitor2 now has the same properties as competitor
        self.assertEqual(competitor2.rating, 1300)
        self.assertEqual(competitor2._k_factor, 24)

    def test_glicko_serialization(self):
        """Test serialization and deserialization of GlickoCompetitor."""
        # Create a competitor
        competitor = GlickoCompetitor(initial_rating=1500, initial_rd=200)

        # Export the state
        state = competitor.export_state()

        # Check that the state has the required fields
        self.assertIn("type", state)
        self.assertIn("version", state)
        self.assertIn("created_at", state)
        self.assertIn("id", state)
        self.assertIn("parameters", state)
        self.assertIn("state", state)

        # Check the type and version
        self.assertEqual(state["type"], "GlickoCompetitor")
        self.assertEqual(state["version"], 1)

        # Check the parameters
        self.assertIn("initial_rating", state["parameters"])
        self.assertEqual(state["parameters"]["initial_rating"], 1500)
        self.assertIn("initial_rd", state["parameters"])
        self.assertEqual(state["parameters"]["initial_rd"], 200)

        # Check the state
        self.assertIn("rating", state["state"])
        self.assertEqual(state["state"]["rating"], 1500)
        self.assertIn("rd", state["state"])
        self.assertEqual(state["state"]["rd"], 200)

        # Create a new competitor from the state
        new_competitor = BaseCompetitor.from_state(state)

        # Check that the new competitor has the same properties
        self.assertIsInstance(new_competitor, GlickoCompetitor)
        self.assertEqual(new_competitor.rating, 1500)
        self.assertEqual(new_competitor.rd, 200)

    def test_colley_matrix_serialization(self):
        """Test serialization and deserialization of ColleyMatrixCompetitor."""
        # Create a competitor
        competitor = ColleyMatrixCompetitor(initial_rating=0.6)

        # Create some match history
        opponent = ColleyMatrixCompetitor(initial_rating=0.5)
        competitor.beat(opponent)
        competitor.beat(opponent)
        opponent.beat(competitor)

        # Export the state
        state = competitor.export_state()

        # Check that the state has the required fields
        self.assertIn("type", state)
        self.assertIn("version", state)
        self.assertIn("created_at", state)
        self.assertIn("id", state)
        self.assertIn("parameters", state)
        self.assertIn("state", state)

        # Check the type and version
        self.assertEqual(state["type"], "ColleyMatrixCompetitor")
        self.assertEqual(state["version"], 1)

        # Check the parameters
        self.assertIn("initial_rating", state["parameters"])
        self.assertEqual(state["parameters"]["initial_rating"], 0.6)

        # Check the state
        self.assertIn("rating", state["state"])
        self.assertAlmostEqual(state["state"]["rating"], competitor.rating)
        self.assertIn("wins", state["state"])
        self.assertEqual(state["state"]["wins"], 2)
        self.assertIn("losses", state["state"])
        self.assertEqual(state["state"]["losses"], 1)
        self.assertIn("ties", state["state"])
        self.assertEqual(state["state"]["ties"], 0)

        # Create a new competitor from the state
        new_competitor = BaseCompetitor.from_state(state)

        # Check that the new competitor has the same properties
        self.assertIsInstance(new_competitor, ColleyMatrixCompetitor)
        self.assertEqual(new_competitor._initial_rating, 0.6)
        self.assertAlmostEqual(new_competitor.rating, competitor.rating)
        self.assertEqual(new_competitor._wins, 2)
        self.assertEqual(new_competitor._losses, 1)
        self.assertEqual(new_competitor._ties, 0)

        # Note: We can't verify _opponents because that can't be exported/imported

    def test_blended_serialization(self):
        """Test serialization and deserialization of BlendedCompetitor."""
        # Create a competitor
        competitor = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
            ]
        )

        # Export the state
        state = competitor.export_state()

        # Check that the state has the required fields
        self.assertIn("type", state)
        self.assertIn("version", state)
        self.assertIn("created_at", state)
        self.assertIn("id", state)
        self.assertIn("parameters", state)
        self.assertIn("state", state)

        # Check the type and version
        self.assertEqual(state["type"], "BlendedCompetitor")
        self.assertEqual(state["version"], 1)

        # Check the parameters
        self.assertIn("blend_mode", state["parameters"])
        self.assertEqual(state["parameters"]["blend_mode"], "mean")
        self.assertIn("competitors", state["parameters"])
        self.assertEqual(len(state["parameters"]["competitors"]), 2)

        # Check the state
        self.assertIn("sub_competitors", state["state"])
        self.assertEqual(len(state["state"]["sub_competitors"]), 2)
        self.assertEqual(state["state"]["sub_competitors"][0]["type"], "EloCompetitor")
        self.assertEqual(state["state"]["sub_competitors"][1]["type"], "GlickoCompetitor")

        # Create a new competitor from the state
        new_competitor = BaseCompetitor.from_state(state)

        # Check that the new competitor has the same properties
        self.assertIsInstance(new_competitor, BlendedCompetitor)
        self.assertEqual(len(new_competitor.sub_competitors), 2)
        self.assertIsInstance(new_competitor.sub_competitors[0], EloCompetitor)
        self.assertIsInstance(new_competitor.sub_competitors[1], GlickoCompetitor)
        self.assertEqual(new_competitor.sub_competitors[0].rating, 1200)
        self.assertEqual(new_competitor.sub_competitors[1].rating, 1500)

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Create a competitor
        competitor = EloCompetitor(initial_rating=1200)

        # Convert to JSON
        json_str = competitor.to_json()

        # Check that the JSON string is valid
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            self.fail("JSON string is not valid")

        # Check that the JSON data has the required fields
        self.assertIn("type", json_data)
        self.assertIn("version", json_data)
        self.assertIn("created_at", json_data)
        self.assertIn("id", json_data)
        self.assertIn("parameters", json_data)
        self.assertIn("state", json_data)

        # Create a new competitor from the JSON string
        new_competitor = EloCompetitor.from_json(json_str)

        # Check that the new competitor has the same properties
        self.assertIsInstance(new_competitor, EloCompetitor)
        self.assertEqual(new_competitor.rating, 1200)

    def test_invalid_state(self):
        """Test handling of invalid state dictionaries."""
        # Test missing required fields
        with self.assertRaises(InvalidStateException):
            BaseCompetitor.from_state({})

        with self.assertRaises(InvalidStateException):
            BaseCompetitor.from_state({"type": "EloCompetitor"})

        # Test invalid type
        with self.assertRaises(InvalidStateException):
            EloCompetitor().import_state({"type": "GlickoCompetitor", "version": 1, "parameters": {}, "state": {}})

        # Test invalid JSON
        with self.assertRaises(InvalidStateException):
            EloCompetitor.from_json("invalid json")

    def test_competitor_registry(self):
        """Test the competitor registry."""
        # Check that all competitor types are registered
        competitor_types = BaseCompetitor.list_competitor_types()
        self.assertIn("EloCompetitor", competitor_types)
        self.assertIn("GlickoCompetitor", competitor_types)
        self.assertIn("ECFCompetitor", competitor_types)
        self.assertIn("DWZCompetitor", competitor_types)
        self.assertIn("BlendedCompetitor", competitor_types)

        # Get a competitor class by name
        elo_class = BaseCompetitor.get_competitor_class("EloCompetitor")
        self.assertEqual(elo_class, EloCompetitor)

        # Create a competitor using the class
        competitor = elo_class(initial_rating=1200)
        self.assertIsInstance(competitor, EloCompetitor)
        self.assertEqual(competitor.rating, 1200)


if __name__ == "__main__":
    unittest.main()
