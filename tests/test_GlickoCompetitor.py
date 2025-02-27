import unittest
import math
from elote import GlickoCompetitor, EloCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException


class TestGlicko(unittest.TestCase):
    def test_Improvement(self):
        initial_rating = 100
        player1 = GlickoCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = GlickoCompetitor(initial_rating=800)
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Decay(self):
        initial_rating = 1500
        player1 = GlickoCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = GlickoCompetitor(initial_rating=1000)
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Expectation(self):
        player1 = GlickoCompetitor(initial_rating=1000)
        player2 = GlickoCompetitor(initial_rating=100)
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))

    def test_initialization(self):
        """Test that the GlickoCompetitor initializes correctly with different parameters."""
        # Test with default parameters
        player = GlickoCompetitor()
        self.assertEqual(player.rating, 1500)
        self.assertEqual(player.rd, 350)

        # Test with custom parameters
        player = GlickoCompetitor(initial_rating=2000, initial_rd=200)
        self.assertEqual(player.rating, 2000)
        self.assertEqual(player.rd, 200)

    def test_transformed_rd(self):
        """Test that the transformed RD is calculated correctly."""
        # Test with default parameters
        player = GlickoCompetitor()
        self.assertEqual(player.tranformed_rd, 350)

        # Test with a small RD
        player = GlickoCompetitor(initial_rd=100)
        # The transformed RD should be sqrt(100^2 + 1^2) = sqrt(10001) ≈ 100.005
        self.assertAlmostEqual(player.tranformed_rd, math.sqrt(100**2 + 1**2), places=2)

        # Test with a large RD
        player = GlickoCompetitor(initial_rd=400)
        # The transformed RD should be capped at 350
        self.assertEqual(player.tranformed_rd, 350)

    def test_g_function(self):
        """Test that the g function is calculated correctly."""
        # The g function is a mathematical function used in the Glicko rating system
        # g(x) = 1 / sqrt(1 + 3 * q^2 * x^2 / pi^2)
        # where q = ln(10) / 400 ≈ 0.0057565

        # Test with x = 0
        self.assertAlmostEqual(GlickoCompetitor._g(0), 1.0, places=5)

        # Test with x = 100
        expected_g = 1 / math.sqrt(1 + 3 * (0.0057565**2) * (100**2) / (math.pi**2))
        self.assertAlmostEqual(GlickoCompetitor._g(100), expected_g, places=5)

        # Test with x = 350 (typical RD value)
        expected_g = 1 / math.sqrt(1 + 3 * (0.0057565**2) * (350**2) / (math.pi**2))
        self.assertAlmostEqual(GlickoCompetitor._g(350), expected_g, places=5)

    def test_expected_score_calculation(self):
        """Test that the expected score is calculated correctly."""
        # The expected score is calculated as:
        # E = 1 / (1 + 10^(-g(RD_i^2) * (R_i - R_j) / 400))

        # Test with equal ratings
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        self.assertAlmostEqual(player1.expected_score(player2), 0.5, places=5)

        # Test with different ratings
        player1 = GlickoCompetitor(initial_rating=1800, initial_rd=100)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        # Calculate the expected score manually
        g_term = GlickoCompetitor._g(player1.rd**2)
        expected_score = 1 / (1 + 10 ** ((-1 * g_term * (player1.rating - player2.rating)) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected_score, places=5)

        # The sum of expected scores should not necessarily be 1.0 in Glicko
        # because each player uses their own RD in the calculation

    def test_tied_match(self):
        """Test that tied matches update ratings correctly."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        # Store the initial ratings
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Simulate a tie
        player1.tied(player2)

        # Since the players have equal ratings, a tie should not change their ratings much
        self.assertAlmostEqual(player1.rating, initial_rating1, delta=1)
        self.assertAlmostEqual(player2.rating, initial_rating2, delta=1)

        # But their RD should decrease (more certainty)
        self.assertLess(player1.rd, 350)
        self.assertLess(player2.rd, 350)

        # Test with different ratings
        player1 = GlickoCompetitor(initial_rating=1800, initial_rd=100)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        # Store the initial ratings
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Simulate a tie
        player1.tied(player2)

        # The higher-rated player should lose rating, and the lower-rated player should gain rating
        self.assertLess(player1.rating, initial_rating1)
        self.assertGreater(player2.rating, initial_rating2)

    def test_export_state(self):
        """Test that the state can be exported correctly."""
        player = GlickoCompetitor(initial_rating=1800, initial_rd=100)

        # Export the state
        state = player.export_state()

        # Check that the state contains the correct information
        self.assertEqual(state["initial_rating"], 1800)
        self.assertEqual(state["initial_rd"], 100)

        # Check that class variables are included (without underscore prefix)
        self.assertIn("c", state["class_vars"])
        self.assertEqual(state["class_vars"]["c"], 1)
        self.assertIn("q", state["class_vars"])
        self.assertEqual(state["class_vars"]["q"], 0.0057565)

    def test_exceptions(self):
        """Test that the correct exceptions are raised."""
        player1 = GlickoCompetitor(initial_rating=1500)
        player2 = EloCompetitor(initial_rating=1500)

        # Trying to compare different competitor types should raise an exception
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.verify_competitor_types(player2)

        # Trying to beat a different competitor type should raise an exception
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.beat(player2)

        # Trying to tie with a different competitor type should raise an exception
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.tied(player2)
