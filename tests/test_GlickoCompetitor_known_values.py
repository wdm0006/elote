import unittest
from elote import GlickoCompetitor
import math


class TestGlickoKnownValues(unittest.TestCase):
    """Tests for GlickoCompetitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating and RD are set correctly."""
        player = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        self.assertEqual(player.rating, 1500)
        self.assertEqual(player.rd, 350)

        player = GlickoCompetitor(initial_rating=2000, initial_rd=200)
        self.assertEqual(player.rating, 2000)
        self.assertEqual(player.rd, 200)

    def test_transformed_rd(self):
        """Test that transformed_rd property returns the correct value."""
        player = GlickoCompetitor(initial_rating=1500, initial_rd=300)
        # tranformed_rd = min([350, sqrt(rd^2 + c^2)])
        expected = min([350, math.sqrt(300**2 + 1**2)])
        self.assertEqual(player.tranformed_rd, expected)

        # Test with rd > 350
        player = GlickoCompetitor(initial_rating=1500, initial_rd=400)
        # Should be capped at 350
        self.assertEqual(player.tranformed_rd, 350)

    def test_g_function(self):
        """Test the _g function with known values."""
        # g(x) = 1 / sqrt(1 + 3 * q^2 * x^2 / pi^2)
        # where q = 0.0057565

        # Test with x = 0
        g_0 = GlickoCompetitor._g(0)
        expected_g_0 = 1 / math.sqrt(1 + 0)
        self.assertEqual(g_0, expected_g_0)

        # Test with x = 100
        g_100 = GlickoCompetitor._g(100)
        q = 0.0057565
        expected_g_100 = 1 / math.sqrt(1 + 3 * q**2 * 100**2 / math.pi**2)
        self.assertAlmostEqual(g_100, expected_g_100)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        # Equal ratings should give 0.5 expected score
        self.assertAlmostEqual(player1.expected_score(player2), 0.5)

        player1 = GlickoCompetitor(initial_rating=1700, initial_rd=300)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        # Calculate expected value manually
        g_term = GlickoCompetitor._g(player1.rd**2)
        expected = 1 / (1 + 10 ** ((-1 * g_term * (player1.rating - player2.rating)) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected)

    def test_update_competitor_rating(self):
        """Test update_competitor_rating with known values."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)

        # Calculate expected values manually
        E_term = player1.expected_score(player2)
        q = player1._q
        g = player1._g(player2.rd)
        d_squared = (q**2 * (g**2 * E_term * (1 - E_term))) ** -1

        # For a win (s=1)
        s_new_r = player1.rating + (q / (1 / player1.rd**2 + 1 / d_squared)) * g * (1 - E_term)
        s_new_rd = math.sqrt((1 / player1.rd**2 + 1 / d_squared) ** -1)

        # Get the calculated values
        calc_new_r, calc_new_rd = player1.update_competitor_rating(player2, 1)

        # Check that they match
        self.assertAlmostEqual(calc_new_r, s_new_r)
        self.assertAlmostEqual(calc_new_rd, s_new_rd)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)

        # Calculate expected new ratings and RDs
        # For player1 (winner, s=1)
        s_new_r, s_new_rd = player1.update_competitor_rating(player2, 1)

        # For player2 (loser, s=0)
        c_new_r, c_new_rd = player2.update_competitor_rating(player1, 0)

        # Player1 beats player2
        player1.beat(player2)

        # Check new ratings and RDs
        self.assertAlmostEqual(player1.rating, s_new_r)
        self.assertAlmostEqual(player1.rd, s_new_rd)
        self.assertAlmostEqual(player2.rating, c_new_r)
        self.assertAlmostEqual(player2.rd, c_new_rd)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)

        # Calculate expected new ratings and RDs
        # For player1 (tie, s=0.5)
        s_new_r, s_new_rd = player1.update_competitor_rating(player2, 0.5)

        # For player2 (tie, s=0.5)
        c_new_r, c_new_rd = player2.update_competitor_rating(player1, 0.5)

        # Players tie
        player1.tied(player2)

        # Check new ratings and RDs
        self.assertAlmostEqual(player1.rating, s_new_r)
        self.assertAlmostEqual(player1.rd, s_new_rd)
        self.assertAlmostEqual(player2.rating, c_new_r)
        self.assertAlmostEqual(player2.rd, c_new_rd)

    def test_rd_effect(self):
        """Test that RD affects the rating change magnitude."""
        # With high RD (more uncertainty)
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)
        player1.beat(player2)
        rating_change_high_rd = abs(player1.rating - 1500)

        # Reset
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=100)  # Lower RD
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)
        player1.beat(player2)
        rating_change_low_rd = abs(player1.rating - 1500)

        # The rating change with higher RD should be greater
        self.assertGreater(rating_change_high_rd, rating_change_low_rd)


if __name__ == "__main__":
    unittest.main()
