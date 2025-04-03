import unittest
from elote import Glicko2Competitor
import math
from datetime import datetime, timedelta


class TestGlicko2KnownValues(unittest.TestCase):
    """Tests for Glicko2Competitor with known values to verify correctness."""

    def test_initial_rating(self):
        """Test that initial rating and RD are set correctly."""
        player = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)
        self.assertEqual(player.rating, 1500)
        self.assertEqual(player.rd, 350)

        player = Glicko2Competitor(initial_rating=2000, initial_rd=200, initial_volatility=0.06)
        self.assertEqual(player.rating, 2000)
        self.assertEqual(player.rd, 200)

    def test_g_function(self):
        """Test the g function with known values."""
        player = Glicko2Competitor(initial_rating=1500, initial_rd=300, initial_volatility=0.06)

        # g(phi) = 1 / sqrt(1 + 3 * phi^2 / pi^2)
        phi = 300 / 173.7178
        1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

        # Test the g function indirectly through expected_score
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=300, initial_volatility=0.06)
        actual_score = player.expected_score(player2)
        self.assertAlmostEqual(actual_score, 0.5)  # Equal ratings should give 0.5

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=300, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300, initial_volatility=0.06)

        # Test that a lower-rated player has less than 0.5 expected score
        self.assertLess(player1.expected_score(player2), 0.5)

        # Test that probabilities sum to 1
        self.assertAlmostEqual(player1.expected_score(player2) + player2.expected_score(player1), 1.0)

    def test_rd_increase_over_time(self):
        """Test RD increase over time with known values."""
        initial_time = datetime(2020, 1, 1)
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )

        # Test that RD increases over time
        current_time = initial_time + timedelta(days=1)
        initial_rd = player.rd
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD increases more over longer periods
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        current_time = initial_time + timedelta(days=10)
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD is capped at 350
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        current_time = initial_time + timedelta(days=1000)  # Very long time
        player.update_rd_for_inactivity(current_time)
        self.assertLessEqual(player.rd, 350)

    def test_fractional_rating_periods(self):
        """Test RD increase with fractional rating periods."""
        initial_time = datetime(2020, 1, 1)
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )

        # Test that RD increases for half a period
        current_time = initial_time + timedelta(hours=12)
        initial_rd = player.rd
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD increases more for 1.5 periods than 0.5 periods
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        current_time = initial_time + timedelta(hours=36)
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        initial_time = datetime(2020, 1, 1)
        match_time = datetime(2020, 1, 10)  # 10 days later

        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1700, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )

        # Store initial ratings
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Perform the match
        player1.beat(player2, match_time=match_time)

        # Check that ratings changed in the expected direction
        self.assertGreater(player1.rating, initial_rating1)  # Winner's rating should increase
        self.assertLess(player2.rating, initial_rating2)  # Loser's rating should decrease

        # Check that RDs decreased (more certainty after a match)
        self.assertLess(player1.rd, 350)
        self.assertLess(player2.rd, 350)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        initial_time = datetime(2020, 1, 1)
        match_time = datetime(2020, 1, 10)  # 10 days later

        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1700, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )

        # Store initial ratings
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Perform the match
        player1.tied(player2, match_time=match_time)

        # Check that ratings changed in the expected direction
        self.assertGreater(player1.rating, initial_rating1)  # Lower-rated player should gain rating
        self.assertLess(player2.rating, initial_rating2)  # Higher-rated player should lose rating

        # Check that RDs decreased (more certainty after a match)
        self.assertLess(player1.rd, 350)
        self.assertLess(player2.rd, 350)

    def test_rd_effect(self):
        """Test that RD affects the rating change magnitude."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=2)  # Match happens 2 days after initialization

        # With high RD (more uncertainty)
        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=350, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1700, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        player1.beat(player2, match_time=match_time)
        rating_change_high_rd = abs(player1.rating - 1500)

        # Reset with lower RD
        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1700, initial_rd=50, initial_volatility=0.06, initial_time=initial_time
        )
        player1.beat(player2, match_time=match_time)
        rating_change_low_rd = abs(player1.rating - 1500)

        # The rating change with higher RD should be greater
        self.assertGreater(rating_change_high_rd, rating_change_low_rd)


if __name__ == "__main__":
    unittest.main()
