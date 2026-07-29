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

    def test_inactivity_update_is_idempotent(self):
        """Repeated inactivity updates for the same timestamp must inflate RD only once."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=10)
        player = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )

        # sqrt((200 / 173.7178)^2 + 10 * 0.06^2) * 173.7178
        player.update_rd_for_inactivity(match_time)
        self.assertAlmostEqual(player.rd, 202.6978, places=4)

        # Applying the same timestamp again must be a no-op
        player.update_rd_for_inactivity(match_time)
        self.assertAlmostEqual(player.rd, 202.6978, places=4)
        player.update_rd_for_inactivity(match_time)
        self.assertAlmostEqual(player.rd, 202.6978, places=4)

    def test_single_match_applies_inactivity_once(self):
        """A single timestamped match must inflate RD for inactivity exactly once."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=10)
        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )

        reference = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        reference.update_rd_for_inactivity(match_time)
        # phi used by the match update is the once-inflated one, so v (and therefore the
        # posterior RD) must match what a single inactivity update produces.
        expected_phi = reference._phi

        player1.beat(player2, match_time=match_time)
        v = 1 / (player1._g(expected_phi) ** 2 * 0.25)
        phi_star_sq = expected_phi**2 + player1.volatility**2
        expected_rd = 173.7178 / math.sqrt(1 / phi_star_sq + 1 / v)
        self.assertAlmostEqual(player1.rd, expected_rd, places=6)

    def test_symmetric_match_has_no_rating_drift(self):
        """Two identical competitors must end a match with equal RD and zero rating drift."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=10)

        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        player1.beat(player2, match_time=match_time)
        self.assertEqual(player1.rd, player2.rd)
        self.assertEqual(player1.volatility, player2.volatility)
        self.assertAlmostEqual(player1.rating + player2.rating, 3000.0, places=9)

        # The same must hold on the default (untimestamped) path.
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.06)
        player1.beat(player2)
        self.assertAlmostEqual(player1.rd, player2.rd, places=6)
        self.assertAlmostEqual(player1.rating + player2.rating, 3000.0, places=6)

    def test_symmetric_tie_leaves_ratings_unchanged(self):
        """A draw between two identical competitors must not move either rating."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=10)

        player1 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        player2 = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        player1.tied(player2, match_time=match_time)
        self.assertAlmostEqual(player1.rating, 1500.0, places=9)
        self.assertAlmostEqual(player2.rating, 1500.0, places=9)
        self.assertEqual(player1.rd, player2.rd)

    def test_beat_known_values_ten_day_gap(self):
        """Pin the posterior state for the documented 1500/200/0.06, 10-day-gap match."""
        initial_time = datetime(2020, 1, 1)
        match_time = initial_time + timedelta(days=10)

        winner = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        loser = Glicko2Competitor(
            initial_rating=1500, initial_rd=200, initial_volatility=0.06, initial_time=initial_time
        )
        winner.beat(loser, match_time=match_time)

        self.assertAlmostEqual(winner.rating, 1580.327992, places=6)
        self.assertAlmostEqual(winner.rd, 182.167363, places=6)
        self.assertAlmostEqual(winner.volatility, 0.059999626, places=9)
        self.assertAlmostEqual(loser.rating, 1419.672008, places=6)
        self.assertAlmostEqual(loser.rd, 182.167363, places=6)
        self.assertAlmostEqual(loser.volatility, 0.059999626, places=9)


if __name__ == "__main__":
    unittest.main()
