import unittest
from elote import GlickoCompetitor
import math
from datetime import datetime, timedelta


class TestGlickoKnownValues(unittest.TestCase):
    """Tests for GlickoCompetitor with known values to verify correctness."""

    def test_initial_rating(self):
        """Test that initial rating and RD are set correctly."""
        player = GlickoCompetitor(initial_rating=1500, initial_rd=350)
        self.assertEqual(player.rating, 1500)
        self.assertEqual(player.rd, 350)

        player = GlickoCompetitor(initial_rating=2000, initial_rd=200)
        self.assertEqual(player.rating, 2000)
        self.assertEqual(player.rd, 200)

    def test_transformed_rd(self):
        """Test that transformed RD is calculated correctly."""
        player = GlickoCompetitor(initial_rating=1500, initial_rd=300)
        expected_rd = min([350, math.sqrt(300**2 + 34.6**2)])
        self.assertAlmostEqual(player.tranformed_rd, expected_rd)

    def test_g_function(self):
        """Test the g function with known values."""
        player = GlickoCompetitor(initial_rating=1500, initial_rd=300)
        g = player._g(300)
        expected_g = 1 / math.sqrt(1 + 3 * (0.0057565**2) * (300**2) / math.pi**2)
        self.assertAlmostEqual(g, expected_g)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=300)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)

        # Calculate expected score manually per Glickman's formula:
        # E = 1 / (1 + 10^(-g(RD_j)(r_i - r_j)/400)), where RD_j is the opponent's RD.
        g = player2._g(player2.rd)
        E = 1 / (1 + 10 ** ((-g * (1500 - 1700)) / 400))
        self.assertAlmostEqual(player1.expected_score(player2), E)

    def test_expected_score_large_gap(self):
        """A large positive rating gap should give an expected score near 1 (and its mirror near 0)."""
        strong = GlickoCompetitor(initial_rating=2500, initial_rd=350)
        weak = GlickoCompetitor(initial_rating=1500, initial_rd=350)

        self.assertGreater(strong.expected_score(weak), 0.9)
        self.assertLess(weak.expected_score(strong), 0.1)

    def test_expected_score_symmetric(self):
        """Equal ratings and RD should give an expected score of exactly 0.5."""
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        player2 = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        self.assertAlmostEqual(player1.expected_score(player2), 0.5)

    def test_expected_score_monotonic_in_gap(self):
        """Expected score should increase monotonically with the rating gap."""
        base = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        opponents = [GlickoCompetitor(initial_rating=r, initial_rd=200) for r in (1200, 1400, 1500, 1600, 1800)]
        scores = [base.expected_score(o) for o in opponents]
        for earlier, later in zip(scores, scores[1:], strict=False):
            self.assertGreater(earlier, later)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        initial_time = datetime(2020, 1, 1)
        match_time = datetime(2020, 1, 10)  # 10 days later

        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=50, initial_time=initial_time)

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

        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=50, initial_time=initial_time)

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
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350, initial_time=initial_time)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=50, initial_time=initial_time)
        player1.beat(player2, match_time=match_time)
        rating_change_high_rd = abs(player1.rating - 1500)

        # Reset with lower RD
        player1 = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        player2 = GlickoCompetitor(initial_rating=1700, initial_rd=50, initial_time=initial_time)
        player1.beat(player2, match_time=match_time)
        rating_change_low_rd = abs(player1.rating - 1500)

        # The rating change with higher RD should be greater
        self.assertGreater(rating_change_high_rd, rating_change_low_rd)

    def test_rd_increase_over_time(self):
        """Test that RD increases over time."""
        initial_time = datetime(2020, 1, 1)
        player = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)

        # Test that RD increases over time
        current_time = initial_time + timedelta(days=1)
        initial_rd = player.rd
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD increases more over longer periods
        player = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        current_time = initial_time + timedelta(days=10)
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD is capped at 350
        player = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        current_time = initial_time + timedelta(days=1000)  # Very long time
        player.update_rd_for_inactivity(current_time)
        self.assertLessEqual(player.rd, 350)

    def test_fractional_rating_periods(self):
        """Test RD increase with fractional rating periods."""
        initial_time = datetime(2020, 1, 1)
        player = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)

        # Test that RD increases for half a period
        current_time = initial_time + timedelta(hours=12)
        initial_rd = player.rd
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)

        # Test that RD increases more for 1.5 periods than 0.5 periods
        player = GlickoCompetitor(initial_rating=1500, initial_rd=50, initial_time=initial_time)
        current_time = initial_time + timedelta(hours=36)
        player.update_rd_for_inactivity(current_time)
        self.assertGreater(player.rd, initial_rd)


if __name__ == "__main__":
    unittest.main()
