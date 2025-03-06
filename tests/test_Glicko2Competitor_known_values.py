import unittest
from elote import Glicko2Competitor


class TestGlicko2KnownValues(unittest.TestCase):
    """Tests for Glicko2Competitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating, RD, and volatility are set correctly."""
        # Test default initialization
        player = Glicko2Competitor()
        self.assertEqual(player.rating, 1500)
        self.assertEqual(player.rd, 350)
        self.assertEqual(player.volatility, 0.06)

        # Test custom initialization
        player = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.08)
        self.assertEqual(player.rating, 1800)
        self.assertEqual(player.rd, 100)
        self.assertEqual(player.volatility, 0.08)

    def test_scale_conversion(self):
        """Test the conversion functions between rating and Glicko-2 scale."""
        player = Glicko2Competitor()

        # Test rating to mu conversion
        mu = player._rating_to_mu(1500)
        self.assertAlmostEqual(mu, 0, places=5)
        mu = player._rating_to_mu(1800)
        self.assertAlmostEqual(mu, 1.7269, places=4)

        # Test mu to rating conversion
        rating = player._mu_to_rating(0)
        self.assertAlmostEqual(rating, 1500, places=5)
        rating = player._mu_to_rating(1.7269)
        self.assertAlmostEqual(rating, 1800, places=1)

        # Test round-trip conversion
        original_rating = 1800
        mu = player._rating_to_mu(original_rating)
        rating = player._mu_to_rating(mu)
        self.assertAlmostEqual(rating, original_rating, places=5)

        # Test RD to phi conversion
        phi = player._rd_to_phi(350)
        self.assertAlmostEqual(phi, 2.0148, places=4)

        # Test phi to RD conversion
        rd = player._phi_to_rd(2.0148)
        self.assertAlmostEqual(rd, 350, places=1)

        # Test round-trip conversion
        original_rd = 200
        phi = player._rd_to_phi(original_rd)
        rd = player._phi_to_rd(phi)
        self.assertAlmostEqual(rd, original_rd, places=5)

    def test_g_function(self):
        """Test the g function with known values."""
        player = Glicko2Competitor()

        # Test with different phi values
        g_value = player._g(0.5)
        self.assertAlmostEqual(g_value, 0.9640, places=4)

        g_value = player._g(1.0)
        self.assertAlmostEqual(g_value, 0.8757, places=4)

        g_value = player._g(1.5)
        self.assertAlmostEqual(g_value, 0.7706, places=4)

    def test_E_function(self):
        """Test the E function with known values."""
        player = Glicko2Competitor()

        # Test with equal ratings
        e_value = player._E(0, 0, 0.5)
        self.assertAlmostEqual(e_value, 0.5, places=5)

        # Test with different ratings
        e_value = player._E(1.0, 0, 0.5)
        self.assertAlmostEqual(e_value, 0.7239, places=4)

        e_value = player._E(0, 1.0, 0.5)
        self.assertAlmostEqual(e_value, 0.2761, places=4)

        # Test with different g values
        e_value = player._E(1.0, 0, 0.8)
        self.assertAlmostEqual(e_value, 0.7140, places=4)

    def test_expected_score(self):
        """Test expected_score with known values."""
        # Test with equal ratings
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=350)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350)
        self.assertAlmostEqual(player1.expected_score(player2), 0.5, places=5)

        # Test with different ratings
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=100)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350)
        expected_score = player1.expected_score(player2)
        self.assertGreater(expected_score, 0.5)
        self.assertLess(player2.expected_score(player1), 0.5)

    def test_update_ratings_no_matches(self):
        """Test update_ratings with no matches."""
        player = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)

        # Store initial values
        initial_rating = player.rating
        initial_rd = player.rd
        initial_volatility = player.volatility

        # Update ratings with no matches
        player.update_ratings()

        # Rating should remain the same
        self.assertEqual(player.rating, initial_rating)

        # RD should increase (less certainty)
        self.assertGreater(player.rd, initial_rd)

        # Volatility should remain the same
        self.assertEqual(player.volatility, initial_volatility)

    def test_update_ratings_with_match(self):
        """Test update_ratings with a match."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)

        # Store initial values
        initial_rating1 = player1.rating
        initial_rd1 = player1.rd

        # Player1 beats player2 (this already calls update_ratings internally)
        player1.beat(player2)

        # Rating should increase for the winner
        self.assertGreater(player1.rating, initial_rating1)

        # RD should decrease (more certainty)
        self.assertLess(player1.rd, initial_rd1)

        # Match results should be cleared after update
        self.assertEqual(len(player1._match_results), 0)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        # Create players with specific ratings
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300, initial_volatility=0.06)

        # Store initial values
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Player1 beats player2
        player1.beat(player2)

        # Rating should increase for the winner and decrease for the loser
        self.assertGreater(player1.rating, initial_rating1)
        self.assertLess(player2.rating, initial_rating2)

        # Match results should be cleared after update
        self.assertEqual(len(player1._match_results), 0)
        self.assertEqual(len(player2._match_results), 0)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        # Create players with specific ratings
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300, initial_volatility=0.06)

        # Store initial values
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Players tie
        player1.tied(player2)

        # Lower-rated player should gain rating, higher-rated player should lose rating
        self.assertGreater(player1.rating, initial_rating1)
        self.assertLess(player2.rating, initial_rating2)

        # Match results should be cleared after update
        self.assertEqual(len(player1._match_results), 0)
        self.assertEqual(len(player2._match_results), 0)

    def test_rd_effect(self):
        """Test the effect of RD on rating change magnitude."""
        # Create players with different RDs
        player1_low_rd = Glicko2Competitor(initial_rating=1500, initial_rd=50, initial_volatility=0.06)
        player1_high_rd = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)

        # Create separate opponents for each test to avoid reset() issues
        player2_for_low_rd = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.06)
        player2_for_high_rd = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.06)

        # Store initial values
        initial_rating_low_rd = player1_low_rd.rating
        initial_rating_high_rd = player1_high_rd.rating

        # Both players beat their respective opponents
        player1_low_rd.beat(player2_for_low_rd)
        player1_high_rd.beat(player2_for_high_rd)

        # The player with higher RD should have a larger rating change
        rating_change_low_rd = player1_low_rd.rating - initial_rating_low_rd
        rating_change_high_rd = player1_high_rd.rating - initial_rating_high_rd

        self.assertGreater(rating_change_high_rd, rating_change_low_rd)

        # Check that RD changes after a match (may increase for low RD players)
        self.assertNotEqual(player1_low_rd.rd, 50)
        self.assertNotEqual(player1_high_rd.rd, 350)

    def test_volatility_effect(self):
        """Test the effect of volatility on rating changes."""
        # Create players with different volatilities
        player1_low_vol = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.03)
        player1_high_vol = Glicko2Competitor(initial_rating=1500, initial_rd=200, initial_volatility=0.09)

        # Create separate opponents for each test to avoid reset() issues
        player2_for_low_vol = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.06)
        player2_for_high_vol = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.06)

        # Store initial values
        initial_rating_low_vol = player1_low_vol.rating
        initial_rating_high_vol = player1_high_vol.rating

        # Both players beat their respective opponents
        player1_low_vol.beat(player2_for_low_vol)
        player1_high_vol.beat(player2_for_high_vol)

        # The player with higher volatility should have a larger rating change
        rating_change_low_vol = player1_low_vol.rating - initial_rating_low_vol
        rating_change_high_vol = player1_high_vol.rating - initial_rating_high_vol

        self.assertGreater(rating_change_high_vol, rating_change_low_vol)

        # Check that volatility changes after a match
        self.assertNotEqual(player1_low_vol.volatility, 0.03)
        self.assertNotEqual(player1_high_vol.volatility, 0.09)


if __name__ == "__main__":
    unittest.main()
