import unittest
from elote import EloCompetitor


class TestEloKnownValues(unittest.TestCase):
    """Tests for EloCompetitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating is set correctly."""
        player = EloCompetitor(initial_rating=400)
        self.assertEqual(player.rating, 400)

        player = EloCompetitor(initial_rating=1000)
        self.assertEqual(player.rating, 1000)

    def test_transformed_rating(self):
        """Test that transformed_rating property returns the correct value."""
        player = EloCompetitor(initial_rating=400)
        expected = 10 ** (400 / 400)  # Should be 10^1 = 10
        self.assertEqual(player.transformed_rating, expected)

        player = EloCompetitor(initial_rating=800)
        expected = 10 ** (800 / 400)  # Should be 10^2 = 100
        self.assertEqual(player.transformed_rating, expected)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = EloCompetitor(initial_rating=400)
        player2 = EloCompetitor(initial_rating=400)

        # Equal ratings should give 0.5 expected score
        self.assertEqual(player1.expected_score(player2), 0.5)

        player1 = EloCompetitor(initial_rating=400)
        player2 = EloCompetitor(initial_rating=600)

        # Calculate expected values manually
        p1_transformed = 10 ** (400 / 400)  # 10
        p2_transformed = 10 ** (600 / 400)  # 10^1.5 ≈ 31.6228
        expected = p1_transformed / (p1_transformed + p2_transformed)

        self.assertAlmostEqual(player1.expected_score(player2), expected)
        self.assertAlmostEqual(player2.expected_score(player1), 1 - expected)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        player1 = EloCompetitor(initial_rating=400, k_factor=32)
        player2 = EloCompetitor(initial_rating=400, k_factor=32)

        # Store original ratings
        p1_original = player1.rating
        p2_original = player2.rating

        # Calculate expected scores
        win_es = player1.expected_score(player2)  # Should be 0.5
        lose_es = player2.expected_score(player1)  # Should be 0.5

        # Calculate expected new ratings
        p1_new_rating = p1_original + 32 * (1 - win_es)  # 400 + 32 * 0.5 = 416
        p2_new_rating = p2_original + 32 * (0 - lose_es)  # 400 + 32 * -0.5 = 384

        # Player1 beats player2
        player1.beat(player2)

        # Check new ratings
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)

        # Test with different ratings
        player1 = EloCompetitor(initial_rating=500, k_factor=32)
        player2 = EloCompetitor(initial_rating=400, k_factor=32)

        # Store original ratings
        p1_original = player1.rating
        p2_original = player2.rating

        # Calculate expected scores
        win_es = player1.expected_score(player2)
        lose_es = player2.expected_score(player1)

        # Calculate expected new ratings
        p1_new_rating = p1_original + 32 * (1 - win_es)
        p2_new_rating = p2_original + 32 * (0 - lose_es)

        # Player1 beats player2
        player1.beat(player2)

        # Check new ratings
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        player1 = EloCompetitor(initial_rating=400, k_factor=32)
        player2 = EloCompetitor(initial_rating=400, k_factor=32)

        # Store original ratings
        p1_original = player1.rating
        p2_original = player2.rating

        # Calculate expected scores
        win_es = player1.expected_score(player2)  # Should be 0.5
        lose_es = player2.expected_score(player1)  # Should be 0.5

        # Calculate expected new ratings
        p1_new_rating = p1_original + 32 * (0.5 - win_es)  # 400 + 32 * 0 = 400
        p2_new_rating = p2_original + 32 * (0.5 - lose_es)  # 400 + 32 * 0 = 400

        # Players tie
        player1.tied(player2)

        # Check new ratings - should be unchanged for equal ratings
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)

        # Test with different ratings
        player1 = EloCompetitor(initial_rating=500, k_factor=32)
        player2 = EloCompetitor(initial_rating=400, k_factor=32)

        # Store original ratings
        p1_original = player1.rating
        p2_original = player2.rating

        # Calculate expected scores
        win_es = player1.expected_score(player2)
        lose_es = player2.expected_score(player1)

        # Calculate expected new ratings
        p1_new_rating = p1_original + 32 * (0.5 - win_es)  # Should decrease for higher rated player
        p2_new_rating = p2_original + 32 * (0.5 - lose_es)  # Should increase for lower rated player

        # Players tie
        player1.tied(player2)

        # Check new ratings
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)

    def test_k_factor_effect(self):
        """Test that k_factor affects the rating change magnitude."""
        # With k_factor = 32
        player1 = EloCompetitor(initial_rating=400, k_factor=32)
        player2 = EloCompetitor(initial_rating=400, k_factor=32)
        player1.beat(player2)
        rating_change_32 = abs(player1.rating - 400)

        # With k_factor = 16
        player1 = EloCompetitor(initial_rating=400, k_factor=16)
        player2 = EloCompetitor(initial_rating=400, k_factor=16)
        player1.beat(player2)
        rating_change_16 = abs(player1.rating - 400)

        # The rating change with k_factor=32 should be twice the change with k_factor=16
        self.assertAlmostEqual(rating_change_32, 2 * rating_change_16)


if __name__ == "__main__":
    unittest.main()
