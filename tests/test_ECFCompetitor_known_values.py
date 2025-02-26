import unittest
from elote import ECFCompetitor


class TestECFKnownValues(unittest.TestCase):
    """Tests for ECFCompetitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating is set correctly."""
        player = ECFCompetitor(initial_rating=40)
        self.assertEqual(player.rating, 40)

        player = ECFCompetitor(initial_rating=100)
        self.assertEqual(player.rating, 100)

    def test_elo_conversion(self):
        """Test that elo_conversion property returns the correct value."""
        player = ECFCompetitor(initial_rating=40)
        self.assertEqual(player.elo_conversion, 40 * 7.5 + 700)

        player = ECFCompetitor(initial_rating=100)
        self.assertEqual(player.elo_conversion, 100 * 7.5 + 700)

    def test_transformed_elo_rating(self):
        """Test that transformed_elo_rating property returns the correct value."""
        player = ECFCompetitor(initial_rating=40)
        expected = 10 ** ((40 * 7.5 + 700) / 400)
        self.assertAlmostEqual(player.transformed_elo_rating, expected)

        # Test caching - should return the same value without recalculating
        self.assertAlmostEqual(player.transformed_elo_rating, expected)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = ECFCompetitor(initial_rating=40)
        player2 = ECFCompetitor(initial_rating=60)

        # Calculate expected values manually
        p1_transformed = 10 ** ((40 * 7.5 + 700) / 400)
        p2_transformed = 10 ** ((60 * 7.5 + 700) / 400)
        expected = p1_transformed / (p1_transformed + p2_transformed)

        self.assertAlmostEqual(player1.expected_score(player2), expected)

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        player1 = ECFCompetitor(initial_rating=40)
        player2 = ECFCompetitor(initial_rating=60)

        # Player1 beats player2
        player1.beat(player2)

        # After player1 beats player2, player1's rating should be updated
        # The new rating is the mean of the scores in the deque
        # Since we just initialized the deque, it contains [40] initially
        # After beat, it contains [40, 60+50] = [40, 110]
        # So the mean is (40 + 110) / 2 = 75
        self.assertEqual(player1.rating, 75)

        # After player1 beats player2, player2's rating should be updated
        # The new rating is the mean of the scores in the deque
        # Since we just initialized the deque, it contains [60] initially
        # After beat, it contains [60, 40-50] = [60, -10]
        # So the mean is (60 + (-10)) / 2 = 25
        self.assertEqual(player2.rating, 25)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        player1 = ECFCompetitor(initial_rating=40)
        player2 = ECFCompetitor(initial_rating=60)

        # Players tie
        player1.tied(player2)

        # After tie, player1's rating should be updated
        # The new rating is the mean of the scores in the deque
        # Since we just initialized the deque, it contains [40] initially
        # After tie, it contains [40, 60] = [40, 60]
        # So the mean is (40 + 60) / 2 = 50
        self.assertEqual(player1.rating, 50)

        # After tie, player2's rating should be updated
        # The new rating is the mean of the scores in the deque
        # Since we just initialized the deque, it contains [60] initially
        # After tie, it contains [60, 40] = [60, 40]
        # So the mean is (60 + 40) / 2 = 50
        self.assertEqual(player2.rating, 50)

    def test_delta_limit(self):
        """Test that the delta limit is applied correctly."""
        player1 = ECFCompetitor(initial_rating=40)
        player2 = ECFCompetitor(initial_rating=200)  # Rating difference > delta (50)

        # Player1 beats player2
        player1.beat(player2)

        # Since difference > delta, player2's effective rating should be limited
        # The effective rating of player2 is limited to player1's rating + delta = 40 + 50 = 90
        # After beat, player1's scores deque contains [40, 90+50] = [40, 140]
        # So the mean is (40 + 140) / 2 = 90
        self.assertEqual(player1.rating, 90)

    def test_scores_deque_behavior(self):
        """Test that the scores deque behaves correctly with maxlen."""
        player = ECFCompetitor(initial_rating=40)

        # Initialize scores
        if player.scores is None:
            player._ECFCompetitor__initialize_ratings()

        # Add more than _n_periods scores
        for i in range(player._n_periods + 10):
            player._update(i)

        # Check that only the last _n_periods scores are kept
        self.assertEqual(len(player.scores), player._n_periods)

        # Check that the oldest scores were dropped
        self.assertEqual(min(player.scores), player._n_periods + 10 - player._n_periods)


if __name__ == "__main__":
    unittest.main()
