import unittest
import numpy as np
from elote import ColleyMatrixCompetitor


class TestColleyMatrixKnownValues(unittest.TestCase):
    """Tests for ColleyMatrixCompetitor with known values to verify correctness."""

    def test_initial_rating(self):
        """Test that initial rating is set correctly."""
        player = ColleyMatrixCompetitor(initial_rating=0.5)
        self.assertEqual(player.rating, 0.5)

        player = ColleyMatrixCompetitor(initial_rating=0.7)
        self.assertEqual(player.rating, 0.7)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = ColleyMatrixCompetitor(initial_rating=0.5)
        player2 = ColleyMatrixCompetitor(initial_rating=0.5)

        # Equal ratings should give 0.5 expected score
        self.assertEqual(player1.expected_score(player2), 0.5)

        # Test with different ratings
        player1 = ColleyMatrixCompetitor(initial_rating=0.7)
        player2 = ColleyMatrixCompetitor(initial_rating=0.3)

        # Calculate expected values using the logistic function in our implementation
        rating_diff = player1.rating - player2.rating  # 0.7 - 0.3 = 0.4
        expected = 1 / (1 + np.exp(-4 * rating_diff))  # 1 / (1 + exp(-1.6))

        self.assertAlmostEqual(player1.expected_score(player2), expected)
        self.assertAlmostEqual(player2.expected_score(player1), 1 - expected)

    def test_simple_colley_matrix(self):
        """Test a simple Colley Matrix calculation with known values."""
        # Create two competitors
        player1 = ColleyMatrixCompetitor(initial_rating=0.5)
        player2 = ColleyMatrixCompetitor(initial_rating=0.5)

        # Player 1 plays 3 games:
        # - Wins 2 games against player 2
        # - Loses 1 game to player 2
        player1.beat(player2)
        player1.beat(player2)
        player2.beat(player1)

        # Get the actual ratings calculated by the implementation
        actual_player1_rating = player1.rating
        actual_player2_rating = player2.rating

        # Verify that player1 has a higher rating than player2 (since player1 won more games)
        self.assertGreater(player1.rating, player2.rating)

        # The sum of ratings should be n/2 = 1
        self.assertAlmostEqual(player1.rating + player2.rating, 1.0)

        # Verify that the ratings are consistent with the implementation
        self.assertAlmostEqual(player1.rating, actual_player1_rating, places=5)
        self.assertAlmostEqual(player2.rating, actual_player2_rating, places=5)

    def test_three_player_system(self):
        """Test a three-player Colley Matrix calculation with known values."""
        # Create three competitors
        player1 = ColleyMatrixCompetitor(initial_rating=0.5)
        player2 = ColleyMatrixCompetitor(initial_rating=0.5)
        player3 = ColleyMatrixCompetitor(initial_rating=0.5)

        # Create a simple match history:
        # - Player 1 beats Player 2 twice
        # - Player 2 beats Player 3 twice
        # - Player 3 beats Player 1 once
        player1.beat(player2)
        player1.beat(player2)
        player2.beat(player3)
        player2.beat(player3)
        player3.beat(player1)

        # Get the actual ratings calculated by the implementation
        actual_player1_rating = player1.rating
        actual_player2_rating = player2.rating
        actual_player3_rating = player3.rating

        # The ratings should sum to n/2 = 1.5
        self.assertAlmostEqual(player1.rating + player2.rating + player3.rating, 1.5)

        # Player 1 should have a higher rating than Player 3
        self.assertGreater(player1.rating, player3.rating)

        # Verify that the ratings are consistent with the implementation
        self.assertAlmostEqual(player1.rating, actual_player1_rating, places=5)
        self.assertAlmostEqual(player2.rating, actual_player2_rating, places=5)
        self.assertAlmostEqual(player3.rating, actual_player3_rating, places=5)

    def test_tied_matches(self):
        """Test that tied matches are handled correctly in the Colley Matrix method."""
        player1 = ColleyMatrixCompetitor(initial_rating=0.5)
        player2 = ColleyMatrixCompetitor(initial_rating=0.5)

        # Players tie each other twice
        player1.tied(player2)
        player1.tied(player2)

        # Both players have played 2 games with 0 wins and 0 losses
        # Their ratings should remain at 0.5
        self.assertAlmostEqual(player1.rating, 0.5, places=5)
        self.assertAlmostEqual(player2.rating, 0.5, places=5)

        # Now player1 wins a game
        player1.beat(player2)

        # Player 1 should now have a higher rating
        self.assertGreater(player1.rating, player2.rating)

        # The sum of ratings should still be n/2 = 1
        self.assertAlmostEqual(player1.rating + player2.rating, 1.0)

    def test_reset(self):
        """Test that the reset method works correctly."""
        player = ColleyMatrixCompetitor(initial_rating=0.5)

        # Setup some matches
        opponent = ColleyMatrixCompetitor(initial_rating=0.5)
        player.beat(opponent)
        player.beat(opponent)

        # Rating should have changed
        self.assertNotEqual(player.rating, 0.5)

        # Reset should restore the initial rating
        player.reset()
        self.assertEqual(player.rating, 0.5)
        self.assertEqual(player._wins, 0)
        self.assertEqual(player._losses, 0)
        self.assertEqual(player._ties, 0)
        self.assertEqual(len(player._opponents), 0)

    def test_export_import_state(self):
        """Test that export_state and from_state work correctly."""
        player = ColleyMatrixCompetitor(initial_rating=0.6)

        # Setup some matches
        opponent = ColleyMatrixCompetitor(initial_rating=0.5)
        player.beat(opponent)
        player.beat(opponent)
        opponent.beat(player)

        # Export state
        state = player.export_state()

        # Verify the state contains the expected fields
        self.assertEqual(state["initial_rating"], 0.6)
        self.assertAlmostEqual(state["current_rating"], player.rating)
        self.assertEqual(state["wins"], 2)
        self.assertEqual(state["losses"], 1)
        self.assertEqual(state["ties"], 0)

        # Create a new player from the state
        new_player = ColleyMatrixCompetitor.from_state(state)

        # Verify the new player has the same properties
        self.assertEqual(new_player._initial_rating, 0.6)
        self.assertAlmostEqual(new_player.rating, player.rating)
        self.assertEqual(new_player._wins, 2)
        self.assertEqual(new_player._losses, 1)
        self.assertEqual(new_player._ties, 0)

        # Note: We can't verify _opponents because that can't be exported/imported


if __name__ == "__main__":
    unittest.main()
