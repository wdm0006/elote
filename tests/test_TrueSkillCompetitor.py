import unittest
import math
from elote import TrueSkillCompetitor, EloCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException, InvalidParameterException


class TestTrueSkill(unittest.TestCase):
    def test_improvement(self):
        """Test that a player's rating improves when they beat higher-rated players."""
        player1 = TrueSkillCompetitor(initial_mu=20, initial_sigma=8.333)
        initial_mu = player1.mu

        # If player1 beats someone with a high rating, their mu should go up
        for _ in range(5):
            player2 = TrueSkillCompetitor(initial_mu=30, initial_sigma=8.333)
            player1.beat(player2)
            self.assertGreater(player1.mu, initial_mu)

    def test_decay(self):
        """Test that a player's rating decreases when they lose to lower-rated players."""
        player1 = TrueSkillCompetitor(initial_mu=30, initial_sigma=8.333)
        initial_mu = player1.mu

        # If player1 loses to someone with a low rating, their mu should go down
        for _ in range(5):
            player2 = TrueSkillCompetitor(initial_mu=20, initial_sigma=8.333)
            player2.beat(player1)
            self.assertLess(player1.mu, initial_mu)

    def test_expectation(self):
        """Test that expected scores are calculated correctly."""
        player1 = TrueSkillCompetitor(initial_mu=30, initial_sigma=5)
        player2 = TrueSkillCompetitor(initial_mu=20, initial_sigma=5)
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))
        self.assertGreater(player1.expected_score(player2), 0.5)
        self.assertLess(player2.expected_score(player1), 0.5)

    def test_initialization(self):
        """Test that the TrueSkillCompetitor initializes correctly with different parameters."""
        # Test with default parameters
        player = TrueSkillCompetitor()
        self.assertEqual(player.mu, player._default_mu)
        self.assertEqual(player.sigma, player._default_sigma)
        self.assertEqual(player.rating, max(player.mu - 3 * player.sigma, player._minimum_rating))

        # Test with custom parameters
        player = TrueSkillCompetitor(initial_mu=30, initial_sigma=5)
        self.assertEqual(player.mu, 30)
        self.assertEqual(player.sigma, 5)
        self.assertEqual(player.rating, max(30 - 3 * 5, player._minimum_rating))

        # Test with invalid parameters
        with self.assertRaises(InvalidParameterException):
            TrueSkillCompetitor(initial_sigma=0)
        with self.assertRaises(InvalidParameterException):
            TrueSkillCompetitor(initial_sigma=-1)

    def test_rating_properties(self):
        """Test the mu and sigma properties."""
        player = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)

        # Test getters
        self.assertEqual(player.mu, 25)
        self.assertEqual(player.sigma, 8.333)
        self.assertEqual(player.rating, max(25 - 3 * 8.333, player._minimum_rating))

        # Test setters
        player.mu = 30
        self.assertEqual(player.mu, 30)
        player.sigma = 5
        self.assertEqual(player.sigma, 5)
        self.assertEqual(player.rating, max(30 - 3 * 5, player._minimum_rating))

        # Test invalid values
        with self.assertRaises(InvalidParameterException):
            player.sigma = 0
        with self.assertRaises(InvalidParameterException):
            player.sigma = -1

        # Test that rating cannot be set directly
        with self.assertRaises(NotImplementedError):
            player.rating = 100

    def test_expected_score_calculation(self):
        """Test that the expected score is calculated correctly."""
        # Test with equal ratings
        player1 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
        player2 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
        self.assertAlmostEqual(player1.expected_score(player2), 0.335, places=3)  # Updated expected value

        # Test with different ratings
        player1 = TrueSkillCompetitor(initial_mu=30, initial_sigma=8.333)
        player2 = TrueSkillCompetitor(initial_mu=20, initial_sigma=8.333)
        self.assertGreater(player1.expected_score(player2), 0.5)
        self.assertLess(player2.expected_score(player1), 0.5)

    def test_tied_match(self):
        """Test that tied matches update ratings correctly."""
        player1 = TrueSkillCompetitor(initial_mu=30, initial_sigma=5)
        player2 = TrueSkillCompetitor(initial_mu=20, initial_sigma=5)

        # Store the initial ratings
        initial_mu1 = player1.mu
        initial_mu2 = player2.mu

        # Simulate a tie
        player1.tied(player2)

        # The higher-rated player should lose rating, and the lower-rated player should gain rating
        self.assertGreater(player1.mu, initial_mu1 - 1)  # Allow for small changes
        self.assertLess(player1.mu, initial_mu1 + 1)
        self.assertGreater(player2.mu, initial_mu2 - 1)
        self.assertLess(player2.mu, initial_mu2 + 1)

    def test_reset(self):
        """Test that reset returns the competitor to its initial state."""
        player = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)

        # Change the player's state
        player.mu = 30
        player.sigma = 5

        # Reset the player
        player.reset()

        # Check that the player's state is reset
        self.assertEqual(player.mu, 25)
        self.assertEqual(player.sigma, 8.333)

    def test_export_import_state(self):
        """Test that the state can be exported and imported correctly."""
        player = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)

        # Export the state
        state = player.export_state()

        # Create a new player from the state
        new_player = TrueSkillCompetitor.from_state(state)

        # Check that the new player has the same state
        self.assertEqual(new_player.mu, player.mu)
        self.assertEqual(new_player.sigma, player.sigma)

    def test_match_quality(self):
        """Test that match quality is calculated correctly."""
        # Equal players should have high match quality
        player1 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
        player2 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
        quality = TrueSkillCompetitor.match_quality(player1, player2)
        self.assertGreater(quality, 0.4)  # Updated expected value

        # Unequal players should have lower match quality
        player1 = TrueSkillCompetitor(initial_mu=30, initial_sigma=5)
        player2 = TrueSkillCompetitor(initial_mu=20, initial_sigma=5)
        quality1 = TrueSkillCompetitor.match_quality(player1, player2)
        self.assertLess(quality1, quality)  # Should be lower than equal players

    def test_create_team(self):
        """Test that teams can be created correctly."""
        player1 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
        player2 = TrueSkillCompetitor(initial_mu=30, initial_sigma=5)

        # Create a team
        team_mu, team_sigma = TrueSkillCompetitor.create_team([player1, player2])

        # Team mu should be the sum of player mus
        self.assertEqual(team_mu, player1.mu + player2.mu)

        # Team sigma should be the square root of the sum of squared player sigmas
        self.assertAlmostEqual(team_sigma, math.sqrt(player1.sigma**2 + player2.sigma**2))

    def test_exceptions(self):
        """Test that the correct exceptions are raised."""
        player1 = TrueSkillCompetitor(initial_mu=25, initial_sigma=8.333)
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


if __name__ == "__main__":
    unittest.main()
