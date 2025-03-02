import unittest
from elote import Glicko2Competitor
from elote.competitors.base import (
    MissMatchedCompetitorTypesException,
    InvalidParameterException,
    InvalidRatingValueException,
)


class TestGlicko2(unittest.TestCase):
    def test_improvement(self):
        """Test that a player's rating improves when they beat higher-rated players."""
        player1 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)
        initial_rating = player1.rating

        # If player1 beats someone with a high rating, their rating should go up
        for _ in range(5):
            player2 = Glicko2Competitor(initial_rating=1800, initial_rd=350, initial_volatility=0.06)
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_decay(self):
        """Test that a player's rating decreases when they lose to lower-rated players."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=350, initial_volatility=0.06)
        initial_rating = player1.rating

        # If player1 loses to someone with a low rating, their rating should go down
        for _ in range(5):
            player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_expectation(self):
        """Test that expected scores are calculated correctly."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)

        # Player1 should be expected to win more often
        self.assertGreater(player1.expected_score(player2), 0.5)
        self.assertLess(player2.expected_score(player1), 0.5)

        # Equal players should have equal expected scores
        player3 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)
        self.assertAlmostEqual(player1.expected_score(player3), 0.5, places=5)

    def test_initialization(self):
        """Test that the Glicko2Competitor initializes correctly."""
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

        # Test invalid initialization
        with self.assertRaises(InvalidRatingValueException):
            Glicko2Competitor(initial_rating=-100)
        with self.assertRaises(InvalidParameterException):
            Glicko2Competitor(initial_rd=-100)
        with self.assertRaises(InvalidParameterException):
            Glicko2Competitor(initial_volatility=-0.1)

    def test_rating_properties(self):
        """Test the rating, rd, and volatility properties."""
        player = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.08)

        # Test getters
        self.assertEqual(player.rating, 1800)
        self.assertEqual(player.rd, 100)
        self.assertEqual(player.volatility, 0.08)

        # Test setters
        player.rating = 1900
        self.assertEqual(player.rating, 1900)
        player.rd = 150
        self.assertEqual(player.rd, 150)
        player.volatility = 0.07
        self.assertEqual(player.volatility, 0.07)

        # Test invalid setters
        with self.assertRaises(InvalidRatingValueException):
            player.rating = -100
        with self.assertRaises(InvalidParameterException):
            player.rd = -100
        with self.assertRaises(InvalidParameterException):
            player.volatility = -0.1

    def test_reset(self):
        """Test that reset returns the competitor to its initial state."""
        player = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.08)
        player.rating = 1900
        player.rd = 150
        player.volatility = 0.07
        player.reset()
        self.assertEqual(player.rating, 1800)
        self.assertEqual(player.rd, 100)
        self.assertEqual(player.volatility, 0.08)

    def test_export_import_state(self):
        """Test that the state can be exported and imported correctly."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.08)
        player1.rating = 1900
        player1.rd = 150
        player1.volatility = 0.07

        # Export the state
        state = player1.export_state()

        # Create a new competitor from the state
        player2 = Glicko2Competitor.from_state(state)

        # Check that the new competitor has the same state
        self.assertEqual(player2.rating, 1900)
        self.assertEqual(player2.rd, 150)
        self.assertEqual(player2.volatility, 0.07)

    def test_exceptions(self):
        """Test that the correct exceptions are raised."""
        player1 = Glicko2Competitor()
        Glicko2Competitor()

        # Test mismatched competitor types
        class DummyCompetitor:
            pass

        dummy = DummyCompetitor()
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.expected_score(dummy)
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.beat(dummy)
        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.tied(dummy)

    def test_update_ratings(self):
        """Test that update_ratings processes match results correctly."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=250, initial_volatility=0.08)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)

        # Store the ratings before any matches
        pre_match_rating1 = player1.rating

        # Player1 beats player2 (this already calls update_ratings internally)
        player1.beat(player2)

        # Check that the ratings have changed
        self.assertNotEqual(player1.rating, pre_match_rating1)

        # Test that update_ratings with no matches increases RD
        player3 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)
        initial_rd = player3.rd
        player3.update_ratings()  # No matches, should increase RD
        self.assertGreater(player3.rd, initial_rd)

    def test_tied_match(self):
        """Test that tied matches update ratings correctly."""
        player1 = Glicko2Competitor(initial_rating=1800, initial_rd=100, initial_volatility=0.06)
        player2 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)

        # Store the initial ratings
        initial_rating1 = player1.rating
        initial_rating2 = player2.rating

        # Simulate a tie
        player1.tied(player2)

        # The higher-rated player should lose rating, and the lower-rated player should gain rating
        self.assertLess(player1.rating, initial_rating1)
        self.assertGreater(player2.rating, initial_rating2)


if __name__ == "__main__":
    unittest.main()
