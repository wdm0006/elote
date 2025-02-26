import unittest
from elote import DWZCompetitor
import math


class TestDWZKnownValues(unittest.TestCase):
    """Tests for DWZCompetitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating is set correctly."""
        player = DWZCompetitor(initial_rating=400)
        self.assertEqual(player.rating, 400)

        player = DWZCompetitor(initial_rating=1000)
        self.assertEqual(player.rating, 1000)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Calculate expected value manually
        expected = 1 / (1 + 10 ** ((1200 - 1000) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected)

        # Test with different ratings
        player1 = DWZCompetitor(initial_rating=1500)
        player2 = DWZCompetitor(initial_rating=1300)

        expected = 1 / (1 + 10 ** ((1300 - 1500) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected)

    def test_E_calculation(self):
        """Test _E property calculation with known values."""
        # Test with rating < 1300
        player = DWZCompetitor(initial_rating=1000)

        # Calculate E manually
        E0 = (1000 / 1000) ** 4 + player._J
        a = max([0.5, min([1000 / 2000, 1])])
        B = math.exp((1300 - 1000) / 150) - 1
        E = int(a * E0 + B)
        expected = max([5, min([E, 150])])

        self.assertEqual(player._E, expected)

        # Test with rating >= 1300
        player = DWZCompetitor(initial_rating=1500)

        # Calculate E manually
        E0 = (1500 / 1000) ** 4 + player._J
        a = max([0.5, min([1500 / 2000, 1])])
        B = 0  # Since rating >= 1300
        E = int(a * E0 + B)
        expected = max([5, min([E, min([30, 5 * player._count])])])

        self.assertEqual(player._E, expected)

        # Test E caching
        cached_E = player._E
        self.assertEqual(player._E, cached_E)  # Should use cached value

    def test_beat_with_known_values(self):
        """Test beat method with known values."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Store original ratings and counts
        p1_original_rating = player1.rating
        p2_original_rating = player2.rating
        p1_original_count = player1._count
        p2_original_count = player2._count

        # Calculate expected new ratings manually
        p1_expected = player1.expected_score(player2)
        p1_E = player1._E
        p1_new_rating = p1_original_rating + (800 / (p1_E + p1_original_count)) * (1 - p1_expected)

        p2_expected = player2.expected_score(player1)
        p2_E = player2._E
        p2_new_rating = p2_original_rating + (800 / (p2_E + p2_original_count)) * (0 - p2_expected)

        # Player1 beats player2
        player1.beat(player2)

        # Check new ratings and counts
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)
        self.assertEqual(player1._count, p1_original_count + 1)
        self.assertEqual(player2._count, p2_original_count + 1)

    def test_tied_with_known_values(self):
        """Test tied method with known values."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Store original ratings and counts
        p1_original_rating = player1.rating
        p2_original_rating = player2.rating
        p1_original_count = player1._count
        p2_original_count = player2._count

        # Calculate expected new ratings manually
        p1_expected = player1.expected_score(player2)
        p1_E = player1._E
        p1_new_rating = p1_original_rating + (800 / (p1_E + p1_original_count)) * (0.5 - p1_expected)

        p2_expected = player2.expected_score(player1)
        p2_E = player2._E
        p2_new_rating = p2_original_rating + (800 / (p2_E + p2_original_count)) * (0.5 - p2_expected)

        # Players tie
        player1.tied(player2)

        # Check new ratings and counts
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)
        self.assertEqual(player1._count, p1_original_count + 1)
        self.assertEqual(player2._count, p2_original_count + 1)

    def test_cache_invalidation(self):
        """Test that cache is properly invalidated when rating changes."""
        player = DWZCompetitor(initial_rating=1000)

        # Access _E to cache the value
        original_E = player._E

        # Change rating directly
        player.rating = 1500

        # _E should be recalculated
        self.assertNotEqual(player._E, original_E)

        # Access _E again to cache the new value
        new_E = player._E

        # Change count
        player._count = 10

        # _E should be recalculated again
        self.assertNotEqual(player._E, new_E)


if __name__ == "__main__":
    unittest.main()
