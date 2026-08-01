import unittest
import numpy as np
from elote import ColleyMatrixCompetitor, EloCompetitor
from elote.competitors.base import MissMatchedCompetitorTypesException


class TestColleyMatrix(unittest.TestCase):
    @staticmethod
    def _symmetric_connected_population():
        a, b, c, d = (ColleyMatrixCompetitor() for _ in range(4))
        a.beat(b)
        c.beat(d)
        a.tied(c)
        b.tied(d)
        return a, b, c, d

    def test_improvement(self):
        """Test that beating stronger opponents improves rating."""
        initial_rating = 0.5
        player1 = ColleyMatrixCompetitor(initial_rating=initial_rating)

        # If player1 beats someone with a higher rating, their rating should go up
        for _ in range(5):
            player2 = ColleyMatrixCompetitor(initial_rating=0.8)
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_decay(self):
        """Test that losing to weaker opponents decreases rating."""
        initial_rating = 0.8
        player1 = ColleyMatrixCompetitor(initial_rating=initial_rating)

        # If player1 loses to someone with a lower rating, their rating should go down
        for _ in range(5):
            player2 = ColleyMatrixCompetitor(initial_rating=0.2)
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_expectation(self):
        """Test that expected scores are calculated correctly."""
        player1 = ColleyMatrixCompetitor(initial_rating=0.8)
        player2 = ColleyMatrixCompetitor(initial_rating=0.2)

        # Higher rated player should have higher expected score
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))

    def test_network_recalculation(self):
        """Test that ratings are recalculated across the network of connected competitors."""
        # Create a network of 5 competitors
        competitors = [ColleyMatrixCompetitor(initial_rating=0.5) for _ in range(5)]

        # Create some matches to establish a network
        # 0 beats 1, 1 beats 2, 2 beats 3, 3 beats 4, 4 beats 0 (circular)
        competitors[0].beat(competitors[1])
        competitors[1].beat(competitors[2])
        competitors[2].beat(competitors[3])
        competitors[3].beat(competitors[4])
        competitors[4].beat(competitors[0])

        # All competitors have identical records in this circular pattern
        ratings = [c.rating for c in competitors]
        self.assertEqual(len(set(ratings)), 1)

        # Ratings should sum to n/2 = 2.5 (property of Colley Matrix Method)
        self.assertAlmostEqual(sum(ratings), len(competitors) / 2)

        # Additional test: if a new player beats the highest rated player, they should improve
        new_player = ColleyMatrixCompetitor(initial_rating=0.5)
        highest_player = competitors[np.argmax([c.rating for c in competitors])]
        initial_rating = new_player.rating
        new_player.beat(highest_player)
        self.assertGreater(new_player.rating, initial_rating)

    def test_exceptions(self):
        """Test that appropriate exceptions are raised."""
        player1 = ColleyMatrixCompetitor(initial_rating=0.5)
        player2 = EloCompetitor(initial_rating=1000)

        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.verify_competitor_types(player2)

        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.expected_score(player2)

    def test_identical_records_have_identical_ratings(self):
        a, b, c, d = self._symmetric_connected_population()

        self.assertEqual(a.rating, c.rating)
        self.assertEqual(b.rating, d.rating)

    def test_duplicate_ratings_preserve_sum_invariant(self):
        competitors = self._symmetric_connected_population()

        self.assertAlmostEqual(sum(competitor.rating for competitor in competitors), 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
