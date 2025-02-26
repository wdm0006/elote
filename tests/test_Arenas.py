import unittest
from elote import LambdaArena, EloCompetitor


class TestArenas(unittest.TestCase):
    def test_lambda_arena_initialization(self):
        """Test that the LambdaArena initializes correctly with different parameters."""
        # Test with default parameters
        arena = LambdaArena(lambda a, b: a > b)
        self.assertEqual(len(arena.competitors), 0)
        self.assertEqual(arena.base_competitor, EloCompetitor)
        self.assertEqual(arena.base_competitor_kwargs, {})

        # Test with custom competitor class parameters
        arena = LambdaArena(lambda a, b: a > b, base_competitor_kwargs={"initial_rating": 1000})
        self.assertEqual(arena.base_competitor_kwargs, {"initial_rating": 1000})

        # Test with initial state
        initial_state = {"A": {"initial_rating": 1200}, "B": {"initial_rating": 800}}
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)
        self.assertEqual(len(arena.competitors), 2)
        self.assertIn("A", arena.competitors)
        self.assertIn("B", arena.competitors)
        self.assertEqual(arena.competitors["A"].rating, 1200)
        self.assertEqual(arena.competitors["B"].rating, 800)

    def test_lambda_arena_matchup(self):
        """Test that matchups work correctly in the LambdaArena."""

        # Create a simple comparison function
        def compare(a, b, attributes=None):
            if attributes and "force_winner" in attributes:
                return attributes["force_winner"] == a
            return a > b

        arena = LambdaArena(compare)

        # Test a simple matchup where a > b
        arena.matchup(10, 5)
        self.assertEqual(len(arena.competitors), 2)
        self.assertIn(10, arena.competitors)
        self.assertIn(5, arena.competitors)

        # The winner's rating should be higher than the initial rating
        initial_rating = EloCompetitor().rating
        self.assertGreater(arena.competitors[10].rating, initial_rating)

        # Test a matchup with attributes
        arena.matchup("X", "Y", attributes={"force_winner": "Y"})
        self.assertEqual(len(arena.competitors), 4)
        self.assertIn("X", arena.competitors)
        self.assertIn("Y", arena.competitors)

        # Y should have won despite X normally being greater alphabetically
        self.assertGreater(arena.competitors["Y"].rating, initial_rating)

    def test_lambda_arena_tournament(self):
        """Test that tournaments work correctly in the LambdaArena."""
        arena = LambdaArena(lambda a, b: a > b)

        # Create a tournament with multiple matchups
        matchups = [(10, 5), (15, 8), (5, 3), (8, 10)]

        arena.tournament(matchups)

        # Check that all competitors are in the arena
        self.assertEqual(len(arena.competitors), 5)
        for competitor in [3, 5, 8, 10, 15]:
            self.assertIn(competitor, arena.competitors)

        # Check that the history has recorded all bouts
        self.assertEqual(len(arena.history.bouts), 4)

    def test_lambda_arena_expected_score(self):
        """Test that expected scores are calculated correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors with different ratings
        initial_state = {"A": {"initial_rating": 1200}, "B": {"initial_rating": 800}}
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)

        # A should have a higher expected score against B
        self.assertGreater(arena.expected_score("A", "B"), 0.5)
        self.assertLess(arena.expected_score("B", "A"), 0.5)

        # Test with new competitors that aren't in the arena yet
        score_c_d = arena.expected_score("C", "D")
        self.assertAlmostEqual(score_c_d, 0.5, places=2)

        # Now they should be in the arena with default ratings
        self.assertIn("C", arena.competitors)
        self.assertIn("D", arena.competitors)

    def test_lambda_arena_export_state(self):
        """Test that the arena state can be exported correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors and run some matchups
        arena.matchup("A", "B")
        arena.matchup("B", "C")
        arena.matchup("A", "C")

        # Export the state
        state = arena.export_state()

        # Check that all competitors are in the state
        self.assertEqual(len(state), 3)
        self.assertIn("A", state)
        self.assertIn("B", state)
        self.assertIn("C", state)

        # Check that the state contains the correct information
        for competitor in ["A", "B", "C"]:
            self.assertIn("initial_rating", state[competitor])
            self.assertIn("class_vars", state[competitor])

    def test_lambda_arena_leaderboard(self):
        """Test that the leaderboard is generated correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors with different ratings
        initial_state = {
            "A": {"initial_rating": 1200},
            "B": {"initial_rating": 1000},
            "C": {"initial_rating": 800},
        }
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)

        # Get the leaderboard
        leaderboard = arena.leaderboard()

        # Check that the leaderboard is sorted by rating (ascending)
        self.assertEqual(len(leaderboard), 3)
        self.assertEqual(leaderboard[0]["competitor"], "C")
        self.assertEqual(leaderboard[1]["competitor"], "B")
        self.assertEqual(leaderboard[2]["competitor"], "A")

    def test_lambda_arena_clear_history(self):
        """Test that the history can be cleared."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some matchups
        arena.matchup("A", "B")
        arena.matchup("B", "C")

        # Check that the history has recorded the bouts
        self.assertEqual(len(arena.history.bouts), 2)

        # Clear the history
        arena.clear_history()

        # Check that the history is empty
        self.assertEqual(len(arena.history.bouts), 0)

    def test_lambda_arena_set_competitor_class_var(self):
        """Test that competitor class variables can be set."""
        arena = LambdaArena(lambda a, b: a > b)

        # Set a class variable
        arena.set_competitor_class_var("_k_factor", 16)

        # Check that the class variable was set
        self.assertEqual(EloCompetitor._k_factor, 16)

        # Reset the class variable for other tests
        arena.set_competitor_class_var("_k_factor", 32)
