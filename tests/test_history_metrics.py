import unittest
from elote.arenas.base import History, Bout
from elote.competitors.elo import EloCompetitor
from elote.arenas.lambda_arena import LambdaArena
import math


class MockArena:
    """Mock arena class for testing."""

    def __init__(self, history=None):
        self.history = history or History()


class TestHistoryMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.history = History()

        # Clear any existing bouts
        self.history.bouts = []

        # Add bouts with known outcomes to get predictable confusion matrix values
        # True positives: actual='a', predicted > upper_threshold
        self.history.add_bout(Bout("a", "b", 0.8, "a"))
        self.history.add_bout(Bout("a", "b", 0.9, "a"))
        self.history.add_bout(Bout("a", "b", 0.7, "a"))

        # False positive: actual='b', predicted > upper_threshold
        self.history.add_bout(Bout("a", "b", 0.7, "b"))

        # True negatives: actual='b', predicted < lower_threshold
        self.history.add_bout(Bout("a", "b", 0.2, "b"))
        self.history.add_bout(Bout("a", "b", 0.3, "b"))

        # False negative: actual='a', predicted < lower_threshold
        self.history.add_bout(Bout("a", "b", 0.1, "a"))

        # Uncertain predictions (between thresholds) - these should be counted as false positives when not draws
        self.history.add_bout(Bout("a", "b", 0.5, "a"))  # Predicting draw when actual='a' is a false positive
        self.history.add_bout(Bout("a", "b", 0.5, "b"))  # Predicting draw when actual='b' is a false positive

    def test_calculate_metrics_default_thresholds(self):
        """Test that calculate_metrics works with default thresholds."""
        # Use the default thresholds (0.5, 0.5)
        metrics = self.history.calculate_metrics()

        # Check that the metrics dictionary has the expected keys
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("confusion_matrix", metrics)

        # With default thresholds:
        # - Three true positives (0.7, 0.8, 0.9 with actual='a')
        # - Three false positives (0.7 with actual='b' and two 0.5 predictions that should be a/b)
        # - Two true negatives (0.2, 0.3 with actual='b')
        # - One false negative (0.1 with actual='a')
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 3)  # Three true positives (0.7, 0.8, 0.9 with actual='a')
        self.assertEqual(
            cm["fp"], 3
        )  # One false positive (0.7 with actual='b') plus two draw predictions that should be a/b
        self.assertEqual(cm["tn"], 2)  # Two true negatives (0.2, 0.3 with actual='b')
        self.assertEqual(cm["fn"], 1)  # One false negative (0.1 with actual='a')

        # Check the calculated metrics
        total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
        self.assertAlmostEqual(metrics["accuracy"], (cm["tp"] + cm["tn"]) / total, places=5)
        self.assertAlmostEqual(metrics["precision"], cm["tp"] / (cm["tp"] + cm["fp"]), places=5)
        self.assertAlmostEqual(metrics["recall"], cm["tp"] / (cm["tp"] + cm["fn"]), places=5)
        self.assertAlmostEqual(
            metrics["f1"],
            2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]),
            places=5,
        )

    def test_calculate_metrics_custom_thresholds(self):
        """Test that calculate_metrics works with custom thresholds."""
        # Use custom thresholds (0.4, 0.6)
        metrics = self.history.calculate_metrics(lower_threshold=0.4, upper_threshold=0.6)

        # With these thresholds:
        # - Predictions 0.7, 0.8, 0.9 are above the upper threshold (0.6)
        # - Predictions 0.1, 0.2, 0.3 are below the lower threshold (0.4)
        # - Predictions 0.5 are in the uncertain range and count as false positives when not draws

        # Check confusion matrix values
        cm = metrics["confusion_matrix"]

        # True positives: predictions > 0.6 with actual='a'
        # In our setup: 0.7, 0.8, 0.9 with actual='a' = 3 cases
        self.assertEqual(cm["tp"], 3)

        # False positives: predictions > 0.6 with actual='b' plus uncertain predictions that should be a/b
        # In our setup: 0.7 with actual='b' plus two 0.5 predictions = 3 cases
        self.assertEqual(cm["fp"], 3)

        # True negatives: predictions < 0.4 with actual='b'
        # In our setup: 0.2, 0.3 with actual='b' = 2 cases
        self.assertEqual(cm["tn"], 2)

        # False negatives: predictions < 0.4 with actual='a'
        # In our setup: 0.1 with actual='a' = 1 case
        self.assertEqual(cm["fn"], 1)

        # Check metrics
        total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
        self.assertAlmostEqual(metrics["accuracy"], (cm["tp"] + cm["tn"]) / total, places=5)
        self.assertAlmostEqual(metrics["precision"], cm["tp"] / (cm["tp"] + cm["fp"]), places=5)
        self.assertAlmostEqual(metrics["recall"], cm["tp"] / (cm["tp"] + cm["fn"]), places=5)
        self.assertAlmostEqual(
            metrics["f1"],
            2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]),
            places=5,
        )

    def test_calculate_metrics_edge_cases(self):
        """Test that calculate_metrics handles edge cases correctly."""
        # Create a new empty history
        empty_history = History()

        # Calculate metrics for an empty history
        metrics = empty_history.calculate_metrics()

        # Check the confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 0)
        self.assertEqual(cm["fp"], 0)
        self.assertEqual(cm["tn"], 0)
        self.assertEqual(cm["fn"], 0)

        # Check the metrics (should all be 0 or NaN)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertTrue(math.isnan(metrics["precision"]))
        self.assertTrue(math.isnan(metrics["recall"]))
        self.assertTrue(math.isnan(metrics["f1"]))

        # Create a history with only uncertain predictions
        uncertain_history = History()
        uncertain_history.add_bout(Bout("a", "b", 0.5, "a"))  # Should be false positive since actual is 'a'
        uncertain_history.add_bout(Bout("a", "b", 0.5, "b"))  # Should be false positive since actual is 'b'

        # Calculate metrics
        metrics = uncertain_history.calculate_metrics()

        # Check the confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 0)
        self.assertEqual(cm["fp"], 2)  # Both uncertain predictions are false positives since actuals are not draws
        self.assertEqual(cm["tn"], 0)
        self.assertEqual(cm["fn"], 0)

        # Check the metrics
        self.assertEqual(metrics["accuracy"], 0)  # No correct predictions
        self.assertEqual(metrics["precision"], 0)  # No true positives
        self.assertTrue(math.isnan(metrics["recall"]))  # No true positives or false negatives
        self.assertTrue(math.isnan(metrics["f1"]))  # Precision or recall is NaN

    def test_accuracy_by_prior_bouts_basic(self):
        """Test that accuracy_by_prior_bouts works with a simple arena."""
        # Create a mock arena with a history
        arena_history = History()
        arena_history.add_bout(Bout("A", "B", 0.7, 1.0))
        arena_history.add_bout(Bout("A", "C", 0.6, 1.0))
        arena_history.add_bout(Bout("B", "C", 0.4, 0.0))

        mock_arena = MockArena(arena_history)

        # Create a test history
        test_history = History()
        test_history.add_bout(Bout("A", "B", 0.7, 1.0))  # True positive
        test_history.add_bout(Bout("A", "C", 0.7, 0.0))  # False positive
        test_history.add_bout(Bout("B", "C", 0.3, 0.0))  # True negative

        # Calculate accuracy by prior bouts
        result = test_history.accuracy_by_prior_bouts(mock_arena)

        # Check that the result has the expected structure
        self.assertIn("binned", result)

        # Check that the binned dictionary has entries
        binned = result["binned"]
        self.assertGreater(len(binned), 0)

        # Check that the first bin has the expected keys
        first_bin = next(iter(binned.values()))
        self.assertIn("accuracy", first_bin)
        self.assertIn("total", first_bin)
        self.assertIn("min_bouts", first_bin)
        self.assertIn("max_bouts", first_bin)

    def test_accuracy_by_prior_bouts_with_real_arena(self):
        """Test accuracy_by_prior_bouts with a real arena."""

        # Create a real arena with some competitors
        def compare_func(a, b, attributes=None):
            return True  # Always predict a wins

        arena = LambdaArena(compare_func, base_competitor=EloCompetitor)

        # Add some matchups to the arena
        arena.matchup("A", "B")
        arena.matchup("A", "C")
        arena.matchup("B", "C")

        # Create a test history
        test_history = History()
        test_history.add_bout(Bout("A", "B", 0.7, 1.0))  # True positive
        test_history.add_bout(Bout("A", "C", 0.7, 0.0))  # False positive
        test_history.add_bout(Bout("B", "C", 0.3, 0.0))  # True negative

        # Calculate accuracy by prior bouts
        result = test_history.accuracy_by_prior_bouts(arena)

        # Check that the result has the expected structure
        self.assertIn("binned", result)

    def test_accuracy_by_prior_bouts_with_custom_thresholds(self):
        """Test accuracy_by_prior_bouts with custom thresholds."""
        # Create a mock arena with a history
        arena_history = History()
        arena_history.add_bout(Bout("A", "B", 0.7, 1.0))
        arena_history.add_bout(Bout("A", "C", 0.6, 1.0))
        arena_history.add_bout(Bout("B", "C", 0.4, 0.0))

        mock_arena = MockArena(arena_history)

        # Create a test history
        test_history = History()
        test_history.add_bout(Bout("A", "B", 0.7, 1.0))  # True positive
        test_history.add_bout(Bout("A", "C", 0.7, 0.0))  # False positive
        test_history.add_bout(Bout("B", "C", 0.3, 0.0))  # True negative

        # Calculate accuracy by prior bouts with custom thresholds
        result = test_history.accuracy_by_prior_bouts(mock_arena, thresholds=(0.4, 0.6))

        # Check that the result has the expected structure
        self.assertIn("binned", result)

        # Check that the binned dictionary has entries
        binned = result["binned"]
        self.assertGreater(len(binned), 0)

    def test_accuracy_by_prior_bouts_edge_cases(self):
        """Test accuracy_by_prior_bouts with edge cases."""
        # Empty arena history
        mock_arena = MockArena()

        # Empty test history
        empty_history = History()
        result = empty_history.accuracy_by_prior_bouts(mock_arena)

        # Check that the result has the expected structure
        self.assertIn("binned", result)

        # Check that the dictionaries are empty
        self.assertEqual(len(result["binned"]), 0)

        # Arena with no history attribute
        class ArenaWithoutHistory:
            pass

        arena_without_history = ArenaWithoutHistory()

        # Test history with some bouts
        test_history = History()
        test_history.add_bout(Bout("A", "B", 0.7, 1.0))

        # Should not raise an error
        result = test_history.accuracy_by_prior_bouts(arena_without_history)

        # Check that the result has the expected structure
        self.assertIn("binned", result)


if __name__ == "__main__":
    unittest.main()
