import unittest
import numpy as np
from elote.arenas.base import History, Bout, BaseArena
from elote.competitors.elo import EloCompetitor
from elote.arenas.lambda_arena import LambdaArena


class MockArena:
    """Mock arena class for testing."""
    def __init__(self, history=None):
        self.history = history or History()


class TestHistoryMetrics(unittest.TestCase):
    def setUp(self):
        """Set up a history with some bouts for testing."""
        self.history = History()
        
        # Add some bouts with different outcomes
        self.history.add_bout(Bout("A", "B", 0.7, 1.0))  # True positive
        self.history.add_bout(Bout("C", "D", 0.7, 0.0))  # False positive
        self.history.add_bout(Bout("E", "F", 0.3, 0.0))  # True negative
        self.history.add_bout(Bout("G", "H", 0.3, 1.0))  # False negative
        
        # Add some bouts with string outcomes
        self.history.add_bout(Bout("I", "J", 0.8, "win"))  # True positive
        self.history.add_bout(Bout("K", "L", 0.8, "loss"))  # False positive
        self.history.add_bout(Bout("M", "N", 0.2, "loss"))  # True negative
        self.history.add_bout(Bout("O", "P", 0.2, "win"))  # False negative
        
        # Add some bouts with attributes
        self.history.add_bout(Bout("Q", "R", 0.6, 1.0, {"type": "test"}))
        self.history.add_bout(Bout("S", "T", 0.4, 0.0, {"type": "test"}))

    def test_calculate_metrics_default_thresholds(self):
        """Test that calculate_metrics works with default thresholds."""
        metrics = self.history.calculate_metrics()
        
        # Check that the metrics dictionary has the expected keys
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check the confusion matrix
        cm = metrics['confusion_matrix']
        self.assertEqual(cm['tp'], 3)  # Three true positives
        self.assertEqual(cm['fp'], 2)  # Two false positives
        self.assertEqual(cm['tn'], 3)  # Three true negatives
        self.assertEqual(cm['fn'], 2)  # Two false negatives
        
        # Check the metrics
        self.assertAlmostEqual(metrics['accuracy'], 0.6)  # (3 + 3) / 10
        self.assertAlmostEqual(metrics['precision'], 0.6)  # 3 / (3 + 2)
        self.assertAlmostEqual(metrics['recall'], 0.6)  # 3 / (3 + 2)
        self.assertAlmostEqual(metrics['f1'], 0.6)  # 2 * 0.6 * 0.6 / (0.6 + 0.6)

    def test_calculate_metrics_custom_thresholds(self):
        """Test that calculate_metrics works with custom thresholds."""
        # Use thresholds that will classify all bouts
        metrics = self.history.calculate_metrics(lower_threshold=0.4, upper_threshold=0.6)
        
        # Check the confusion matrix
        cm = metrics['confusion_matrix']
        self.assertEqual(cm['tp'], 2)  # Two true positives
        self.assertEqual(cm['fp'], 2)  # Two false positives
        self.assertEqual(cm['tn'], 3)  # Three true negatives
        self.assertEqual(cm['fn'], 2)  # Two false negatives
        self.assertEqual(cm['undecided'], 0)  # No undecided bouts with these thresholds
        
        # Check the metrics
        self.assertAlmostEqual(metrics['accuracy'], 0.5555555555555556)  # (2 + 3) / 9 (since there are 9 decided bouts)
        
        # Use thresholds that will classify all bouts as undecided
        metrics = self.history.calculate_metrics(lower_threshold=0.0, upper_threshold=1.0)
        
        # Check the confusion matrix
        cm = metrics['confusion_matrix']
        self.assertEqual(cm['tp'], 0)
        self.assertEqual(cm['fp'], 0)
        self.assertEqual(cm['tn'], 0)
        self.assertEqual(cm['fn'], 0)
        self.assertEqual(cm['undecided'], 10)  # All bouts
        
        # Check the metrics
        self.assertAlmostEqual(metrics['accuracy'], 0.0)  # No correct predictions

    def test_calculate_metrics_edge_cases(self):
        """Test calculate_metrics with edge cases."""
        # Empty history
        empty_history = History()
        metrics = empty_history.calculate_metrics()
        
        # Check that the metrics are all zero
        self.assertAlmostEqual(metrics['accuracy'], 0.0)
        self.assertAlmostEqual(metrics['precision'], 0.0)
        self.assertAlmostEqual(metrics['recall'], 0.0)
        self.assertAlmostEqual(metrics['f1'], 0.0)
        
        # History with only undecided bouts
        undecided_history = History()
        undecided_history.add_bout(Bout("A", "B", 0.5, 1.0))
        metrics = undecided_history.calculate_metrics()
        
        # Check that the metrics are all zero
        self.assertAlmostEqual(metrics['accuracy'], 0.0)
        self.assertAlmostEqual(metrics['precision'], 0.0)
        self.assertAlmostEqual(metrics['recall'], 0.0)
        self.assertAlmostEqual(metrics['f1'], 0.0)

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
        self.assertIn('by_bout_count', result)
        self.assertIn('binned', result)
        
        # Check that the by_bout_count dictionary has entries
        by_bout_count = result['by_bout_count']
        self.assertGreater(len(by_bout_count), 0)
        
        # Check that the binned dictionary has entries
        binned = result['binned']
        self.assertGreater(len(binned), 0)
        
        # Check that the first bin has the expected keys
        first_bin = next(iter(binned.values()))
        self.assertIn('accuracy', first_bin)
        self.assertIn('total', first_bin)
        self.assertIn('tp', first_bin)
        self.assertIn('fp', first_bin)
        self.assertIn('tn', first_bin)
        self.assertIn('fn', first_bin)
        self.assertIn('do_nothing', first_bin)
        self.assertIn('min_bouts', first_bin)
        self.assertIn('max_bouts', first_bin)

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
        self.assertIn('by_bout_count', result)
        self.assertIn('binned', result)
        
        # Check that the by_bout_count dictionary has entries
        by_bout_count = result['by_bout_count']
        self.assertGreater(len(by_bout_count), 0)

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
        self.assertIn('by_bout_count', result)
        self.assertIn('binned', result)
        
        # Check that the by_bout_count dictionary has entries
        by_bout_count = result['by_bout_count']
        self.assertGreater(len(by_bout_count), 0)
        
        # Check that the binned dictionary has entries
        binned = result['binned']
        self.assertGreater(len(binned), 0)

    def test_accuracy_by_prior_bouts_edge_cases(self):
        """Test accuracy_by_prior_bouts with edge cases."""
        # Empty arena history
        mock_arena = MockArena()
        
        # Empty test history
        empty_history = History()
        result = empty_history.accuracy_by_prior_bouts(mock_arena)
        
        # Check that the result has the expected structure
        self.assertIn('by_bout_count', result)
        self.assertIn('binned', result)
        
        # Check that the dictionaries are empty
        self.assertEqual(len(result['by_bout_count']), 0)
        self.assertEqual(len(result['binned']), 0)
        
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
        self.assertIn('by_bout_count', result)
        self.assertIn('binned', result)
        
        # Check that the by_bout_count dictionary has entries
        by_bout_count = result['by_bout_count']
        self.assertGreater(len(by_bout_count), 0)


if __name__ == '__main__':
    unittest.main() 