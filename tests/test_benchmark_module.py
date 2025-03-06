import unittest
from unittest.mock import patch, MagicMock
import logging
from elote.benchmark import evaluate_competitor, benchmark_competitors
from elote.competitors.elo import EloCompetitor
from elote.competitors.glicko import GlickoCompetitor
from elote.arenas.base import History, Bout
from elote.datasets.base import DataSplit


class TestBenchmarkModule(unittest.TestCase):
    def setUp(self):
        """Set up test data for benchmark functions."""
        # Create a mock data split
        self.train_data = [
            ("A", "B", 1.0, None, {"score_a": 10, "score_b": 5}),
            ("A", "C", 1.0, None, {"score_a": 8, "score_b": 3}),
            ("B", "C", 0.0, None, {"score_a": 4, "score_b": 7}),
        ]

        self.test_data = [
            ("A", "B", 1.0, None, {"score_a": 9, "score_b": 4}),
            ("A", "C", 0.0, None, {"score_a": 5, "score_b": 6}),
            ("B", "C", 0.0, None, {"score_a": 3, "score_b": 8}),
        ]

        self.data_split = DataSplit(self.train_data, self.test_data)

        # Create a comparison function
        self.compare_func = (
            lambda a, b, attributes=None: attributes.get("score_a", 0) > attributes.get("score_b", 0)
            if attributes
            else None
        )

        # Set up logging
        logging.basicConfig(level=logging.ERROR)  # Suppress log messages during tests

    @patch("elote.benchmark.train_arena_with_dataset")
    @patch("elote.benchmark.evaluate_arena_with_dataset")
    def test_evaluate_competitor(self, mock_evaluate, mock_train):
        """Test that evaluate_competitor works correctly."""
        # Create a mock history with some bouts
        mock_history = History()
        mock_history.add_bout(Bout("A", "B", 0.7, 1.0))
        mock_history.add_bout(Bout("A", "C", 0.7, 0.0))
        mock_history.add_bout(Bout("B", "C", 0.3, 0.0))

        # Mock the optimize_thresholds method to return a known value
        mock_history.optimize_thresholds = MagicMock(return_value=(0.75, (0.4, 0.6)))

        # Mock the calculate_metrics method to return known values
        mock_history.calculate_metrics = MagicMock(
            side_effect=[
                # First call with default thresholds
                {
                    "accuracy": 0.67,
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.67,
                    "confusion_matrix": {"tp": 1, "fp": 1, "tn": 1, "fn": 0, "undecided": 0},
                },
                # Second call with optimized thresholds
                {
                    "accuracy": 0.75,
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1": 0.67,
                    "confusion_matrix": {"tp": 1, "fp": 0, "tn": 1, "fn": 1, "undecided": 0},
                },
            ]
        )

        # Mock the accuracy_by_prior_bouts method
        mock_history.accuracy_by_prior_bouts = MagicMock(
            return_value={
                "by_bout_count": {0: {"accuracy": 0.5, "total": 2}},
                "binned": {0: {"accuracy": 0.5, "total": 2, "min_bouts": 0, "max_bouts": 4}},
            }
        )

        # Set up the mock return values
        mock_train.return_value = MagicMock()  # Return a mock arena
        mock_evaluate.return_value = mock_history

        # Call the function
        result = evaluate_competitor(
            competitor_class=EloCompetitor,
            data_split=self.data_split,
            comparison_function=self.compare_func,
            competitor_name="Test Elo",
            competitor_params={"k_factor": 32},
            optimize_thresholds=True,
        )

        # Check that the function was called with the correct arguments
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()

        # Check that the result has the expected keys
        self.assertIn("name", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertIn("accuracy_opt", result)
        self.assertIn("optimized_thresholds", result)
        self.assertIn("train_time", result)
        self.assertIn("eval_time", result)
        self.assertIn("top_teams", result)
        self.assertIn("accuracy_by_prior_bouts", result)

        # Check that the values are correct
        self.assertEqual(result["name"], "Test Elo")
        self.assertEqual(result["accuracy"], 0.67)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 0.67)
        self.assertEqual(result["accuracy_opt"], 0.75)
        self.assertEqual(result["optimized_thresholds"], (0.4, 0.6))

    @patch("elote.benchmark.evaluate_competitor")
    def test_benchmark_competitors(self, mock_evaluate):
        """Test that benchmark_competitors works correctly."""
        # Set up the mock return values
        mock_evaluate.side_effect = [
            {
                "name": "Elo",
                "accuracy": 0.67,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "accuracy_opt": 0.75,
                "optimized_thresholds": (0.4, 0.6),
                "train_time": 0.1,
                "eval_time": 0.1,
                "top_teams": [{"competitor": "A", "rating": 1600}],
                "accuracy_by_prior_bouts": {"by_bout_count": {}, "binned": {}},
            },
            {
                "name": "Glicko",
                "accuracy": 0.75,
                "precision": 0.67,
                "recall": 0.67,
                "f1": 0.67,
                "accuracy_opt": 0.83,
                "optimized_thresholds": (0.3, 0.7),
                "train_time": 0.2,
                "eval_time": 0.2,
                "top_teams": [{"competitor": "A", "rating": 1800}],
                "accuracy_by_prior_bouts": {"by_bout_count": {}, "binned": {}},
            },
        ]

        # Define competitor configs
        competitor_configs = [
            {"class": EloCompetitor, "name": "Elo", "params": {"k_factor": 32}},
            {"class": GlickoCompetitor, "name": "Glicko", "params": {}},
        ]

        # Call the function
        results = benchmark_competitors(
            competitor_configs=competitor_configs,
            data_split=self.data_split,
            comparison_function=self.compare_func,
            optimize_thresholds=True,
        )

        # Check that evaluate_competitor was called twice with the correct arguments
        self.assertEqual(mock_evaluate.call_count, 2)

        # First call should be for Elo
        args1, kwargs1 = mock_evaluate.call_args_list[0]
        self.assertEqual(kwargs1["competitor_class"], EloCompetitor)
        self.assertEqual(kwargs1["competitor_name"], "Elo")
        self.assertEqual(kwargs1["competitor_params"], {"k_factor": 32})

        # Second call should be for Glicko
        args2, kwargs2 = mock_evaluate.call_args_list[1]
        self.assertEqual(kwargs2["competitor_class"], GlickoCompetitor)
        self.assertEqual(kwargs2["competitor_name"], "Glicko")
        self.assertEqual(kwargs2["competitor_params"], {})

        # Check that the results list has the correct length and content
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "Elo")
        self.assertEqual(results[1]["name"], "Glicko")

    @patch("elote.benchmark.train_arena_with_dataset")
    @patch("elote.benchmark.evaluate_arena_with_dataset")
    def test_evaluate_competitor_without_optimization(self, mock_evaluate, mock_train):
        """Test that evaluate_competitor works without threshold optimization."""
        # Create a mock history with some bouts
        mock_history = History()
        mock_history.add_bout(Bout("A", "B", 0.7, 1.0))
        mock_history.add_bout(Bout("A", "C", 0.7, 0.0))
        mock_history.add_bout(Bout("B", "C", 0.3, 0.0))

        # Mock the calculate_metrics method to return known values
        mock_history.calculate_metrics = MagicMock(
            return_value={
                "accuracy": 0.67,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "confusion_matrix": {"tp": 1, "fp": 1, "tn": 1, "fn": 0, "undecided": 0},
            }
        )

        # Set up the mock return values
        mock_train.return_value = MagicMock()  # Return a mock arena
        mock_evaluate.return_value = mock_history

        # Call the function with optimize_thresholds=False
        result = evaluate_competitor(
            competitor_class=EloCompetitor,
            data_split=self.data_split,
            comparison_function=self.compare_func,
            competitor_name="Test Elo",
            competitor_params={"k_factor": 32},
            optimize_thresholds=False,
        )

        # Check that the result has the expected keys
        self.assertIn("name", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertNotIn("accuracy_opt", result)  # Should not be present
        self.assertNotIn("optimized_thresholds", result)  # Should not be present
        self.assertNotIn("accuracy_by_prior_bouts", result)  # Should not be present

    @patch("elote.benchmark.evaluate_competitor")
    def test_benchmark_competitors_without_optimization(self, mock_evaluate):
        """Test that benchmark_competitors works without threshold optimization."""
        # Set up the mock return values
        mock_evaluate.side_effect = [
            {
                "name": "Elo",
                "accuracy": 0.67,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "train_time": 0.1,
                "eval_time": 0.1,
                "top_teams": [{"competitor": "A", "rating": 1600}],
            },
            {
                "name": "Glicko",
                "accuracy": 0.75,
                "precision": 0.67,
                "recall": 0.67,
                "f1": 0.67,
                "train_time": 0.2,
                "eval_time": 0.2,
                "top_teams": [{"competitor": "A", "rating": 1800}],
            },
        ]

        # Define competitor configs
        competitor_configs = [
            {"class": EloCompetitor, "name": "Elo", "params": {"k_factor": 32}},
            {"class": GlickoCompetitor, "name": "Glicko", "params": {}},
        ]

        # Call the function with optimize_thresholds=False
        results = benchmark_competitors(
            competitor_configs=competitor_configs,
            data_split=self.data_split,
            comparison_function=self.compare_func,
            optimize_thresholds=False,
        )

        # Check that evaluate_competitor was called with optimize_thresholds=False
        args, kwargs = mock_evaluate.call_args_list[0]
        self.assertFalse(kwargs["optimize_thresholds"])

        # Check that the results list has the correct length and content
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "Elo")
        self.assertEqual(results[1]["name"], "Glicko")


if __name__ == "__main__":
    unittest.main()
