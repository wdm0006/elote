import unittest
from unittest.mock import patch, MagicMock
import logging
from elote.benchmark import BENCHMARK_INITIAL_RATING, evaluate_competitor, benchmark_competitors
from elote.competitors.elo import EloCompetitor
from elote.competitors.dwz import DWZCompetitor
from elote.competitors.glicko import GlickoCompetitor
from elote.competitors.trueskill import TrueSkillCompetitor
from elote.competitors.colley import ColleyMatrixCompetitor
from elote.competitors.massey import MasseyCompetitor
from elote.competitors.keener import KeenerCompetitor
from elote.arenas.base import History, Bout
from elote.datasets.base import DataSplit
from elote.datasets.synthetic import SyntheticDataset


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


def _always_true(a, b, attributes=None):
    """Comparison function for the dataset helpers.

    ``train_arena_with_dataset`` encodes the real outcome in the argument order it hands to
    ``arena.matchup``, so the comparison function only has to agree that the first argument
    won. It must accept ``attributes`` as a keyword because rows carry attributes.
    """
    return True


class TestBenchmarkEndToEnd(unittest.TestCase):
    """Exercises evaluate_competitor against real data, with nothing mocked out."""

    @classmethod
    def setUpClass(cls):
        dataset = SyntheticDataset(num_competitors=20, num_matchups=200, seed=42)
        cls.data_split = dataset.time_split(test_ratio=0.3)

    def test_class_variables_are_restored_after_evaluation(self):
        """evaluate_competitor must leave no residue on the competitor class.

        GlickoCompetitor is used because no other test in this module benchmarks it, so a
        leak from an earlier test cannot mask the assertions here.
        """
        # _minimum_rating is inherited from BaseCompetitor rather than declared on
        # GlickoCompetitor, so restoring it means removing the attribute again.
        self.assertNotIn("_minimum_rating", vars(GlickoCompetitor))
        before_minimum = GlickoCompetitor._minimum_rating
        before_c = GlickoCompetitor._c

        evaluate_competitor(
            competitor_class=GlickoCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            competitor_params={"c": 20.0},
            optimize_thresholds=False,
        )

        self.assertNotIn("_minimum_rating", vars(GlickoCompetitor))
        self.assertEqual(GlickoCompetitor._minimum_rating, before_minimum)
        self.assertEqual(GlickoCompetitor._c, before_c)

    def test_unknown_competitor_params_are_not_left_behind(self):
        """A class variable that did not exist before the run must not exist after it."""
        self.assertNotIn("_benchmark_probe", vars(EloCompetitor))

        evaluate_competitor(
            competitor_class=EloCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            competitor_params={"benchmark_probe": 123},
            optimize_thresholds=False,
        )

        self.assertNotIn("_benchmark_probe", vars(EloCompetitor))

    def test_competitors_start_from_the_common_initial_rating(self):
        """The competitors the arena actually built must start at the benchmark rating.

        Asserted on real competitor objects rather than on the class attribute: every
        __init__ assigns self._initial_rating from its own argument, so a class attribute
        set on the competitor class is shadowed and proves nothing.
        """
        for competitor_class in (EloCompetitor, DWZCompetitor):
            with self.subTest(competitor=competitor_class.__name__):
                result = evaluate_competitor(
                    competitor_class=competitor_class,
                    data_split=self.data_split,
                    comparison_function=_always_true,
                    optimize_thresholds=False,
                )
                competitors = list(result["arena"].competitors.values())
                self.assertGreater(len(competitors), 0)
                for competitor in competitors:
                    self.assertEqual(competitor._initial_rating, BENCHMARK_INITIAL_RATING)

    def test_competitor_without_initial_rating_argument(self):
        """A class whose __init__ takes no initial_rating must not be handed one."""
        result = evaluate_competitor(
            competitor_class=TrueSkillCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            optimize_thresholds=False,
        )

        competitors = list(result["arena"].competitors.values())
        self.assertGreater(len(competitors), 0)
        for competitor in competitors:
            self.assertIsInstance(competitor, TrueSkillCompetitor)

    def test_end_to_end_metrics_are_non_degenerate(self):
        """A real train/evaluate run must produce plausible, fully accounted-for metrics."""
        result = evaluate_competitor(
            competitor_class=EloCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            optimize_thresholds=True,
        )

        arena = result["arena"]
        scored_bouts = sum(
            1 for a, b, _, _, _ in self.data_split.test if a in arena.competitors and b in arena.competitors
        )
        self.assertGreater(scored_bouts, 0)
        self.assertEqual(len(result["history"].bouts), scored_bouts)

        confusion_matrix = result["confusion_matrix"]
        self.assertEqual(sum(confusion_matrix.values()), scored_bouts)

        self.assertGreater(result["accuracy"], 0.0)
        self.assertLess(result["accuracy"], 1.0)
        expected_accuracy = (confusion_matrix["tp"] + confusion_matrix["tn"]) / scored_bouts
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

        # Optimized thresholds can only improve on the default ones.
        self.assertGreaterEqual(result["accuracy_opt"], result["accuracy"])
        lower, upper = result["optimized_thresholds"]
        self.assertLessEqual(lower, upper)

        self.assertEqual(len(result["top_teams"]), 5)
        ratings = [team["rating"] for team in result["top_teams"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertGreater(ratings[0], ratings[-1])

    def test_keener_runs_end_to_end_through_the_benchmark(self):
        """Keener is a global-fit system, so the benchmark must not force an initial rating.

        evaluate_competitor detects global-fit classes by the presence of
        _recalculate_ratings and skips the common start for them; forcing 1500 on a system
        whose ratings average 1.0 would give degenerate first-bout predictions.
        """
        result = evaluate_competitor(
            competitor_class=KeenerCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            optimize_thresholds=True,
        )

        competitors = list(result["arena"].competitors.values())
        self.assertGreater(len(competitors), 0)
        for competitor in competitors:
            self.assertEqual(competitor._initial_rating, KeenerCompetitor._default_initial_rating)
            self.assertGreater(competitor.rating, 0.0)

        self.assertGreater(result["accuracy"], 0.5)
        self.assertGreaterEqual(result["accuracy_opt"], result["accuracy"])

    def test_massey_preserves_signed_ratings_through_benchmark_paths(self):
        """Massey's zero-mean scale must survive both public benchmark entry points."""
        result = evaluate_competitor(
            competitor_class=MasseyCompetitor,
            data_split=self.data_split,
            comparison_function=_always_true,
            optimize_thresholds=False,
        )

        ratings = [competitor.rating for competitor in result["arena"].competitors.values()]
        self.assertLess(min(ratings), 0.0)
        self.assertGreater(max(ratings), 0.0)
        self.assertEqual(len(result["history"].bouts), 60)
        self.assertEqual(result["confusion_matrix"], {"tp": 28, "fp": 1, "tn": 29, "fn": 2})
        self.assertEqual(result["accuracy"], 0.95)

        results = benchmark_competitors(
            [{"class": MasseyCompetitor, "name": "Massey", "params": {}}],
            self.data_split,
            _always_true,
            optimize_thresholds=False,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Massey")
        self.assertEqual(results[0]["accuracy"], result["accuracy"])

    def test_keener_can_be_compared_against_another_global_fit_system(self):
        """benchmark_competitors must accept Keener alongside its siblings."""
        results = benchmark_competitors(
            [
                {"class": KeenerCompetitor, "name": "Keener", "params": {}},
                {"class": ColleyMatrixCompetitor, "name": "Colley", "params": {}},
            ],
            self.data_split,
            _always_true,
            optimize_thresholds=False,
        )

        self.assertEqual([entry["name"] for entry in results], ["Keener", "Colley"])
        for entry in results:
            with self.subTest(system=entry["name"]):
                self.assertGreater(entry["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
