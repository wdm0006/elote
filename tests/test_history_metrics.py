import random
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

    def test_calculate_metrics_includes_draws(self):
        history = History()
        history.bouts = [
            Bout("a", "b", 0.8, "win"),
            Bout("a", "b", 0.5, 0.5),
            Bout("a", "b", 0.5, "tie"),
            Bout("a", "b", 0.5, "draw"),
            Bout("a", "b", 0.8, "loss"),
            Bout("a", "b", 0.5, "win"),
            Bout("a", "b", 0.2, "loss"),
            Bout("a", "b", 0.2, "win"),
            Bout("a", "b", 0.8, 0.5),
            Bout("a", "b", 0.2, "tie"),
            Bout("a", "b", 0.8, "draw"),
            Bout("a", "b", None, "draw"),
            Bout("a", "b", 0.5, None),
            Bout("a", "b", 0.5, "unknown"),
        ]

        metrics = history.calculate_metrics(lower_threshold=0.4, upper_threshold=0.6)

        self.assertEqual(metrics["confusion_matrix"], {"tp": 4, "fp": 2, "tn": 1, "fn": 4})
        self.assertAlmostEqual(metrics["accuracy"], 5 / 11)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)

    def test_optimize_thresholds_evaluates_draws(self):
        history = History()
        history.add_bout(Bout("a", "b", 0.5, "draw"))

        accuracy, thresholds = history.optimize_thresholds(initial_thresholds=(0.5, 0.5))

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(thresholds, [0.5, 0.5])

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


def _brute_force_best(history):
    """Exhaustively evaluate every candidate threshold pair via calculate_metrics.

    Returns the best accuracy reachable by any (lower, upper) pair.
    """
    probabilities = sorted({bout.predicted_outcome for bout in history.bouts if bout.predicted_outcome is not None})
    candidates = (
        [probabilities[0] - 0.1]
        + [(probabilities[k - 1] + probabilities[k]) / 2 for k in range(1, len(probabilities))]
        + [probabilities[-1] + 0.1]
    )
    return max(
        history.calculate_metrics(lower, upper)["accuracy"]
        for lower in candidates
        for upper in candidates
        if lower <= upper
    )


def _mixed_history():
    """A history whose optimal thresholds are neither (0.5, 0.5) nor a pure split."""
    history = History()
    for probability, outcome in [
        (0.08, "loss"),
        (0.17, "loss"),
        (0.24, "win"),
        (0.31, "loss"),
        (0.42, "draw"),
        (0.46, "loss"),
        (0.53, "draw"),
        (0.57, "draw"),
        (0.61, "win"),
        (0.68, "draw"),
        (0.74, "win"),
        (0.79, "loss"),
        (0.86, "win"),
        (0.93, "win"),
    ]:
        history.add_bout(Bout("a", "b", probability, outcome))
    return history


def _narrow_band_history():
    """A history whose only perfect thresholds are a narrow band far from (0.5, 0.5).

    No bout has a probability between 0.34 and 0.62, so accuracy is constant in a wide
    neighbourhood of the default starting point. The single perfect answer requires the
    lower threshold in (0.28, 0.30) and the upper in (0.32, 0.34) -- two 2%-wide
    windows, which only an exact sweep locates reliably.
    """
    history = History()
    bouts = (
        [(round(0.02 + 0.04 * step, 2), "loss") for step in range(7)]  # 0.02 .. 0.26
        + [(0.28, "loss")]
        + [(0.30, "draw")] * 8
        + [(0.32, "draw")] * 8
        + [(0.34, "win")]
        + [(round(0.62 + 0.04 * step, 2), "win") for step in range(10)]  # 0.62 .. 0.98
    )
    for probability, outcome in bouts:
        history.add_bout(Bout("a", "b", probability, outcome))
        if outcome != "draw":
            history.add_bout(Bout("a", "b", probability, outcome))
    return history


class TestOptimizeThresholds(unittest.TestCase):
    def test_optimize_thresholds_is_deterministic(self):
        """Repeated calls on one unchanged history return identical results."""
        for name, history in [("mixed", _mixed_history()), ("narrow_band", _narrow_band_history())]:
            with self.subTest(history=name):
                results = [history.optimize_thresholds() for _ in range(5)]

                for result in results[1:]:
                    self.assertEqual(result, results[0])

    def test_optimize_thresholds_matches_brute_force(self):
        """The returned accuracy equals the exhaustively computed optimum."""
        for name, history in [("mixed", _mixed_history()), ("narrow_band", _narrow_band_history())]:
            with self.subTest(history=name):
                accuracy, thresholds = history.optimize_thresholds()

                self.assertEqual(accuracy, _brute_force_best(history))
                self.assertEqual(accuracy, history.calculate_metrics(*thresholds)["accuracy"])

    def test_optimize_thresholds_finds_draw_band(self):
        """A history whose optimum is a draw band returns lower < upper."""
        history = _narrow_band_history()

        accuracy, thresholds = history.optimize_thresholds()

        # Only a band with 0.28 < lower < 0.30 and 0.32 < upper < 0.34 classifies
        # every bout correctly; a decisive split tops out below 1.0.
        self.assertEqual(accuracy, 1.0)
        self.assertLess(thresholds[0], thresholds[1])
        self.assertTrue(0.28 < thresholds[0] < 0.30, thresholds)
        self.assertTrue(0.32 < thresholds[1] < 0.34, thresholds)
        self.assertLess(history.calculate_metrics(0.5, 0.5)["accuracy"], accuracy)

    def test_optimize_thresholds_never_worse_than_initial(self):
        """The optimum is never below the accuracy of the supplied thresholds."""
        history = _mixed_history()

        for initial in [(0.5, 0.5), (0.2, 0.8), (0.0, 1.0), (0.45, 0.46)]:
            with self.subTest(initial_thresholds=initial):
                baseline = history.calculate_metrics(*initial)["accuracy"]
                accuracy, thresholds = history.optimize_thresholds(initial_thresholds=initial)

                self.assertGreaterEqual(accuracy, baseline)
                self.assertEqual(accuracy, history.calculate_metrics(*thresholds)["accuracy"])

    def test_optimize_thresholds_handles_decisive_history(self):
        """A history with no draws is optimized by a single split point."""
        history = History()
        for probability, outcome in [(0.1, "loss"), (0.2, "loss"), (0.7, "win"), (0.9, "win")]:
            history.add_bout(Bout("a", "b", probability, outcome))

        accuracy, thresholds = history.optimize_thresholds()

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(accuracy, history.calculate_metrics(*thresholds)["accuracy"])

    def test_optimize_thresholds_ignores_method_argument(self):
        """The deprecated method argument no longer changes the answer."""
        history = _mixed_history()

        self.assertEqual(history.optimize_thresholds(method="Nelder-Mead"), history.optimize_thresholds())

    def test_random_search_is_reproducible_with_seed(self):
        """random_search returns identical results for a fixed seed."""
        history = _mixed_history()

        first = history.random_search(trials=50, seed=7)
        second = history.random_search(trials=50, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first[1][0], first[1][1])


def _repeated_history(outcome, predicted_outcome=0.9, count=20):
    """Build a history of identical bouts, so every bin lands in one bucket."""
    history = History()
    for _ in range(count):
        history.add_bout(Bout("a", "b", predicted_outcome, outcome))
    return history


def _arena_with_online_eval_history(training_matchups=200, eval_matchups=160, seed=11):
    """Train an arena, then record further ``matchup`` calls as an online evaluation history.

    ``LambdaArena.matchup`` records the strings ``"win"``/``"loss"``/``"tie"`` as the bout
    outcome, which is the path the float-only dataset helpers never exercise.
    """
    rng = random.Random(seed)
    strengths = {"c%d" % i: rng.gauss(0, 200) for i in range(12)}

    def comparison(a, b, attributes=None):
        return strengths[a] + rng.gauss(0, 100) > strengths[b] + rng.gauss(0, 100)

    arena = LambdaArena(comparison, base_competitor=EloCompetitor)
    ids = list(strengths)
    for _ in range(training_matchups):
        a, b = rng.sample(ids, 2)
        arena.matchup(a, b)

    training_history = arena.history
    arena.history = History()
    for _ in range(eval_matchups):
        a, b = rng.sample(ids, 2)
        arena.matchup(a, b)
    eval_history = arena.history
    arena.history = training_history

    return arena, eval_history


class TestAccuracyByPriorBouts(unittest.TestCase):
    """``accuracy_by_prior_bouts`` must read outcomes the way every other metric does."""

    def test_string_outcomes_score_identically_to_float_outcomes(self):
        """A history of arena-recorded "win" bouts reports the same accuracy as 1.0 bouts."""
        arena = MockArena()

        string_binned = _repeated_history("win").accuracy_by_prior_bouts(arena, bin_size=100)["binned"]
        float_binned = _repeated_history(1.0).accuracy_by_prior_bouts(arena, bin_size=100)["binned"]

        self.assertEqual(float_binned[0]["accuracy"], 1.0)
        self.assertEqual(string_binned[0]["accuracy"], 1.0)
        self.assertEqual(string_binned, float_binned)

    def test_string_losses_score_identically_to_float_outcomes(self):
        """The "loss" spelling must be scored against the lower threshold, not dropped."""
        arena = MockArena()

        string_binned = _repeated_history("loss", predicted_outcome=0.1).accuracy_by_prior_bouts(arena, bin_size=100)
        float_binned = _repeated_history(0.0, predicted_outcome=0.1).accuracy_by_prior_bouts(arena, bin_size=100)

        self.assertEqual(string_binned["binned"][0]["accuracy"], 1.0)
        self.assertEqual(string_binned, float_binned)

    def test_tie_outcomes_are_counted_inside_the_draw_band(self):
        """Drawn bouts recorded as "tie" count as correct when the prediction is in the band."""
        arena = MockArena()
        history = History()
        for _ in range(12):
            history.add_bout(Bout("a", "b", 0.5, "tie"))
        for _ in range(8):
            history.add_bout(Bout("a", "b", 0.5, "win"))

        binned = history.accuracy_by_prior_bouts(arena, thresholds=(0.4, 0.6), bin_size=100)["binned"]

        self.assertEqual(binned[0]["total"], 20)
        self.assertAlmostEqual(binned[0]["accuracy"], 12 / 20)

    def test_unrecognized_outcomes_are_skipped_not_counted_wrong(self):
        """A bout whose outcome does not normalize is skipped rather than scored as incorrect."""
        arena = MockArena()
        history = _repeated_history("win")
        for _ in range(5):
            history.add_bout(Bout("a", "b", 0.9, "nonsense"))

        binned = history.accuracy_by_prior_bouts(arena, bin_size=100)["binned"]

        self.assertEqual(binned[0]["total"], 20)
        self.assertEqual(binned[0]["accuracy"], 1.0)

    def test_arena_driven_bins_agree_with_calculate_metrics(self):
        """The bout-weighted mean of the bins equals the accuracy of the same history."""
        arena, eval_history = _arena_with_online_eval_history()

        # Every bout carries an arena string outcome, so the pre-fix code scored zero of them.
        self.assertTrue(all(isinstance(bout.outcome, str) for bout in eval_history.bouts))

        expected = eval_history.calculate_metrics()["accuracy"]
        self.assertGreater(expected, 0.5)

        binned = eval_history.accuracy_by_prior_bouts(arena, bin_size=25)["binned"]
        self.assertGreater(len(binned), 1)
        self.assertTrue(all(bin_data["accuracy"] is not None for bin_data in binned.values()))

        total = sum(bin_data["total"] for bin_data in binned.values())
        weighted = sum(bin_data["accuracy"] * bin_data["total"] for bin_data in binned.values())

        self.assertEqual(total, len(eval_history.bouts))
        self.assertAlmostEqual(weighted / total, expected)


class _ExplodingArena:
    """An arena whose history must never be read, so eager validation is provable."""

    @property
    def history(self):
        raise AssertionError("bin_size must be validated before any bout is processed")


def _history_with_leading_misses(count, misses):
    """``count`` bouts between one pair, the first ``misses`` of them predicted wrong."""
    history = History()
    for index in range(count):
        history.add_bout(Bout("a", "b", 0.9, "loss" if index < misses else "win"))
    return history


class TestAccuracyByPriorBoutsSmallBins(unittest.TestCase):
    """Every non-empty bin reports its observed accuracy, however few bouts it holds."""

    def test_single_bout_bin_reports_its_observed_accuracy(self):
        """One bout is enough to report ``correct / total``."""
        arena = MockArena()

        hit = _history_with_leading_misses(1, 0).accuracy_by_prior_bouts(arena, bin_size=100)["binned"]
        miss = _history_with_leading_misses(1, 1).accuracy_by_prior_bouts(arena, bin_size=100)["binned"]

        self.assertEqual(hit[0]["total"], 1)
        self.assertEqual(hit[0]["accuracy"], 1.0)
        self.assertEqual(miss[0]["total"], 1)
        self.assertEqual(miss[0]["accuracy"], 0.0)

    def test_accuracy_is_continuous_across_the_former_eleven_bout_floor(self):
        """Bins of 1 through 11 bouts all report the exact observed ratio.

        The old code returned ``None`` for every total of 10 or fewer, so a bin jumped
        from missing to rated on the eleventh bout.
        """
        arena = MockArena()

        for count in range(1, 12):
            with self.subTest(count=count):
                binned = _history_with_leading_misses(count, 1).accuracy_by_prior_bouts(arena, bin_size=100)["binned"]

                self.assertEqual(binned[0]["total"], count)
                self.assertAlmostEqual(binned[0]["accuracy"], (count - 1) / count)

    def test_weighted_bins_aggregate_across_prior_bout_counts(self):
        """A bin spanning several prior-bout counts reports their combined ratio."""
        arena = MockArena()

        # Ten bouts between one pair, so each prior-bout count 0-9 holds exactly one bout.
        # bin_size=5 groups counts 0-4 and 5-9 into two five-bout bins.
        binned = _history_with_leading_misses(10, 2).accuracy_by_prior_bouts(arena, bin_size=5)["binned"]

        self.assertEqual(sorted(binned), [0, 1])
        self.assertEqual((binned[0]["min_bouts"], binned[0]["max_bouts"]), (0, 4))
        self.assertEqual((binned[1]["min_bouts"], binned[1]["max_bouts"]), (5, 9))
        self.assertEqual(binned[0]["total"], 5)
        self.assertEqual(binned[1]["total"], 5)
        self.assertAlmostEqual(binned[0]["accuracy"], 3 / 5)
        self.assertAlmostEqual(binned[1]["accuracy"], 1.0)

        total = sum(bin_data["total"] for bin_data in binned.values())
        weighted = sum(bin_data["accuracy"] * bin_data["total"] for bin_data in binned.values())
        self.assertAlmostEqual(weighted / total, 8 / 10)

    def test_bins_weight_prior_bout_counts_by_their_bout_totals(self):
        """A bin is the combined ``correct / total``, not the mean of its counts' accuracies.

        The four fresh pairs land on prior-bout count 0 and the repeat lands on count 1,
        so an unweighted mean over the two counts would report 0.625 instead of 0.4.
        """
        arena = MockArena()
        history = History()
        history.add_bout(Bout("p0", "q0", 0.9, "win"))
        for index in range(1, 4):
            history.add_bout(Bout("p%d" % index, "q%d" % index, 0.9, "loss"))
        history.add_bout(Bout("p0", "q0", 0.9, "win"))

        binned = history.accuracy_by_prior_bouts(arena, bin_size=5)["binned"]

        self.assertEqual(sorted(binned), [0])
        self.assertEqual(binned[0]["total"], 5)
        self.assertAlmostEqual(binned[0]["accuracy"], 2 / 5)

    def test_invalid_bin_sizes_are_rejected_before_any_bout_is_read(self):
        """Zero, negative, non-integral and boolean bin sizes raise eagerly."""
        history = _history_with_leading_misses(4, 1)

        for bin_size in (0, -1, -5, 2.5, 5.0, True, False, "5", None):
            with self.subTest(bin_size=bin_size):
                with self.assertRaises(ValueError) as caught:
                    history.accuracy_by_prior_bouts(_ExplodingArena(), bin_size=bin_size)

                self.assertIn("bin_size must be a positive integer", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
