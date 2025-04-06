import abc
import random
from typing import Dict, Any, List, Tuple, Optional

from elote.logging import logger


class BaseArena:
    """Base abstract class for all arena implementations.

    Arenas manage competitions between multiple competitors, handling matchups,
    tournaments, and leaderboard generation. This class defines the interface
    that all arena implementations must follow.
    """

    @abc.abstractmethod
    def set_competitor_class_var(self, name: str, value: Any) -> None:
        """Set a class variable on all competitors in this arena.

        This method allows for global configuration of all competitors
        managed by this arena.

        Args:
            name (str): The name of the class variable to set.
            value: The value to set for the class variable.
        """
        pass

    @abc.abstractmethod
    def tournament(self, matchups: List[Tuple[Any, Any]]) -> None:
        """Run a tournament with the given matchups.

        A tournament consists of multiple matchups between competitors.

        Args:
            matchups (list): A list of matchup pairs to process.
        """
        pass

    @abc.abstractmethod
    def matchup(self, a: Any, b: Any) -> Any:
        """Process a single matchup between two competitors.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.

        Returns:
            The result of the matchup.
        """
        pass

    @abc.abstractmethod
    def leaderboard(self) -> List[Tuple[Any, float]]:
        """Generate a leaderboard of all competitors.

        Returns:
            list: A sorted list of competitors and their ratings.
        """
        pass

    @abc.abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this arena for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this arena's current state.
        """
        pass


class History:
    """Tracks the history of bouts (matchups) and provides analysis methods.

    This class stores the results of matchups and provides methods to analyze
    the performance of the rating system.
    """

    def __init__(self) -> None:
        """Initialize an empty history of bouts."""
        self.bouts: List[Bout] = []
        logger.debug("History initialized.")

    def add_bout(self, bout: "Bout") -> None:
        """Add a bout to the history.

        Args:
            bout (Bout): The bout object to add to the history.
        """
        self.bouts.append(bout)
        logger.debug("Added bout between %s and %s", bout.a, bout.b)

    def report_results(self, lower_threshold: float = 0.5, upper_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Generate a report of the results in this history.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            list: A list of dictionaries containing the results of each bout.
        """
        report = list()
        logger.info("Generating results report for %d bouts", len(self.bouts))
        for bout in self.bouts:
            report.append(
                {
                    "predicted_winnder": bout.predicted_winner(lower_threshold, upper_threshold),
                    "predicted_loser": bout.predicted_loser(lower_threshold, upper_threshold),
                    "probability": bout.predicted_outcome * 100,
                    "actual_winner": bout.actual_winner(),
                    "correct": bout.predicted_winner(lower_threshold, upper_threshold) == bout.actual_winner(),
                }
            )
        return report

    def confusion_matrix(self, lower_threshold: float = 0.45, upper_threshold: float = 0.55) -> Dict[str, int]:
        """
        Calculate the confusion matrix for the history of bouts.

        Args:
            lower_threshold: The lower threshold for prediction (below this is a prediction for the second competitor)
            upper_threshold: The upper threshold for prediction (above this is a prediction for the first competitor)

        Returns:
            A dictionary with confusion matrix metrics: {'tp': int, 'fp': int, 'tn': int, 'fn': int}
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        skipped_bouts = 0

        logger.info(
            "Calculating confusion matrix for %d bouts with thresholds: [%.2f, %.2f]",
            len(self.bouts),
            lower_threshold,
            upper_threshold,
        )

        for bout in self.bouts:
            # Extract the actual winner and predicted probability
            actual_winner = bout.actual_winner()
            predicted_prob = bout.predicted_outcome

            # Skip if we don't have both actual and predicted values
            if actual_winner is None or predicted_prob is None:
                skipped_bouts += 1
                continue

            # Convert predicted_prob to float if it's a string or None
            if isinstance(predicted_prob, str):
                try:
                    predicted_prob = float(predicted_prob)
                except ValueError:
                    # If conversion fails, skip this bout
                    logger.warning(
                        "Could not convert predicted_prob '%s' to float, skipping bout.", bout.predicted_outcome
                    )
                    skipped_bouts += 1
                    continue
            elif predicted_prob is None:
                skipped_bouts += 1
                continue

            # Determine the predicted outcome
            if predicted_prob > upper_threshold:
                predicted_winner = "a"
            elif predicted_prob < lower_threshold:
                predicted_winner = "b"
            else:
                predicted_winner = "draw"

            # Normalize actual winner to 'a', 'b', or 'draw'
            if isinstance(actual_winner, str):
                actual_winner = actual_winner.lower()
                if actual_winner in ["a", "win", "true", "1"]:
                    actual_winner = "a"
                elif actual_winner in ["b", "loss", "false", "0"]:
                    actual_winner = "b"
                else:
                    actual_winner = "draw"
            elif isinstance(actual_winner, (int, float)):
                if actual_winner == 1:
                    actual_winner = "a"
                elif actual_winner == 0:
                    actual_winner = "b"
                elif actual_winner == 0.5:
                    actual_winner = "draw"
                else:
                    # Skip if actual winner is not a recognized value
                    logger.debug("Unrecognized actual_winner value: %s, skipping bout.", bout.outcome)
                    skipped_bouts += 1
                    continue
            else:
                # Skip if actual winner is not a recognized type
                logger.debug("Unrecognized actual_winner type: %s, skipping bout.", type(bout.outcome))
                skipped_bouts += 1
                continue

            # Update confusion matrix
            if predicted_winner == "draw":
                if actual_winner == "draw":
                    true_positives += 1  # Correctly predicted draw
                else:
                    false_positives += 1  # Incorrectly predicted draw
            elif actual_winner == "draw":
                false_negatives += 1  # Failed to predict draw
            elif predicted_winner == "a":
                if actual_winner == "a":
                    true_positives += 1
                else:
                    false_positives += 1
            elif predicted_winner == "b":
                if actual_winner == "b":
                    true_negatives += 1
                else:
                    false_negatives += 1

        if skipped_bouts > 0:
            logger.info("Skipped %d bouts during confusion matrix calculation due to missing data.", skipped_bouts)

        # Return results as a dictionary
        return {"tp": true_positives, "fp": false_positives, "tn": true_negatives, "fn": false_negatives}

    def random_search(self, trials: int = 1000) -> Tuple[float, List[float]]:
        """Search for optimal prediction thresholds using random sampling.

        This method performs a random search to find the best lower and upper
        thresholds that maximize the overall accuracy, including draws.

        Args:
            trials (int): The number of random threshold pairs to try.

        Returns:
            tuple: A tuple containing (best_accuracy, best_thresholds).
        """
        best_accuracy = 0
        best_thresholds = [0.5, 0.5]  # Initialize with default values
        num_bouts = len(self.bouts)
        logger.info("Performing random search for optimal thresholds with %d trials on %d bouts.", trials, num_bouts)

        if num_bouts == 0:
            logger.warning("Cannot perform random search on empty history.")
            return best_accuracy, best_thresholds

        for i in range(trials):
            # Find min and max predicted outcomes in history
            predicted_outcomes = [bout.predicted_outcome for bout in self.bouts if bout.predicted_outcome is not None]
            min_outcome = min(predicted_outcomes) if predicted_outcomes else 0
            max_outcome = max(predicted_outcomes) if predicted_outcomes else 1

            # Generate two random numbers between min and max
            thresholds = sorted(
                [
                    min_outcome + random.random() * (max_outcome - min_outcome),
                    min_outcome + random.random() * (max_outcome - min_outcome),
                ]
            )

            # Calculate metrics with the random thresholds
            metrics = self.calculate_metrics(*thresholds)
            accuracy = metrics["accuracy"]

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = thresholds
                logger.debug(
                    "Random search trial %d: New best accuracy %.4f with thresholds [%.2f, %.2f]",
                    i + 1,
                    best_accuracy,
                    *best_thresholds,
                )

        logger.info(
            "Random search complete. Best accuracy: %.4f with thresholds [%.2f, %.2f]", best_accuracy, *best_thresholds
        )
        return best_accuracy, best_thresholds

    def optimize_thresholds(
        self, method: str = "L-BFGS-B", initial_thresholds: Tuple[float, float] = (0.5, 0.5)
    ) -> Tuple[float, List[float]]:
        """Optimize prediction thresholds using scipy.optimize.

        This method uses scipy's optimization algorithms to find the best thresholds
        for maximizing prediction accuracy.

        Args:
            method (str): The optimization method to use (e.g., 'L-BFGS-B', 'Nelder-Mead')
            initial_thresholds (tuple): Initial guess for (lower_threshold, upper_threshold)

        Returns:
            tuple: (best_accuracy, best_thresholds) where:
                - best_accuracy: The accuracy achieved with the optimized thresholds
                - best_thresholds: List of [lower_threshold, upper_threshold]
        """
        from scipy import optimize

        num_bouts = len(self.bouts)
        logger.info("Optimizing thresholds using scipy ('%s') on %d bouts.", method, num_bouts)

        if num_bouts == 0:
            logger.warning("Cannot optimize thresholds on empty history.")
            return 0.0, list(initial_thresholds)

        # Find min and max predicted outcomes in history
        predicted_outcomes = [bout.predicted_outcome for bout in self.bouts if bout.predicted_outcome is not None]
        if not predicted_outcomes:
            logger.warning("No valid predicted outcomes found in history. Cannot optimize.")
            return 0.0, list(initial_thresholds)
        min_outcome = min(predicted_outcomes)
        max_outcome = max(predicted_outcomes)
        logger.debug("Predicted outcome range: [%.4f, %.4f]", min_outcome, max_outcome)

        # Define the objective function to minimize (negative accuracy)
        def objective(thresholds: List[float]) -> float:
            # Ensure thresholds are sorted
            sorted_thresholds = sorted(thresholds)
            metrics = self.calculate_metrics(*sorted_thresholds)
            return -metrics["accuracy"]  # Negative because we want to maximize accuracy

        # Calculate baseline accuracy with initial thresholds
        baseline_metrics = self.calculate_metrics(*initial_thresholds)
        baseline_accuracy = baseline_metrics["accuracy"]
        logger.debug(
            "Baseline accuracy with initial thresholds [%.2f, %.2f]: %.4f", *initial_thresholds, baseline_accuracy
        )

        # Use initial_thresholds as the initial guess
        initial_guess = list(initial_thresholds)

        # Bounds for the thresholds
        bounds = [(min_outcome, max_outcome), (min_outcome, max_outcome)]

        # Run multiple optimizations with different methods and starting points
        best_accuracy = baseline_accuracy
        best_thresholds = list(initial_thresholds)

        # Try different optimization methods
        methods = [method]
        if method != "L-BFGS-B":
            methods.append("L-BFGS-B")
        if "Nelder-Mead" not in methods:
            methods.append("Nelder-Mead")

        for opt_method in methods:
            try:
                # Run the optimization with current method
                logger.debug("Running optimization with method: %s", opt_method)
                result = optimize.minimize(
                    objective,
                    initial_guess,
                    method=opt_method,
                    bounds=bounds if opt_method != "Nelder-Mead" else None,
                    options={"maxiter": 1000},
                )

                # Get the thresholds and ensure they're sorted
                opt_thresholds = sorted(result.x)

                # Calculate the accuracy
                metrics = self.calculate_metrics(*opt_thresholds)
                accuracy = metrics["accuracy"]

                # Update best if better than current best
                if accuracy > best_accuracy:
                    logger.debug(
                        "Optimization (%s): New best accuracy %.4f with thresholds [%.2f, %.2f]",
                        opt_method,
                        accuracy,
                        *opt_thresholds,
                    )
                    best_accuracy = accuracy
                    best_thresholds = opt_thresholds

                # Try with a few random starting points
                for j in range(3):
                    random_guess = [
                        random.uniform(min_outcome, max_outcome),
                        random.uniform(min_outcome, max_outcome),
                    ]
                    random_guess.sort()
                    result = optimize.minimize(
                        objective, random_guess, method=opt_method, bounds=bounds, options={"disp": False}
                    )
                    if result.success:
                        opt_thresholds = sorted(result.x)
                        metrics = self.calculate_metrics(*opt_thresholds)
                        accuracy = metrics["accuracy"]
                        if accuracy > best_accuracy:
                            logger.debug(
                                "Optimization (%s, random start %d): New best accuracy %.4f with thresholds [%.2f, %.2f]",
                                opt_method,
                                j + 1,
                                accuracy,
                                *opt_thresholds,
                            )
                            best_accuracy = accuracy
                            best_thresholds = opt_thresholds
            except Exception as e:
                # Skip if optimization fails
                logger.warning("Optimization with method '%s' failed: %s", opt_method, e)
                continue

        # If optimized accuracy is worse than baseline, use baseline
        if best_accuracy < baseline_accuracy:
            logger.info(
                "Optimized accuracy (%.4f) is worse than baseline (%.4f). Reverting to baseline thresholds.",
                best_accuracy,
                baseline_accuracy,
            )
            return baseline_accuracy, list(initial_thresholds)

        logger.info(
            "Optimization complete. Best accuracy: %.4f with thresholds [%.2f, %.2f]", best_accuracy, *best_thresholds
        )
        return best_accuracy, list(best_thresholds)

    def calculate_metrics(self, lower_threshold: float = 0.5, upper_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate performance metrics based on the confusion matrix.

        Args:
            lower_threshold: The lower threshold for prediction (below this is a prediction for the second competitor)
            upper_threshold: The upper threshold for prediction (above this is a prediction for the first competitor)

        Returns:
            A dictionary with metrics including accuracy, precision, recall, F1 score, and the confusion matrix
        """
        # Get the confusion matrix
        cm = self.confusion_matrix(lower_threshold, upper_threshold)
        logger.debug("Calculated confusion matrix: %s", cm)

        # Extract values from the confusion matrix
        tp = cm["tp"]
        fp = cm["fp"]
        tn = cm["tn"]
        fn = cm["fn"]

        # Calculate total predictions
        total = tp + fp + tn + fn

        # Calculate accuracy
        accuracy = (tp + tn) / total if total > 0 else 0

        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

        logger.debug(
            "Calculated metrics: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f", accuracy, precision, recall, f1
        )

        # Return all metrics
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}

    def calculate_metrics_with_draws(
        self, lower_threshold: float = 0.33, upper_threshold: float = 0.66
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics for the bout history, treating predictions
        between thresholds as explicit draw predictions.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1 score, and draw metrics.
        """
        # Count the different prediction outcomes
        total_bouts = 0
        correct_predictions = 0
        true_draw = 0
        false_draw = 0
        skipped_bouts = 0

        logger.info(
            "Calculating metrics with draws for %d bouts using thresholds [%.2f, %.2f]",
            len(self.bouts),
            lower_threshold,
            upper_threshold,
        )

        for bout in self.bouts:
            # Skip if we don't have both actual and predicted values
            if bout.actual_winner() is None or bout.predicted_outcome is None:
                skipped_bouts += 1
                continue

            total_bouts += 1

            # Determine predicted outcome
            if bout.predicted_outcome > upper_threshold:
                predicted = "a"
            elif bout.predicted_outcome < lower_threshold:
                predicted = "b"
            else:
                predicted = "draw"

            # Determine actual outcome
            actual = bout.actual_winner()
            if isinstance(actual, (int, float)):
                if actual == 1:
                    actual = "a"
                elif actual == 0:
                    actual = "b"
                elif actual == 0.5:
                    actual = "draw"
            elif isinstance(actual, str):
                actual = actual.lower()
                if actual in ["a", "win", "true", "1"]:
                    actual = "a"
                elif actual in ["b", "loss", "false", "0"]:
                    actual = "b"
                else:
                    actual = "draw"

            # Count correct predictions
            if predicted == actual:
                correct_predictions += 1
                if predicted == "draw":
                    true_draw += 1
            elif predicted == "draw":
                false_draw += 1

        if total_bouts == 0:
            logger.warning("No valid bouts found to calculate metrics with draws.")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "true_draw": 0,
                "false_draw": 0,
                "draw_rate": 0,
                "draw_accuracy": 0,
                "confusion_matrix": self.confusion_matrix(lower_threshold, upper_threshold),
            }

        # Calculate overall accuracy
        accuracy = correct_predictions / total_bouts if total_bouts > 0 else 0

        # Get standard metrics from confusion matrix
        cm = self.confusion_matrix(lower_threshold, upper_threshold)

        # Return combined metrics
        return {
            "accuracy": accuracy,
            "precision": cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0,
            "recall": cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0,
            "f1": 2 * cm["tp"] / (2 * cm["tp"] + cm["fp"] + cm["fn"])
            if (2 * cm["tp"] + cm["fp"] + cm["fn"]) > 0
            else 0,
            "true_draw": true_draw,
            "false_draw": false_draw,
            "draw_rate": (true_draw + false_draw) / total_bouts if total_bouts > 0 else 0,
            "draw_accuracy": true_draw / (true_draw + false_draw) if (true_draw + false_draw) > 0 else 0,
            "confusion_matrix": cm,
        }

    def accuracy_by_prior_bouts(
        self, arena: "BaseArena", thresholds: Optional[Tuple[float, float]] = None, bin_size: int = 5
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate accuracy based on the number of prior bouts for each competitor.

        This method analyzes how accuracy changes as competitors participate in more bouts,
        properly accounting for draws as a third outcome category.

        Args:
            arena (BaseArena): The arena containing the competitors and their history
            thresholds (tuple, optional): Tuple of (lower_threshold, upper_threshold) for predictions
            bin_size (int): Size of bins for grouping bout counts

        Returns:
            dict: A dictionary with 'binned' key containing binned accuracy data
        """
        # Default thresholds if not provided
        if thresholds is None:
            lower_threshold, upper_threshold = 0.5, 0.5
            logger.debug(
                "Using default thresholds for accuracy by prior bouts: [%.2f, %.2f]", lower_threshold, upper_threshold
            )
        else:
            lower_threshold, upper_threshold = thresholds
            logger.debug(
                "Using provided thresholds for accuracy by prior bouts: [%.2f, %.2f]", lower_threshold, upper_threshold
            )

        # Track the number of bouts for each competitor
        competitor_bout_counts: Dict[Any, int] = {}

        # Count all bouts from arena's history (which includes training data)
        if hasattr(arena, "history") and hasattr(arena.history, "bouts"):
            logger.debug("Populating initial bout counts from arena history (%d bouts)", len(arena.history.bouts))
            for bout in arena.history.bouts:
                competitor_bout_counts[bout.a] = competitor_bout_counts.get(bout.a, 0) + 1
                competitor_bout_counts[bout.b] = competitor_bout_counts.get(bout.b, 0) + 1
        else:
            logger.warning("Arena history not found or empty. Initial bout counts will be zero.")

        # Track accuracy by minimum bout count
        accuracy_by_min_bouts = {}
        logger.info("Calculating accuracy by prior bouts for %d evaluation bouts.", len(self.bouts))

        # Process each bout in the evaluation history
        for bout in self.bouts:
            # Get the current bout count for each competitor
            a_count = competitor_bout_counts.get(bout.a, 0)
            b_count = competitor_bout_counts.get(bout.b, 0)

            # Determine the minimum bout count between the two competitors
            min_bout_count = min(a_count, b_count)

            # Initialize the bucket if it doesn't exist
            if min_bout_count not in accuracy_by_min_bouts:
                accuracy_by_min_bouts[min_bout_count] = {"correct": 0, "total": 0}

            # Check if prediction is correct, properly handling draws
            is_predicted_draw = lower_threshold <= bout.predicted_outcome <= upper_threshold
            is_actual_draw = bout.outcome == 0.5

            if (
                (is_predicted_draw and is_actual_draw)
                or (bout.predicted_outcome > upper_threshold and bout.outcome == 1.0)
                or (bout.predicted_outcome < lower_threshold and bout.outcome == 0.0)
            ):
                accuracy_by_min_bouts[min_bout_count]["correct"] += 1

            accuracy_by_min_bouts[min_bout_count]["total"] += 1

            # Update bout counts for both competitors for subsequent bouts in evaluation
            competitor_bout_counts[bout.a] = a_count + 1
            competitor_bout_counts[bout.b] = b_count + 1

        # Calculate accuracy for each bucket
        for _bout_count, metrics in accuracy_by_min_bouts.items():
            if metrics["total"] > 0:
                metrics["accuracy"] = metrics["correct"] / metrics["total"]
            else:
                metrics["accuracy"] = None

        # Group data into bins for smoother visualization
        binned_data = {}
        for count, metrics in accuracy_by_min_bouts.items():
            bin_index = count // bin_size
            if bin_index not in binned_data:
                logger.debug(
                    "Creating new bin %d (counts %d-%d)",
                    bin_index,
                    bin_index * bin_size,
                    (bin_index + 1) * bin_size - 1,
                )
                binned_data[bin_index] = {"accuracy_sum": 0, "total": 0}

            if metrics["accuracy"] is not None:
                binned_data[bin_index]["accuracy_sum"] += metrics["accuracy"] * metrics["total"]
            binned_data[bin_index]["total"] += metrics["total"]

        # Calculate average accuracy for each bin
        for bin_idx, bin_data in binned_data.items():
            if bin_data["total"] > 10:
                bin_data["accuracy"] = bin_data["accuracy_sum"] / bin_data["total"]
                del bin_data["accuracy_sum"]
            else:
                bin_data["accuracy"] = None
                del bin_data["accuracy_sum"]

            # Add bin range information
            bin_data["min_bouts"] = bin_idx * bin_size
            bin_data["max_bouts"] = (bin_idx + 1) * bin_size - 1

        # Return only the binned data in the expected format
        logger.info("Completed accuracy by prior bouts calculation with bin size %d.", bin_size)
        return {"binned": binned_data}

    def get_calibration_data(self, n_bins: int = 10) -> Tuple[List[float], List[float]]:
        """Compute calibration data from the bout history.

        This method extracts predicted probabilities and actual outcomes from the bout history
        and prepares them for calibration curve plotting.

        Args:
            n_bins (int): Number of bins to use for calibration curve.

        Returns:
            tuple: (y_true, y_prob) where:
                - y_true: List of actual outcomes (1.0 for wins, 0.0 for losses)
                - y_prob: List of predicted probabilities
        """
        # Extract predicted probabilities and actual outcomes
        y_prob = []
        y_true = []
        skipped_bouts = 0
        logger.info("Extracting calibration data for %d bouts.", len(self.bouts))

        for bout in self.bouts:
            if bout.predicted_outcome is None or bout.outcome is None:
                skipped_bouts += 1
                continue
            y_prob.append(bout.predicted_outcome)
            # Convert outcomes to binary format (1.0 for wins, 0.0 for losses/draws)
            y_true.append(1.0 if bout.outcome == 1.0 else 0.0)

        if skipped_bouts > 0:
            logger.warning("Skipped %d bouts while extracting calibration data due to missing values.", skipped_bouts)

        logger.debug("Extracted %d valid data points for calibration.", len(y_true))
        return y_true, y_prob


class Bout:
    """A single bout between two competitors."""

    def __init__(
        self,
        a: Any,
        b: Any,
        predicted_outcome: Optional[float],
        outcome: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a bout.

        Args:
            a: The first competitor
            b: The second competitor
            predicted_outcome: The predicted probability of a winning
            outcome: The actual outcome of the bout
            attributes: Optional dictionary of additional attributes
        """
        self.a = a
        self.b = b
        self.predicted_outcome = predicted_outcome
        self.outcome = outcome
        self.attributes = attributes or {}

    def actual_winner(self) -> Optional[str]:
        """
        Return the actual winner of the bout based on the outcome.

        Returns:
            str or None: 'a' if a won, 'b' if b won, None if it was a draw or unclear
        """
        if isinstance(self.outcome, str):
            outcome_lower = self.outcome.lower()
            if outcome_lower in ["win", "won", "1", "a", "true", "t", "yes", "y"]:
                return "a"
            elif outcome_lower in ["loss", "lost", "0", "b", "false", "f", "no", "n"]:
                return "b"
            elif outcome_lower in ["draw", "tie", "tied", "draw", "0.5", "d", "equal", "eq"]:
                return None
        elif isinstance(self.outcome, (int, float)):
            if self.outcome == 1:
                return "a"
            elif self.outcome == 0:
                return "b"
            elif self.outcome == 0.5:
                return None

        return None

    def true_positive(self, threshold: float = 0.5) -> bool:
        """Check if this bout is a true positive prediction.

        A true positive occurs when the model correctly predicts a win.

        Args:
            threshold (float): The probability threshold for a positive prediction.

        Returns:
            bool: True if this bout is a true positive, False otherwise.
        """
        if self.predicted_outcome is None or self.outcome is None:
            return False
        if self.predicted_outcome > threshold:
            if isinstance(self.outcome, str):
                return bool(self.outcome == "win")
            else:
                return bool(self.outcome == 1.0)
        return False

    def false_positive(self, threshold: float = 0.5) -> bool:
        """Check if this bout is a false positive prediction.

        A false positive occurs when the model incorrectly predicts a win.

        Args:
            threshold (float): The probability threshold for a positive prediction.

        Returns:
            bool: True if this bout is a false positive, False otherwise.
        """
        if self.predicted_outcome is None or self.outcome is None:
            return False
        if self.predicted_outcome > threshold:
            if isinstance(self.outcome, str):
                return bool(self.outcome != "win")
            else:
                return bool(self.outcome != 1.0)
        return False

    def true_negative(self, threshold: float = 0.5) -> bool:
        """Check if this bout is a true negative prediction.

        A true negative occurs when the model correctly predicts a non-win.

        Args:
            threshold (float): The probability threshold for a negative prediction.

        Returns:
            bool: True if this bout is a true negative, False otherwise.
        """
        if self.predicted_outcome is None or self.outcome is None:
            return False
        if self.predicted_outcome <= threshold:
            if isinstance(self.outcome, str):
                return bool(self.outcome == "loss")
            else:
                return bool(self.outcome == 0.0)
        return False

    def false_negative(self, threshold: float = 0.5) -> bool:
        """Check if this bout is a false negative prediction.

        A false negative occurs when the model incorrectly predicts a non-win.

        Args:
            threshold (float): The probability threshold for a negative prediction.

        Returns:
            bool: True if this bout is a false negative, False otherwise.
        """
        if self.predicted_outcome is None or self.outcome is None:
            return False
        if self.predicted_outcome <= threshold:
            if isinstance(self.outcome, str):
                return bool(self.outcome != "loss")
            else:
                return bool(self.outcome != 0.0)
        return False

    def predicted_winner(self, lower_threshold: float = 0.5, upper_threshold: float = 0.5) -> Optional[str]:
        """Determine the predicted winner of this bout.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            str: The identifier of the predicted winner, or None if no winner is predicted.
        """
        if self.predicted_outcome > upper_threshold:
            return self.a.lower() if isinstance(self.a, str) else self.a
        elif self.predicted_outcome < lower_threshold:
            return self.b.lower() if isinstance(self.b, str) else self.b
        else:
            return None

    def predicted_loser(self, lower_threshold: float = 0.5, upper_threshold: float = 0.5) -> Optional[str]:
        """Determine the predicted loser of this bout.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            str: The identifier of the predicted loser, or None if no loser is predicted.
        """
        if self.predicted_outcome > upper_threshold:
            return self.b.lower() if isinstance(self.b, str) else self.b
        elif self.predicted_outcome < lower_threshold:
            return self.a.lower() if isinstance(self.a, str) else self.a
        else:
            return None
