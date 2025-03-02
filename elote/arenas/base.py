import abc
import random


class BaseArena:
    """Base abstract class for all arena implementations.

    Arenas manage competitions between multiple competitors, handling matchups,
    tournaments, and leaderboard generation. This class defines the interface
    that all arena implementations must follow.
    """

    @abc.abstractmethod
    def set_competitor_class_var(self, name, value):
        """Set a class variable on all competitors in this arena.

        This method allows for global configuration of all competitors
        managed by this arena.

        Args:
            name (str): The name of the class variable to set.
            value: The value to set for the class variable.
        """
        pass

    @abc.abstractmethod
    def tournament(self, matchups):
        """Run a tournament with the given matchups.

        A tournament consists of multiple matchups between competitors.

        Args:
            matchups (list): A list of matchup pairs to process.
        """
        pass

    @abc.abstractmethod
    def matchup(self, a, b):
        """Process a single matchup between two competitors.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.

        Returns:
            The result of the matchup.
        """
        pass

    @abc.abstractmethod
    def leaderboard(self):
        """Generate a leaderboard of all competitors.

        Returns:
            list: A sorted list of competitors and their ratings.
        """
        pass

    @abc.abstractmethod
    def export_state(self):
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

    def __init__(self):
        """Initialize an empty history of bouts."""
        self.bouts = []

    def add_bout(self, bout):
        """Add a bout to the history.

        Args:
            bout (Bout): The bout object to add to the history.
        """
        self.bouts.append(bout)

    def report_results(self, lower_threshold=0.5, upper_threshold=0.5):
        """Generate a report of the results in this history.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            list: A list of dictionaries containing the results of each bout.
        """
        report = list()
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

    def confusion_matrix(self, lower_threshold=0.5, upper_threshold=0.5, attribute_filter=None):
        """Calculate confusion matrix metrics for the bout history.

        This method calculates true positives, false positives, true negatives,
        false negatives, and correctly/incorrectly predicted draws based on the prediction thresholds.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.
            attribute_filter (dict, optional): Filter bouts by attributes.

        Returns:
            dict: A dictionary containing confusion matrix metrics including true/false draws.
        """
        tp, fp, tn, fn, true_draw, false_draw = 0, 0, 0, 0, 0, 0
        
        for bout in self.bouts:
            match = True
            if attribute_filter:
                for key, value in attribute_filter.items():
                    if bout.attributes.get(key) != value:
                        match = False
            
            if not match:
                continue
                
            # Check if the prediction is a draw (between thresholds)
            is_predicted_draw = lower_threshold <= bout.predicted_outcome <= upper_threshold
            
            # Check if the actual outcome is a draw (0.5 in numeric format)
            is_actual_draw = bout.outcome == 0.5
            
            if is_predicted_draw:
                # Predicted a draw
                if is_actual_draw:
                    true_draw += 1  # Correctly predicted a draw
                else:
                    false_draw += 1  # Incorrectly predicted a draw
            else:
                # Predicted a win for one side
                if bout.predicted_outcome > upper_threshold:
                    # Predicted a win for player A
                    if bout.outcome == 1.0:
                        tp += 1  # Correctly predicted A wins
                    elif bout.outcome == 0.0:
                        fp += 1  # Incorrectly predicted A wins (B actually won)
                    else:  # bout.outcome == 0.5
                        fp += 1  # Incorrectly predicted A wins (it was a draw)
                else:  # bout.predicted_outcome < lower_threshold
                    # Predicted a win for player B
                    if bout.outcome == 0.0:
                        tn += 1  # Correctly predicted B wins
                    elif bout.outcome == 1.0:
                        fn += 1  # Incorrectly predicted B wins (A actually won)
                    else:  # bout.outcome == 0.5
                        fn += 1  # Incorrectly predicted B wins (it was a draw)
        
        # Calculate total correct and total predictions
        total_correct = tp + tn + true_draw
        total = tp + fp + tn + fn + true_draw + false_draw
        
        # Calculate accuracy
        accuracy = total_correct / total if total > 0 else 0
        
        return {
            "tp": tp,  # True positives (correctly predicted A wins)
            "fp": fp,  # False positives (incorrectly predicted A wins)
            "tn": tn,  # True negatives (correctly predicted B wins)
            "fn": fn,  # False negatives (incorrectly predicted B wins)
            "true_draw": true_draw,  # Correctly predicted draws
            "false_draw": false_draw,  # Incorrectly predicted draws
            "accuracy": accuracy,  # Overall accuracy including draws
            "total_correct": total_correct,  # Total correct predictions
            "total": total  # Total predictions
        }

    def random_search(self, trials=1000):
        """Search for optimal prediction thresholds using random sampling.

        This method performs a random search to find the best lower and upper
        thresholds that maximize the overall accuracy, including draws.

        Args:
            trials (int): The number of random threshold pairs to try.

        Returns:
            tuple: A tuple containing (best_accuracy, best_thresholds).
        """
        best_accuracy, best_thresholds = 0, list()
        for _ in range(trials):
            # Find min and max predicted outcomes in history
            predicted_outcomes = [bout.predicted_outcome for bout in self.bouts]
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

        return best_accuracy, best_thresholds

    def optimize_thresholds(self, method="L-BFGS-B", initial_thresholds=(0.5, 0.5)):
        """Search for optimal prediction thresholds using scipy optimization.

        This method uses scipy's optimize module to find the best lower and upper
        thresholds that maximize the overall accuracy, including draws.

        Args:
            method (str): The optimization method to use (default: 'L-BFGS-B').
            initial_thresholds (tuple): Initial guess for (lower, upper) thresholds.

        Returns:
            tuple: A tuple containing (best_accuracy, best_thresholds).
        """
        from scipy import optimize

        # Find min and max predicted outcomes in history
        predicted_outcomes = [bout.predicted_outcome for bout in self.bouts]
        min_outcome = min(predicted_outcomes) if predicted_outcomes else 0
        max_outcome = max(predicted_outcomes) if predicted_outcomes else 1

        # Define the objective function to minimize (negative accuracy)
        def objective(thresholds):
            # Ensure thresholds are sorted
            sorted_thresholds = sorted(thresholds)
            metrics = self.calculate_metrics(*sorted_thresholds)
            return -metrics["accuracy"]  # Negative because we want to maximize accuracy

        # Calculate baseline accuracy with initial thresholds
        baseline_metrics = self.calculate_metrics(*initial_thresholds)
        baseline_accuracy = baseline_metrics["accuracy"]

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
                    best_accuracy = accuracy
                    best_thresholds = opt_thresholds
                    
                # Try with a few random starting points
                for _ in range(3):
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
                            best_accuracy = accuracy
                            best_thresholds = opt_thresholds
            except Exception:
                # Skip if optimization fails
                continue

        # If optimized accuracy is worse than baseline, use baseline
        if best_accuracy < baseline_accuracy:
            return baseline_accuracy, initial_thresholds

        return best_accuracy, best_thresholds

    def calculate_metrics(self, lower_threshold=0.5, upper_threshold=0.5):
        """Calculate evaluation metrics based on the bout history.

        This method calculates accuracy, precision, recall, and F1 score based on
        the confusion matrix. It now properly handles draws as a third outcome category.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        cm = self.confusion_matrix(lower_threshold, upper_threshold)
        
        # Extract values from confusion matrix
        tp = cm["tp"]
        fp = cm["fp"]
        tn = cm["tn"]
        fn = cm["fn"]
        true_draw = cm["true_draw"]
        false_draw = cm["false_draw"]
        
        # Calculate metrics for wins/losses
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate draw metrics if applicable
        draw_precision = true_draw / (true_draw + false_draw) if (true_draw + false_draw) > 0 else 0
        
        # Calculate overall accuracy including draws
        total_correct = tp + tn + true_draw
        total_predictions = tp + fp + tn + fn + true_draw + false_draw
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "draw_precision": draw_precision,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "true_draws": true_draw,
            "false_draws": false_draw
        }

    def calculate_metrics_with_draws(self, lower_threshold=0.33, upper_threshold=0.66):
        """Calculate evaluation metrics for the bout history, treating predictions
        between thresholds as explicit draw predictions.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1 score, and draw metrics.
        """
        cm = self.confusion_matrix(lower_threshold, upper_threshold)
        
        # Extract values from confusion matrix
        tp = cm["tp"]
        fp = cm["fp"]
        tn = cm["tn"]
        fn = cm["fn"]
        true_draw = cm["true_draw"]
        false_draw = cm["false_draw"]
        
        # Calculate precision, recall, and F1 for wins (not including draws)
        precision_wins = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_wins = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_wins = 2 * precision_wins * recall_wins / (precision_wins + recall_wins) if (precision_wins + recall_wins) > 0 else 0
        
        # Calculate precision, recall, and F1 for draws
        precision_draws = true_draw / (true_draw + false_draw) if (true_draw + false_draw) > 0 else 0
        recall_draws = true_draw / (true_draw + fp + fn) if (true_draw + fp + fn) > 0 else 0
        f1_draws = 2 * precision_draws * recall_draws / (precision_draws + recall_draws) if (precision_draws + recall_draws) > 0 else 0
        
        # Calculate overall metrics
        total = tp + fp + tn + fn + true_draw + false_draw
        accuracy = (tp + tn + true_draw) / total if total > 0 else 0
        
        # Calculate macro-averaged precision, recall, and F1
        precision_macro = (precision_wins + precision_draws) / 2
        recall_macro = (recall_wins + recall_draws) / 2
        f1_macro = (f1_wins + f1_draws) / 2
        
        return {
            "accuracy": accuracy,
            "precision": precision_wins,  # For backward compatibility
            "recall": recall_wins,  # For backward compatibility
            "f1": f1_wins,  # For backward compatibility
            "precision_wins": precision_wins,
            "recall_wins": recall_wins,
            "f1_wins": f1_wins,
            "precision_draws": precision_draws,
            "recall_draws": recall_draws,
            "f1_draws": f1_draws,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
        }

    def accuracy_by_prior_bouts(self, arena, thresholds=None, bin_size=5):
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
        else:
            lower_threshold, upper_threshold = thresholds

        # Track the number of bouts for each competitor
        competitor_bout_counts = {}

        # Count all bouts from arena's history (which includes training data)
        if hasattr(arena, "history") and hasattr(arena.history, "bouts"):
            for bout in arena.history.bouts:
                competitor_bout_counts[bout.a] = competitor_bout_counts.get(bout.a, 0) + 1
                competitor_bout_counts[bout.b] = competitor_bout_counts.get(bout.b, 0) + 1

        # Track accuracy by minimum bout count
        accuracy_by_min_bouts = {}

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
            
            if (is_predicted_draw and is_actual_draw) or \
               (bout.predicted_outcome > upper_threshold and bout.outcome == 1.0) or \
               (bout.predicted_outcome < lower_threshold and bout.outcome == 0.0):
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
                binned_data[bin_index] = {"accuracy_sum": 0, "total": 0}

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
        return {"binned": binned_data}
        
    def get_calibration_data(self, n_bins=10):
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
        y_prob = [bout.predicted_outcome for bout in self.bouts]
        
        # Convert outcomes to binary format (1.0 for wins, 0.0 for losses)
        # Note: For calibration curves, we treat draws (0.5) as losses (0.0)
        # since we're evaluating the calibration of the predicted probability
        # that player A wins.
        y_true = [1.0 if bout.outcome == 1.0 else 0.0 for bout in self.bouts]
        
        return y_true, y_prob


class Bout:
    """Represents a single matchup (bout) between two competitors.

    This class stores the competitors, the predicted outcome, the actual outcome,
    and any additional attributes of the bout.
    """

    def __init__(self, a, b, predicted_outcome, outcome, attributes=None):
        """Initialize a bout between two competitors.

        Args:
            a: The first competitor.
            b: The second competitor.
            predicted_outcome (float): The predicted probability of a winning.
            outcome (str or float): The actual outcome ("win", "loss", "draw" or 1.0, 0.0, 0.5).
            attributes (dict, optional): Additional attributes of this bout.
        """
        self.a = a
        self.b = b
        self.predicted_outcome = predicted_outcome
        self.outcome = outcome
        self.attributes = attributes or dict()

    def true_positive(self, threshold=0.5):
        """Check if this bout is a true positive prediction.

        A true positive occurs when the model correctly predicts a win.

        Args:
            threshold (float): The probability threshold for a positive prediction.

        Returns:
            bool: True if this bout is a true positive, False otherwise.
        """
        if self.predicted_outcome > threshold:
            if isinstance(self.outcome, str):
                return self.outcome == "win"
            else:
                return self.outcome == 1.0
        return False

    def false_positive(self, threshold=0.5):
        """Check if this bout is a false positive prediction.

        A false positive occurs when the model incorrectly predicts a win.

        Args:
            threshold (float): The probability threshold for a positive prediction.

        Returns:
            bool: True if this bout is a false positive, False otherwise.
        """
        if self.predicted_outcome > threshold:
            if isinstance(self.outcome, str):
                return self.outcome != "win"
            else:
                return self.outcome != 1.0
        return False

    def true_negative(self, threshold=0.5):
        """Check if this bout is a true negative prediction.

        A true negative occurs when the model correctly predicts a non-win.

        Args:
            threshold (float): The probability threshold for a negative prediction.

        Returns:
            bool: True if this bout is a true negative, False otherwise.
        """
        if self.predicted_outcome <= threshold:
            if isinstance(self.outcome, str):
                return self.outcome == "loss"
            else:
                return self.outcome == 0.0
        return False

    def false_negative(self, threshold=0.5):
        """Check if this bout is a false negative prediction.

        A false negative occurs when the model incorrectly predicts a non-win.

        Args:
            threshold (float): The probability threshold for a negative prediction.

        Returns:
            bool: True if this bout is a false negative, False otherwise.
        """
        if self.predicted_outcome <= threshold:
            if isinstance(self.outcome, str):
                return self.outcome != "loss"
            else:
                return self.outcome != 0.0
        return False

    def predicted_winner(self, lower_threshold=0.5, upper_threshold=0.5):
        """Determine the predicted winner of this bout.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            str: The identifier of the predicted winner, or None if no winner is predicted.
        """
        if self.predicted_outcome > upper_threshold:
            return self.a
        elif self.predicted_outcome < lower_threshold:
            return self.b
        else:
            return None

    def predicted_loser(self, lower_threshold=0.5, upper_threshold=0.5):
        """Determine the predicted loser of this bout.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.

        Returns:
            str: The identifier of the predicted loser, or None if no loser is predicted.
        """
        if self.predicted_outcome > upper_threshold:
            return self.b
        elif self.predicted_outcome < lower_threshold:
            return self.a
        else:
            return None

    def actual_winner(self):
        """Determine the actual winner of this bout.

        Returns:
            str: The identifier of the actual winner, or None if no winner is determined.
        """
        if isinstance(self.outcome, str):
            if self.outcome == "win":
                return self.a
            elif self.outcome == "loss":
                return self.b
            else:
                return None
        else:  # numeric outcome
            if self.outcome == 1.0:
                return self.a
            elif self.outcome == 0.0:
                return self.b
            else:
                return None
