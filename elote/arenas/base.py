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
        and false negatives based on the prediction thresholds.

        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.
            attribute_filter (dict, optional): Filter bouts by attributes.

        Returns:
            tuple: A tuple containing (true_positives, false_positives, true_negatives,
                  false_negatives, do_nothing_count).
        """
        tp, fp, tn, fn, do_nothing = 0, 0, 0, 0, 0
        for bout in self.bouts:
            match = True
            if attribute_filter:
                for key, value in attribute_filter.items():
                    if bout.attributes.get(key) != value:
                        match = False
            if match:
                if upper_threshold > bout.predicted_outcome > lower_threshold:
                    do_nothing += 1
                else:
                    if bout.true_positive(upper_threshold):
                        tp += 1
                    if bout.false_positive(upper_threshold):
                        fp += 1
                    if bout.true_negative(lower_threshold):
                        tn += 1
                    if bout.false_negative(lower_threshold):
                        fn += 1

        return tp, fp, tn, fn, do_nothing

    def random_search(self, trials=1000):
        """Search for optimal prediction thresholds using random sampling.

        This method performs a random search to find the best lower and upper
        thresholds that maximize the overall accuracy.

        Args:
            trials (int): The number of random threshold pairs to try.

        Returns:
            tuple: A tuple containing (best_net_performance, best_thresholds).
        """
        best_accuracy, best_thresholds = 0, list()
        for _ in range(trials):
            # Find min and max predicted outcomes in history
            predicted_outcomes = [bout.predicted_outcome for bout in self.bouts]
            min_outcome = min(predicted_outcomes) if predicted_outcomes else 0
            max_outcome = max(predicted_outcomes) if predicted_outcomes else 1
            
            # Generate two random numbers between min and max
            thresholds = sorted([
                min_outcome + random.random() * (max_outcome - min_outcome),
                min_outcome + random.random() * (max_outcome - min_outcome)
            ])
            tp, fp, tn, fn, do_nothing = self.confusion_matrix(*thresholds)
            total = tp + fp + tn + fn + do_nothing
            accuracy = (tp + tn) / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = thresholds

        return best_accuracy, best_thresholds
    
    def optimize_thresholds(self, method='L-BFGS-B'):
        """Search for optimal prediction thresholds using scipy optimization.

        This method uses scipy's optimize module to find the best lower and upper
        thresholds that maximize the overall accuracy.

        Args:
            method (str): The optimization method to use (default: 'L-BFGS-B').

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
            tp, fp, tn, fn, do_nothing = self.confusion_matrix(*sorted_thresholds)
            total = tp + fp + tn + fn + do_nothing
            accuracy = (tp + tn) / total if total > 0 else 0
            return -accuracy  # Negative because we want to maximize accuracy
        
        # Calculate baseline accuracy with (0.5, 0.5) thresholds
        baseline_tp, baseline_fp, baseline_tn, baseline_fn, baseline_do_nothing = self.confusion_matrix(0.5, 0.5)
        baseline_total = baseline_tp + baseline_fp + baseline_tn + baseline_fn + baseline_do_nothing
        baseline_accuracy = (baseline_tp + baseline_tn) / baseline_total if baseline_total > 0 else 0
        
        # Use (0.5, 0.5) as the initial guess
        initial_guess = [0.5, 0.5]
        
        # Bounds for the thresholds
        bounds = [(min_outcome, max_outcome), (min_outcome, max_outcome)]
        
        # Run multiple optimizations with different methods and starting points
        best_accuracy = baseline_accuracy
        best_thresholds = [0.5, 0.5]
        
        # Try different optimization methods
        methods = [method]
        if method != 'L-BFGS-B':
            methods.append('L-BFGS-B')
        if 'Nelder-Mead' not in methods:
            methods.append('Nelder-Mead')
        
        for opt_method in methods:
            try:
                # Run the optimization with current method
                result = optimize.minimize(
                    objective, 
                    initial_guess, 
                    method=opt_method,
                    bounds=bounds if opt_method != 'Nelder-Mead' else None,
                    options={'maxiter': 1000}
                )
                
                # Get the thresholds and ensure they're sorted
                opt_thresholds = sorted(result.x)
                
                # Calculate the accuracy
                tp, fp, tn, fn, do_nothing = self.confusion_matrix(*opt_thresholds)
                total = tp + fp + tn + fn + do_nothing
                accuracy = (tp + tn) / total if total > 0 else 0
                
                # Update best if better than current best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = opt_thresholds
            except Exception:
                # Skip if optimization fails
                continue
        
        # If optimized accuracy is worse than baseline, use baseline
        if best_accuracy < baseline_accuracy:
            return baseline_accuracy, [0.5, 0.5]
        
        return best_accuracy, best_thresholds
    
    def calculate_metrics(self, lower_threshold=0.5, upper_threshold=0.5):
        """Calculate common evaluation metrics for the bout history.
        
        Args:
            lower_threshold (float): The lower probability threshold for predictions.
            upper_threshold (float): The upper probability threshold for predictions.
            
        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        tp, fp, tn, fn, do_nothing = self.confusion_matrix(lower_threshold, upper_threshold)
        
        total = tp + fp + tn + fn + do_nothing
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'undecided': do_nothing
            }
        }
    
    def accuracy_by_prior_bouts(self, arena, thresholds=None, bin_size=5):
        """Calculate accuracy based on the number of prior bouts for each competitor.
        
        This method analyzes how the accuracy of predictions changes based on how many
        previous matchups each competitor has participated in.
        
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
        if hasattr(arena, 'history') and hasattr(arena.history, 'bouts'):
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
                accuracy_by_min_bouts[min_bout_count] = {
                    'correct': 0,
                    'total': 0
                }
            
            # Check if prediction is correct
            if bout.predicted_outcome > upper_threshold and bout.outcome == 1.0:
                accuracy_by_min_bouts[min_bout_count]['correct'] += 1
            elif bout.predicted_outcome < lower_threshold and bout.outcome == 0.0:
                accuracy_by_min_bouts[min_bout_count]['correct'] += 1
            accuracy_by_min_bouts[min_bout_count]['total'] += 1
            
            # Update bout counts for both competitors for subsequent bouts in evaluation
            competitor_bout_counts[bout.a] = a_count + 1
            competitor_bout_counts[bout.b] = b_count + 1
        
        # Calculate accuracy for each bucket
        for bout_count, metrics in accuracy_by_min_bouts.items():
            if metrics['total'] > 0:
                metrics['accuracy'] = metrics['correct'] / metrics['total']
            else:
                metrics['accuracy'] = None
        
        # Group data into bins for smoother visualization
        binned_data = {}
        for count, metrics in accuracy_by_min_bouts.items():
            bin_index = count // bin_size
            if bin_index not in binned_data:
                binned_data[bin_index] = {
                    'accuracy_sum': 0,
                    'total': 0
                }
            
            binned_data[bin_index]['accuracy_sum'] += metrics['accuracy'] * metrics['total']
            binned_data[bin_index]['total'] += metrics['total']
        
        # Calculate average accuracy for each bin
        for bin_idx, bin_data in binned_data.items():
            if bin_data['total'] > 10:
                bin_data['accuracy'] = bin_data['accuracy_sum'] / bin_data['total']
                del bin_data['accuracy_sum']
            else:
                bin_data['accuracy'] = None
                del bin_data['accuracy_sum']
            
            # Add bin range information
            bin_data['min_bouts'] = bin_idx * bin_size
            bin_data['max_bouts'] = (bin_idx + 1) * bin_size - 1
        
        # Return only the binned data in the expected format
        return {
            'binned': binned_data
        }


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
