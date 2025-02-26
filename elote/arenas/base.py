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
        thresholds that maximize the net performance (TP + TN - FP - FN).
        
        Args:
            trials (int): The number of random threshold pairs to try.
            
        Returns:
            tuple: A tuple containing (best_net_performance, best_thresholds).
        """
        best_net, best_thresholds = 0, list()
        for _ in range(trials):
            thresholds = sorted([random.random(), random.random()])
            tp, fp, tn, fn, do_nothing = self.confusion_matrix(*thresholds)
            net = tp + tn - fn - fp
            if net > best_net:
                best_thresholds = thresholds

        return best_net, best_thresholds


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
            outcome (str): The actual outcome ("win", "loss", or "draw").
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
        if self.predicted_outcome > threshold and self.outcome == "win":
            return True
        else:
            return False

    def false_positive(self, threshold=0.5):
        """Check if this bout is a false positive prediction.
        
        A false positive occurs when the model incorrectly predicts a win.
        
        Args:
            threshold (float): The probability threshold for a positive prediction.
            
        Returns:
            bool: True if this bout is a false positive, False otherwise.
        """
        if self.predicted_outcome > threshold and self.outcome != "win":
            return True
        else:
            return False

    def true_negative(self, threshold=0.5):
        """Check if this bout is a true negative prediction.
        
        A true negative occurs when the model correctly predicts a non-win.
        
        Args:
            threshold (float): The probability threshold for a negative prediction.
            
        Returns:
            bool: True if this bout is a true negative, False otherwise.
        """
        if self.predicted_outcome <= threshold and self.outcome == "loss":
            return True
        else:
            return False

    def false_negative(self, threshold=0.5):
        """Check if this bout is a false negative prediction.
        
        A false negative occurs when the model incorrectly predicts a non-win.
        
        Args:
            threshold (float): The probability threshold for a negative prediction.
            
        Returns:
            bool: True if this bout is a false negative, False otherwise.
        """
        if self.predicted_outcome <= threshold and self.outcome != "loss":
            return True
        else:
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
        if self.outcome == "win":
            return self.a
        elif self.outcome == "loss":
            return self.b
        else:
            return None
