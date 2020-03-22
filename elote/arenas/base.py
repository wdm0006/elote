import abc
import random


class BaseArena:
    @abc.abstractmethod
    def set_competitor_class_var(self, name, value):
        pass

    @abc.abstractmethod
    def tournament(self, matchups):
        pass

    @abc.abstractmethod
    def matchup(self, a, b):
        pass

    @abc.abstractmethod
    def leaderboard(self):
        pass

    @abc.abstractmethod
    def export_state(self):
        pass


class History:
    def __init__(self):
        self.bouts = []

    def add_bout(self, bout):
        self.bouts.append(bout)

    def report_results(self, lower_threshold=0.5, upper_threshold=0.5):
        print('predicted_winner,predicted_loser,probability,actual_winner,correct')
        for bout in self.bouts:
            print('%s,%s,%s,%s,%s' % (
                bout.predicted_winner(lower_threshold, upper_threshold),
                bout.predicted_loser(lower_threshold, upper_threshold),
                bout.predicted_outcome * 100,
                bout.actual_winner(),
                bout.predicted_winner(lower_threshold, upper_threshold) == bout.actual_winner()
            ))

    def confusion_matrix(self, lower_threshold=0.5, upper_threshold=0.5, attribute_filter=None):
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
        best_net, best_thresholds = 0, list()
        for _ in range(trials):
            thresholds = sorted([random.random(), random.random()])
            tp, fp, tn, fn, do_nothing = self.confusion_matrix(*thresholds)
            net = tp + tn - fn - fp
            if net > best_net:
                best_thresholds = thresholds

        return best_net, best_thresholds


class Bout:
    def __init__(self, a, b, predicted_outcome, outcome, attributes=None):
        self.a = a
        self.b = b
        self.predicted_outcome = predicted_outcome
        self.outcome = outcome
        self.attributes = attributes or dict()

    def true_positive(self, threshold=0.5):
        if self.predicted_outcome > threshold and self.outcome == 'win':
            return True
        else:
            return False

    def false_positive(self, threshold=0.5):
        if self.predicted_outcome > threshold and self.outcome != 'win':
            return True
        else:
            return False

    def true_negative(self, threshold=0.5):
        if self.predicted_outcome <= threshold and self.outcome == 'loss':
            return True
        else:
            return False

    def false_negative(self, threshold=0.5):
        if self.predicted_outcome <= threshold and self.outcome != 'loss':
            return True
        else:
            return False

    def predicted_winner(self, lower_treshold=0.5, upper_threshold=0.5):
        if self.predicted_outcome > upper_threshold:
            return self.a
        elif self.predicted_outcome < lower_treshold:
            return self.b
        else:
            return None

    def predicted_loser(self, lower_treshold=0.5, upper_threshold=0.5):
        if self.predicted_outcome > upper_threshold:
            return self.b
        elif self.predicted_outcome < lower_treshold:
            return self.a
        else:
            return None

    def actual_winner(self):
        if self.outcome == 'win':
            return self.a
        elif self.outcome == 'loss':
            return self.b
        else:
            return None