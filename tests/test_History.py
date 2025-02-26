import unittest
from elote.arenas.base import History, Bout


class TestHistory(unittest.TestCase):
    def test_history_initialization(self):
        """Test that the History class initializes correctly."""
        history = History()
        self.assertEqual(len(history.bouts), 0)

    def test_add_bout(self):
        """Test that bouts can be added to the history."""
        history = History()

        # Create a bout
        bout = Bout("A", "B", 0.7, "win")

        # Add the bout to the history
        history.add_bout(bout)

        # Check that the bout was added
        self.assertEqual(len(history.bouts), 1)
        self.assertEqual(history.bouts[0], bout)

        # Add another bout
        bout2 = Bout("C", "D", 0.3, "loss")
        history.add_bout(bout2)

        # Check that both bouts are in the history
        self.assertEqual(len(history.bouts), 2)
        self.assertEqual(history.bouts[0], bout)
        self.assertEqual(history.bouts[1], bout2)

    def test_report_results(self):
        """Test that the history can report results correctly."""
        history = History()

        # Add some bouts with different outcomes
        history.add_bout(Bout("A", "B", 0.7, "win"))  # True positive
        history.add_bout(Bout("C", "D", 0.7, "loss"))  # False positive
        history.add_bout(Bout("E", "F", 0.3, "loss"))  # True negative
        history.add_bout(Bout("G", "H", 0.3, "win"))  # False negative

        # Get the report
        report = history.report_results()

        # Check that the report contains the correct information
        self.assertEqual(len(report), 4)

        # Check that each report entry has the expected keys
        for entry in report:
            self.assertIn("predicted_winnder", entry)  # Note: there's a typo in the original code
            self.assertIn("predicted_loser", entry)
            self.assertIn("probability", entry)
            self.assertIn("actual_winner", entry)
            self.assertIn("correct", entry)

    def test_report_results_with_thresholds(self):
        """Test that the history can report results with custom thresholds."""
        history = History()

        # Add some bouts
        history.add_bout(Bout("A", "B", 0.8, "win"))
        history.add_bout(Bout("C", "D", 0.6, "win"))
        history.add_bout(Bout("E", "F", 0.4, "loss"))
        history.add_bout(Bout("G", "H", 0.2, "loss"))

        # With default thresholds (0.5, 0.5)
        report1 = history.report_results()

        # With custom thresholds (0.7, 0.3)
        report2 = history.report_results(lower_threshold=0.3, upper_threshold=0.7)

        # Check that the reports have different correctness values
        correct_count1 = sum(1 for r in report1 if r["correct"])
        correct_count2 = sum(1 for r in report2 if r["correct"])

        # The correctness should be different because the thresholds change what is considered a "win"
        self.assertNotEqual(correct_count1, correct_count2)

    def test_confusion_matrix(self):
        """Test that the confusion matrix is calculated correctly."""
        history = History()

        # Add some bouts with different outcomes
        history.add_bout(Bout("A", "B", 0.7, "win"))  # True positive
        history.add_bout(Bout("C", "D", 0.7, "loss"))  # False positive
        history.add_bout(Bout("E", "F", 0.3, "loss"))  # True negative
        history.add_bout(Bout("G", "H", 0.3, "win"))  # False negative

        # Get the confusion matrix
        tp, fp, tn, fn, do_nothing = history.confusion_matrix()

        # Check the counts
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(tn, 1)
        self.assertEqual(fn, 1)
        self.assertEqual(do_nothing, 0)

        # Test with custom thresholds
        tp, fp, tn, fn, do_nothing = history.confusion_matrix(lower_threshold=0.4, upper_threshold=0.6)

        # With these thresholds, the bouts with predicted_outcome=0.3 and 0.7 should be counted,
        # but the bouts with predicted_outcome=0.4 and 0.6 should be in do_nothing
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(tn, 1)
        self.assertEqual(fn, 1)
        self.assertEqual(do_nothing, 0)

    def test_random_search(self):
        """Test that random search finds good thresholds."""
        history = History()

        # Add some bouts with different outcomes
        history.add_bout(Bout("A", "B", 0.9, "win"))  # True positive
        history.add_bout(Bout("C", "D", 0.8, "loss"))  # False positive
        history.add_bout(Bout("E", "F", 0.2, "loss"))  # True negative
        history.add_bout(Bout("G", "H", 0.1, "win"))  # False negative

        # Run random search
        best_net, best_thresholds = history.random_search(trials=100)

        # Check that the best thresholds are sorted
        self.assertLessEqual(best_thresholds[0], best_thresholds[1])

        # Check that the best net is at least 0
        self.assertGreaterEqual(best_net, 0)


class TestBout(unittest.TestCase):
    def test_bout_initialization(self):
        """Test that the Bout class initializes correctly."""
        bout = Bout("A", "B", 0.7, "win")
        self.assertEqual(bout.a, "A")
        self.assertEqual(bout.b, "B")
        self.assertEqual(bout.predicted_outcome, 0.7)
        self.assertEqual(bout.outcome, "win")
        self.assertEqual(bout.attributes, {})

        # Test with attributes
        attributes = {"importance": "high"}
        bout = Bout("C", "D", 0.3, "loss", attributes=attributes)
        self.assertEqual(bout.attributes, attributes)

    def test_true_positive(self):
        """Test the true_positive method."""
        # A true positive is when the predicted outcome is > threshold and the actual outcome is a win
        bout = Bout("A", "B", 0.7, "win")
        self.assertTrue(bout.true_positive())

        # Not a true positive if the outcome is not a win
        bout = Bout("A", "B", 0.7, "loss")
        self.assertFalse(bout.true_positive())

        # Not a true positive if the predicted outcome is <= threshold
        bout = Bout("A", "B", 0.5, "win")
        self.assertFalse(bout.true_positive())

        # Test with a custom threshold
        bout = Bout("A", "B", 0.6, "win")
        self.assertFalse(bout.true_positive(threshold=0.7))

    def test_false_positive(self):
        """Test the false_positive method."""
        # A false positive is when the predicted outcome is > threshold but the actual outcome is not a win
        bout = Bout("A", "B", 0.7, "loss")
        self.assertTrue(bout.false_positive())

        # Not a false positive if the outcome is a win
        bout = Bout("A", "B", 0.7, "win")
        self.assertFalse(bout.false_positive())

        # Not a false positive if the predicted outcome is <= threshold
        bout = Bout("A", "B", 0.5, "loss")
        self.assertFalse(bout.false_positive())

        # Test with a custom threshold
        bout = Bout("A", "B", 0.6, "loss")
        self.assertFalse(bout.false_positive(threshold=0.7))

    def test_true_negative(self):
        """Test the true_negative method."""
        # A true negative is when the predicted outcome is <= threshold and the actual outcome is a loss
        bout = Bout("A", "B", 0.3, "loss")
        self.assertTrue(bout.true_negative())

        # Not a true negative if the outcome is not a loss
        bout = Bout("A", "B", 0.3, "win")
        self.assertFalse(bout.true_negative())

        # Not a true negative if the predicted outcome is > threshold
        bout = Bout("A", "B", 0.6, "loss")
        self.assertFalse(bout.true_negative())

        # Test with a custom threshold
        bout = Bout("A", "B", 0.4, "loss")
        self.assertFalse(bout.true_negative(threshold=0.3))

    def test_false_negative(self):
        """Test the false_negative method."""
        # A false negative is when the predicted outcome is <= threshold but the actual outcome is not a loss
        bout = Bout("A", "B", 0.3, "win")
        self.assertTrue(bout.false_negative())

        # Not a false negative if the outcome is a loss
        bout = Bout("A", "B", 0.3, "loss")
        self.assertFalse(bout.false_negative())

        # Not a false negative if the predicted outcome is > threshold
        bout = Bout("A", "B", 0.6, "win")
        self.assertFalse(bout.false_negative())

        # Test with a custom threshold
        bout = Bout("A", "B", 0.4, "win")
        self.assertFalse(bout.false_negative(threshold=0.3))

    def test_predicted_winner(self):
        """Test the predicted_winner method."""
        # If predicted_outcome > upper_threshold, a is the predicted winner
        bout = Bout("A", "B", 0.7, "win")
        self.assertEqual(bout.predicted_winner(), "A")

        # If predicted_outcome < lower_threshold, b is the predicted winner
        bout = Bout("A", "B", 0.3, "win")
        self.assertEqual(bout.predicted_winner(), "B")

        # If lower_threshold <= predicted_outcome <= upper_threshold, there is no predicted winner
        bout = Bout("A", "B", 0.5, "win")
        self.assertIsNone(bout.predicted_winner())

        # Test with custom thresholds
        bout = Bout("A", "B", 0.6, "win")
        self.assertEqual(bout.predicted_winner(lower_threshold=0.4, upper_threshold=0.7), None)
        self.assertEqual(bout.predicted_winner(lower_threshold=0.4, upper_threshold=0.5), "A")

    def test_predicted_loser(self):
        """Test the predicted_loser method."""
        # If predicted_outcome > upper_threshold, b is the predicted loser
        bout = Bout("A", "B", 0.7, "win")
        self.assertEqual(bout.predicted_loser(), "B")

        # If predicted_outcome < lower_threshold, a is the predicted loser
        bout = Bout("A", "B", 0.3, "win")
        self.assertEqual(bout.predicted_loser(), "A")

        # If lower_threshold <= predicted_outcome <= upper_threshold, there is no predicted loser
        bout = Bout("A", "B", 0.5, "win")
        self.assertIsNone(bout.predicted_loser())

        # Test with custom thresholds
        bout = Bout("A", "B", 0.6, "win")
        self.assertEqual(bout.predicted_loser(lower_threshold=0.4, upper_threshold=0.7), None)
        self.assertEqual(bout.predicted_loser(lower_threshold=0.4, upper_threshold=0.5), "B")

    def test_actual_winner(self):
        """Test the actual_winner method."""
        # If outcome is 'win', a is the actual winner
        bout = Bout("A", "B", 0.7, "win")
        self.assertEqual(bout.actual_winner(), "A")

        # If outcome is 'loss', b is the actual winner
        bout = Bout("A", "B", 0.3, "loss")
        self.assertEqual(bout.actual_winner(), "B")

        # If outcome is 'tie', there is no actual winner
        bout = Bout("A", "B", 0.5, "tie")
        self.assertIsNone(bout.actual_winner())
