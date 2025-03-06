import unittest
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from elote.arenas.base import History, Bout
from elote.visualization import (
    compute_calibration_data,
    plot_calibration_curve,
    plot_calibration_comparison,
)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        """Set up test data for calibration functions."""
        # Create a temporary directory for saving plots
        self.temp_dir = tempfile.mkdtemp()

        # Create test histories with different calibration properties
        self.histories = {
            "Well Calibrated": self._create_well_calibrated_history(),
            "Overconfident": self._create_overconfident_history(),
            "Underconfident": self._create_underconfident_history(),
        }

    def tearDown(self):
        """Clean up after tests."""
        # Close all matplotlib figures
        plt.close("all")

        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def _create_well_calibrated_history(self, n_samples=1000):
        """Create a history with well-calibrated predictions."""
        history = History()

        # Generate random probabilities
        np.random.seed(42)
        probabilities = np.random.uniform(0, 1, n_samples)

        # For each probability p, generate outcome 1 with probability p
        for i, p in enumerate(probabilities):
            outcome = 1.0 if np.random.random() < p else 0.0
            bout = Bout(f"A{i}", f"B{i}", p, outcome)
            history.add_bout(bout)

        return history

    def _create_overconfident_history(self, n_samples=1000):
        """Create a history with overconfident predictions."""
        history = History()

        # Generate random probabilities
        np.random.seed(43)
        probabilities = np.random.uniform(0, 1, n_samples)

        # Make predictions more extreme (closer to 0 or 1)
        overconfident_probs = np.power(probabilities, 0.5)  # Push toward 1
        overconfident_probs[probabilities < 0.5] = 1 - np.power(
            1 - probabilities[probabilities < 0.5], 0.5
        )  # Push toward 0

        # For each probability p, generate outcome 1 with probability p
        for i, (p, overconfident_p) in enumerate(zip(probabilities, overconfident_probs)):
            outcome = 1.0 if np.random.random() < p else 0.0
            bout = Bout(f"A{i}", f"B{i}", overconfident_p, outcome)
            history.add_bout(bout)

        return history

    def _create_underconfident_history(self, n_samples=1000):
        """Create a history with underconfident predictions."""
        history = History()

        # Generate random probabilities
        np.random.seed(44)
        probabilities = np.random.uniform(0, 1, n_samples)

        # Make predictions less extreme (closer to 0.5)
        underconfident_probs = 0.5 + (probabilities - 0.5) * 0.5  # Shrink toward 0.5

        # For each probability p, generate outcome 1 with probability p
        for i, (p, underconfident_p) in enumerate(zip(probabilities, underconfident_probs)):
            outcome = 1.0 if np.random.random() < p else 0.0
            bout = Bout(f"A{i}", f"B{i}", underconfident_p, outcome)
            history.add_bout(bout)

        return history

    def test_get_calibration_data(self):
        """Test that get_calibration_data returns the expected data format."""
        for _name, history in self.histories.items():
            y_true, y_prob = history.get_calibration_data()

            # Check that the data has the expected shape
            self.assertEqual(len(y_true), len(history.bouts))
            self.assertEqual(len(y_prob), len(history.bouts))

            # Check that y_true contains only 0s and 1s
            self.assertTrue(all(y in [0.0, 1.0] for y in y_true))

            # Check that y_prob contains values between 0 and 1
            self.assertTrue(all(0 <= p <= 1 for p in y_prob))

    def test_compute_calibration_data(self):
        """Test that compute_calibration_data returns the expected data format."""
        for _name, history in self.histories.items():
            prob_true, prob_pred = compute_calibration_data(history, n_bins=10)

            # Check that the data has a reasonable shape (may not be exactly 10 bins due to empty bins)
            self.assertGreater(len(prob_true), 0)
            self.assertGreater(len(prob_pred), 0)
            self.assertEqual(len(prob_true), len(prob_pred))

            # Check that prob_true and prob_pred contain values between 0 and 1
            self.assertTrue(all(0 <= p <= 1 for p in prob_true))
            self.assertTrue(all(0 <= p <= 1 for p in prob_pred))

    def test_plot_calibration_curve(self):
        """Test that plot_calibration_curve creates a figure."""
        # Test without saving
        fig = plot_calibration_curve(self.histories)
        self.assertIsNotNone(fig)

        # Test with saving
        save_path = os.path.join(self.temp_dir, "calibration_curve.png")
        fig = plot_calibration_curve(self.histories, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

        # Test with custom parameters
        fig = plot_calibration_curve(
            self.histories,
            figsize=(12, 10),
            title="Custom Title",
            n_bins=5,
        )
        self.assertEqual(fig.get_figwidth(), 12)
        self.assertEqual(fig.get_figheight(), 10)
        self.assertEqual(fig.axes[0].get_title(), "Custom Title")

    def test_plot_calibration_comparison(self):
        """Test that plot_calibration_comparison creates a figure."""
        # Test without saving
        fig = plot_calibration_comparison(self.histories)
        self.assertIsNotNone(fig)

        # Test with saving
        save_path = os.path.join(self.temp_dir, "calibration_comparison.png")
        fig = plot_calibration_comparison(self.histories, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

        # Test with custom parameters
        fig = plot_calibration_comparison(
            self.histories,
            figsize=(16, 10),
            title="Custom Title",
            n_bins=5,
        )
        self.assertEqual(fig.get_figwidth(), 16)
        self.assertEqual(fig.get_figheight(), 10)
        self.assertEqual(fig.axes[0].get_title(), "Custom Title")

    def test_with_empty_history(self):
        """Test that calibration functions handle empty histories gracefully."""
        empty_history = History()

        # Test get_calibration_data
        y_true, y_prob = empty_history.get_calibration_data()
        self.assertEqual(len(y_true), 0)
        self.assertEqual(len(y_prob), 0)

        # Test compute_calibration_data
        # This should not raise an error, but return empty arrays
        prob_true, prob_pred = compute_calibration_data(empty_history)
        self.assertEqual(len(prob_true), 0)
        self.assertEqual(len(prob_pred), 0)

        # Test plot_calibration_curve with empty history
        empty_histories = {"Empty": empty_history}
        fig = plot_calibration_curve(empty_histories)
        self.assertIsNotNone(fig)

        # Test plot_calibration_comparison with empty history
        fig = plot_calibration_comparison(empty_histories)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
