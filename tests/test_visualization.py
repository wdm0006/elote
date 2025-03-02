import unittest
import os
import tempfile
import matplotlib.pyplot as plt
from elote.visualization import (
    plot_rating_system_comparison,
    plot_optimized_accuracy_comparison,
    plot_accuracy_by_prior_bouts,
)


class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test data for visualization functions."""
        # Sample results for rating system comparison
        self.results = [
            {
                "name": "System A",
                "accuracy": 0.75,
                "precision": 0.80,
                "recall": 0.70,
                "f1": 0.75,
                "optimized_accuracy": 0.78,
            },
            {
                "name": "System B",
                "accuracy": 0.65,
                "precision": 0.70,
                "recall": 0.60,
                "f1": 0.65,
                "optimized_accuracy": 0.68,
            },
            {
                "name": "System C",
                "accuracy": 0.85,
                "precision": 0.90,
                "recall": 0.80,
                "f1": 0.85,
                "optimized_accuracy": 0.88,
            },
        ]

        # Sample data for accuracy by prior bouts
        self.accuracy_by_prior_bouts = {
            "System A": {
                "binned": {
                    0: {"accuracy": 0.60, "total": 100, "min_bouts": 0, "max_bouts": 4},
                    1: {"accuracy": 0.70, "total": 100, "min_bouts": 5, "max_bouts": 9},
                    2: {"accuracy": 0.80, "total": 100, "min_bouts": 10, "max_bouts": 14},
                }
            },
            "System B": {
                "binned": {
                    0: {"accuracy": 0.55, "total": 100, "min_bouts": 0, "max_bouts": 4},
                    1: {"accuracy": 0.65, "total": 100, "min_bouts": 5, "max_bouts": 9},
                    2: {"accuracy": 0.75, "total": 100, "min_bouts": 10, "max_bouts": 14},
                }
            },
        }

        # Create a temporary directory for saving plots
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Close all matplotlib figures
        plt.close("all")

        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_plot_rating_system_comparison(self):
        """Test that plot_rating_system_comparison creates a figure."""
        # Test without saving
        fig = plot_rating_system_comparison(self.results)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # 2x2 grid of subplots

        # Test with saving
        save_path = os.path.join(self.temp_dir, "rating_comparison.png")
        fig = plot_rating_system_comparison(self.results, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

        # Test with custom figsize and title
        fig = plot_rating_system_comparison(self.results, figsize=(10, 8), title="Custom Title")
        self.assertEqual(fig.get_figwidth(), 10)
        self.assertEqual(fig.get_figheight(), 8)
        self.assertEqual(fig._suptitle.get_text(), "Custom Title")

    def test_plot_optimized_accuracy_comparison(self):
        """Test that plot_optimized_accuracy_comparison creates a figure."""
        # Test without saving
        fig = plot_optimized_accuracy_comparison(self.results)
        self.assertIsNotNone(fig)

        # Test with saving
        save_path = os.path.join(self.temp_dir, "optimized_accuracy.png")
        fig = plot_optimized_accuracy_comparison(self.results, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

        # Test with custom figsize and title
        fig = plot_optimized_accuracy_comparison(self.results, figsize=(8, 6), title="Custom Title")
        self.assertEqual(fig.get_figwidth(), 8)
        self.assertEqual(fig.get_figheight(), 6)
        self.assertEqual(fig.axes[0].get_title(), "Custom Title")

    def test_plot_accuracy_by_prior_bouts_with_binned_data(self):
        """Test that plot_accuracy_by_prior_bouts works with binned data."""
        # Test without saving
        fig = plot_accuracy_by_prior_bouts(self.accuracy_by_prior_bouts)
        self.assertIsNotNone(fig)

        # Test with saving
        save_path = os.path.join(self.temp_dir, "accuracy_by_bouts.png")
        fig = plot_accuracy_by_prior_bouts(self.accuracy_by_prior_bouts, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

        # Test with custom parameters
        fig = plot_accuracy_by_prior_bouts(self.accuracy_by_prior_bouts, figsize=(12, 8), title="Custom Title")
        self.assertEqual(fig.get_figwidth(), 12)
        self.assertEqual(fig.get_figheight(), 8)
        self.assertEqual(fig.axes[0].get_title(), "Custom Title")

    def test_plot_accuracy_by_prior_bouts_with_empty_data(self):
        """Test that plot_accuracy_by_prior_bouts handles empty data gracefully."""
        # Empty data
        empty_data = {}
        fig = plot_accuracy_by_prior_bouts(empty_data)
        self.assertIsNotNone(fig)

        # Data with empty bins
        data_with_empty_bins = {"System A": {"binned": {}}}
        fig = plot_accuracy_by_prior_bouts(data_with_empty_bins)
        self.assertIsNotNone(fig)

    def test_plot_functions_with_invalid_data(self):
        """Test that visualization functions handle invalid data gracefully."""
        # Invalid results for rating system comparison
        invalid_results = [
            {"name": "System A"}  # Missing metrics
        ]

        # Should not raise an error, but might not plot anything
        fig = plot_rating_system_comparison(invalid_results)
        self.assertIsNotNone(fig)

        # Invalid results for optimized accuracy comparison
        fig = plot_optimized_accuracy_comparison(invalid_results)
        self.assertIsNotNone(fig)

        # Invalid data for accuracy by prior bouts
        invalid_bout_data = {
            "System A": {}  # Missing binned data
        }
        fig = plot_accuracy_by_prior_bouts(invalid_bout_data)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
