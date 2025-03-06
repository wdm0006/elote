"""
Visualization utilities for elote.

This module provides functions for visualizing the performance of rating systems,
including comparison charts and accuracy analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def plot_rating_system_comparison(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Comparison of Rating Systems Performance",
):
    """
    Create bar charts comparing the performance of different rating systems.

    Args:
        results: List of dictionaries containing evaluation results for each rating system.
                Each dict should have 'name', 'accuracy', 'precision', 'recall', 'f1' keys.
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.

    Returns:
        The matplotlib figure object.
    """
    # Filter out results without required metrics
    valid_results = [r for r in results if all(k in r for k in ["name", "accuracy", "precision", "recall", "f1"])]

    # Sort results by accuracy (descending)
    valid_results = sorted(valid_results, key=lambda x: x["accuracy"], reverse=True)

    # Extract data for plotting
    names = [r["name"] for r in valid_results]
    accuracy = [r["accuracy"] for r in valid_results]
    precision = [r["precision"] for r in valid_results]
    recall = [r["recall"] for r in valid_results]
    f1 = [r["f1"] for r in valid_results]

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot accuracy
    x = np.arange(len(names))
    axes[0, 0].bar(x, accuracy, color="skyblue")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right")

    # Plot precision
    axes[0, 1].bar(x, precision, color="lightgreen")
    axes[0, 1].set_title("Precision")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right")

    # Plot recall
    axes[1, 0].bar(x, recall, color="salmon")
    axes[1, 0].set_title("Recall")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_xlabel("Rating System")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")

    # Plot F1 score
    axes[1, 1].bar(x, f1, color="gold")
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Rating System")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save or display
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_optimized_accuracy_comparison(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Accuracy with Optimized Thresholds",
):
    """
    Create a bar chart comparing the accuracy of different rating systems with optimized thresholds.

    Args:
        results: List of dictionaries containing evaluation results for each rating system.
                Each dict should have 'name', 'accuracy', and 'optimized_accuracy' keys.
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.

    Returns:
        The matplotlib figure object.
    """
    # Filter out results without required metrics
    valid_results = [r for r in results if all(k in r for k in ["name", "accuracy", "optimized_accuracy"])]

    # Sort results by optimized accuracy (descending)
    valid_results = sorted(valid_results, key=lambda x: x["optimized_accuracy"], reverse=True)

    # Extract data for plotting
    names = [r["name"] for r in valid_results]
    optimized_accuracy = [r["optimized_accuracy"] for r in valid_results]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot the bars
    ax.bar(np.arange(len(names)), optimized_accuracy, color="purple")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Rating System")
    plt.xticks(np.arange(len(names)), names, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save or display
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_accuracy_by_prior_bouts(
    results_by_prior_bouts: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Accuracy vs. Prior Bout Count by Rating System",
):
    """
    Create a line chart showing accuracy vs. prior bout count for each competitor type.

    Args:
        results_by_prior_bouts: Dictionary mapping competitor names to their accuracy by prior bout data.
                               Each value should be the output of History.accuracy_by_prior_bouts().
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.

    Returns:
        The matplotlib figure object.
    """
    plt.figure(figsize=figsize)

    # Define a color map for the different competitor types
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    # Plot a line for each competitor type
    for i, (competitor_name, bout_data) in enumerate(results_by_prior_bouts.items()):
        # Skip if no data is available
        if not bout_data or "binned" not in bout_data:
            continue

        binned_data = bout_data["binned"]
        bin_indices = sorted(binned_data.keys())

        if not bin_indices:
            continue

        # Extract x and y values for plotting
        x_values = [(binned_data[idx]["min_bouts"] + binned_data[idx]["max_bouts"]) // 2 for idx in bin_indices]
        y_values = [binned_data[idx]["accuracy"] for idx in bin_indices]

        # Plot the line with binned data
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=colors[i % len(colors)],
            label=competitor_name,
        )

    # Add labels and title
    plt.xlabel("Prior Bouts for Competitors", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    # Set y-axis limits to match typical accuracy range
    plt.ylim(0, 1)

    # Save or display
    if save_path:
        plt.savefig(save_path)

    return plt.gcf()


def compute_calibration_data(history, n_bins=10):
    """
    Compute calibration data from a history of bouts without using scikit-learn.

    Args:
        history: A History object containing bouts with predicted outcomes and actual outcomes.
        n_bins: Number of bins to use for calibration curve.

    Returns:
        tuple: (prob_true, prob_pred) where:
            - prob_true: The fraction of positive samples in each bin
            - prob_pred: The mean predicted probability in each bin
    """
    # Get the raw data from the history
    y_true, y_prob = history.get_calibration_data(n_bins=n_bins)

    if len(y_true) == 0:
        return np.array([]), np.array([])

    # Create bins of equal width
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1

    # Ensure all indices are within valid range (0 to n_bins-1)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate mean predicted probability and fraction of positives for each bin
    prob_pred = np.zeros(n_bins)
    prob_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(len(y_prob)):
        bin_idx = bin_indices[i]
        prob_pred[bin_idx] += y_prob[i]
        prob_true[bin_idx] += y_true[i]
        bin_counts[bin_idx] += 1

    # Calculate means, avoiding division by zero
    for i in range(n_bins):
        if bin_counts[i] > 0:
            prob_pred[i] /= bin_counts[i]
            prob_true[i] /= bin_counts[i]
        else:
            # If a bin is empty, set both values to NaN
            prob_pred[i] = np.nan
            prob_true[i] = np.nan

    # Remove NaN values
    valid_indices = ~np.isnan(prob_pred)
    prob_pred = prob_pred[valid_indices]
    prob_true = prob_true[valid_indices]

    return prob_true, prob_pred


def plot_calibration_curve(
    histories: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Calibration Curves",
    n_bins: int = 10,
):
    """
    Create a calibration curve plot for multiple rating systems.

    Args:
        histories: Dictionary mapping rating system names to their History objects.
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.
        n_bins: Number of bins to use for calibration curve.

    Returns:
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the perfectly calibrated line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Define a color map for the different rating systems
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    # Plot calibration curve for each rating system
    for i, (name, history) in enumerate(histories.items()):
        prob_true, prob_pred = compute_calibration_data(history, n_bins=n_bins)

        if len(prob_true) > 0:
            # Plot the calibration curve
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                color=colors[i % len(colors)],
                label=f"{name}",
                linewidth=2,
            )

    # Add labels and title
    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)

    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_calibration_comparison(
    histories: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
    title: str = "Calibration Comparison",
    n_bins: int = 10,
):
    """
    Create a plot with calibration curves and histograms of predicted probabilities.

    This is similar to sklearn's calibration_display but customized for elote.

    Args:
        histories: Dictionary mapping rating system names to their History objects.
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.
        n_bins: Number of bins to use for calibration curve and histogram.

    Returns:
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

    # Define a color map for the different rating systems
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    # Plot the perfectly calibrated line
    axes[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot calibration curve and histogram for each rating system
    for i, (name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]

        # Compute calibration curve
        prob_true, prob_pred = compute_calibration_data(history, n_bins=n_bins)

        if len(prob_true) > 0:
            # Plot the calibration curve
            axes[0].plot(
                prob_pred,
                prob_true,
                "s-",
                color=color,
                label=f"{name}",
                linewidth=2,
            )

        # Extract predicted probabilities for histogram
        y_prob = [bout.predicted_outcome for bout in history.bouts]

        if len(y_prob) > 0:
            # Plot histogram of predicted probabilities
            axes[1].hist(
                y_prob,
                range=(0, 1),
                bins=n_bins,
                histtype="step",
                lw=2,
                color=color,
                label=name,
                density=True,
            )

    # Configure calibration curve subplot
    axes[0].set_xlabel("Mean predicted probability", fontsize=12)
    axes[0].set_ylabel("Fraction of positives", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend(loc="best", fontsize=10)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Configure histogram subplot
    axes[1].set_xlabel("Predicted probability", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)

    return fig
