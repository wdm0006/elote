"""
Visualization utilities for elote.

This module provides functions for visualizing the performance of rating systems,
including comparison charts and accuracy analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def plot_rating_system_comparison(results: List[Dict[str, Any]], 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10),
                                 title: str = 'Comparison of Rating Systems Performance'):
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
    valid_results = [r for r in results if all(k in r for k in ['name', 'accuracy', 'precision', 'recall', 'f1'])]
    
    # Sort results by accuracy (descending)
    valid_results = sorted(valid_results, key=lambda x: x['accuracy'], reverse=True)
    
    # Extract data for plotting
    names = [r['name'] for r in valid_results]
    accuracy = [r['accuracy'] for r in valid_results]
    precision = [r['precision'] for r in valid_results]
    recall = [r['recall'] for r in valid_results]
    f1 = [r['f1'] for r in valid_results]
    
    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plot accuracy
    x = np.arange(len(names))
    axes[0, 0].bar(x, accuracy, color='skyblue')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot precision
    axes[0, 1].bar(x, precision, color='lightgreen')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot recall
    axes[1, 0].bar(x, recall, color='salmon')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Rating System')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot F1 score
    axes[1, 1].bar(x, f1, color='gold')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel('Rating System')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_optimized_accuracy_comparison(results: List[Dict[str, Any]], 
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 6),
                                      title: str = 'Accuracy with Optimized Thresholds'):
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
    valid_results = [r for r in results if all(k in r for k in ['name', 'accuracy', 'optimized_accuracy'])]
    
    # Sort results by optimized accuracy (descending)
    valid_results = sorted(valid_results, key=lambda x: x['optimized_accuracy'], reverse=True)
    
    # Extract data for plotting
    names = [r['name'] for r in valid_results]
    optimized_accuracy = [r['optimized_accuracy'] for r in valid_results]
    
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plot the bars
    ax.bar(np.arange(len(names)), optimized_accuracy, color='purple')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Rating System')
    plt.xticks(np.arange(len(names)), names, rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_accuracy_by_prior_bouts(results_by_prior_bouts: Dict[str, Dict], 
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 8),
                                title: str = 'Accuracy vs. Prior Bout Count by Rating System'):
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
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot a line for each competitor type
    for i, (competitor_name, bout_data) in enumerate(results_by_prior_bouts.items()):
        # Skip if no data is available
        if not bout_data or 'binned' not in bout_data:
            continue
            
        binned_data = bout_data['binned']
        bin_indices = sorted(binned_data.keys())
        
        if not bin_indices:
            continue
            
        # Extract x and y values for plotting
        x_values = [(binned_data[idx]['min_bouts'] + binned_data[idx]['max_bouts']) // 2 
                   for idx in bin_indices]
        y_values = [binned_data[idx]['accuracy'] for idx in bin_indices]
        
        # Plot the line with binned data
        plt.plot(x_values, y_values, marker='o', linestyle='-', 
                linewidth=2, markersize=8,
                color=colors[i % len(colors)], label=competitor_name)
    
    # Add labels and title
    plt.xlabel('Prior Bouts for Competitors', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set y-axis limits to match typical accuracy range
    plt.ylim(0, 1)
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf() 