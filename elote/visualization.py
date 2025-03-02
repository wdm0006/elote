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
    # Sort results by accuracy (descending)
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]
    
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
    Create a bar chart comparing the optimized accuracy of different rating systems.
    
    Args:
        results: List of dictionaries containing evaluation results for each rating system.
                Each dict should have 'name' and 'accuracy_opt' keys.
        save_path: Optional path to save the figure. If None, the figure is displayed instead.
        figsize: Figure size as (width, height) in inches.
        title: Title for the figure.
        
    Returns:
        The matplotlib figure object.
    """
    # Sort by optimized accuracy
    results = sorted(results, key=lambda x: x.get('accuracy_opt', 0), reverse=True)
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracy_opt = [r.get('accuracy_opt', 0) for r in results]
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Plot the bars
    plt.bar(np.arange(len(names)), accuracy_opt, color='purple')
    plt.title(title)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Rating System')
    plt.xticks(np.arange(len(names)), names, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()


def plot_accuracy_by_prior_bouts(results_by_prior_bouts: Dict[str, Dict], 
                                max_bouts: int = 30,
                                bin_size: int = 5,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 8),
                                title: str = 'Accuracy vs. Prior Bout Count by Rating System'):
    """
    Create a line chart showing accuracy vs. prior bout count for each competitor type.
    
    Args:
        results_by_prior_bouts: Dictionary mapping competitor names to their accuracy by prior bout data.
                               Each value should be the output of History.accuracy_by_prior_bouts().
        max_bouts: Maximum number of prior bouts to include in the plot.
        bin_size: Size of bins for grouping bout counts.
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
        # If the data is already binned, use it directly
        if 'binned' in bout_data:
            binned_data = bout_data['binned']
            bin_indices = sorted([idx for idx in binned_data.keys() 
                                if binned_data[idx]['max_bouts'] <= max_bouts])
            
            if not bin_indices:
                continue
                
            bin_labels = [(binned_data[idx]['min_bouts'] + binned_data[idx]['max_bouts']) // 2 
                         for idx in bin_indices]
            bin_accuracies = [binned_data[idx]['accuracy'] for idx in bin_indices]
            
            # Plot the line with binned data
            plt.plot(bin_labels, bin_accuracies, marker='o', linestyle='-', 
                    linewidth=2, markersize=8,
                    color=colors[i % len(colors)], label=competitor_name)
        
        # If the data is not binned, bin it now
        else:
            # Extract bout counts and corresponding accuracies
            raw_data = bout_data.get('by_bout_count', bout_data)
            bout_counts = sorted([count for count in raw_data.keys() if count <= max_bouts])
            
            # Skip if no data points
            if not bout_counts:
                continue
            
            # Group data into bins
            binned_data = {}
            for count in bout_counts:
                bin_index = count // bin_size
                if bin_index not in binned_data:
                    binned_data[bin_index] = {'accuracy': 0, 'total': 0}
                
                metrics = raw_data[count]
                if isinstance(metrics, dict) and 'accuracy' in metrics and 'total' in metrics:
                    binned_data[bin_index]['accuracy'] += metrics['accuracy'] * metrics['total']
                    binned_data[bin_index]['total'] += metrics['total']
                else:
                    # If metrics is just a value, assume it's the accuracy and weight is 1
                    binned_data[bin_index]['accuracy'] += metrics
                    binned_data[bin_index]['total'] += 1
            
            # Calculate average accuracy for each bin
            bin_counts = sorted(binned_data.keys())
            bin_labels = [(bin_idx * bin_size) + (bin_size // 2) for bin_idx in bin_counts]
            bin_accuracies = []
            
            for bin_idx in bin_counts:
                if binned_data[bin_idx]['total'] > 0:
                    avg_accuracy = binned_data[bin_idx]['accuracy'] / binned_data[bin_idx]['total']
                    bin_accuracies.append(avg_accuracy)
                else:
                    bin_accuracies.append(0)
            
            # Plot the line with binned data
            plt.plot(bin_labels, bin_accuracies, marker='o', linestyle='-', 
                    linewidth=2, markersize=8,
                    color=colors[i % len(colors)], label=competitor_name)
    
    # Add labels and title
    plt.xlabel('Prior Bouts for Competitors (binned)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set x-axis to show bin centers
    plt.xticks([(i * bin_size) + (bin_size // 2) for i in range((max_bouts // bin_size) + 1)], 
              [f"{i*bin_size}-{(i+1)*bin_size-1}" for i in range((max_bouts // bin_size) + 1)],
              rotation=45)
    
    # Set y-axis limits to match typical accuracy range
    plt.ylim(0, 1)
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf() 