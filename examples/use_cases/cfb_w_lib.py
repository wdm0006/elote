"""
Example of using the CollegeFootballDataset to train and evaluate multiple rating systems.

This example demonstrates how to use the CollegeFootballDataset to train and evaluate
different rating systems for college football teams and compare their performance.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from elote import (
    LambdaArena, 
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    BlendedCompetitor,
    CollegeFootballDataset,
    train_and_evaluate_arena
)
from elote.arenas.base import History, Bout


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cfb_example')


def progress_callback(phase, current, total):
    """Callback function to report progress during training and evaluation."""
    if current == 0:
        logger.info(f"Starting {phase} phase...")
    elif current == total:
        logger.info(f"Completed {phase} phase.")
    elif current % 500 == 0:
        logger.info(f"{phase.capitalize()} progress: {current}/{total} ({current/total:.1%})")


def compare_teams(team_a, team_b, attributes=None):
    """Compare two teams based on their scores."""
    logger.debug(f"compare_teams called with team_a={team_a}, team_b={team_b}, attributes={attributes}")
    
    if attributes is None:
        logger.warning(f"No attributes provided for matchup between {team_a} and {team_b}")
        return None
    
    # Check if the expected attributes are present
    if 'home_score' not in attributes or 'away_score' not in attributes:
        logger.warning(f"Missing score attributes in matchup between {team_a} and {team_b}. "
                      f"Available attributes: {attributes}")
        return None
    
    # In our dataset, team_a is always the home team
    # and attributes contain home_score and away_score
    home_points = attributes.get("home_score", 0)
    away_points = attributes.get("away_score", 0)
    
    # Log the comparison details
    result = home_points > away_points
    logger.debug(f"Home team ({team_a}) scored {home_points}, Away team ({team_b}) scored {away_points}. "
                f"Home team won: {result}")
    
    # Return True if home team (team_a) won, False otherwise
    return result


def evaluate_competitor(competitor_class, data_split, competitor_name, competitor_params=None):
    """Train and evaluate a specific competitor type."""
    if competitor_params is None:
        competitor_params = {}
    
    logger.info(f"\n{'='*20} Evaluating {competitor_name} {'='*20}")
    
    # Create the arena with the specified rating system
    arena = LambdaArena(compare_teams, base_competitor=competitor_class)
    
    # Set competitor parameters
    if competitor_class == EloCompetitor:
        arena.set_competitor_class_var("_k_factor", 24)
    
    # Set common parameters
    arena.set_competitor_class_var("_minimum_rating", 0)
    arena.set_competitor_class_var("_initial_rating", 1500)
    
    # Set any additional parameters
    for param, value in competitor_params.items():
        arena.set_competitor_class_var(f"_{param}", value)
    
    # Train the arena on training data
    for i, (team_a, team_b, outcome, timestamp, attributes) in enumerate(data_split.train):
        arena.matchup(team_a, team_b, attributes)
        if i % 500 == 0 and i > 0:
            logger.info(f"Train progress: {i}/{len(data_split.train)} ({i/len(data_split.train):.1%})")
    
    logger.info("Completed train phase.")
    
    # Evaluate on test data, updating ratings as we go
    history = History()
    for i, (team_a, team_b, outcome, timestamp, attributes) in enumerate(data_split.test):
        # Get prediction before updating
        predicted_outcome = arena.expected_score(team_a, team_b)
        
        # Record the bout in history
        history.add_bout(Bout(team_a, team_b, predicted_outcome, outcome, attributes))
        
        # Update the model with this matchup
        arena.matchup(team_a, team_b, attributes)
        
        if i % 500 == 0 and i > 0:
            logger.info(f"Eval progress: {i}/{len(data_split.test)} ({i/len(data_split.test):.1%})")
    
    logger.info("Completed eval phase.")
    
    # Debug: Check some predictions
    logger.info(f"\nSample predictions for {competitor_name}:")
    for i, bout in enumerate(history.bouts[:5]):
        logger.info(f"Game {i+1}: {bout.a} vs {bout.b}")
        logger.info(f"  Predicted outcome: {bout.predicted_outcome:.4f}")
        logger.info(f"  Actual outcome: {bout.outcome}")
        logger.info(f"  Prediction: {'Home win' if bout.predicted_outcome > 0.5 else 'Away win'}")
        logger.info(f"  Correct: {(bout.predicted_outcome > 0.5 and bout.outcome == 1.0) or (bout.predicted_outcome < 0.5 and bout.outcome == 0.0)}")
    
    # Debug: Check distribution of predicted outcomes
    predicted_outcomes = [bout.predicted_outcome for bout in history.bouts]
    logger.info(f"\nPredicted outcome distribution for {competitor_name}:")
    logger.info(f"  Min: {min(predicted_outcomes):.4f}")
    logger.info(f"  Max: {max(predicted_outcomes):.4f}")
    logger.info(f"  Mean: {sum(predicted_outcomes)/len(predicted_outcomes):.4f}")
    logger.info(f"  Values > 0.5: {sum(1 for p in predicted_outcomes if p > 0.5)}/{len(predicted_outcomes)}")
    logger.info(f"  Values < 0.5: {sum(1 for p in predicted_outcomes if p < 0.5)}/{len(predicted_outcomes)}")
    logger.info(f"  Values = 0.5: {sum(1 for p in predicted_outcomes if p == 0.5)}/{len(predicted_outcomes)}")
    
    # Calculate metrics
    tp, fp, tn, fn, do_nothing = history.confusion_matrix()

        # Print confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"True Positives: {tp}")
    logger.info(f"False Positives: {fp}")
    logger.info(f"True Negatives: {tn}")
    logger.info(f"False Negatives: {fn}")
    logger.info(f"Undecided: {do_nothing}")
    
    total = tp + fp + tn + fn + do_nothing
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy with optimized thresholds
    _, thresholds = history.optimize_thresholds()
    tp_opt, fp_opt, tn_opt, fn_opt, do_nothing_opt = history.confusion_matrix(*thresholds)
    
    # Print confusion matrix
    logger.info(f"\nConfusion Matrix (with optimized thresholds):")
    logger.info(f"True Positives: {tp_opt}")
    logger.info(f"False Positives: {fp_opt}")
    logger.info(f"True Negatives: {tn_opt}")
    logger.info(f"False Negatives: {fn_opt}")
    logger.info(f"Undecided: {do_nothing_opt}")

    total_opt = tp_opt + fp_opt + tn_opt + fn_opt + do_nothing_opt
    accuracy_opt = (tp_opt + tn_opt) / total_opt if total_opt > 0 else 0
    
        # Print metrics
    logger.info(f"\nEvaluation results for {competitor_name}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy with optimized thresholds {thresholds}: {accuracy_opt:.4f}")

    # Print top teams
    logger.info(f"\nTop 5 teams according to {competitor_name}:")
    rankings = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get("rating"))[:5]
    for idx, item in enumerate(rankings):
        logger.info(f"{idx+1}. {item.get('competitor')}: {item.get('rating'):.1f}")
        
    return {
        'name': competitor_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy_opt': accuracy_opt,
        'confusion_matrix': {
            'tp': tp_opt,
            'fp': fp_opt,
            'tn': tn_opt,
            'fn': fn_opt,
            'undecided': do_nothing_opt
        }
    }


def plot_results(results):
    """Create bar charts comparing the performance of different rating systems."""
    # Sort results by accuracy (descending)
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]
    accuracy_opt = [r['accuracy_opt'] for r in results]
    
    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison of Rating Systems Performance', fontsize=16)
    
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
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('rating_systems_comparison.png')
    logger.info("Comparison chart saved as 'rating_systems_comparison.png'")
    
    # Create a separate chart for optimized accuracy
    plt.figure(figsize=(10, 6))
    
    # Sort by optimized accuracy for this chart
    sorted_indices = np.argsort(accuracy_opt)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_accuracy_opt = [accuracy_opt[i] for i in sorted_indices]
    
    plt.bar(np.arange(len(sorted_names)), sorted_accuracy_opt, color='purple')
    plt.title('Accuracy with Optimized Thresholds')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Rating System')
    plt.xticks(np.arange(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('optimized_accuracy_comparison.png')
    logger.info("Optimized accuracy chart saved as 'optimized_accuracy_comparison.png'")


def calculate_accuracy_by_prior_bouts(arena, history, thresholds=None):
    """Calculate accuracy based on the number of prior bouts for each competitor.
    
    Args:
        arena: The trained arena containing competitors
        history: The History object with bout results
        thresholds: Optional tuple of (lower_threshold, upper_threshold) for optimized thresholds
        
    Returns:
        dict: A dictionary mapping bout counts to accuracy metrics
    """
    # Default thresholds if not provided
    if thresholds is None:
        lower_threshold, upper_threshold = 0.5, 0.5
    else:
        lower_threshold, upper_threshold = thresholds
    
    # Track the number of bouts for each competitor
    # This should include all bouts from training, not just evaluation
    competitor_bout_counts = {}
    
    # Count all bouts from arena's history (which includes training data)
    for bout in arena.history.bouts:
        competitor_bout_counts[bout.a] = competitor_bout_counts.get(bout.a, 0) + 1
        competitor_bout_counts[bout.b] = competitor_bout_counts.get(bout.b, 0) + 1
    
    # Track accuracy metrics by minimum bout count
    accuracy_by_min_bouts = {}
    
    # Process each bout in the evaluation history
    for bout in history.bouts:
        # Get the current bout count for each competitor (including training data)
        a_count = competitor_bout_counts.get(bout.a, 0)
        b_count = competitor_bout_counts.get(bout.b, 0)
        
        # Determine the minimum bout count between the two competitors
        min_bout_count = min(a_count, b_count)
        
        # Initialize the bucket if it doesn't exist
        if min_bout_count not in accuracy_by_min_bouts:
            accuracy_by_min_bouts[min_bout_count] = {
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'do_nothing': 0, 'total': 0
            }
        
        # Update metrics based on prediction outcome
        if upper_threshold > bout.predicted_outcome > lower_threshold:
            # Undecided - count as do_nothing but still include in total
            accuracy_by_min_bouts[min_bout_count]['do_nothing'] += 1
            accuracy_by_min_bouts[min_bout_count]['total'] += 1
        else:
            # Update the appropriate counter
            if bout.true_positive(upper_threshold):
                accuracy_by_min_bouts[min_bout_count]['tp'] += 1
            elif bout.false_positive(upper_threshold):
                accuracy_by_min_bouts[min_bout_count]['fp'] += 1
            elif bout.true_negative(lower_threshold):
                accuracy_by_min_bouts[min_bout_count]['tn'] += 1
            elif bout.false_negative(lower_threshold):
                accuracy_by_min_bouts[min_bout_count]['fn'] += 1
            
            accuracy_by_min_bouts[min_bout_count]['total'] += 1
        
        # Update bout counts for both competitors for subsequent bouts in evaluation
        competitor_bout_counts[bout.a] = a_count + 1
        competitor_bout_counts[bout.b] = b_count + 1
    
    # Calculate accuracy for each bucket
    for bout_count, metrics in accuracy_by_min_bouts.items():
        if metrics['total'] > 0:
            # Include only true positives and true negatives in the numerator
            # but include all predictions (including do_nothing) in the denominator
            metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / metrics['total']
        else:
            metrics['accuracy'] = 0
    
    return accuracy_by_min_bouts


def plot_accuracy_by_prior_bouts(results_by_prior_bouts, max_bouts=30):
    """Create a line chart showing accuracy vs. prior bout count for each competitor type.
    
    Args:
        results_by_prior_bouts: Dictionary mapping competitor names to their accuracy by prior bout data
        max_bouts: Maximum number of prior bouts to include in the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Define a color map for the different competitor types
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Group data into bins for smoother visualization
    bin_size = 5  # Group every 5 bouts together
    
    # Plot a line for each competitor type
    for i, (competitor_name, bout_data) in enumerate(results_by_prior_bouts.items()):
        # Extract bout counts and corresponding accuracies
        bout_counts = sorted([count for count in bout_data.keys() if count <= max_bouts])
        
        # Skip if no data points
        if not bout_counts:
            continue
        
        # Group data into bins
        binned_data = {}
        for count in bout_counts:
            bin_index = count // bin_size
            if bin_index not in binned_data:
                binned_data[bin_index] = {'accuracy': 0, 'total': 0}
            
            metrics = bout_data[count]
            binned_data[bin_index]['accuracy'] += metrics['accuracy'] * metrics['total']
            binned_data[bin_index]['total'] += metrics['total']
        
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
    plt.ylabel('Accuracy (Including Undecided Predictions)', fontsize=12)
    plt.title('Accuracy vs. Prior Bout Count by Rating System', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set x-axis to show bin centers
    plt.xticks([(i * bin_size) + (bin_size // 2) for i in range((max_bouts // bin_size) + 1)], 
               [f"{i*bin_size}-{(i+1)*bin_size-1}" for i in range((max_bouts // bin_size) + 1)],
               rotation=45)
    
    # Set y-axis limits to match typical accuracy range
    plt.ylim(0, 1)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('accuracy_by_prior_bouts.png')
    logger.info("Accuracy by prior bouts chart saved as 'accuracy_by_prior_bouts.png'")


def main():
    # Create dataset
    logger.info("Loading college football dataset...")
    dataset = CollegeFootballDataset(start_year=2015, end_year=2021)
    
    # Split the dataset into training and test sets (80% train, 20% test)
    logger.info("Splitting dataset into training and test sets...")
    data_split = dataset.time_split(test_ratio=0.5)
    logger.info(f"Split complete: {len(data_split.train)} train games, {len(data_split.test)} test games")
    
    # Define the competitor types to evaluate
    competitors = [
        {'class': EloCompetitor, 'name': 'Elo', 'params': {'k_factor': 24}},
        {'class': GlickoCompetitor, 'name': 'Glicko', 'params': {}},
        {'class': Glicko2Competitor, 'name': 'Glicko-2', 'params': {}},
        {'class': TrueSkillCompetitor, 'name': 'TrueSkill', 'params': {}},
        {'class': ECFCompetitor, 'name': 'ECF', 'params': {}},
        {'class': DWZCompetitor, 'name': 'DWZ', 'params': {}}
    ]
    
    # Evaluate each competitor type
    results = []
    accuracy_by_prior_bouts = {}
    
    for competitor in competitors:
        # Train and evaluate the competitor
        result = evaluate_competitor(
            competitor['class'], 
            data_split, 
            competitor['name'], 
            competitor['params']
        )
        results.append(result)
        
        # Create a new arena for calculating accuracy by prior bouts
        # We need to train it on both training and test data to get accurate bout counts
        arena = LambdaArena(compare_teams, base_competitor=competitor['class'])
        
        # Set competitor parameters
        if competitor['class'] == EloCompetitor:
            arena.set_competitor_class_var("_k_factor", 24)
        
        # Set common parameters
        arena.set_competitor_class_var("_minimum_rating", 0)
        arena.set_competitor_class_var("_initial_rating", 1500)
        
        # Set any additional parameters
        for param, value in competitor.get('params', {}).items():
            arena.set_competitor_class_var(f"_{param}", value)
        
        # First train on the training data
        logger.info(f"Training {competitor['name']} for accuracy by prior bouts analysis...")
        for team_a, team_b, outcome, timestamp, attributes in data_split.train:
            arena.matchup(team_a, team_b, attributes)
        
        # Then evaluate on test data
        test_history = History()
        for team_a, team_b, outcome, timestamp, attributes in data_split.test:
            # Get prediction before updating
            predicted_outcome = arena.expected_score(team_a, team_b)
            # Update the model
            arena.matchup(team_a, team_b, attributes)
            # Record the bout in test history
            test_history.add_bout(Bout(team_a, team_b, predicted_outcome, outcome, attributes))
        
        # Get the optimized thresholds
        _, thresholds = test_history.optimize_thresholds()
        
        # Calculate accuracy by prior bouts
        bout_data = calculate_accuracy_by_prior_bouts(arena, test_history, thresholds)
        accuracy_by_prior_bouts[competitor['name']] = bout_data
        
        # Log some information about the accuracy by prior bouts
        logger.info(f"\nAccuracy by prior bouts for {competitor['name']}:")
        # Show data for selected bout counts to avoid too much output
        display_counts = sorted([c for c in bout_data.keys() if c % 5 == 0 and c <= 50])
        for bout_count in display_counts:
            if bout_count in bout_data:
                metrics = bout_data[bout_count]
                if metrics['total'] > 5:  # Only show buckets with enough samples
                    logger.info(f"  Prior bouts: {bout_count}, Accuracy: {metrics['accuracy']:.4f}, Samples: {metrics['total']}")
    
    # Plot the standard results
    plot_results(results)
    
    # Plot accuracy by prior bouts
    plot_accuracy_by_prior_bouts(accuracy_by_prior_bouts)
    
    # Print overall summary
    logger.info("\n===== OVERALL SUMMARY =====")
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        logger.info(f"{result['name']}: Accuracy={result['accuracy']:.4f}, "
                   f"Optimized Accuracy={result['accuracy_opt']:.4f}, "
                   f"F1 Score={result['f1']:.4f}")


if __name__ == "__main__":
    main()
