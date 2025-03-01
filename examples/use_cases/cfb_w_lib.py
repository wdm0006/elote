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
    
    # Train and evaluate the arena
    trained_arena, history = train_and_evaluate_arena(
        arena,
        data_split,
        batch_size=500,
        progress_callback=progress_callback
    )
    
    # Calculate metrics
    tp, fp, tn, fn, do_nothing = history.confusion_matrix()
    total = tp + fp + tn + fn + do_nothing
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy with optimized thresholds
    _, thresholds = history.random_search(trials=1000)
    tp_opt, fp_opt, tn_opt, fn_opt, do_nothing_opt = history.confusion_matrix(*thresholds)
    total_opt = tp_opt + fp_opt + tn_opt + fn_opt + do_nothing_opt
    accuracy_opt = (tp_opt + tn_opt) / total_opt if total_opt > 0 else 0
    
    # Print top teams
    logger.info(f"\nTop 5 teams according to {competitor_name}:")
    rankings = sorted(trained_arena.leaderboard(), reverse=True, key=lambda x: x.get("rating"))[:5]
    for idx, item in enumerate(rankings):
        logger.info(f"{idx+1}. {item.get('competitor')}: {item.get('rating'):.1f}")
    
    # Print metrics
    logger.info(f"\nEvaluation results for {competitor_name}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy with optimized thresholds {thresholds}: {accuracy_opt:.4f}")
    
    # Print confusion matrix
    logger.info(f"\nConfusion Matrix (with optimized thresholds):")
    logger.info(f"True Positives: {tp_opt}")
    logger.info(f"False Positives: {fp_opt}")
    logger.info(f"True Negatives: {tn_opt}")
    logger.info(f"False Negatives: {fn_opt}")
    logger.info(f"Undecided: {do_nothing_opt}")
    
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
    axes[0, 0].bar(names, accuracy, color='skyblue')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot precision
    axes[0, 1].bar(names, precision, color='lightgreen')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot recall
    axes[1, 0].bar(names, recall, color='salmon')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Rating System')
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Plot F1 score
    axes[1, 1].bar(names, f1, color='gold')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel('Rating System')
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('rating_systems_comparison.png')
    logger.info("Comparison chart saved as 'rating_systems_comparison.png'")
    
    # Create a separate chart for optimized accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracy_opt, color='purple')
    plt.title('Accuracy with Optimized Thresholds')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Rating System')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('optimized_accuracy_comparison.png')
    logger.info("Optimized accuracy chart saved as 'optimized_accuracy_comparison.png'")


def main():
    # Create dataset
    logger.info("Loading college football dataset...")
    dataset = CollegeFootballDataset(start_year=2015, end_year=2021)
    
    # Split the dataset into training and test sets (80% train, 20% test)
    logger.info("Splitting dataset into training and test sets...")
    data_split = dataset.time_split(test_ratio=0.2)
    logger.info(f"Split complete: {len(data_split.train)} train games, {len(data_split.test)} test games")
    
    # Log some sample data to verify structure
    logger.debug("Sample training data:")
    for i, (team_a, team_b, outcome, timestamp, attributes) in enumerate(data_split.train[:5]):
        logger.debug(f"Game {i+1}: {team_a} vs {team_b}, outcome={outcome}, attributes={attributes}")
    
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
    for competitor in competitors:
        result = evaluate_competitor(
            competitor['class'], 
            data_split, 
            competitor['name'], 
            competitor['params']
        )
        results.append(result)
    
    # Plot the results
    plot_results(results)
    
    # Print overall summary
    logger.info("\n===== OVERALL SUMMARY =====")
    for result in sorted(results, key=lambda x: x['accuracy_opt'], reverse=True):
        logger.info(f"{result['name']}: Accuracy={result['accuracy']:.4f}, "
                   f"Optimized Accuracy={result['accuracy_opt']:.4f}, "
                   f"F1 Score={result['f1']:.4f}")


if __name__ == "__main__":
    main()
