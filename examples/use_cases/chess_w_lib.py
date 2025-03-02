"""
Example of using the ChessDataset to train and evaluate multiple rating systems.

This example demonstrates how to use the ChessDataset to train and evaluate
different rating systems for chess players and compare their performance.
"""

import logging
import os
from elote import (
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    ChessDataset,
)
from elote.benchmark import evaluate_competitor
from elote.visualization import (
    plot_rating_system_comparison,
    plot_accuracy_by_prior_bouts,
    plot_optimized_accuracy_comparison,
    plot_calibration_curve,
    plot_calibration_comparison,
)


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chess_example")


def progress_callback(phase, current, total):
    """Callback function to report progress during training and evaluation."""
    if current == 0:
        logger.info(f"Starting {phase} phase...")
    elif current == total:
        logger.info(f"Completed {phase} phase.")
    elif current % 500 == 0:
        logger.info(f"{phase.capitalize()} progress: {current}/{total} ({current / total:.1%})")


def compare_players(player_a, player_b, attributes=None):
    """Compare two chess players based on game outcome.
    
    In chess dataset:
    - player_a is always the white player
    - player_b is always the black player
    - The outcome is stored in the dataset tuple during training/evaluation
      - 1.0 means white (player_a) won
      - 0.0 means black (player_b) won
      - 0.5 means draw
    
    During training and evaluation, this function is called by the benchmark system
    with the outcome already determined from the dataset. We just need to return
    the appropriate boolean value based on the outcome.
    
    When called from train_arena_with_dataset:
      - If outcome is 1.0 (white wins), it calls arena.matchup(a, b)
      - If outcome is 0.0 (black wins), it calls arena.matchup(b, a)
      - If outcome is 0.5 (draw), it handles it specially
    
    When called from evaluate_arena_with_dataset:
      - It gets the expected score from arena.expected_score(a, b)
      - It creates a Bout object with the expected and actual outcomes
    
    In both cases, we don't need to determine the outcome here - it's already
    determined from the dataset. We just need to return the appropriate boolean.
    """
    logger.debug(f"compare_players called with player_a={player_a}, player_b={player_b}, attributes={attributes}")
    
    # In the benchmark system, this function is called in a way where we don't
    # actually need to determine the outcome - it's already determined from the dataset.
    # The function is only used to determine the winner in arena.matchup().
    
    # Since player_a is white and player_b is black, and in the dataset:
    # - 1.0 means white wins
    # - 0.0 means black wins
    # - 0.5 means draw
    
    # For this example, we'll just return True to indicate player_a (white) wins.
    # This is just a placeholder - in a real system, we would determine the winner
    # based on ratings or other factors.
    
    # Return True to indicate player_a (white) wins
    return True


def ensure_dir_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def log_sample_predictions(competitor_name, history):
    """Log sample predictions for a competitor."""
    if not history or not hasattr(history, "bouts") or not history.bouts:
        return

    logger.info(f"\nSample predictions for {competitor_name}:")
    for i, bout in enumerate(history.bouts[:5]):
        logger.info(f"Game {i + 1}: {bout.a} (White) vs {bout.b} (Black)")
        logger.info(f"  Predicted outcome: {bout.predicted_outcome:.4f}")
        logger.info(f"  Actual outcome: {bout.outcome}")
        
        # Interpret the prediction
        if bout.predicted_outcome > 0.66:
            prediction = "White win"
        elif bout.predicted_outcome < 0.33:
            prediction = "Black win"
        else:
            prediction = "Draw"
            
        logger.info(f"  Prediction: {prediction}")
        
        # Determine if prediction was correct
        correct = False
        if bout.predicted_outcome > 0.66 and bout.outcome == 1.0:
            correct = True  # Predicted white win, white won
        elif bout.predicted_outcome < 0.33 and bout.outcome == 0.0:
            correct = True  # Predicted black win, black won
        elif 0.33 <= bout.predicted_outcome <= 0.66 and bout.outcome == 0.5:
            correct = True  # Predicted draw, was a draw
            
        logger.info(f"  Correct: {correct}")


def log_prediction_distribution(competitor_name, history):
    """Log distribution of predicted outcomes for a competitor."""
    if not history or not hasattr(history, "bouts") or not history.bouts:
        return

    predicted_outcomes = [bout.predicted_outcome for bout in history.bouts]
    logger.info(f"\nPredicted outcome distribution for {competitor_name}:")
    logger.info(f"  Min: {min(predicted_outcomes):.4f}")
    logger.info(f"  Max: {max(predicted_outcomes):.4f}")
    logger.info(f"  Mean: {sum(predicted_outcomes) / len(predicted_outcomes):.4f}")
    logger.info(f"  Values > 0.66 (White win): {sum(1 for p in predicted_outcomes if p > 0.66)}/{len(predicted_outcomes)}")
    logger.info(f"  Values < 0.33 (Black win): {sum(1 for p in predicted_outcomes if p < 0.33)}/{len(predicted_outcomes)}")
    logger.info(f"  0.33 <= Values <= 0.66 (Draw): {sum(1 for p in predicted_outcomes if 0.33 <= p <= 0.66)}/{len(predicted_outcomes)}")


def log_confusion_matrix(confusion_matrix):
    """Log confusion matrix metrics."""
    logger.info("\nConfusion Matrix:")
    logger.info(f"True Positives: {confusion_matrix.get('tp', 0)}")
    logger.info(f"False Positives: {confusion_matrix.get('fp', 0)}")
    logger.info(f"True Negatives: {confusion_matrix.get('tn', 0)}")
    logger.info(f"False Negatives: {confusion_matrix.get('fn', 0)}")
    logger.info(f"Undecided: {confusion_matrix.get('undecided', 0)}")


def log_metrics(competitor_name, result):
    """Log evaluation metrics for a competitor."""
    logger.info(f"\nEvaluation results for {competitor_name}:")
    logger.info(f"Accuracy: {result.get('accuracy', 0):.4f}")
    logger.info(f"Precision: {result.get('precision', 0):.4f}")
    logger.info(f"Recall: {result.get('recall', 0):.4f}")
    logger.info(f"F1 Score: {result.get('f1', 0):.4f}")

    if "accuracy_opt" in result and "optimized_thresholds" in result:
        logger.info(
            f"Accuracy with optimized thresholds {result['optimized_thresholds']}: {result['accuracy_opt']:.4f}"
        )


def log_top_players(competitor_name, top_players):
    """Log top players according to a competitor."""
    logger.info(f"\nTop 5 players according to {competitor_name}:")
    for idx, item in enumerate(top_players[:5]):
        logger.info(f"{idx + 1}. {item.get('competitor')}: {item.get('rating'):.1f}")


def generate_visualizations(results, histories_and_arenas, image_dir):
    """Generate and save visualization charts."""
    # Prepare data for the optimized accuracy chart
    for result in results:
        if "accuracy_opt" in result:
            result["optimized_accuracy"] = result["accuracy_opt"]

    # Generate and save charts
    charts = [
        {
            "name": "rating_systems_comparison",
            "title": "Comparison of Rating Systems Performance",
            "function": plot_rating_system_comparison,
            "data": results,
            "params": {},
        },
        {
            "name": "optimized_accuracy_comparison",
            "title": "Optimized Accuracy with Optimized Thresholds",
            "function": plot_optimized_accuracy_comparison,
            "data": results,
            "params": {},
        },
        {
            "name": "accuracy_by_prior_bouts",
            "title": "Accuracy vs. Prior Bout Count by Rating System",
            "function": plot_accuracy_by_prior_bouts,
            "data": {
                name: data["history"].accuracy_by_prior_bouts(data["arena"], data["thresholds"], bin_size=25)
                for name, data in histories_and_arenas.items()
            },
            "params": {},
        },
        {
            "name": "calibration_curves",
            "title": "Calibration Curves for Rating Systems",
            "function": plot_calibration_curve,
            "data": {name: data["history"] for name, data in histories_and_arenas.items()},
            "params": {"n_bins": 10},
        },
        {
            "name": "calibration_comparison",
            "title": "Calibration Comparison for Rating Systems",
            "function": plot_calibration_comparison,
            "data": {name: data["history"] for name, data in histories_and_arenas.items()},
            "params": {"n_bins": 10},
        },
    ]

    for chart in charts:
        logger.info(f"Generating {chart['name']} chart...")
        chart_path = os.path.join(image_dir, f"{chart['name']}.png")
        chart["function"](chart["data"], save_path=chart_path, title=chart["title"], **chart["params"])
        logger.info(f"{chart['name']} chart saved as '{chart_path}'")


def main():
    # Create image directory
    image_dir = "images/chess"
    ensure_dir_exists(image_dir)

    # Create dataset
    logger.info("Loading chess dataset...")
    dataset = ChessDataset(max_games=10000, year=2013, month=1)

    # Split the dataset into training and test sets
    logger.info("Splitting dataset into training and test sets...")
    data_split = dataset.time_split(test_ratio=0.3)
    logger.info(f"Split complete: {len(data_split.train)} train games, {len(data_split.test)} test games")

    # Define the competitor types to evaluate with proper parameter names
    # Note: evaluate_competitor adds an underscore prefix to parameter names
    competitors = [
        {"class": EloCompetitor, "name": "Elo", "params": {"k_factor": 32, "initial_rating": 1500}},  # Standard chess k-factor
        {"class": GlickoCompetitor, "name": "Glicko", "params": {"initial_rating": 1500}},
        {"class": Glicko2Competitor, "name": "Glicko-2", "params": {"initial_rating": 1500}},
        {"class": TrueSkillCompetitor, "name": "TrueSkill", "params": {"initial_rating": 1500}},
        {"class": ECFCompetitor, "name": "ECF", "params": {"initial_rating": 1500}},  # English Chess Federation rating
        {"class": DWZCompetitor, "name": "DWZ", "params": {"initial_rating": 1500}},  # German Chess Federation rating
    ]

    # Evaluate each competitor type
    results = []
    histories_and_arenas = {}

    for competitor in competitors:
        # Use the library's evaluate_competitor function
        logger.info(f"\n{'=' * 20} Evaluating {competitor['name']} {'=' * 20}")
        result = evaluate_competitor(
            competitor_class=competitor["class"],
            data_split=data_split,
            comparison_function=compare_players,
            competitor_name=competitor["name"],
            competitor_params=competitor["params"],
            batch_size=500,
            progress_callback=progress_callback,
            optimize_thresholds=True,
        )
        results.append(result)

        # Log various aspects of the results
        history = result.get("history", None)
        log_sample_predictions(competitor["name"], history)
        log_prediction_distribution(competitor["name"], history)

        if "confusion_matrix" in result:
            log_confusion_matrix(result["confusion_matrix"])

        log_metrics(competitor["name"], result)

        if "top_teams" in result:
            log_top_players(competitor["name"], result["top_teams"])

        # Store history and arena for later visualization
        if "history" in result and "arena" in result:
            histories_and_arenas[competitor["name"]] = {
                "history": result["history"],
                "arena": result["arena"],
                "thresholds": result.get("optimized_thresholds", (0.5, 0.5)),
            }

    # Generate all visualizations
    generate_visualizations(results, histories_and_arenas, image_dir)

    # Print overall summary
    logger.info("\n===== OVERALL SUMMARY =====")
    for result in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
        logger.info(
            f"{result['name']}: Accuracy={result.get('accuracy', 0):.4f}, "
            f"Optimized Accuracy={result.get('accuracy_opt', 0):.4f}, "
            f"F1 Score={result.get('f1', 0):.4f}"
        )


if __name__ == "__main__":
    main() 