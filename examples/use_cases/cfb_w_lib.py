"""
Example of using the CollegeFootballDataset to train and evaluate multiple rating systems.

This example demonstrates how to use the CollegeFootballDataset to train and evaluate
different rating systems for college football teams and compare their performance.
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
    ColleyMatrixCompetitor,
    CollegeFootballDataset,
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
logger = logging.getLogger("cfb_example")


def progress_callback(phase, current, total):
    """Callback function to report progress during training and evaluation."""
    if current == 0:
        logger.info(f"Starting {phase} phase...")
    elif current == total:
        logger.info(f"Completed {phase} phase.")
    elif current % 500 == 0:
        logger.info(f"{phase.capitalize()} progress: {current}/{total} ({current / total:.1%})")


def compare_teams(team_a, team_b, attributes=None):
    """Compare two teams based on their scores."""
    logger.debug(f"compare_teams called with team_a={team_a}, team_b={team_b}, attributes={attributes}")

    if attributes is None:
        logger.warning(f"No attributes provided for matchup between {team_a} and {team_b}")
        return None

    # Check if the expected attributes are present
    if "home_score" not in attributes or "away_score" not in attributes:
        logger.warning(f"Missing score attributes for matchup between {team_a} and {team_b}")
        return None

    # In our dataset, team_a is always the home team
    # and attributes contain home_score and away_score
    home_points = attributes.get("home_score", 0)
    away_points = attributes.get("away_score", 0)

    # Determine the winner
    if home_points > away_points:
        return True  # team_a (home) wins
    elif away_points > home_points:
        return False  # team_b (away) wins
    else:
        return None  # tie


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
        logger.info(f"Game {i + 1}: {bout.a} vs {bout.b}")
        logger.info(f"  Predicted outcome: {bout.predicted_outcome:.4f}")
        logger.info(f"  Actual outcome: {bout.outcome}")
        logger.info(f"  Correct: {bout.predicted_winner() == bout.actual_winner()}")


def log_prediction_distribution(competitor_name, history):
    """Log distribution of predicted outcomes for a competitor."""
    if not history or not hasattr(history, "bouts") or not history.bouts:
        return

    predicted_outcomes = [bout.predicted_outcome for bout in history.bouts]
    logger.info(f"\nPredicted outcome distribution for {competitor_name}:")
    logger.info(f"  Min: {min(predicted_outcomes):.4f}")
    logger.info(f"  Max: {max(predicted_outcomes):.4f}")
    logger.info(f"  Mean: {sum(predicted_outcomes) / len(predicted_outcomes):.4f}")
    logger.info(f"  Values > 0.75: {sum(1 for p in predicted_outcomes if p > 0.75)}/{len(predicted_outcomes)}")
    logger.info(f"  Values < 0.25: {sum(1 for p in predicted_outcomes if p < 0.25)}/{len(predicted_outcomes)}")
    logger.info(
        f"  0.25 <= Values <= 0.75: {sum(1 for p in predicted_outcomes if 0.25 <= p <= 0.75)}/{len(predicted_outcomes)}"
    )


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


def log_top_teams(competitor_name, top_teams):
    """Log top teams according to a competitor."""
    logger.info(f"\nTop 5 teams according to {competitor_name}:")
    for idx, item in enumerate(top_teams[:5]):
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
    image_dir = "images/cfb"
    ensure_dir_exists(image_dir)

    # Create dataset
    logger.info("Loading college football dataset...")
    dataset = CollegeFootballDataset(start_year=2015, end_year=2022)

    # Split the dataset into training and test sets
    logger.info("Splitting dataset into training and test sets...")
    data_split = dataset.time_split(test_ratio=0.3)
    logger.info(f"Split complete: {len(data_split.train)} train games, {len(data_split.test)} test games")

    # Define the competitor types to evaluate
    competitors = [
        {"class": EloCompetitor, "name": "Elo", "params": {"k_factor": 32, "initial_rating": 1500}},
        {"class": GlickoCompetitor, "name": "Glicko", "params": {"initial_rating": 1500}},
        {"class": Glicko2Competitor, "name": "Glicko-2", "params": {"initial_rating": 1500}},
        {"class": TrueSkillCompetitor, "name": "TrueSkill", "params": {"initial_rating": 1500}},
        {"class": ECFCompetitor, "name": "ECF", "params": {"initial_rating": 1500}},
        {"class": DWZCompetitor, "name": "DWZ", "params": {"initial_rating": 1500}},
        {"class": ColleyMatrixCompetitor, "name": "Colley Matrix", "params": {"initial_rating": 0.5}},
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
            comparison_function=compare_teams,
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
            log_top_teams(competitor["name"], result["top_teams"])

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
