"""
Example of using datasets with different rating algorithms.

This example demonstrates how to use the datasets module to evaluate different rating algorithms.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt

from elote import (
    LambdaArena,
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    SyntheticDataset,
    ChessDataset,
    CollegeFootballDataset,
    train_and_evaluate_arena,
)


def progress_callback(phase, current, total):
    """Callback function for reporting progress."""
    if current == 0:
        print(f"\nStarting {phase} phase...")
    elif current == total:
        print(f"\nCompleted {phase} phase.")


def evaluate_algorithms_on_dataset(dataset_name, dataset, test_ratio=0.2, seed=42):
    """
    Evaluate different rating algorithms on a dataset.

    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
    """
    print(f"\n=== Evaluating algorithms on {dataset_name} dataset ===")

    # Split the dataset into train and test sets
    print(f"Splitting dataset with test_ratio={test_ratio}...")
    data_split = dataset.time_split(test_ratio=test_ratio)
    print(f"Split complete: {len(data_split.train)} train samples, {len(data_split.test)} test samples")

    # Define the algorithms to evaluate
    algorithms = [
        ("Elo", EloCompetitor, {"initial_rating": 1500}),
        ("Glicko", GlickoCompetitor, {"initial_rating": 1500}),
        ("Glicko-2", Glicko2Competitor, {"initial_rating": 1500}),
        ("TrueSkill", TrueSkillCompetitor, {}),
    ]

    # Evaluate each algorithm
    results = []

    for algo_name, competitor_class, competitor_kwargs in algorithms:
        print(f"\nEvaluating {algo_name}...")
        start_time = time.time()

        # Create an arena with the algorithm
        arena = LambdaArena(
            lambda a, b, attributes=None: True,  # Dummy function, not used in this example
            base_competitor=competitor_class,
            base_competitor_kwargs=competitor_kwargs,
        )

        # Train and evaluate the arena
        _, history = train_and_evaluate_arena(
            arena,
            data_split,
            batch_size=1000,
            progress_callback=progress_callback,
        )

        # Calculate metrics
        metrics = history.calculate_metrics()
        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Store results
        results.append(
            {
                "Algorithm": algo_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Time (s)": elapsed_time,
            }
        )

        print(f"{algo_name} evaluation complete in {elapsed_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print results table
    print("\nResults:")
    print(results_df.to_string(index=False))

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot accuracy, precision, recall, F1
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, 1)
        plt.bar([x + i * 0.2 for x in range(len(algorithms))], results_df[metric], width=0.2, label=metric)

    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.title(f"Performance Metrics on {dataset_name} Dataset")
    plt.xticks([i + 0.3 for i in range(len(algorithms))], results_df["Algorithm"])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot time
    plt.subplot(1, 2, 2)
    plt.bar(results_df["Algorithm"], results_df["Time (s)"])
    plt.xlabel("Algorithm")
    plt.ylabel("Time (s)")
    plt.title(f"Execution Time on {dataset_name} Dataset")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_results.png")
    plt.close()


def main():
    """Main function."""
    # Evaluate on synthetic dataset
    print("Generating synthetic dataset...")
    synthetic_dataset = SyntheticDataset(
        num_competitors=100,
        num_matchups=5000,
        skill_distribution="normal",
        skill_mean=1500,
        skill_std=300,
        noise_std=100,
        draw_probability=0.1,
        time_span_days=365,
        seed=42,
    )
    evaluate_algorithms_on_dataset("Synthetic", synthetic_dataset, test_ratio=0.2, seed=42)

    # Evaluate on chess dataset
    print("\nLoading chess dataset...")
    chess_dataset = ChessDataset(max_games=5000, year=2013, month=1)
    evaluate_algorithms_on_dataset("Chess", chess_dataset, test_ratio=0.2, seed=42)

    # Evaluate on college football dataset
    print("\nLoading college football dataset...")
    football_dataset = CollegeFootballDataset(start_year=2015, end_year=2022)
    evaluate_algorithms_on_dataset("College Football", football_dataset, test_ratio=0.2, seed=42)


if __name__ == "__main__":
    main()
