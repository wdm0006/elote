"""
Benchmarking utilities for elote.

This module provides functions for benchmarking and comparing different rating systems
using consistent evaluation metrics and visualization.
"""

import logging
from typing import Dict, List, Type, Optional, Any, Callable
import time

from elote.arenas.lambda_arena import LambdaArena
from elote.competitors.base import BaseCompetitor
from elote.datasets.base import DataSplit
from elote.datasets.utils import train_arena_with_dataset, evaluate_arena_with_dataset


logger = logging.getLogger(__name__)


def evaluate_competitor(
    competitor_class: Type[BaseCompetitor],
    data_split: DataSplit,
    comparison_function: Callable,
    competitor_name: str = None,
    competitor_params: Dict[str, Any] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    optimize_thresholds: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a specific competitor type.

    Args:
        competitor_class: The competitor class to evaluate
        data_split: DataSplit object containing train and test sets
        comparison_function: Function to compare competitors (used in LambdaArena)
        competitor_name: Name for the competitor (defaults to class name if None)
        competitor_params: Dictionary of parameters to set on the competitor
        batch_size: Number of matchups to process in each batch
        progress_callback: Callback function for reporting progress
        optimize_thresholds: Whether to optimize prediction thresholds

    Returns:
        Dictionary containing evaluation results
    """
    if competitor_params is None:
        competitor_params = {}

    if competitor_name is None:
        competitor_name = competitor_class.__name__

    logger.info(f"Evaluating {competitor_name}...")

    # Create the arena with the specified rating system
    arena = LambdaArena(comparison_function, base_competitor=competitor_class)

    # Set common parameters
    arena.set_competitor_class_var("_minimum_rating", 0)
    arena.set_competitor_class_var("_initial_rating", 1500)

    # Set any additional parameters
    for param, value in competitor_params.items():
        arena.set_competitor_class_var(f"_{param}", value)

    # Train the arena on training data
    start_time = time.time()

    if progress_callback:

        def train_progress(current, total):
            return progress_callback("train", current, total)
    else:
        train_progress = None

    train_arena_with_dataset(arena, data_split.train, batch_size=batch_size, progress_callback=train_progress)

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate on test data
    start_time = time.time()

    if progress_callback:

        def eval_progress(current, total):
            return progress_callback("eval", current, total)
    else:
        eval_progress = None

    history = evaluate_arena_with_dataset(
        arena, data_split.test, batch_size=batch_size, progress_callback=eval_progress
    )

    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    # Calculate metrics with default thresholds
    metrics = history.calculate_metrics()

    # Optimize thresholds if requested
    if optimize_thresholds:
        best_accuracy, best_thresholds = history.optimize_thresholds()
        optimized_metrics = history.calculate_metrics(*best_thresholds)
        metrics["accuracy_opt"] = optimized_metrics["accuracy"]
        metrics["optimized_thresholds"] = best_thresholds

    # Add competitor info and timing
    metrics["name"] = competitor_name
    metrics["train_time"] = train_time
    metrics["eval_time"] = eval_time

    # Get top teams
    top_teams = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get("rating"))[:5]
    metrics["top_teams"] = top_teams

    # Add history and arena to metrics
    metrics["history"] = history
    metrics["arena"] = arena

    # Calculate accuracy by prior bouts if optimize_thresholds is True
    if optimize_thresholds:
        thresholds = best_thresholds
        bout_data = history.accuracy_by_prior_bouts(arena, thresholds)
        metrics["accuracy_by_prior_bouts"] = bout_data

    # Log results
    logger.info(f"Results for {competitor_name}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")

    if optimize_thresholds:
        logger.info(f"  Optimized Accuracy: {metrics['accuracy_opt']:.4f}")
        logger.info(f"  Optimized Thresholds: {metrics['optimized_thresholds']}")

    return metrics


def benchmark_competitors(
    competitor_configs: List[Dict[str, Any]],
    data_split: DataSplit,
    comparison_function: Callable,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    optimize_thresholds: bool = True,
) -> List[Dict[str, Any]]:
    """
    Benchmark multiple competitor types against each other.

    Args:
        competitor_configs: List of dictionaries with keys 'class', 'name', and 'params'
        data_split: DataSplit object containing train and test sets
        comparison_function: Function to compare competitors (used in LambdaArena)
        batch_size: Number of matchups to process in each batch
        progress_callback: Callback function for reporting progress
        optimize_thresholds: Whether to optimize prediction thresholds

    Returns:
        List of dictionaries containing evaluation results for each competitor
    """
    results = []

    for config in competitor_configs:
        competitor_class = config["class"]
        competitor_name = config.get("name", competitor_class.__name__)
        competitor_params = config.get("params", {})

        result = evaluate_competitor(
            competitor_class=competitor_class,
            data_split=data_split,
            comparison_function=comparison_function,
            competitor_name=competitor_name,
            competitor_params=competitor_params,
            batch_size=batch_size,
            progress_callback=progress_callback,
            optimize_thresholds=optimize_thresholds,
        )

        results.append(result)

    # Print overall summary
    logger.info("\n===== OVERALL SUMMARY =====")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        summary = f"{result['name']}: Accuracy={result['accuracy']:.4f}"

        if optimize_thresholds:
            summary += f", Optimized Accuracy={result['accuracy_opt']:.4f}"

        summary += f", F1 Score={result['f1']:.4f}"
        logger.info(summary)

    return results
