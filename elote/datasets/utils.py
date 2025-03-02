"""
Utility functions for using datasets with arenas.

This module provides utility functions for using datasets with arenas for evaluating different rating algorithms.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import datetime

from elote.arenas.base import BaseArena, Bout, History
from elote.datasets.base import DataSplit


def train_arena_with_dataset(
    arena: BaseArena,
    train_data: List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]],
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BaseArena:
    """
    Train an arena with a dataset.

    Args:
        arena: The arena to train
        train_data: List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        batch_size: Number of matchups to process in each batch (for progress reporting)
        progress_callback: Callback function for reporting progress (current, total)

    Returns:
        The trained arena
    """
    # Sort by timestamp if available
    train_data_with_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in train_data if ts is not None]
    train_data_without_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in train_data if ts is None]

    if train_data_with_time:
        # Sort by timestamp
        train_data_with_time.sort(key=lambda x: x[3])
        # Combine sorted data with data without timestamps
        sorted_data = train_data_with_time + train_data_without_time
    else:
        sorted_data = train_data

    # Process in batches if requested
    if batch_size is None:
        batch_size = len(sorted_data)

    total_batches = (len(sorted_data) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sorted_data))
        batch = sorted_data[start_idx:end_idx]

        # Process each matchup
        for a, b, outcome, _, attributes in batch:
            if outcome == 1.0:
                # A wins
                arena.matchup(a, b, attributes=attributes)
            elif outcome == 0.0:
                # B wins
                arena.matchup(b, a, attributes=attributes)
            else:
                # Draw - we need to handle this specially
                # First, get the competitors
                if a not in arena.competitors:
                    arena.competitors[a] = arena.base_competitor(**arena.base_competitor_kwargs)
                if b not in arena.competitors:
                    arena.competitors[b] = arena.base_competitor(**arena.base_competitor_kwargs)

                # Then, record the draw
                arena.competitors[a].tied(arena.competitors[b])

        # Report progress
        if progress_callback is not None:
            progress_callback(end_idx, len(sorted_data))

    return arena


def evaluate_arena_with_dataset(
    arena: BaseArena,
    test_data: List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]],
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> History:
    """
    Evaluate an arena with a dataset.

    Args:
        arena: The arena to evaluate
        test_data: List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        batch_size: Number of matchups to process in each batch (for progress reporting)
        progress_callback: Callback function for reporting progress (current, total)

    Returns:
        History object containing the evaluation results
    """
    # Create a new history object
    history = History()

    # Sort by timestamp if available
    test_data_with_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in test_data if ts is not None]
    test_data_without_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in test_data if ts is None]

    if test_data_with_time:
        # Sort by timestamp
        test_data_with_time.sort(key=lambda x: x[3])
        # Combine sorted data with data without timestamps
        sorted_data = test_data_with_time + test_data_without_time
    else:
        sorted_data = test_data

    # Process in batches if requested
    if batch_size is None:
        batch_size = len(sorted_data)

    total_batches = (len(sorted_data) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sorted_data))
        batch = sorted_data[start_idx:end_idx]

        # Process each matchup
        for a, b, outcome, _, attributes in batch:
            # Skip if either competitor is not in the arena
            if a not in arena.competitors or b not in arena.competitors:
                continue

            # Get the expected outcome
            expected_score = arena.expected_score(a, b)

            # Create a bout object
            bout = Bout(a, b, expected_score, outcome, attributes)

            # Add to history
            history.add_bout(bout)

        # Report progress
        if progress_callback is not None:
            progress_callback(end_idx, len(sorted_data))

    return history


def train_and_evaluate_arena(
    arena: BaseArena,
    data_split: DataSplit,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[BaseArena, History]:
    """
    Train and evaluate an arena with a dataset split.

    Args:
        arena: The arena to train and evaluate
        data_split: DataSplit object containing train and test sets
        batch_size: Number of matchups to process in each batch (for progress reporting)
        progress_callback: Callback function for reporting progress (phase, current, total)

    Returns:
        Tuple of (trained_arena, history)
    """
    # Train the arena
    if progress_callback:

        def train_progress(current, total):
            return progress_callback("train", current, total)
    else:
        train_progress = None

    trained_arena = train_arena_with_dataset(
        arena, data_split.train, batch_size=batch_size, progress_callback=train_progress
    )

    # Evaluate the arena
    if progress_callback:

        def eval_progress(current, total):
            return progress_callback("eval", current, total)
    else:
        eval_progress = None

    history = evaluate_arena_with_dataset(
        trained_arena, data_split.test, batch_size=batch_size, progress_callback=eval_progress
    )

    return trained_arena, history
