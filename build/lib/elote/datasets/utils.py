"""
Utility functions for using datasets with arenas.

This module provides utility functions for using datasets with arenas for evaluating different rating algorithms.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import datetime
import math
import numbers

from elote.arenas.base import BaseArena, Bout, History
from elote.datasets.base import DataSplit
from elote.logging import logger


def _scores_from_attributes(
    attributes: Optional[Dict[str, Any]], score_keys: Tuple[str, str]
) -> Optional[Tuple[float, float]]:
    """Read a row's two point scores out of its attributes, in ``(a, b)`` order.

    Real feeds have gaps, so anything short of two usable numbers means "no scores for this
    row" rather than an error: a missing attributes mapping, a missing key, or a value that
    is not a finite real number. Payloads that are well-formed but wrong -- negative, or
    disagreeing with the recorded outcome -- are still rejected downstream by
    :func:`elote.competitors.base.validate_scores`.

    Args:
        attributes: The row's attributes mapping, which may be None.
        score_keys: The attribute keys holding the two scores, as ``(a_key, b_key)``.

    Returns:
        tuple of float, or None: The two scores in ``(a, b)`` order, or None.
    """
    if not attributes:
        return None

    values: List[float] = []
    for key in score_keys:
        value = attributes.get(key)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        values.append(as_float)

    return values[0], values[1]


def train_arena_with_dataset(
    arena: BaseArena,
    train_data: List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]],
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    score_keys: Optional[Tuple[str, str]] = None,
) -> BaseArena:
    """
    Train an arena with a dataset.

    Args:
        arena: The arena to train
        train_data: List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        batch_size: Number of matchups to process in each batch (for progress reporting)
        progress_callback: Callback function for reporting progress (current, total)
        score_keys: Optional pair of attribute keys naming each row's two point scores, in
            ``(a_score_key, b_score_key)`` order -- for example
            ``("home_score", "away_score")`` for
            :class:`~elote.datasets.football.CollegeFootballDataset`. When supplied, rows
            carrying both scores are trained with an explicit outcome and the score payload,
            so margin-aware systems (Massey, Keener) see the real margins instead of unit
            ones. Rows without usable scores are trained exactly as they are with the
            default of None, which leaves behaviour unchanged.

    Returns:
        The trained arena
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

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

    if not sorted_data:
        return arena

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
            scores = None if score_keys is None else _scores_from_attributes(attributes, score_keys)

            if outcome == 1.0:
                # A wins
                if scores is None:
                    arena.matchup(a, b, attributes=attributes)
                else:
                    arena.matchup(a, b, attributes=attributes, outcome=1.0, scores=scores)
            elif outcome == 0.0:
                # B wins. The call is reversed so b is the winner, so the score pair has to
                # be reversed with it to stay in the arena's caller order.
                if scores is None:
                    arena.matchup(b, a, attributes=attributes)
                else:
                    arena.matchup(b, a, attributes=attributes, outcome=1.0, scores=(scores[1], scores[0]))
            else:
                # Draw
                arena.matchup(a, b, attributes=attributes, outcome=0.5, scores=scores)

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
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

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

    if not sorted_data:
        return history

    # Process in batches if requested
    if batch_size is None:
        batch_size = len(sorted_data)

    total_batches = (len(sorted_data) + batch_size - 1) // batch_size
    skipped_bouts = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sorted_data))
        batch = sorted_data[start_idx:end_idx]

        # Process each matchup
        for a, b, outcome, _, attributes in batch:
            # Skip if either competitor is not in the arena
            if a not in arena.competitors or b not in arena.competitors:
                skipped_bouts += 1
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

    if skipped_bouts > 0:
        logger.warning(
            "Skipped %d/%d evaluation bouts: competitor not found in training history.",
            skipped_bouts,
            len(sorted_data),
        )
    if not history.bouts:
        logger.warning(
            "Evaluation history is empty after evaluating %d test bouts; metrics are not meaningful.",
            len(sorted_data),
        )

    return history


def train_and_evaluate_arena(
    arena: BaseArena,
    data_split: DataSplit,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    *,
    score_keys: Optional[Tuple[str, str]] = None,
) -> Tuple[BaseArena, History]:
    """
    Train and evaluate an arena with a dataset split.

    Args:
        arena: The arena to train and evaluate
        data_split: DataSplit object containing train and test sets
        batch_size: Number of matchups to process in each batch (for progress reporting)
        progress_callback: Callback function for reporting progress (phase, current, total)
        score_keys: Optional pair of attribute keys naming each row's two point scores,
            forwarded to :func:`train_arena_with_dataset`. Evaluation only reads expected
            scores, so it has no use for them.

    Returns:
        Tuple of (trained_arena, history)
    """
    # Train the arena
    if progress_callback:

        def train_progress(current: int, total: int) -> None:
            return progress_callback("train", current, total)
    else:
        train_progress = None

    trained_arena = train_arena_with_dataset(
        arena, data_split.train, batch_size=batch_size, progress_callback=train_progress, score_keys=score_keys
    )

    # Evaluate the arena
    if progress_callback:

        def eval_progress(current: int, total: int) -> None:
            return progress_callback("eval", current, total)
    else:
        eval_progress = None

    history = evaluate_arena_with_dataset(
        trained_arena, data_split.test, batch_size=batch_size, progress_callback=eval_progress
    )

    return trained_arena, history
