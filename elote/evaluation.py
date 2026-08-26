"""Walk-forward evaluation and hyperparameter search for rating systems.

:func:`~elote.benchmark.evaluate_competitor` trains on one split and then predicts a held-out
split with **frozen** ratings. That answers "how well do these ratings survive going stale",
which is a real question but rarely the one being asked. The usual question is how a system
performs in the way it would actually be used: predict the next round of results from
everything that has happened so far, then fold those results in and step forward.

This module provides that protocol, the metrics that can see a system's calibration as well
as its picks, and a grid search over competitor parameters.

Example:
    >>> from elote import EloCompetitor, walk_forward, group_by_period
    >>> periods = group_by_period(rows)                       # doctest: +SKIP
    >>> report = walk_forward(EloCompetitor, periods)         # doctest: +SKIP
    >>> report.accuracy, report.log_loss                      # doctest: +SKIP
"""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from elote.arenas.lambda_arena import LambdaArena
from elote.competitors.base import BaseCompetitor, InvalidParameterException
from elote.datasets.utils import train_arena_with_dataset
from elote.logging import logger

__all__ = ["WalkForwardReport", "TuningResult", "group_by_period", "walk_forward", "tune"]

# A dataset row, as produced by every dataset in :mod:`elote.datasets`.
Row = Tuple[Any, Any, float, Optional[datetime], Optional[Dict[str, Any]]]

_PROBABILITY_EPS = 1e-12


def _validate_competitor_params(competitor_class: Type[BaseCompetitor], names: Iterable[str]) -> None:
    for name in names:
        class_variable = f"_{name}"
        if hasattr(competitor_class, class_variable):
            continue
        default_name = f"default_{name}"
        if hasattr(competitor_class, f"_{default_name}"):
            route = (
                f"use {default_name!r} to tune its class-level default; "
                f"fixed constructor arguments such as {name!r} go in base_competitor_kwargs"
            )
        else:
            route = f"constructor arguments such as {name!r} go in base_competitor_kwargs"
        raise InvalidParameterException(
            f"{competitor_class.__name__} has no class variable {class_variable!r}; "
            f"{route}, not competitor_params"
        )


@dataclass(frozen=True)
class WalkForwardReport:
    """Metrics from a walk-forward run.

    Attributes:
        predictions: Bouts that were both scored and predictable.
        skipped: Bouts skipped because a competitor had not been seen yet.
        draws: Drawn bouts, excluded from every metric below.
        accuracy: Fraction of predictions on the correct side of 0.5.
        log_loss: Mean negative log likelihood. Sees calibration; accuracy does not.
        brier: Mean squared error of the predicted probability.
        by_period: ``(period_index, predictions, accuracy)`` per scored period.
    """

    predictions: int
    skipped: int
    draws: int
    accuracy: float
    log_loss: float
    brier: float
    by_period: Tuple[Tuple[int, int, float], ...] = field(default=())

    def __str__(self) -> str:
        return (
            f"{self.predictions} predictions: accuracy {self.accuracy:.4f}, "
            f"log loss {self.log_loss:.4f}, Brier {self.brier:.4f}"
        )


@dataclass(frozen=True)
class TuningResult:
    """One point of a :func:`tune` grid search."""

    params: Dict[str, Any]
    report: WalkForwardReport

    def __str__(self) -> str:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{rendered}: {self.report}"


def _period_key(when: Optional[datetime]) -> Tuple[int, int]:
    if when is None:
        return (0, 0)
    day = when.date() if isinstance(when, datetime) else when
    if not isinstance(day, date):
        return (0, 0)
    iso = day.isocalendar()
    return (iso[0], iso[1])


def group_by_period(
    rows: Iterable[Row],
    key: Optional[Callable[[Row], Any]] = None,
) -> List[List[Row]]:
    """Group dataset rows into chronologically ordered periods.

    A period is the unit of "predict, then learn": everything inside one is predicted before
    any of it is used for fitting, which is what stops a result informing a bet placed on the
    same afternoon.

    Args:
        rows: Dataset rows, in any order.
        key: Maps a row to its period. Defaults to the ISO calendar week of the row's
            timestamp, which suits weekly league sports. Rows without a usable timestamp are
            collected into one leading period.

    Returns:
        A list of periods, each a list of rows, ordered by period key.
    """
    grouping = key if key is not None else (lambda row: _period_key(row[3]))
    buckets: "OrderedDict[Any, List[Row]]" = OrderedDict()
    for row in rows:
        buckets.setdefault(grouping(row), []).append(row)
    return [buckets[k] for k in sorted(buckets)]


def walk_forward(
    competitor_class: Type[BaseCompetitor],
    periods: Sequence[Sequence[Row]],
    *,
    competitor_params: Optional[Dict[str, Any]] = None,
    base_competitor_kwargs: Optional[Dict[str, Any]] = None,
    comparison_function: Optional[Callable[..., Any]] = None,
    score_keys: Optional[Tuple[str, str]] = None,
    warmup: int = 0,
) -> WalkForwardReport:
    """Predict each period from everything before it, then learn that period.

    Args:
        competitor_class: The rating system to evaluate.
        periods: Ordered periods of dataset rows, as produced by :func:`group_by_period`.
        competitor_params: Existing class-level knobs to set for the duration of the run,
            without the leading underscore. ``{"default_w2": 100.0}`` sets
            ``_default_w2``. Constructor arguments instead belong in
            ``base_competitor_kwargs``.
        base_competitor_kwargs: Constructor keyword arguments for every competitor.
        comparison_function: Arena comparison function. Defaults to one that reports the
            recorded outcome, which is what a dataset row already carries.
        score_keys: ``(a_score_key, b_score_key)`` naming each row's two point scores, for
            the margin-aware systems.
        warmup: Leading periods used for fitting but not scored, so a system is not judged
            on predictions made with no history.

    Returns:
        WalkForwardReport: Metrics over every scored, predictable bout.

    Raises:
        ValueError: If ``warmup`` is negative or not smaller than the number of periods.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if periods and warmup >= len(periods):
        raise ValueError(f"warmup ({warmup}) leaves no periods to score (have {len(periods)})")

    _validate_competitor_params(competitor_class, (competitor_params or {}).keys())

    comparison = comparison_function if comparison_function is not None else (lambda a, b, attributes=None: True)
    arena = LambdaArena(
        comparison,
        base_competitor=competitor_class,
        base_competitor_kwargs=dict(base_competitor_kwargs or {}),
    )

    overrides = {f"_{name}": value for name, value in (competitor_params or {}).items()}
    originals = {name: getattr(competitor_class, name) for name in overrides}

    predictions = skipped = draws = 0
    log_loss_total = brier_total = 0.0
    correct = 0
    by_period: List[Tuple[int, int, float]] = []

    try:
        for name, value in overrides.items():
            setattr(competitor_class, name, value)

        for index, period in enumerate(periods):
            scored = index >= warmup
            period_correct = period_count = 0

            if scored:
                for a, b, outcome, _when, _attributes in period:
                    if outcome is None:
                        continue
                    if outcome == 0.5:
                        draws += 1
                        continue
                    if a not in arena.competitors or b not in arena.competitors:
                        skipped += 1
                        continue
                    probability = arena.expected_score(a, b)
                    probability = min(max(probability, _PROBABILITY_EPS), 1.0 - _PROBABILITY_EPS)
                    actual = 1.0 if outcome > 0.5 else 0.0
                    hit = (probability > 0.5) == (actual > 0.5)
                    predictions += 1
                    correct += hit
                    period_correct += hit
                    period_count += 1
                    log_loss_total -= actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability)
                    brier_total += (probability - actual) ** 2

            train_arena_with_dataset(arena, list(period), score_keys=score_keys)

            if scored and period_count:
                by_period.append((index, period_count, period_correct / period_count))
    finally:
        for name, value in originals.items():
            setattr(competitor_class, name, value)

    if not predictions:
        logger.warning("Walk-forward produced no scored predictions (skipped %d, draws %d).", skipped, draws)
        return WalkForwardReport(0, skipped, draws, float("nan"), float("nan"), float("nan"), ())

    return WalkForwardReport(
        predictions=predictions,
        skipped=skipped,
        draws=draws,
        accuracy=correct / predictions,
        log_loss=log_loss_total / predictions,
        brier=brier_total / predictions,
        by_period=tuple(by_period),
    )


def tune(
    competitor_class: Type[BaseCompetitor],
    param_grid: Dict[str, Sequence[Any]],
    periods: Sequence[Sequence[Row]],
    *,
    metric: str = "log_loss",
    **walk_forward_kwargs: Any,
) -> List[TuningResult]:
    """Grid-search competitor parameters against a walk-forward run.

    ``metric`` defaults to ``log_loss`` deliberately. Accuracy is a rank statistic: it only
    asks which side of 0.5 a prediction landed on, so any parameter that changes confidence
    without changing order is invisible to it. Pythagorean's exponent is exactly such a
    parameter, and tuning it on accuracy reports every value as equally good.

    Args:
        competitor_class: The rating system to tune.
        param_grid: Parameter names (without the leading underscore) to sequences of values.
        periods: Ordered periods, as for :func:`walk_forward`.
        metric: ``"log_loss"``, ``"brier"`` or ``"accuracy"``.
        **walk_forward_kwargs: Forwarded to :func:`walk_forward`.

    Returns:
        Every combination, best first.

    Raises:
        ValueError: If ``metric`` is unknown or ``param_grid`` is empty.
    """
    if metric not in {"log_loss", "brier", "accuracy"}:
        raise ValueError(f"unknown metric {metric!r}; expected 'log_loss', 'brier' or 'accuracy'")
    if not param_grid:
        raise ValueError("param_grid must not be empty")

    _validate_competitor_params(competitor_class, param_grid)

    names = sorted(param_grid)
    results: List[TuningResult] = []
    for values in product(*(param_grid[name] for name in names)):
        params = dict(zip(names, values, strict=True))
        report = walk_forward(competitor_class, periods, competitor_params=params, **walk_forward_kwargs)
        logger.info("tune %s -> %s", params, report)
        results.append(TuningResult(params=params, report=report))

    higher_is_better = metric == "accuracy"
    results.sort(key=lambda r: getattr(r.report, metric), reverse=higher_is_better)
    return results
