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
from elote.datasets.utils import _scores_from_attributes, train_arena_with_dataset
from elote.logging import logger

__all__ = [
    "ReliabilityBin",
    "WalkForwardReport",
    "ExpandingWindowFold",
    "expanding_window_evaluate",
    "TuningResult",
    "WalkForwardComparison",
    "group_by_period",
    "walk_forward",
    "compare_walk_forward",
    "tune",
]

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
class ReliabilityBin:
    """One equal-width probability bin of a reliability table.

    The bin covers ``[lower, upper)``; the final bin of a table also includes 1.0.

    Attributes:
        lower: Inclusive lower bound.
        upper: Exclusive upper bound (inclusive for the final bin).
        count: Scored predictions whose probability fell in the bin.
        mean_predicted: Mean predicted probability that the first side wins, or ``None`` if empty.
        observed_rate: Fraction of the bin's bouts the first side actually won, or ``None`` if empty.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: Optional[float]
    observed_rate: Optional[float]


def _validate_calibration_bins(calibration_bins: Any) -> None:
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, int):
        raise ValueError(f"calibration_bins must be a positive integer, got {calibration_bins!r}")
    if calibration_bins <= 0:
        raise ValueError(f"calibration_bins must be a positive integer, got {calibration_bins!r}")


def _bin_index(probability: float, bins: int) -> int:
    index = min(int(probability * bins), bins - 1)
    if index + 1 < bins and probability >= (index + 1) / bins:
        index += 1
    elif index > 0 and probability < index / bins:
        index -= 1
    return index


def _build_reliability(
    bins: int, counts: List[int], predicted_sums: List[float], wins: List[int]
) -> Tuple[ReliabilityBin, ...]:
    return tuple(
        ReliabilityBin(
            lower=i / bins,
            upper=(i + 1) / bins,
            count=counts[i],
            mean_predicted=predicted_sums[i] / counts[i] if counts[i] else None,
            observed_rate=wins[i] / counts[i] if counts[i] else None,
        )
        for i in range(bins)
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
        reliability: Equal-width :class:`ReliabilityBin` records over exactly the
            ``predictions`` population (decisive, predictable, post-warmup bouts; draws are
            excluded). Binned on the original prediction, before the log-loss clamp.
    """

    predictions: int
    skipped: int
    draws: int
    accuracy: float
    log_loss: float
    brier: float
    by_period: Tuple[Tuple[int, int, float], ...] = field(default=())
    reliability: Tuple[ReliabilityBin, ...] = field(default=())

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
    calibration_bins: int = 10,
) -> WalkForwardReport:
    """Predict each period from everything before it, then learn that period.

    Systems that override :meth:`BaseCompetitor.apply_rating_period` learn through
    :meth:`LambdaArena.rating_period`; systems that inherit the default implementation keep
    the existing sequential dataset-training path. A period-native system receives the
    maximum usable row timestamp as ``period_end`` (or ``None`` when there is none), so time
    is resolved per period rather than per row. Sequential systems continue to receive each
    row's timestamp individually.

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
        calibration_bins: Number of equal-width bins for ``WalkForwardReport.reliability``.

    Returns:
        WalkForwardReport: Metrics over every scored, predictable bout.

    Raises:
        ValueError: If ``warmup`` is negative or not smaller than the number of periods, or
            ``calibration_bins`` is not a positive integer.
    """
    _validate_calibration_bins(calibration_bins)
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
    bin_counts = [0] * calibration_bins
    bin_predicted = [0.0] * calibration_bins
    bin_wins = [0] * calibration_bins

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
                    raw_probability = arena.expected_score(a, b)
                    slot = _bin_index(raw_probability, calibration_bins)
                    probability = min(max(raw_probability, _PROBABILITY_EPS), 1.0 - _PROBABILITY_EPS)
                    actual = 1.0 if outcome > 0.5 else 0.0
                    hit = (probability > 0.5) == (actual > 0.5)
                    bin_counts[slot] += 1
                    bin_predicted[slot] += raw_probability
                    bin_wins[slot] += int(actual)
                    predictions += 1
                    correct += hit
                    period_correct += hit
                    period_count += 1
                    log_loss_total -= actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability)
                    brier_total += (probability - actual) ** 2

            if competitor_class.apply_rating_period.__func__ is BaseCompetitor.apply_rating_period.__func__:
                train_arena_with_dataset(arena, list(period), score_keys=score_keys)
            else:
                period_rows = [
                    (
                        a,
                        b,
                        outcome,
                        None if score_keys is None else _scores_from_attributes(attributes, score_keys),
                    )
                    for a, b, outcome, _when, attributes in period
                    if outcome is not None
                ]
                timestamps = [when for _a, _b, _outcome, when, _attributes in period if isinstance(when, datetime)]
                arena.rating_period(period_rows, period_end=max(timestamps, default=None))

            if scored and period_count:
                by_period.append((index, period_count, period_correct / period_count))
    finally:
        for name, value in originals.items():
            setattr(competitor_class, name, value)

    if not predictions:
        logger.warning("Walk-forward produced no scored predictions (skipped %d, draws %d).", skipped, draws)
        return WalkForwardReport(
            0,
            skipped,
            draws,
            float("nan"),
            float("nan"),
            float("nan"),
            (),
            _build_reliability(calibration_bins, bin_counts, bin_predicted, bin_wins),
        )

    return WalkForwardReport(
        predictions=predictions,
        skipped=skipped,
        draws=draws,
        accuracy=correct / predictions,
        log_loss=log_loss_total / predictions,
        brier=brier_total / predictions,
        by_period=tuple(by_period),
        reliability=_build_reliability(calibration_bins, bin_counts, bin_predicted, bin_wins),
    )


@dataclass(frozen=True)
class ExpandingWindowFold:
    """One expanding-window replay and its scoring report.

    Training covers ``[0, train_end)`` and scoring covers ``[train_end, test_end)``.
    Indexes in ``report.by_period`` retain their positions in the supplied schedule.
    Bout counts (predictions, skips and draws) are carried by ``report``.
    """

    train_end: int
    test_end: int
    report: WalkForwardReport

    @property
    def training_periods(self) -> int:
        """Number of leading periods used only for training."""
        return self.train_end

    @property
    def scoring_periods(self) -> int:
        """Number of periods scored with adaptive predict-then-learn updates."""
        return self.test_end - self.train_end


def expanding_window_evaluate(
    competitor_class: Type[BaseCompetitor],
    periods: Sequence[Sequence[Row]],
    folds: Sequence[Tuple[int, int]],
    *,
    competitor_params: Optional[Dict[str, Any]] = None,
    base_competitor_kwargs: Optional[Dict[str, Any]] = None,
    comparison_function: Optional[Callable[..., Any]] = None,
    score_keys: Optional[Tuple[str, str]] = None,
    calibration_bins: int = 10,
) -> Tuple[ExpandingWindowFold, ...]:
    """Evaluate a fixed configuration across explicit expanding development windows.

    Each fold starts a fresh :func:`walk_forward` replay of ``periods[:test_end]``
    with ``warmup=train_end``. Scoring is adaptive: each complete period is predicted
    before learning its results. Earlier scoring windows can become training history
    in later folds, so folds are dependent rather than independent trials.

    :param competitor_class: The rating system to evaluate.
    :param periods: Ordered development periods. Keep reserved final data outside.
    :param folds: Non-empty sequence of exclusive ``(train_end, test_end)`` indexes.
        Require ``0 < train_end < test_end <= len(periods)`` and ordered,
        non-overlapping scoring windows. Gaps between scoring windows are allowed.
    :param competitor_params: Fixed class knobs, forwarded to :func:`walk_forward`.
    :param base_competitor_kwargs: Fixed constructor options, forwarded to :func:`walk_forward`.
    :param comparison_function: Arena comparison function, forwarded to :func:`walk_forward`.
    :param score_keys: Point-score attribute names, forwarded to :func:`walk_forward`.
    :param calibration_bins: Reliability bin count, forwarded to :func:`walk_forward`.
    :returns: Immutable fold records in supplied order. No pooled aggregate, model
        selection, confidence intervals or other uncertainty claims are computed.
    :raises ValueError: If folds are empty, malformed, use non-integer indexes (including
        booleans), violate bounds, or have unordered/overlapping scoring windows.
        Every boundary is validated before any fold is evaluated.
    """
    boundaries = tuple(folds)
    if not boundaries:
        raise ValueError("folds must contain at least one scoring window")
    previous_end = 0
    for boundary in boundaries:
        if not isinstance(boundary, (tuple, list)) or len(boundary) != 2:
            raise ValueError("each fold must be a (train_end, test_end) pair")
        train_end, test_end = boundary
        if any(isinstance(index, bool) or not isinstance(index, int) for index in boundary):
            raise ValueError("fold boundaries must be integers, excluding booleans")
        if not 0 < train_end < test_end <= len(periods):
            raise ValueError("fold boundaries must satisfy 0 < train_end < test_end <= len(periods)")
        if train_end < previous_end:
            raise ValueError("scoring windows must be ordered and non-overlapping")
        previous_end = test_end

    return tuple(
        ExpandingWindowFold(
            train_end,
            test_end,
            walk_forward(
                competitor_class,
                periods[:test_end],
                warmup=train_end,
                competitor_params=competitor_params,
                base_competitor_kwargs=base_competitor_kwargs,
                comparison_function=comparison_function,
                score_keys=score_keys,
                calibration_bins=calibration_bins,
            ),
        )
        for train_end, test_end in boundaries
    )


@dataclass(frozen=True)
class WalkForwardComparison:
    """Walk-forward reports for several systems run over the same periods and warmup.

    Attributes:
        reports: Label to :class:`WalkForwardReport`, in the order the systems were given.
        warmup: Leading periods used for fitting but not scored, shared by every system.
        periods: Number of periods every system was run over.
        same_population: ``True`` when every system scored the same number of bouts and
            skipped and drew the same numbers. When ``False`` the figures describe different
            row populations and should not be compared as-is.
    """

    reports: Dict[str, WalkForwardReport]
    warmup: int
    periods: int
    same_population: bool

    def ranking(self) -> List[Dict[str, Any]]:
        """Rows sorted by log loss, best first; systems with no scored bouts (NaN) come last.

        Every row carries the protocol (``warmup``, ``periods``) and the scored-row counts.
        """
        rows = [
            {
                "system": label,
                "log_loss": report.log_loss,
                "brier": report.brier,
                "accuracy": report.accuracy,
                "predictions": report.predictions,
                "skipped": report.skipped,
                "draws": report.draws,
                "warmup": self.warmup,
                "periods": self.periods,
            }
            for label, report in self.reports.items()
        ]
        return sorted(rows, key=lambda row: (math.isnan(row["log_loss"]), row["log_loss"]))

    def __str__(self) -> str:
        rows = self.ranking()
        width = max([len("system"), *(len(row["system"]) for row in rows)])
        header = f"{'system':<{width}}  {'log loss':>9}  {'Brier':>7}  {'accuracy':>8}  {'scored':>6}  {'skipped':>7}  {'warmup':>6}"
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(
                f"{row['system']:<{width}}  {row['log_loss']:>9.4f}  {row['brier']:>7.4f}  "
                f"{row['accuracy']:>8.4f}  {row['predictions']:>6}  {row['skipped']:>7}  {row['warmup']:>6}"
            )
        if not self.same_population:
            lines.append("WARNING: systems scored different row populations; figures are not directly comparable.")
        return "\n".join(lines)


_SYSTEM_SPEC_KEYS = frozenset({"competitor_params", "base_competitor_kwargs"})


def _parse_system(label: Any, spec: Any) -> Tuple[Type[BaseCompetitor], Dict[str, Any], Dict[str, Any]]:
    if isinstance(spec, tuple):
        if len(spec) != 2 or not isinstance(spec[1], dict):
            raise InvalidParameterException(f"system {label!r} must be a class or a (class, options dict) pair")
        competitor_class, options = spec
        unknown = set(options) - _SYSTEM_SPEC_KEYS
        if unknown:
            raise InvalidParameterException(
                f"system {label!r} has unknown options {sorted(unknown)}; "
                f"expected only {sorted(_SYSTEM_SPEC_KEYS)}"
            )
    else:
        competitor_class, options = spec, {}
    if not (isinstance(competitor_class, type) and issubclass(competitor_class, BaseCompetitor)):
        raise InvalidParameterException(f"system {label!r} is not a BaseCompetitor subclass: {competitor_class!r}")
    params = dict(options.get("competitor_params") or {})
    kwargs = dict(options.get("base_competitor_kwargs") or {})
    _validate_competitor_params(competitor_class, params.keys())
    return competitor_class, params, kwargs


def compare_walk_forward(
    systems: Dict[str, Any],
    periods: Sequence[Sequence[Row]],
    *,
    warmup: int = 0,
    score_keys: Optional[Tuple[str, str]] = None,
) -> WalkForwardComparison:
    """Run :func:`walk_forward` for each system on the same periods and warmup.

    Args:
        systems: Label to a competitor class, or to a ``(class, options)`` pair where
            ``options`` may hold ``competitor_params`` and ``base_competitor_kwargs`` exactly
            as :func:`walk_forward` takes them.
        periods: Ordered periods of dataset rows, as produced by :func:`group_by_period`.
        warmup: Leading periods used for fitting but not scored, applied to every system.
        score_keys: ``(a_score_key, b_score_key)`` naming each row's two point scores.

    Returns:
        WalkForwardComparison: Each system's report, a log-loss ranking and a printable table.

    Raises:
        InvalidParameterException: If ``systems`` is empty or any entry is malformed, is not a
            competitor class, or names a parameter its class does not have. Raised before any
            system is evaluated.
        ValueError: If ``warmup`` is negative or not smaller than the number of periods.
    """
    if not systems:
        raise InvalidParameterException("systems must contain at least one entry")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if periods and warmup >= len(periods):
        raise ValueError(f"warmup ({warmup}) leaves no periods to score (have {len(periods)})")

    parsed = {label: _parse_system(label, spec) for label, spec in systems.items()}

    reports = {
        label: walk_forward(
            competitor_class,
            periods,
            competitor_params=params,
            base_competitor_kwargs=kwargs,
            score_keys=score_keys,
            warmup=warmup,
        )
        for label, (competitor_class, params, kwargs) in parsed.items()
    }
    populations = {(r.predictions, r.skipped, r.draws) for r in reports.values()}
    return WalkForwardComparison(
        reports=reports,
        warmup=warmup,
        periods=len(periods),
        same_population=len(populations) == 1,
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
