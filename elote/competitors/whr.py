"""Whole-History Rating (WHR) for time-aware paired comparisons."""

import math
from datetime import date, datetime
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar, cast

import numpy as np

from elote.competitors.base import BaseCompetitor, InvalidParameterException

T = TypeVar("T", bound="WholeHistoryRatingCompetitor")


class WholeHistoryRatingCompetitor(BaseCompetitor):
    """A time-aware Bradley-Terry maximum-a-posteriori rating curve.

    One latent rating is kept for every distinct playing day. Consecutive ratings are
    linked by a Wiener-process prior whose variance is ``w2 * elapsed_days`` in Elo
    points squared. Results use the Bradley-Terry likelihood, and a bounded sequence
    of per-competitor tridiagonal Newton updates fits the connected component lazily.

    Serialized state preserves the fitted per-day curve and day index, but cannot
    preserve object references in the game graph. On the first result after restore,
    the latest restored rating becomes the initial rating of a fresh history; restored
    lifetime games are therefore never silently combined with a reset graph.

    Args:
        w2: Per-day Wiener-process variance in Elo points squared.
        initial_rating: Rating before any games, on the Elo scale.
        max_iterations: Maximum component-wide Newton sweeps per lazy fit.
        precision: Stop when the largest Elo-scale Newton step is below this value.

    Reference: Coulom, R. (2008), *Whole-History Rating: A Bayesian Rating System
    for Players of Time-Varying Strength*.
    """

    _minimum_rating: ClassVar[float] = 0.0
    _scale: ClassVar[float] = 400.0 / math.log(10.0)
    _default_w2: ClassVar[float] = 300.0
    _default_initial_rating: ClassVar[float] = 1500.0
    _default_max_iterations: ClassVar[int] = 20
    _default_precision: ClassVar[float] = 0.1

    def __init__(
        self,
        w2: Optional[float] = None,
        initial_rating: Optional[float] = None,
        max_iterations: Optional[int] = None,
        precision: Optional[float] = None,
    ) -> None:
        super().__init__()
        w2 = self._default_w2 if w2 is None else w2
        initial_rating = self._default_initial_rating if initial_rating is None else initial_rating
        max_iterations = self._default_max_iterations if max_iterations is None else max_iterations
        precision = self._default_precision if precision is None else precision
        self._validate_parameters(w2, initial_rating, max_iterations, precision)
        self._w2 = float(w2)
        self._initial_rating = float(initial_rating)
        self._max_iterations = int(max_iterations)
        self._precision = float(precision)
        self._days: List[date] = []
        self._ratings: List[float] = []
        self._day_index: Dict[date, int] = {}
        self._games: List[List[Tuple["WholeHistoryRatingCompetitor", date, float]]] = []
        self._opponents: Set["WholeHistoryRatingCompetitor"] = set()
        self._dirty = False
        self._restored = False
        self._last_activity: Optional[datetime] = None

    @staticmethod
    def _validate_parameters(w2: float, initial_rating: float, max_iterations: int, precision: float) -> None:
        if not math.isfinite(w2) or w2 <= 0:
            raise InvalidParameterException("w2 must be positive and finite")
        if not math.isfinite(initial_rating):
            raise InvalidParameterException("initial_rating must be finite")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise InvalidParameterException("max_iterations must be a positive integer")
        if not math.isfinite(precision) or precision <= 0:
            raise InvalidParameterException("precision must be positive and finite")

    @property
    def rating(self) -> float:
        self._ensure_fit()
        return self._ratings[-1] if self._ratings else self._initial_rating

    @rating.setter
    def rating(self, value: float) -> None:
        if not math.isfinite(value):
            raise InvalidParameterException("rating must be finite")
        if self._ratings:
            self._ratings[-1] = float(value)
        else:
            self._initial_rating = float(value)

    @property
    def num_games(self) -> int:
        return sum(len(games) for games in self._games)

    def rating_at(self, when: datetime | date) -> float:
        """Return the fitted rating on, or most recently before, ``when``."""
        self._ensure_fit()
        day = when.date() if isinstance(when, datetime) else when
        candidates = [i for i, played in enumerate(self._days) if played <= day]
        return self._initial_rating if not candidates else self._ratings[candidates[-1]]

    def rating_history(self) -> List[Tuple[date, float]]:
        """Return a chronological copy of the fitted ``(day, rating)`` curve."""
        self._ensure_fit()
        return list(zip(self._days, self._ratings, strict=True))

    def expected_score(self, competitor: BaseCompetitor) -> float:
        self.verify_competitor_types(competitor)
        other = cast(WholeHistoryRatingCompetitor, competitor)
        self._ensure_fit()
        other._ensure_fit()
        high, low = (self.rating, other.rating) if self.rating >= other.rating else (other.rating, self.rating)
        numerator = math.exp((low - high) / self._scale)
        high_probability = min(1.0 / (1.0 + numerator), math.nextafter(1.0, 0.0))
        return high_probability if self.rating >= other.rating else 1.0 - high_probability

    def beat(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 1.0)
        self._record(cast(WholeHistoryRatingCompetitor, competitor), 1.0, match_time)

    def lost_to(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.0)
        competitor.beat(
            self,
            match_time,
            scores=None if validated is None else (validated[1], validated[0]),
        )

    def tied(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.5)
        self._record(cast(WholeHistoryRatingCompetitor, competitor), 0.5, match_time)

    def _record(self, opponent: "WholeHistoryRatingCompetitor", score: float, when: Optional[datetime]) -> None:
        if self._restored:
            self._restart_from_restored_rating()
        if opponent._restored:
            opponent._restart_from_restored_rating()
        timestamp = when or datetime.now()
        day = timestamp.date()
        own_index = self._node(day)
        opponent_index = opponent._node(day)
        self._games[own_index].append((opponent, day, score))
        opponent._games[opponent_index].append((self, day, 1.0 - score))
        self._opponents.add(opponent)
        opponent._opponents.add(self)
        self._last_activity = timestamp
        opponent._last_activity = timestamp
        for member in self._component():
            member._dirty = True

    def _restart_from_restored_rating(self) -> None:
        latest = self._ratings[-1] if self._ratings else self._initial_rating
        self._initial_rating = latest
        self._days, self._ratings, self._games = [], [], []
        self._day_index, self._opponents = {}, set()
        self._restored = False

    def _node(self, day: date) -> int:
        if day in self._day_index:
            return self._day_index[day]
        rating = self._ratings[-1] if self._ratings else self._initial_rating
        position = 0
        while position < len(self._days) and self._days[position] < day:
            position += 1
        self._days.insert(position, day)
        self._ratings.insert(position, rating)
        self._games.insert(position, [])
        self._day_index = {value: i for i, value in enumerate(self._days)}
        return position

    def _component(self) -> List["WholeHistoryRatingCompetitor"]:
        result: List[WholeHistoryRatingCompetitor] = []
        pending = [self]
        seen: Set[WholeHistoryRatingCompetitor] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            pending.extend(current._opponents - seen)
        return result

    def _ensure_fit(self) -> None:
        component = self._component()
        if not any(member._dirty for member in component):
            return
        for _ in range(self._max_iterations):
            largest = 0.0
            for member in component:
                largest = max(largest, member._newton_update())
            if largest < self._precision:
                break
        for member in component:
            member._dirty = False

    def _newton_update(self) -> float:
        n = len(self._ratings)
        if not n:
            return 0.0
        gradient = np.zeros(n)
        diagonal = np.zeros(n)
        off_diagonal = np.zeros(max(0, n - 1))
        anchor_weight = 1.0 / self._w2
        gradient[0] += (self._initial_rating - self._ratings[0]) * anchor_weight
        diagonal[0] += anchor_weight
        for i, games in enumerate(self._games):
            for opponent, day, score in games:
                opponent_rating = opponent._ratings[opponent._day_index[day]]
                difference = (self._ratings[i] - opponent_rating) / self._scale
                probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, difference))))
                gradient[i] += (score - probability) / self._scale
                diagonal[i] += probability * (1.0 - probability) / (self._scale * self._scale)
        for i in range(n - 1):
            variance = self._w2 * (self._days[i + 1] - self._days[i]).days
            weight = 1.0 / variance
            difference = self._ratings[i + 1] - self._ratings[i]
            gradient[i] += difference * weight
            gradient[i + 1] -= difference * weight
            diagonal[i] += weight
            diagonal[i + 1] += weight
            off_diagonal[i] = -weight
        matrix = np.diag(diagonal)
        if n > 1:
            matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        matrix += np.eye(n) * 1e-12
        step = np.linalg.solve(matrix, gradient)
        step = np.clip(step, -200.0, 200.0)
        self._ratings = [rating + float(delta) for rating, delta in zip(self._ratings, step, strict=True)]
        return float(np.max(np.abs(step)))

    def _export_parameters(self) -> Dict[str, Any]:
        return {
            "w2": self._w2,
            "initial_rating": self._initial_rating,
            "max_iterations": self._max_iterations,
            "precision": self._precision,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        self._ensure_fit()
        return {
            "rating": self.rating,
            "days": [day.isoformat() for day in self._days],
            "ratings": list(self._ratings),
            "day_index": {day.isoformat(): index for day, index in self._day_index.items()},
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        self._w2 = float(parameters.get("w2", self._default_w2))
        self._initial_rating = float(parameters.get("initial_rating", self._default_initial_rating))
        self._max_iterations = int(parameters.get("max_iterations", self._default_max_iterations))
        self._precision = float(parameters.get("precision", self._default_precision))

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        self._days = [date.fromisoformat(value) for value in state.get("days", [])]
        self._ratings = [float(value) for value in state.get("ratings", [])]
        self._day_index = {day: i for i, day in enumerate(self._days)}
        self._games = [[] for _ in self._days]
        self._opponents = set()
        value = state.get("last_activity")
        self._last_activity = datetime.fromisoformat(value) if value else None
        self._dirty = False
        self._restored = bool(self._days)

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        return cls(**parameters)

    def reset(self) -> None:
        self._days, self._ratings, self._games = [], [], []
        self._day_index, self._opponents = {}, set()
        self._dirty = self._restored = False
        self._last_activity = None

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        mapping = {
            "w2": "_default_w2",
            "initial_rating": "_default_initial_rating",
            "max_iterations": "_default_max_iterations",
            "precision": "_default_precision",
        }
        values = {key: kwargs.get(key, getattr(cls, name)) for key, name in mapping.items()}
        cls._validate_parameters(**values)
        for key, value in kwargs.items():
            if key not in mapping:
                raise InvalidParameterException(f"Unknown class parameter: {key}")
            setattr(cls, mapping[key], value)

    def configure(self, **kwargs: Any) -> None:
        """Validate and update this instance's fitting parameters."""
        allowed = {"w2", "initial_rating", "max_iterations", "precision"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise InvalidParameterException(f"Unknown instance parameter: {unknown.pop()}")
        values = {
            "w2": kwargs.get("w2", self._w2),
            "initial_rating": kwargs.get("initial_rating", self._initial_rating),
            "max_iterations": kwargs.get("max_iterations", self._max_iterations),
            "precision": kwargs.get("precision", self._precision),
        }
        self._validate_parameters(**values)
        self._w2 = float(values["w2"])
        self._initial_rating = float(values["initial_rating"])
        self._max_iterations = int(values["max_iterations"])
        self._precision = float(values["precision"])
        self._dirty = bool(self._days)

    def __eq__(self, other: Any) -> bool:
        return self is other

    __hash__ = object.__hash__
