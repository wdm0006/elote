"""Pythagorean expectation implementation for the Elote library.

Pythagorean expectation is the oldest points-based rating in the sports canon. Bill James
introduced it for baseball in the early 1980s, observing that a team's winning percentage is
predicted remarkably well by the runs it scored and the runs it allowed:

.. math::

   w = \\frac{PF^{k}}{PF^{k} + PA^{k}}

The name comes from the resemblance to the Pythagorean theorem in the original ``k = 2``
form. The exponent is the one free parameter, and it is fitted per sport: 2 for baseball
(James' original), around 2.37 for American football, and around 14 for basketball, where
scores are much larger and a point is worth correspondingly less.

Two properties make it unusual in this library.

1. **The rating is already a win expectation.** Every other shipped system produces a
   strength score that has to be mapped through a logistic, a normal CDF or a share before it
   can be read as a probability. A Pythagorean rating is a number in ``[0, 1]`` that reads
   directly as "the fraction of games this competitor should win against the field it has
   played".
2. **It ignores the opponent graph entirely.** A competitor's rating depends only on its own
   accumulated points for and points against, never on who supplied them. That makes it the
   cheapest system here -- a constant-time update with no graph, no matrix and no refit --
   and also its main limitation: it makes no strength-of-schedule adjustment at all.

Two competitors are compared through the standard log5 combination of their two win
expectations,

.. math::

   E_a = \\frac{w_a - w_a w_b}{w_a + w_b - 2 w_a w_b}

which is Bill James' formula for the probability that ``a`` beats ``b`` given the rate at
which each of them beats the field.

References:
- James, B. (1981). *The Bill James Baseball Abstract*. The original ``k = 2`` formulation.
- Schatz, A. (2003). "Pythagoras on the Gridiron", Football Outsiders. The ``k = 2.37`` fit
  for American football used as the default here.
- Miller, S. J. (2007). "A Derivation of the Pythagorean Won-Loss Formula in Baseball".
  *Chance*, 20(1), 40-48. Derives the formula from a Weibull model of scoring.
"""

from typing import Dict, Any, ClassVar, Optional, Sequence, Type, TypeVar, cast

from elote.competitors.base import BaseCompetitor, InvalidParameterException
from elote.logging import logger

T = TypeVar("T", bound="PythagoreanCompetitor")


class PythagoreanCompetitor(BaseCompetitor):
    """Pythagorean expectation competitor.

    Rates a competitor from the points it has scored and the points it has allowed. Unlike
    Colley, Massey, Keener and Bradley-Terry, nothing is fitted across a population: each
    result adds to two running totals, and the rating is read straight off them. Unlike Elo
    and the other incremental systems, the update does not depend on the opponent at all.

    **Scores.** ``beat`` / ``lost_to`` / ``tied`` accept the common optional ``scores``
    payload, the two competitors' scores in caller order. Points are what this system is
    built on, so supplying real ones is the intended use. When they are omitted the same
    unit-score fallback the rest of the library uses applies: the winner is credited ``1``
    and the loser ``0``, and a draw credits each side ``0.5``. On unit scores the rating
    degenerates into a smoothed winning percentage, which is still a usable baseline.

    **Degenerate inputs.** A fresh competitor has ``PF = PA = 0`` and an unbeaten one has
    ``PA = 0``; the first makes the rating ``0/0`` and the second makes it exactly ``1``,
    which in turn makes log5 ``0/0``. Both are handled by a small symmetric prior added to
    each accumulator rather than by clamping the output: the rating stays a continuous,
    strictly monotone function of the real totals, a fresh competitor is exactly ``0.5``, and
    every rating is strictly inside ``(0, 1)``, so log5 is always defined.

    Key characteristics:
    - The rating is a win expectation in [0, 1], not a strength score
    - Constant-time update: no opponent graph, no refit, no order dependence
    - Reads real scores; falls back to unit scores when they are omitted
    - Makes no strength-of-schedule adjustment whatsoever

    Class Attributes:
        _minimum_rating (float): The minimum allowed rating value. Default: 0.0. The rating
            is a probability, so the inherited Elo-style floor of 100 would saturate it.
        _default_initial_rating (float): The rating of a competitor that has not played.
            Default: 0.5, which the symmetric prior produces exactly.
        _exponent (float): Pythagorean ``k``. Default: 2.37, the standard fit for American
            football (Football Outsiders). Use 2 for baseball, or a much larger value for a
            high-scoring sport such as basketball.
        _prior_points (float): The symmetric prior added to both accumulators, in points.
            Default: 1.0 -- comparable to the unit scores, negligible against real ones.
    """

    _minimum_rating: ClassVar[float] = 0.0
    _default_initial_rating: ClassVar[float] = 0.5
    _exponent: ClassVar[float] = 2.37
    _prior_points: ClassVar[float] = 1.0

    def __init__(self, exponent: Optional[float] = None):
        """Initialize a Pythagorean competitor.

        There is deliberately no ``initial_rating`` argument: the rating is derived from the
        points totals rather than stored, and a caller-supplied starting rating on a ``[0, 1]``
        scale has no points totals that would produce it.

        Args:
            exponent (float, optional): The Pythagorean exponent for this competitor. If
                None, the class exponent is used. Default: None.

        Raises:
            InvalidParameterException: If the exponent is not positive.
        """
        super().__init__()

        if exponent is not None and exponent <= 0:
            raise InvalidParameterException("Pythagorean exponent must be positive")

        self._exponent = exponent if exponent is not None else PythagoreanCompetitor._exponent
        self._points_for = 0.0
        self._points_against = 0.0
        self._num_games = 0
        logger.debug("Initialized PythagoreanCompetitor with exponent %.4f", self._exponent)

    def __repr__(self) -> str:
        """Return a string representation of this competitor."""
        return (
            f"<PythagoreanCompetitor: rating={self.rating:.4f}, "
            f"points={self._points_for:.1f}-{self._points_against:.1f}, games={self._num_games}>"
        )

    def __str__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<PythagoreanCompetitor: rating={self.rating:.4f}>"

    @property
    def rating(self) -> float:
        """Get the current rating of this competitor.

        The rating is the Pythagorean win expectation ``PF^k / (PF^k + PA^k)``, computed on
        the prior-adjusted totals. It is evaluated in the equivalent ratio form
        ``1 / (1 + (PA/PF)^k)``, which cannot overflow for large totals and returns exactly
        0.5 whenever the two totals are equal.

        Returns:
            float: The current rating, strictly inside (0, 1).
        """
        scored = self._points_for + self._prior_points
        allowed = self._points_against + self._prior_points
        try:
            powered = (allowed / scored) ** self._exponent
        except OverflowError:
            # Needs a points ratio around 1e130, so this is a guard rather than a case:
            # such a competitor has been outscored beyond anything the float scale can
            # express, and its expectation is zero to every digit available.
            powered = float("inf")
        return float(1.0 / (1.0 + powered))

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor.

        Not supported: the rating is derived from the points totals, so there is no single
        pair of totals that a given rating corresponds to.

        Args:
            value (float): The new rating value.

        Raises:
            NotImplementedError: Always.
        """
        logger.warning("Attempted to set rating directly on PythagoreanCompetitor, which is not supported.")
        raise NotImplementedError(
            "Cannot directly set the rating of a PythagoreanCompetitor. It is derived from points for and against."
        )

    @property
    def num_games(self) -> int:
        """Get the total number of games played by this competitor.

        Returns:
            int: The total number of games played.
        """
        return self._num_games

    @staticmethod
    def _log5(higher: float, lower: float) -> float:
        """Combine two win expectations with log5, given in descending order.

        The denominator is formed as the sum of the two numerators rather than as the
        textbook ``w_a + w_b - 2 w_a w_b``. The two are the same expression, but summing the
        already-rounded numerators keeps the quotient inside ``[0.5, 1]`` in floating point:
        both numerators are non-negative and the larger one is at least the smaller, so the
        denominator is at least the numerator and at most twice it. Written the textbook way,
        a heavily lopsided pair rounds to 1.0000000000000002.

        Args:
            higher: The larger of the two win expectations.
            lower: The smaller of the two win expectations.

        Returns:
            float: The probability that the ``higher`` competitor wins, in [0.5, 1].
        """
        product = higher * lower
        higher_edge = higher - product
        lower_edge = lower - product
        denominator = higher_edge + lower_edge
        if denominator <= 0.0:
            # Only reachable when both expectations are exactly 0 or exactly 1, which the
            # prior rules out for ratings this class produces. Two competitors that agree
            # completely are an even matchup.
            return 0.5
        return higher_edge / denominator

    def expected_score(self, competitor: "BaseCompetitor") -> float:
        """Calculate the expected score against another competitor.

        Both ratings are already win expectations, so they are combined with the standard
        log5 formula. The direction with the larger rating is the one actually computed and
        the other is taken as its complement, which makes the two argument orders sum to
        exactly 1.0 and makes two equal ratings give exactly 0.5.

        Args:
            competitor (BaseCompetitor): The competitor to compare against.

        Returns:
            float: The expected score (probability of winning).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        mine = self.rating
        theirs = competitor.rating
        if mine >= theirs:
            return self._log5(mine, theirs)
        return 1.0 - self._log5(theirs, mine)

    def _record_game(self, scored: float, allowed: float) -> None:
        """Add one game's points to this competitor's running totals.

        Args:
            scored: The points this competitor scored.
            allowed: The points this competitor conceded.
        """
        self._points_for += scored
        self._points_against += allowed
        self._num_games += 1

    def beat(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has won against the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. When omitted the unit scores ``1`` and
                ``0`` are recorded.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` does not describe a win for this competitor.
        """
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 1.0)
        opponent = cast(PythagoreanCompetitor, competitor)
        mine, theirs = (1.0, 0.0) if validated is None else validated

        self._record_game(mine, theirs)
        opponent._record_game(theirs, mine)
        logger.debug("Recorded win %.3f-%.3f for %s over %s", mine, theirs, self, opponent)

    def tied(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. They must be equal. When omitted each
                side is credited the unit draw score ``0.5``.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` does not describe a draw.
        """
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.5)
        opponent = cast(PythagoreanCompetitor, competitor)
        mine, theirs = (0.5, 0.5) if validated is None else validated

        self._record_game(mine, theirs)
        opponent._record_game(theirs, mine)
        logger.debug("Recorded draw %.3f-%.3f between %s and %s", mine, theirs, self, opponent)

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "exponent": self._exponent if self._exponent != self.__class__._exponent else None,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        The rating itself is not exported: it is a pure function of the two totals and the
        exponent, so restoring the totals restores the rating exactly.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "points_for": self._points_for,
            "points_against": self._points_against,
            "num_games": self._num_games,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        exponent = parameters.get("exponent", None)
        if exponent is not None and exponent <= 0:
            raise InvalidParameterException("Pythagorean exponent must be positive")
        self._exponent = exponent if exponent is not None else PythagoreanCompetitor._exponent

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        points_for = state.get("points_for", 0.0)
        points_against = state.get("points_against", 0.0)
        if points_for < 0 or points_against < 0:
            raise InvalidParameterException("Point totals cannot be negative")
        self._points_for = float(points_for)
        self._points_against = float(points_against)
        self._num_games = int(state.get("num_games", 0))

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            PythagoreanCompetitor: A new competitor instance.
        """
        return cls(exponent=parameters.get("exponent", None))

    def reset(self) -> None:
        """Reset this competitor to its initial state."""
        logger.info("Resetting PythagoreanCompetitor to initial state.")
        self._points_for = 0.0
        self._points_against = 0.0
        self._num_games = 0

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        """Configure class-level parameters for this rating system.

        Overrides the base implementation to validate the Pythagorean-specific parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        if "exponent" in kwargs and kwargs["exponent"] <= 0:
            raise InvalidParameterException("Pythagorean exponent must be positive")
        if "prior_points" in kwargs and kwargs["prior_points"] <= 0:
            raise InvalidParameterException("prior_points must be positive")
        super().configure_class(**kwargs)
