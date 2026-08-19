"""
Massey Ratings implementation for the Elote library.

The Massey method is the least-squares counterpart to
:class:`~elote.competitors.colley.ColleyMatrixCompetitor`. Where Colley solves a
ridge-regularized win-percentage model whose ratings are bounded to [0, 1], Massey solves an
unregularized least-squares system whose ratings live on a signed *margin* scale: the fitted
rating difference ``r_i - r_j`` is the model's predicted margin when ``i`` plays ``j``.

The system is

.. math::

   M r = p

where ``M = D - A`` (``D_ii`` is the number of games played by ``i`` and ``A_ij`` the number of
games between ``i`` and ``j``) and ``p_i`` is ``i``'s cumulative margin. ``M`` is a graph
Laplacian, so its rows sum to zero and it is singular by construction. The standard fix is
applied: the last row is replaced with all ones and the last entry of ``p`` with zero, which
pins the ratings to zero mean and makes the solution unique on a connected schedule.

References:
- Massey, K. (1997). Statistical Models Applied to the Rating of Sports Teams.
  Bluefield College undergraduate honors thesis.
- Langville, A. N., & Meyer, C. D. (2012). Who's #1? The Science of Rating and Ranking.
  Princeton University Press, chapter 2.
"""

from typing import Dict, Any, ClassVar, Type, TypeVar, List, Optional, cast, Set, Sequence

import numpy as np

from elote.competitors.base import BaseCompetitor, InvalidParameterException, InvalidRatingValueException
from elote.logging import logger

T = TypeVar("T", bound="MasseyCompetitor")


class MasseyCompetitor(BaseCompetitor):
    """Massey Ratings competitor.

    Massey's method assigns every competitor a rating such that the difference between two
    ratings is a least-squares estimate of the margin by which one would beat the other.
    Like Colley and Bradley-Terry -- and unlike Elo -- ratings are not nudged after each
    result; the whole connected group is re-fit from the complete match history, which makes
    the method order independent.

    **Margins.** ``beat`` / ``lost_to`` / ``tied`` accept the common optional ``scores``
    payload, the two competitors' scores in caller order. When it is supplied this is genuine
    margin-of-victory Massey -- the form used in college football rankings -- and the margin
    contributed by a game is ``self_score - competitor_score``. When it is omitted the
    implementation falls back to *unit margins*: a win contributes ``+1`` to the winner's
    cumulative margin and ``-1`` to the loser's. A draw contributes ``0`` either way, while
    still counting as a game played for both.

    Key characteristics:
    - Global least-squares fit (does not depend on the order of results)
    - Ratings are zero mean within a connected group, so roughly half of them are negative
    - The rating difference is directly interpretable as a predicted margin

    Class Attributes:
        _minimum_rating (float): The minimum allowed rating value. Default: ``-inf``. Massey
            ratings are zero mean and routinely negative, so no floor is imposed.
        _default_initial_rating (float): Default initial rating. Default: 0.0.
        _expected_score_scale (float): Logistic scale used by :meth:`expected_score` to turn a
            rating difference into a win probability. Default: 2.0, chosen so that the widest
            plausible unit-margin gap maps to roughly the same probability as the widest gap
            under Colley's ``1 / (1 + exp(-4 * diff))``.
        _round_decimals (int): Number of decimal places fitted ratings are canonicalized to, so
            that solver noise cannot make mathematically identical records differ. Default: 13.
    """

    _minimum_rating: ClassVar[float] = float("-inf")
    _default_initial_rating: ClassVar[float] = 0.0
    _expected_score_scale: ClassVar[float] = 2.0
    _round_decimals: ClassVar[int] = 13

    def __init__(self, initial_rating: Optional[float] = None):
        """Initialize a new Massey competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 0.0.
        """
        super().__init__()  # Call base class constructor

        self._initial_rating = initial_rating if initial_rating is not None else self._default_initial_rating
        self._rating = self._initial_rating
        self._wins = 0
        self._losses = 0
        self._ties = 0
        # Cumulative margin -- the right-hand side ``p`` of the Massey system. With unit
        # margins this is simply wins minus losses; with real scores it is the sum of
        # ``self_score - opponent_score`` over every game played.
        self._point_differential = 0.0
        self._opponents: Dict["MasseyCompetitor", int] = {}  # Opponent -> num games
        self._margins_for: Dict["MasseyCompetitor", float] = {}
        # Unique ID for hashing (instances are used as dict keys in the match graph).
        self._id = id(self)
        logger.debug("Initialized MasseyCompetitor %d with initial rating %.3f", self._id, self._initial_rating)

    @property
    def rating(self) -> float:
        """Get the current rating of this competitor.

        Returns:
            float: The current rating.
        """
        return self._rating

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor.

        Args:
            value (float): The new rating value.

        Raises:
            InvalidRatingValueException: If the rating value is below the minimum rating.
        """
        if value < self._minimum_rating:
            logger.warning(
                "Attempted to set rating %.3f below minimum %.3f for %d", value, self._minimum_rating, self._id
            )
            raise InvalidRatingValueException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = value

    @property
    def num_games(self) -> int:
        """Get the total number of games played by this competitor.

        Returns:
            int: The total number of games played.
        """
        return self._wins + self._losses + self._ties

    def expected_score(self, competitor: "BaseCompetitor") -> float:
        """Calculate the expected score against another competitor.

        Massey ratings are on a margin scale rather than a probability scale, so the predicted
        margin ``r_self - r_competitor`` is squashed through a logistic function to obtain a
        win probability. Two competitors with equal ratings give exactly 0.5, and the two
        argument orders are exactly complementary.

        Args:
            competitor (BaseCompetitor): The competitor to compare against.

        Returns:
            float: The expected score (probability of winning).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        rating_diff = self.rating - competitor.rating
        # tanh form: symmetric in the sign of rating_diff, so the two argument orders sum to
        # exactly 1.0 in floating point.
        return float(0.5 * (1.0 + np.tanh(0.5 * self._expected_score_scale * rating_diff)))

    def beat(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has won against the given competitor.

        The result is recorded in the match graph and the whole connected group is re-fit.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. When supplied, the margin contributed to
                the Massey system is the real ``self_score - competitor_score``; when
                omitted the unit margin ``+1 / -1`` is used.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s beat %s", self, competitor)
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 1.0)
        opponent = cast(MasseyCompetitor, competitor)
        margin = 1.0 if validated is None else validated[0] - validated[1]

        self._wins += 1
        self._point_differential += margin
        self._opponents[opponent] = self._opponents.get(opponent, 0) + 1
        self._margins_for[opponent] = self._margins_for.get(opponent, 0.0) + margin

        opponent._losses += 1
        opponent._point_differential -= margin
        opponent._opponents[self] = opponent._opponents.get(self, 0) + 1
        opponent._margins_for[self] = opponent._margins_for.get(self, 0.0) - margin

        logger.debug("Recorded win for %d, loss for %d", self._id, opponent._id)
        self._recalculate_ratings()

    def tied(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        A draw contributes nothing to either cumulative margin, but counts as a game played
        for both competitors.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. They must be equal, so a drawn game
                contributes a zero margin whether or not scores are supplied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s tied with %s", self, competitor)
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.5)
        opponent = cast(MasseyCompetitor, competitor)

        self._ties += 1
        self._opponents[opponent] = self._opponents.get(opponent, 0) + 1
        self._margins_for.setdefault(opponent, 0.0)

        opponent._ties += 1
        opponent._opponents[self] = opponent._opponents.get(self, 0) + 1
        opponent._margins_for.setdefault(self, 0.0)

        logger.debug("Recorded tie for %d and %d", self._id, opponent._id)
        self._recalculate_ratings()

    def lost_to(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has lost to the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. Reversed before being passed to the
                winner's :meth:`beat`.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s lost to %s", self, competitor)
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.0)
        competitor.beat(self, scores=None if validated is None else (validated[1], validated[0]))

    def _get_connected_competitors(self) -> List["MasseyCompetitor"]:
        """Get all competitors connected to this competitor in the match graph.

        Returns:
            List[MasseyCompetitor]: A list of all connected competitors.
        """
        visited: Set["MasseyCompetitor"] = set()
        to_visit: List["MasseyCompetitor"] = [self]
        all_competitors = []

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            all_competitors.append(current)
            for opponent in current._opponents:
                if opponent not in visited:
                    to_visit.append(opponent)

        logger.debug("Found %d connected competitors", len(all_competitors))
        return all_competitors

    def _recalculate_ratings(self) -> None:
        """Re-fit the Massey ratings for all connected competitors.

        Builds ``M = D - A`` and the cumulative-margin vector ``p``, replaces the last row of
        ``M`` with all ones (and the last entry of ``p`` with zero) to remove the singularity,
        and solves the resulting system. The constraint pins the ratings to zero mean.
        """
        competitors = self._get_connected_competitors()
        n = len(competitors)
        if n <= 1:
            logger.debug("Only one competitor in network, skipping recalculation.")
            return

        idx = {comp: i for i, comp in enumerate(competitors)}

        M = np.zeros((n, n), dtype=np.float64)
        p = np.zeros(n, dtype=np.float64)
        for i, comp in enumerate(competitors):
            M[i, i] = sum(comp._opponents.values())
            p[i] = sum(comp._margins_for.values())
            for opponent, games in comp._opponents.items():
                j = idx.get(opponent)
                if j is not None:
                    M[i, j] = -games

        # M is a graph Laplacian and therefore singular. Replacing one equation with the
        # zero-sum constraint makes the system non-singular; the replaced equation is implied
        # by the remaining ones, so the solution is unaffected by which row is chosen.
        M[n - 1, :] = 1.0
        p[n - 1] = 0.0

        try:
            r = np.linalg.solve(M, p)
        except np.linalg.LinAlgError as e:
            logger.warning(
                "Massey matrix is singular (%s). Falling back to average-margin ratings for %d competitors.",
                str(e),
                n,
            )
            self._fallback_rating_calculation(competitors)
            return

        # The solver's pivoting depends on the row order, which is the order competitors were
        # discovered in, so two runs over the same results can differ in the last few bits.
        # Canonicalizing to _round_decimals places makes mathematically identical records
        # compare exactly equal and makes the fit order independent.
        r = np.round(r, decimals=self._round_decimals)

        for i, comp in enumerate(competitors):
            comp.rating = float(r[i])

    def _fallback_rating_calculation(self, competitors: List["MasseyCompetitor"]) -> None:
        """Assign average-margin ratings when the linear system cannot be solved.

        Args:
            competitors: The connected group of competitors to rate.
        """
        ratings = [(comp._point_differential / comp.num_games) if comp.num_games else 0.0 for comp in competitors]
        mean_rating = sum(ratings) / len(ratings)
        for comp, value in zip(competitors, ratings, strict=True):
            comp.rating = value - mean_rating

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_rating": self._initial_rating,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "rating": self._rating,
            "wins": self._wins,
            "losses": self._losses,
            "ties": self._ties,
            "point_differential": self._point_differential,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.
        """
        self._initial_rating = parameters.get("initial_rating", self._default_initial_rating)
        self._rating = self._initial_rating

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Note: opponent references cannot be restored from serialization, so the match graph is
        reset; the rating and aggregate counts are preserved.

        Args:
            state (dict): A dictionary containing state variables.
        """
        self._rating = state.get("rating", self._initial_rating)
        self._wins = state.get("wins", 0)
        self._losses = state.get("losses", 0)
        self._ties = state.get("ties", 0)
        self._point_differential = state.get("point_differential", float(self._wins - self._losses))
        self._opponents = {}
        self._margins_for = {}

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            MasseyCompetitor: A new competitor instance.
        """
        return cls(initial_rating=parameters.get("initial_rating", cls._default_initial_rating))

    def reset(self) -> None:
        """Reset this competitor to its initial state."""
        logger.info("Resetting MasseyCompetitor %d to initial state.", self._id)
        self._rating = self._initial_rating
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._point_differential = 0.0
        self._opponents = {}
        self._margins_for = {}

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        """Configure class-level parameters for this rating system.

        Overrides the base implementation to validate the expected-score scale.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        if "expected_score_scale" in kwargs and kwargs["expected_score_scale"] <= 0:
            raise InvalidParameterException("expected_score_scale must be positive")
        super().configure_class(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<MasseyCompetitor: rating={self._rating:.3f}, W/L/T={self._wins}/{self._losses}/{self._ties}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<MasseyCompetitor: rating={self._rating:.3f}>"

    def __eq__(self, other: Any) -> bool:
        """Check if two competitors are the same object."""
        return self is other

    def __hash__(self) -> int:
        """Get a hash value for this competitor based on its object identity."""
        return object.__hash__(self)
