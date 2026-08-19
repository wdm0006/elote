"""Keener Ratings implementation for the Elote library.

Keener's method is the score-based member of the classical global-fit family. Where
:class:`~elote.competitors.colley.ColleyMatrixCompetitor` and
:class:`~elote.competitors.massey.MasseyCompetitor` solve a linear system, Keener builds a
square *preference matrix* from the points competitors have scored on one another and reads
the ratings off that matrix's dominant eigenvector.

The construction has four steps. Let ``S_ij`` be the total number of points ``i`` has scored
against ``j`` across all their meetings.

1. **Smoothed preference.** Raw score shares are unstable for lopsided or barely-played
   pairs, so Keener applies Laplace smoothing:

   .. math::

      a_{ij} = \\frac{S_{ij} + 1}{S_{ij} + S_{ji} + 2}

   A pair that has never met gives ``a_ij = 1/2`` -- no evidence either way.

2. **Skew transform.** ``a_ij`` is pushed away from the middle by Keener's skew function

   .. math::

      h(x) = \\frac{1}{2} + \\frac{1}{2}\\,\\mathrm{sgn}\\!\\left(x - \\frac{1}{2}\\right)
             \\sqrt{\\left|2x - 1\\right|}

   which is monotone, fixes ``0``, ``1/2`` and ``1``, and satisfies ``h(x) + h(1 - x) = 1``,
   so the matrix stays antisymmetric about ``1/2``. Its square root damps the influence of
   very large margins, which is what stops the method rewarding running up the score without
   limit.

3. **Games-played normalization.** Row ``i`` is divided by the number of games ``i`` played,
   so a competitor cannot accumulate rating merely by playing more often.

4. **Stabilization.** A small positive constant is added to every entry. Keener's own
   ``A + eps * E`` perturbation makes the matrix strictly positive, so Perron-Frobenius
   applies: the dominant eigenvalue is real and simple and its eigenvector is strictly
   positive and unique up to scale. Without it a schedule whose graph is bipartite or
   otherwise imprimitive has no single dominant eigenvector to read.

The ratings of a connected group are then that dominant eigenvector, scaled so they average
exactly ``1.0``.

References:
- Keener, J. P. (1993). The Perron-Frobenius Theorem and the Ranking of Football Teams.
  SIAM Review, 35(1), 80-93.
- Langville, A. N., & Meyer, C. D. (2012). Who's #1? The Science of Rating and Ranking.
  Princeton University Press, chapter 4.
"""

import math
from typing import Dict, Any, ClassVar, Optional, Sequence, Type, TypeVar, List, cast, Set

import numpy as np

from elote.competitors.base import BaseCompetitor, InvalidParameterException, InvalidRatingValueException
from elote.logging import logger

T = TypeVar("T", bound="KeenerCompetitor")


class KeenerCompetitor(BaseCompetitor):
    """Keener Ratings competitor.

    Keener rates a connected population from the dominant eigenvector of a pairwise
    preference matrix built from points scored. Like Colley, Massey and Bradley-Terry -- and
    unlike Elo -- ratings are not nudged after each result; the whole connected group is
    re-fit from the complete match history, which makes the method order independent.

    **Scores.** ``beat`` / ``lost_to`` / ``tied`` accept the common optional ``scores``
    payload, the two competitors' scores in caller order. Keener is a score-based method, so
    supplying real scores is what it is built for. When they are omitted the implementation
    falls back to the same unit scores the rest of the library uses: the winner is credited
    ``1`` and the loser ``0``, and a draw credits each side ``0.5``. Unit-score Keener still
    produces a sensible ranking, but it only sees who beat whom.

    Key characteristics:
    - Global eigenvector fit (does not depend on the order of results)
    - Ratings are strictly positive and average exactly 1.0 within a connected group
    - Large margins have a damped, not linear, influence thanks to the skew transform

    Class Attributes:
        _minimum_rating (float): The minimum allowed rating value. Default: 0.0. Keener
            ratings are strictly positive, so no Elo-style floor applies.
        _default_initial_rating (float): Default initial rating. Default: 1.0, which is the
            mean the fitted ratings are normalized to, so an unplayed competitor sits exactly
            at the population average.
        _perturbation (float): Keener's ``eps``, added to every matrix entry so the matrix is
            strictly positive and Perron-Frobenius applies. Default: 1e-4.
        _expected_score_scale (float): Logistic scale applied to the log rating ratio by
            :meth:`expected_score`. Default: 1.0, which is the plain Keener share
            ``r_a / (r_a + r_b)``.
        _round_decimals (int): Number of decimal places fitted ratings are canonicalized to,
            so that solver noise cannot make mathematically identical records differ.
            Default: 10, chosen because the eigen-solve's row-order noise is around 5e-15 --
            five orders of magnitude below the rounding grid.
    """

    _minimum_rating: ClassVar[float] = 0.0
    _default_initial_rating: ClassVar[float] = 1.0
    _perturbation: ClassVar[float] = 1e-4
    _expected_score_scale: ClassVar[float] = 1.0
    _round_decimals: ClassVar[int] = 10

    # Ratings are strictly positive by construction; this only guards expected_score against
    # a rating a caller has explicitly set to zero.
    _probability_floor: ClassVar[float] = 1e-300

    def __init__(self, initial_rating: Optional[float] = None):
        """Initialize a new Keener competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor.
                Default: 1.0.

        Raises:
            InvalidRatingValueException: If the initial rating is not positive.
        """
        super().__init__()

        self._initial_rating = initial_rating if initial_rating is not None else self._default_initial_rating
        if self._initial_rating <= 0:
            raise InvalidRatingValueException("Keener ratings are strictly positive; initial_rating must be > 0")
        self._rating = self._initial_rating
        self._wins = 0
        self._losses = 0
        self._ties = 0
        # Aggregate score totals across every game played. These are the serializable
        # summary of the score history; the per-opponent breakdown below is not.
        self._points_for = 0.0
        self._points_against = 0.0
        self._opponents: Dict["KeenerCompetitor", int] = {}  # Opponent -> num games
        # Opponent -> total points this competitor has scored against that opponent. This is
        # the S_ij the preference matrix is built from.
        self._scores_for: Dict["KeenerCompetitor", float] = {}
        # Unique ID for hashing (instances are used as dict keys in the match graph).
        self._id = id(self)
        logger.debug("Initialized KeenerCompetitor %d with initial rating %.6f", self._id, self._initial_rating)

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
                "Attempted to set rating %.6f below minimum %.6f for %d", value, self._minimum_rating, self._id
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

        Keener ratings are positive strengths rather than probabilities, so the natural
        mapping is the share ``r_self / (r_self + r_competitor)``. That share is computed
        here as a logistic of the log rating ratio, which is the same quantity written in a
        form that is exactly complementary in floating point: two equal ratings give exactly
        0.5, and the two argument orders sum to exactly 1.0.

        Args:
            competitor (BaseCompetitor): The competitor to compare against.

        Returns:
            float: The expected score (probability of winning).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        mine = max(self.rating, self._probability_floor)
        theirs = max(competitor.rating, self._probability_floor)
        log_ratio = math.log(mine) - math.log(theirs)
        return float(0.5 * (1.0 + np.tanh(0.5 * self._expected_score_scale * log_ratio)))

    def beat(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has won against the given competitor.

        The scores are recorded in the match graph and the whole connected group is re-fit.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. When omitted the unit scores ``1`` and
                ``0`` are recorded.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` does not describe a win for this competitor.
        """
        logger.debug("Competitor %s beat %s", self, competitor)
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 1.0)
        opponent = cast(KeenerCompetitor, competitor)
        mine, theirs = (1.0, 0.0) if validated is None else validated

        self._wins += 1
        opponent._losses += 1
        self._record_game(opponent, mine, theirs)

        logger.debug("Recorded win for %d (%.3f-%.3f), loss for %d", self._id, mine, theirs, opponent._id)
        self._recalculate_ratings()

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
        logger.debug("Competitor %s tied with %s", self, competitor)
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.5)
        opponent = cast(KeenerCompetitor, competitor)
        mine, theirs = (0.5, 0.5) if validated is None else validated

        self._ties += 1
        opponent._ties += 1
        self._record_game(opponent, mine, theirs)

        logger.debug("Recorded tie for %d and %d (%.3f-%.3f)", self._id, opponent._id, mine, theirs)
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
            ValueError: If ``scores`` does not describe a loss for this competitor.
        """
        logger.debug("Competitor %s lost to %s", self, competitor)
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.0)
        competitor.beat(self, scores=None if validated is None else (validated[1], validated[0]))

    def _record_game(self, opponent: "KeenerCompetitor", mine: float, theirs: float) -> None:
        """Record one game's scores on both sides of the match graph.

        Args:
            opponent: The other competitor in the game.
            mine: The points this competitor scored.
            theirs: The points the opponent scored.
        """
        self._points_for += mine
        self._points_against += theirs
        self._opponents[opponent] = self._opponents.get(opponent, 0) + 1
        self._scores_for[opponent] = self._scores_for.get(opponent, 0.0) + mine

        opponent._points_for += theirs
        opponent._points_against += mine
        opponent._opponents[self] = opponent._opponents.get(self, 0) + 1
        opponent._scores_for[self] = opponent._scores_for.get(self, 0.0) + theirs

    def _get_connected_competitors(self) -> List["KeenerCompetitor"]:
        """Get all competitors connected to this competitor in the match graph.

        Returns:
            List[KeenerCompetitor]: A list of all connected competitors.
        """
        visited: Set["KeenerCompetitor"] = set()
        to_visit: List["KeenerCompetitor"] = [self]
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

    @classmethod
    def _preference_matrix(cls, scores: "np.ndarray", pair_games: "np.ndarray", games: "np.ndarray") -> "np.ndarray":
        """Build Keener's stabilized preference matrix from a raw score matrix.

        Args:
            scores: ``S``, where ``S[i, j]`` is the total number of points ``i`` scored
                against ``j``.
            pair_games: ``G``, where ``G[i, j]`` is the number of games ``i`` played
                against ``j``.
            games: The number of games each competitor played in total.

        Returns:
            numpy.ndarray: The strictly positive preference matrix.
        """
        totals = scores + scores.T
        # Laplace-smoothed score share.
        proportions = (scores + 1.0) / (totals + 2.0)

        # Keener's skew transform: monotone, fixes 1/2, and h(x) + h(1 - x) == 1.
        centered = 2.0 * proportions - 1.0
        preferences = 0.5 + 0.5 * np.sign(centered) * np.sqrt(np.abs(centered))

        # A pair that never met expresses no preference, and neither does a competitor
        # against itself. Leaving unplayed pairs at the "no evidence" value h(1/2) = 1/2
        # instead would swamp the row: on a sparse schedule the n - games_i entries for
        # opponents never faced dominate the games_i real ones, and the fit degenerates
        # into a ranking by 1 / games_played.
        preferences = np.where(pair_games > 0, preferences, 0.0)
        np.fill_diagonal(preferences, 0.0)

        # Divide by games played, making each row the competitor's average preference per
        # game, so a longer schedule is not itself worth rating.
        preferences = preferences / np.maximum(games, 1.0)[:, None]

        # Keener's eps * E perturbation: makes the matrix strictly positive, so its dominant
        # eigenvalue is simple and its eigenvector strictly positive. This is also what
        # connects competitors who never met, which the mask above disconnected.
        return cast("np.ndarray", preferences + cls._perturbation)

    def _recalculate_ratings(self) -> None:
        """Re-fit the Keener ratings for all connected competitors.

        Builds the preference matrix for the connected group, takes the eigenvector of its
        dominant eigenvalue, and scales the result so the group's ratings average 1.0.
        """
        competitors = self._get_connected_competitors()
        n = len(competitors)
        if n <= 1:
            logger.debug("Only one competitor in network, skipping recalculation.")
            return

        idx = {comp: i for i, comp in enumerate(competitors)}

        scores = np.zeros((n, n), dtype=np.float64)
        pair_games = np.zeros((n, n), dtype=np.float64)
        games = np.zeros(n, dtype=np.float64)
        for i, comp in enumerate(competitors):
            games[i] = comp.num_games
            for opponent, points in comp._scores_for.items():
                j = idx.get(opponent)
                if j is not None:
                    scores[i, j] = points
            for opponent, count in comp._opponents.items():
                j = idx.get(opponent)
                if j is not None:
                    pair_games[i, j] = count

        matrix = self._preference_matrix(scores, pair_games, games)

        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
        except np.linalg.LinAlgError as e:
            logger.warning(
                "Keener eigenvector solve failed (%s). Falling back to mean-preference ratings for %d competitors.",
                str(e),
                n,
            )
            self._fallback_rating_calculation(competitors, matrix)
            return

        # The matrix is strictly positive, so by Perron-Frobenius the eigenvalue of largest
        # magnitude is real, simple and positive, and its eigenvector can be taken positive.
        dominant = int(np.argmax(eigenvalues.real))
        vector = np.real(eigenvectors[:, dominant])
        if vector.sum() < 0:
            # LAPACK returns eigenvectors up to sign; orient the Perron vector positive.
            vector = -vector

        if not np.all(np.isfinite(vector)) or vector.sum() <= 0:
            logger.warning("Keener eigenvector was degenerate for %d competitors; falling back.", n)
            self._fallback_rating_calculation(competitors, matrix)
            return

        vector = np.maximum(vector, self._probability_floor)
        ratings = vector * (n / vector.sum())

        # LAPACK's pivoting follows row order, and row order is the order competitors were
        # discovered in, so two runs over the same results can differ in the last few bits.
        # Canonicalizing makes mathematically identical records compare exactly equal and
        # makes the fit order independent.
        ratings = np.round(ratings, decimals=self._round_decimals)

        for i, comp in enumerate(competitors):
            comp.rating = float(ratings[i])

    def _fallback_rating_calculation(self, competitors: List["KeenerCompetitor"], matrix: "np.ndarray") -> None:
        """Assign mean-preference ratings when the eigenvector cannot be computed.

        Args:
            competitors: The connected group of competitors to rate.
            matrix: The preference matrix that was built for them.
        """
        means = matrix.sum(axis=1)
        total = float(means.sum())
        n = len(competitors)
        if total <= 0:
            values = np.full(n, 1.0)
        else:
            values = means * (n / total)
        for comp, value in zip(competitors, values, strict=True):
            comp.rating = float(max(value, self._probability_floor))

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
            "points_for": self._points_for,
            "points_against": self._points_against,
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
        reset; the rating and the aggregate score totals are preserved.

        Args:
            state (dict): A dictionary containing state variables.
        """
        self._rating = state.get("rating", self._initial_rating)
        self._wins = state.get("wins", 0)
        self._losses = state.get("losses", 0)
        self._ties = state.get("ties", 0)
        self._points_for = state.get("points_for", 0.0)
        self._points_against = state.get("points_against", 0.0)
        self._opponents = {}
        self._scores_for = {}

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            KeenerCompetitor: A new competitor instance.
        """
        return cls(initial_rating=parameters.get("initial_rating", cls._default_initial_rating))

    def reset(self) -> None:
        """Reset this competitor to its initial state."""
        logger.info("Resetting KeenerCompetitor %d to initial state.", self._id)
        self._rating = self._initial_rating
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._points_for = 0.0
        self._points_against = 0.0
        self._opponents = {}
        self._scores_for = {}

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        """Configure class-level parameters for this rating system.

        Overrides the base implementation to validate the Keener-specific parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        if "expected_score_scale" in kwargs and kwargs["expected_score_scale"] <= 0:
            raise InvalidParameterException("expected_score_scale must be positive")
        if "perturbation" in kwargs and kwargs["perturbation"] <= 0:
            raise InvalidParameterException("perturbation must be positive")
        super().configure_class(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<KeenerCompetitor: rating={self._rating:.4f}, W/L/T={self._wins}/{self._losses}/{self._ties}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<KeenerCompetitor: rating={self._rating:.4f}>"

    def __eq__(self, other: Any) -> bool:
        """Check if two competitors are the same object."""
        return self is other

    def __hash__(self) -> int:
        """Get a hash value for this competitor based on its object identity."""
        return object.__hash__(self)
