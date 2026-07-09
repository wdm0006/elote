"""
Bradley-Terry model implementation for the Elote library.

The Bradley-Terry model is a probabilistic model for paired comparisons. Each competitor
is assigned a latent strength, and the probability that competitor ``i`` beats competitor
``j`` is ``p_i / (p_i + p_j)``. Working in the log domain with ``p_i = exp(beta_i)`` this is
the logistic function ``sigmoid(beta_i - beta_j)`` -- the same functional form as the Elo
expected score.

Unlike Elo, which nudges ratings incrementally after every game, the Bradley-Terry strengths
are obtained by maximum likelihood estimation over the full set of observed comparisons. This
implementation follows the same approach as :class:`~elote.competitors.colley.ColleyMatrixCompetitor`:
each competitor accumulates its match history and, after every result, the strengths of the
whole connected component are re-fit. The fit uses the minorization-maximization (MM) update of
Hunter (2004) with geometric-mean normalization and a small amount of regularization so that a
unique, finite solution exists even when a competitor is undefeated or winless within its
component.

References:
- Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The
  Method of Paired Comparisons. Biometrika, 39(3/4), 324-345.
- Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models. The Annals of
  Statistics, 32(1), 384-406.
"""

import math
from typing import Dict, Any, ClassVar, Type, TypeVar, List, Optional, cast, Set

from elote.competitors.base import BaseCompetitor, InvalidParameterException
from elote.logging import logger

T = TypeVar("T", bound="BradleyTerryCompetitor")


class BradleyTerryCompetitor(BaseCompetitor):
    """Bradley-Terry model competitor.

    The Bradley-Terry model assigns each competitor a latent strength and models the
    probability that one competitor beats another as a function of the difference of their
    log-strengths. Unlike Elo, which updates ratings incrementally, Bradley-Terry re-fits the
    strengths of every connected competitor via maximum likelihood after each result.

    Ratings are reported on an Elo-like scale via ``rating = anchor + scale * beta``, where
    ``beta`` is the internal log-strength (re-centered so the mean log-strength of a component
    is zero). Choosing ``scale = 400 / ln(10)`` makes :meth:`expected_score` numerically
    identical to the Elo expected score, so ratings are directly comparable to Elo ratings.

    Key characteristics:
    - Global maximum-likelihood fit (does not depend on the order of results)
    - Considers only wins and losses (ties are counted as half a win for each side)
    - Regularized so a unique, finite fit exists even for undefeated/winless competitors

    Class Attributes:
        _minimum_rating (float): The minimum allowed rating value. Default: 0.0.
        _anchor_rating (float): Rating assigned to the mean log-strength. Default: 1500.0.
        _scale (float): Points per unit of log-strength. Default: 400 / ln(10).
        _reg (float): Regularization strength (virtual wins/losses against an average
            phantom opponent). Default: 0.1.
        _max_iter (int): Maximum MM iterations per fit. Default: 10000.
        _tol (float): Convergence tolerance on the max change in log-strength. Default: 1e-8.
    """

    _minimum_rating: ClassVar[float] = 0.0
    _anchor_rating: ClassVar[float] = 1500.0
    _scale: ClassVar[float] = 400.0 / math.log(10.0)
    _reg: ClassVar[float] = 0.1
    _max_iter: ClassVar[int] = 10000
    _tol: ClassVar[float] = 1e-8

    def __init__(self, initial_rating: Optional[float] = None):
        """Initialize a new Bradley-Terry competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor, on the
                Elo-like reporting scale. Default: the anchor rating (1500).

        Raises:
            InvalidParameterException: If the regularization or iteration parameters are invalid.
        """
        super().__init__()  # Call base class constructor

        self._initial_rating = initial_rating if initial_rating is not None else self._anchor_rating
        # Internal log-strength; derived so that rating == anchor + scale * beta.
        self._beta = (self._initial_rating - self._anchor_rating) / self._scale
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._opponents: Dict["BradleyTerryCompetitor", float] = {}  # Opponent -> num games
        self._head_to_head: Dict["BradleyTerryCompetitor", float] = {}  # Opponent -> wins against
        # Unique ID for hashing (instances are used as dict keys in the match graph).
        self._id = id(self)
        logger.debug(
            "Initialized BradleyTerryCompetitor %d with initial rating %.3f", self._id, self._initial_rating
        )

    @property
    def rating(self) -> float:
        """Get the current rating of this competitor on the Elo-like reporting scale.

        Returns:
            float: The current rating.
        """
        return self._anchor_rating + self._scale * self._beta

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor on the Elo-like reporting scale.

        Args:
            value (float): The new rating value.
        """
        logger.debug("Setting rating for competitor %d to %.3f", self._id, value)
        self._beta = (value - self._anchor_rating) / self._scale

    @property
    def num_games(self) -> int:
        """Get the total number of games played by this competitor.

        Returns:
            int: The total number of games played.
        """
        return self._wins + self._losses + self._ties

    def expected_score(self, competitor: "BaseCompetitor") -> float:
        """Calculate the expected score against another competitor.

        Args:
            competitor (BaseCompetitor): The competitor to compare against.

        Returns:
            float: The expected score (probability of winning).
        """
        self.verify_competitor_types(competitor)
        other = cast(BradleyTerryCompetitor, competitor)
        return 1.0 / (1.0 + math.exp(-(self._beta - other._beta)))

    def beat(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has won against the given competitor.

        The result is recorded on both competitors and the strengths of the entire connected
        component are re-fit by maximum likelihood.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s beat %s", self, competitor)
        self.verify_competitor_types(competitor)
        opponent = cast(BradleyTerryCompetitor, competitor)

        self._wins += 1
        self._opponents[opponent] = self._opponents.get(opponent, 0) + 1
        self._head_to_head[opponent] = self._head_to_head.get(opponent, 0) + 1

        opponent._losses += 1
        opponent._opponents[self] = opponent._opponents.get(self, 0) + 1
        opponent._head_to_head.setdefault(self, 0)

        logger.debug("Recorded win for %d, loss for %d", self._id, opponent._id)
        self._recalculate_ratings()

    def tied(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        A tie is counted as half a win for each side.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s tied with %s", self, competitor)
        self.verify_competitor_types(competitor)
        opponent = cast(BradleyTerryCompetitor, competitor)

        self._ties += 1
        self._opponents[opponent] = self._opponents.get(opponent, 0) + 1
        self._head_to_head[opponent] = self._head_to_head.get(opponent, 0) + 0.5

        opponent._ties += 1
        opponent._opponents[self] = opponent._opponents.get(self, 0) + 1
        opponent._head_to_head[self] = opponent._head_to_head.get(self, 0) + 0.5

        logger.debug("Recorded tie for %d and %d", self._id, opponent._id)
        self._recalculate_ratings()

    def lost_to(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has lost to the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        logger.debug("Competitor %s lost to %s", self, competitor)
        self.verify_competitor_types(competitor)
        competitor.beat(self)

    def _get_connected_competitors(self) -> List["BradleyTerryCompetitor"]:
        """Get all competitors connected to this competitor in the match graph.

        Returns:
            List[BradleyTerryCompetitor]: A list of all connected competitors.
        """
        visited: Set["BradleyTerryCompetitor"] = set()
        to_visit: List["BradleyTerryCompetitor"] = [self]
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
        """Re-fit the Bradley-Terry strengths for all connected competitors.

        Uses the minorization-maximization (MM) update of Hunter (2004) with geometric-mean
        normalization and a small regularization term (virtual results against an average phantom
        opponent) so that the maximum-likelihood estimate exists and is unique even when a
        competitor is undefeated or winless within its component.
        """
        competitors = self._get_connected_competitors()
        n = len(competitors)
        if n <= 1:
            logger.debug("Only one competitor in network, skipping recalculation.")
            return

        idx = {comp: i for i, comp in enumerate(competitors)}

        # Win counts: wins[i][j] = number of times i beat j (ties count as 0.5 to each side).
        wins = [[0.0] * n for _ in range(n)]
        for i, comp in enumerate(competitors):
            for opponent, w in comp._head_to_head.items():
                j = idx.get(opponent)
                if j is not None:
                    wins[i][j] += w

        # Total games between each pair (symmetric).
        games = [[wins[i][j] + wins[j][i] for j in range(n)] for i in range(n)]

        # Total wins per competitor (ties are already counted as half in the win matrix).
        total_wins = [math.fsum(wins[i]) for i in range(n)]

        # Strengths p_i = exp(beta_i). The Bradley-Terry log-likelihood is concave with a
        # unique maximum, so we seed from a flat, well-conditioned starting point rather than
        # warm-starting from the (possibly extreme) current strengths.
        p = [1.0] * n

        reg = self._reg
        for iteration in range(self._max_iter):
            new_p = [0.0] * n
            for i in range(n):
                denominator = 0.0
                for j in range(n):
                    if i == j or games[i][j] == 0:
                        continue
                    denominator += games[i][j] / (p[i] + p[j])
                # Regularization: a virtual win and loss against a phantom opponent of unit
                # strength, guaranteeing a finite, unique solution even for undefeated or
                # winless competitors.
                numerator = total_wins[i] + reg
                denominator += 2.0 * reg / (p[i] + 1.0)
                new_p[i] = numerator / denominator if denominator > 0 else p[i]

            # Normalize by the geometric mean to fix the overall scale.
            log_sum = sum(math.log(v) for v in new_p if v > 0)
            geo_mean = math.exp(log_sum / n)
            if geo_mean > 0:
                new_p = [v / geo_mean for v in new_p]

            max_delta = max(abs(math.log(new_p[i]) - math.log(p[i])) for i in range(n) if new_p[i] > 0)
            p = new_p
            if max_delta < self._tol:
                logger.debug("Bradley-Terry fit converged in %d iterations", iteration + 1)
                break

        # Write back re-centered log-strengths (mean beta == 0).
        betas = [math.log(v) for v in p]
        mean_beta = sum(betas) / n
        for i, comp in enumerate(competitors):
            comp._beta = betas[i] - mean_beta

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
            "rating": self.rating,
            "beta": self._beta,
            "wins": self._wins,
            "losses": self._losses,
            "ties": self._ties,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.
        """
        self._initial_rating = parameters.get("initial_rating", self._anchor_rating)
        self._beta = (self._initial_rating - self._anchor_rating) / self._scale

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Note: opponent references cannot be restored from serialization, so the match graph is
        reset; the rating and aggregate win/loss/tie counts are preserved.

        Args:
            state (dict): A dictionary containing state variables.
        """
        if "beta" in state:
            self._beta = state["beta"]
        elif "rating" in state:
            self._beta = (state["rating"] - self._anchor_rating) / self._scale
        self._wins = state.get("wins", 0)
        self._losses = state.get("losses", 0)
        self._ties = state.get("ties", 0)
        self._opponents = {}
        self._head_to_head = {}

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            BradleyTerryCompetitor: A new competitor instance.
        """
        return cls(initial_rating=parameters.get("initial_rating", cls._anchor_rating))

    def reset(self) -> None:
        """Reset this competitor to its initial state."""
        logger.info("Resetting BradleyTerryCompetitor %d to initial state.", self._id)
        self._beta = (self._initial_rating - self._anchor_rating) / self._scale
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._opponents = {}
        self._head_to_head = {}

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        """Configure class-level parameters for this rating system.

        Overrides the base implementation to validate the regularization and iteration
        parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        if kwargs.get("reg", 0.0) < 0:
            raise InvalidParameterException("reg must be non-negative")
        if kwargs.get("max_iter", 1) < 1:
            raise InvalidParameterException("max_iter must be at least 1")
        if kwargs.get("scale", 1.0) <= 0:
            raise InvalidParameterException("scale must be positive")
        super().configure_class(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of this competitor."""
        return (
            f"<BradleyTerryCompetitor: rating={self.rating:.3f}, "
            f"W/L/T={self._wins}/{self._losses}/{self._ties}>"
        )

    def __str__(self) -> str:
        """Return a string representation of this competitor."""
        return f"<BradleyTerryCompetitor: rating={self.rating:.3f}>"

    def __eq__(self, other: Any) -> bool:
        """Check if two competitors are the same object (by unique ID)."""
        if not isinstance(other, BradleyTerryCompetitor):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        """Get a hash value for this competitor based on its unique ID."""
        return hash(self._id)
