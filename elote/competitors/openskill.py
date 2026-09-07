"""OpenSkill rating system competitor.

A native implementation of the Weng-Lin family of online ranking algorithms
(Weng & Lin, "A Bayesian Approximation Method for Online Ranking", JMLR 2011),
exposing the Plackett-Luce model (Algorithm 4) behind elote's unified
competitor interface.

Unlike Elo or Glicko, whose updates are defined over a pair of competitors,
Weng-Lin updates are defined over the ordered participant set of a whole bout
in one closed-form step (no inner convergence loop). The competitor therefore
follows the period-native Glicko-Boost template: ``beat``/``lost_to``/``tied``
route a single result through :meth:`OpenSkillCompetitor.apply_rating_period`
as a one-row period, and multi-bout periods apply each bout in row order.

Beliefs live per instance as ``_mu``/``_sigma``. Every write bypasses the
``rating`` property (its setter refuses writes and the property derives the
conservative ordinal ``mu - 3*sigma`` from the state), so the
``_minimum_rating`` guard used by scalar rating systems can never corrupt the
mu-approx-25 scale this family uses.
"""

import math
import numbers
from statistics import NormalDist
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    MissMatchedCompetitorTypesException,
    validate_scores,
)
from elote.logging import logger

T = TypeVar("T", bound="OpenSkillCompetitor")

_normal = NormalDist()


def _phi_major(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return _normal.cdf(x)


class OpenSkillCompetitor(BaseCompetitor):
    """OpenSkill rating system competitor (Weng-Lin, Algorithm 4: Plackett-Luce).

    Each competitor carries a Gaussian belief about its own skill: a mean
    ``mu`` and a standard deviation ``sigma``. The displayed :attr:`rating` is
    the conservative ordinal ``mu - 3*sigma``, matching the convention of the
    reference ``openskill.py`` implementation. Updates are closed-form
    Bayesian approximations computed over a whole bout's ordered participant
    set, generalizing Elo-style pairwise play to ranked fields with ties.

    Class Attributes:
        _beta (float): Skill-vs-chance deviation baked into every comparison.
            Default: 25/6.
        _tau (float): Additive dynamics parameter; uncertainty is inflated by
            ``tau`` before every update so sigma never collapses to zero.
            Default: 25/300.
        _kappa (float): Floor on the variance multiplier of a posterior,
            keeping ``sigma`` strictly positive. Default: 0.0001.
        _default_mu (float): Default prior mean. Default: 25.0.
        _default_sigma (float): Default prior standard deviation. Default: 25/3.
        _supported_models (tuple of str): Model variants accepted by the
            ``model`` parameter. Only ``plackett_luce`` is implemented; the
            Bradley-Terry and Thurstone-Mosteller variants of the family will
            join this selector in a follow-up.
    """

    _beta: ClassVar[float] = 25.0 / 6.0
    _tau: ClassVar[float] = 25.0 / 300.0
    _kappa: ClassVar[float] = 0.0001
    _default_mu: ClassVar[float] = 25.0
    _default_sigma: ClassVar[float] = 25.0 / 3.0
    _supported_models: ClassVar[Tuple[str, ...]] = ("plackett_luce",)

    def __init__(self, initial_mu: Optional[float] = None, initial_sigma: Optional[float] = None, model: str = "plackett_luce"):
        """Initialize an OpenSkill competitor.

        Args:
            initial_mu (float, optional): The prior mean skill value.
                Default: _default_mu.
            initial_sigma (float, optional): The prior skill standard deviation.
                Default: _default_sigma.
            model (str, optional): Model variant from the Weng-Lin family.
                Default: ``plackett_luce``.

        Raises:
            InvalidParameterException: If the model variant is unknown or the
                initial sigma is not positive.
        """
        super().__init__()

        if model not in self._supported_models:
            raise InvalidParameterException(
                f"Unknown OpenSkill model variant: {model!r}. Supported variants: {', '.join(self._supported_models)}"
            )
        if initial_mu is None:
            initial_mu = self._default_mu
        if initial_sigma is None:
            initial_sigma = self._default_sigma
        if initial_sigma <= 0:
            raise InvalidParameterException("Initial sigma must be positive")

        # Store initial values for reset
        self._initial_mu = initial_mu
        self._initial_sigma = initial_sigma
        self._model = model

        # Current skill parameters
        self._mu = initial_mu
        self._sigma = initial_sigma
        logger.debug("Initialized OpenSkillCompetitor: mu=%.2f, sigma=%.2f, model=%s", self._mu, self._sigma, self._model)

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<OpenSkillCompetitor: mu={self._mu:.2f}, sigma={self._sigma:.2f}, rating={self.rating:.2f}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<OpenSkillCompetitor: rating={self.rating:.2f}>"

    # ------------------------------------------------------------------
    # State surface
    # ------------------------------------------------------------------

    @property
    def rating(self) -> float:
        """Get the current conservative rating of this competitor.

        The ordinal ``mu - 3*sigma`` underestimates true skill with ~99.7%
        confidence, so it only rises as demonstrated results shrink sigma.

        Returns:
            float: The current conservative rating.
        """
        return self._mu - 3 * self._sigma

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor.

        Ratings are derived from ``mu`` and ``sigma`` and cannot be written
        directly. This implementation raises an exception.

        Args:
            value (float): The new rating value.

        Raises:
            NotImplementedError: Always, as setting the rating directly is not supported.
        """
        logger.warning("Attempted to set rating directly on OpenSkillCompetitor, which is not supported.")
        raise NotImplementedError("Cannot directly set the rating of an OpenSkillCompetitor. Set mu and sigma instead.")

    @property
    def mu(self) -> float:
        """Get the current mean skill value of this competitor.

        Returns:
            float: The current mean skill value.
        """
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        """Set the current mean skill value of this competitor.

        Args:
            value (float): The new mean skill value.
        """
        self._mu = value
        logger.debug("Set OpenSkillCompetitor mu to %.2f", value)

    @property
    def sigma(self) -> float:
        """Get the current standard deviation of this competitor.

        Returns:
            float: The current standard deviation.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Set the current standard deviation of this competitor.

        Args:
            value (float): The new standard deviation value.

        Raises:
            InvalidParameterException: If the standard deviation is not positive.
        """
        if value <= 0:
            logger.error("Attempted to set non-positive sigma: %.2f", value)
            raise InvalidParameterException("Sigma must be positive")
        self._sigma = value
        logger.debug("Set OpenSkillCompetitor sigma to %.2f", value)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        This is the two-team special case of the model's win probability: the
        probability that this competitor outranks the opponent in a fresh bout.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        denominator = math.sqrt(2 * self._beta**2 + self._sigma**2 + competitor._sigma**2)
        return _phi_major((self._mu - competitor._mu) / denominator)

    # ------------------------------------------------------------------
    # Bout-level update (the algorithm's native shape)
    # ------------------------------------------------------------------

    @classmethod
    def apply_bout(cls, participants: Sequence["OpenSkillCompetitor"], *, ranks: Optional[Sequence[float]] = None) -> None:
        """Apply one bout over an ordered participant set.

        This is the update shape the Weng-Lin math is defined over: every
        participant's belief is updated once, in closed form, from the whole
        bout's outcome. Pairwise results are the two-participant special case;
        genuine N-way bouts enter here.

        Args:
            participants (sequence of OpenSkillCompetitor): The bout's
                participants, in any order (``ranks`` carries the outcome).
            ranks (sequence of float, optional): Finishing ranks, lower is
                better, equal ranks are ties. Defaults to ``0..n-1``, i.e. the
                participants given in finishing order.

        Raises:
            ValueError: If the bout has fewer than two participants, a
                participant appears twice, or ``ranks`` does not match the
                participant list.
            MissMatchedCompetitorTypesException: If a participant is not an
                ``OpenSkillCompetitor``.
        """
        bout = list(participants)
        if len(bout) < 2:
            raise ValueError(f"a bout needs at least two participants, got {len(bout)}")
        if len({id(competitor) for competitor in bout}) != len(bout):
            raise ValueError("the same competitor cannot appear twice in one bout")
        for competitor in bout:
            if not isinstance(competitor, cls):
                raise MissMatchedCompetitorTypesException(
                    f"{cls.__name__}.apply_bout only accepts {cls.__name__} competitors, got {type(competitor).__name__}"
                )

        if ranks is None:
            bout_ranks: List[int] = list(range(len(bout)))
        else:
            if len(ranks) != len(bout):
                raise ValueError(f"ranks must have one entry per participant, got {len(ranks)} for {len(bout)}")
            for rank in ranks:
                if isinstance(rank, bool) or not isinstance(rank, numbers.Real):
                    raise ValueError(f"ranks must contain only numbers, got {rank!r}")
            bout_ranks = [int(rank) for rank in ranks]

        cls._solve_bout(bout, bout_ranks)

    @classmethod
    def _solve_bout(cls, participants: List["OpenSkillCompetitor"], ranks: Sequence[int]) -> None:
        """Run the closed-form Weng-Lin update over one bout's participant set.

        Single-player teams only: each participant is its own team, so team
        strength aggregates reduce to the member's own belief. Every factor of
        the update reads a pre-update snapshot of the beliefs; the results are
        written back afterwards through direct ``_mu``/``_sigma`` assignment --
        never through the ``rating`` property, so ``_minimum_rating`` clamping
        (built for 1000+-scale Elo systems) can never clip the mu-approx-25
        scale this family uses.

        Args:
            participants (list of OpenSkillCompetitor): The bout's participants.
            ranks (sequence of int): Finishing ranks per participant, lower is
                better, equal ranks are ties.
        """
        k = len(participants)

        # Pre-update snapshot: the whole update is computed from the beliefs as
        # of the start of the bout.
        mus = [competitor._mu for competitor in participants]
        sigmas = [competitor._sigma for competitor in participants]

        # Additive dynamics: inflate uncertainty by tau first. The inflated
        # sigma is what the update consumes and what the posterior builds on,
        # so dynamics compound across bouts exactly as in the reference
        # implementation.
        inflated_sigmas = [math.sqrt(sigma**2 + cls._tau**2) for sigma in sigmas]

        # One-player teams: team strength is the member's own belief.
        team_mus = list(mus)
        team_sigma_squares = [sigma**2 for sigma in inflated_sigmas]

        # Square root of the collective skill-plus-chance variance.
        c = math.sqrt(sum(sigma_squared + cls._beta**2 for sigma_squared in team_sigma_squares))

        # sum_q[q]: sum of exp(mu/c) over every team ranked at or below team q.
        sum_q = [0.0] * k
        for i in range(k):
            exp_mu_over_c = math.exp(team_mus[i] / c)
            for q in range(k):
                if ranks[i] >= ranks[q]:
                    sum_q[q] += exp_mu_over_c

        # a[q]: how many teams share team q's rank.
        a = [sum(1 for rank in ranks if rank == ranks[q]) for q in range(k)]

        updates: List[Tuple[float, float]] = []
        for i in range(k):
            omega = 0.0
            delta = 0.0
            exp_mu_over_c = math.exp(team_mus[i] / c)

            for q in range(k):
                if ranks[q] <= ranks[i]:
                    ratio = exp_mu_over_c / sum_q[q]
                    delta += ratio * (1 - ratio) / a[q]
                    if q == i:
                        omega += (1 - ratio) / a[q]
                    else:
                        omega -= ratio / a[q]

            omega *= team_sigma_squares[i] / c
            delta *= team_sigma_squares[i] / c**2
            # Default Plackett-Luce gamma: sqrt(team variance) / c.
            delta *= math.sqrt(team_sigma_squares[i]) / c

            # One-player teams distribute the full team update to the member.
            new_mu = mus[i] + omega
            new_sigma = inflated_sigmas[i] * math.sqrt(max(1 - delta, cls._kappa))
            updates.append((new_mu, new_sigma))

        # Tied teams share an outcome, so their average mu change is applied to
        # every member of every tied team; sigma stays per-team.
        rank_groups: Dict[float, List[int]] = {}
        for i, rank in enumerate(ranks):
            rank_groups.setdefault(rank, []).append(i)
        for indices in rank_groups.values():
            if len(indices) > 1:
                average_change = sum(updates[i][0] - mus[i] for i in indices) / len(indices)
                for i in indices:
                    updates[i] = (mus[i] + average_change, updates[i][1])

        for participant, (new_mu, new_sigma) in zip(participants, updates, strict=True):
            participant._mu = new_mu
            participant._sigma = new_sigma

    # ------------------------------------------------------------------
    # Period update (the batch hook the arenas call)
    # ------------------------------------------------------------------

    @classmethod
    def _validate_period(
        cls,
        results: Sequence[Tuple[BaseCompetitor, BaseCompetitor, float, Optional[Sequence[float]]]],
    ) -> List[Tuple[List["OpenSkillCompetitor"], Tuple[int, ...]]]:
        """Validate every row of a period up front and turn rows into bouts.

        Mirrors the up-front validation of the Glicko-Boost period schedule: a
        malformed row anywhere in the batch fails the whole period before any
        belief is touched.

        Raises:
            ValueError: If an outcome is not ``1.0``, ``0.0`` or ``0.5``, if a
                score payload is invalid or inconsistent with its outcome, or
                if a row contains another rating system.
            MissMatchedCompetitorTypesException: If a row mixes rating systems.
        """
        bouts: List[Tuple[List["OpenSkillCompetitor"], Tuple[int, ...]]] = []
        for competitor_a, competitor_b, outcome, scores in results:
            if outcome not in (1.0, 0.0, 0.5):
                raise ValueError(f"outcome must be one of 1.0, 0.0 or 0.5, got {outcome!r}")
            competitor_a.verify_competitor_types(competitor_b)
            if not isinstance(competitor_a, cls) or not isinstance(competitor_b, cls):
                raise MissMatchedCompetitorTypesException(
                    f"{cls.__name__}.apply_rating_period only accepts {cls.__name__} competitors"
                )
            validate_scores(scores, outcome)

            if outcome == 1.0:
                ranks = (0, 1)
            elif outcome == 0.0:
                ranks = (1, 0)
            else:
                ranks = (0, 0)
            bouts.append(([competitor_a, competitor_b], ranks))
        return bouts

    @classmethod
    def apply_rating_period(
        cls,
        results: Sequence[Tuple[BaseCompetitor, BaseCompetitor, float, Optional[Sequence[float]]]],
        *,
        period_end: Optional[Any] = None,
    ) -> None:
        """Apply results that share one rating period, using the Weng-Lin bout update.

        This is where the whole algorithm lives: the pairwise methods route a
        single result through here as a one-row period, so every caller gets
        the same formulas. Each row is one two-participant bout, applied in row
        order; a bout's update always reads the beliefs as of its start, so a
        repeated sequence of bouts reproduces the reference implementation's
        per-bout ``rate`` calls exactly.

        Args:
            results: ``(competitor_a, competitor_b, outcome, scores)`` tuples.
                Outcomes use ``1.0`` for an A win, ``0.0`` for a B win and
                ``0.5`` for a draw; scores follow the usual optional
                caller-order contract and are validated but not consumed.
            period_end: Accepted for interface parity with the arena's period
                plumbing. The Weng-Lin update has no temporal dynamics term
                (uncertainty inflation is handled by ``_tau``), so it is unused.

        Raises:
            ValueError: If an outcome or score payload is invalid.
            MissMatchedCompetitorTypesException: If a row contains another rating system.
        """
        bouts = cls._validate_period(results)
        if not bouts:
            return

        for participants, ranks in bouts:
            cls._solve_bout(participants, ranks)
        logger.debug("Applied an OpenSkill rating period over %d bouts", len(bouts))

    # ------------------------------------------------------------------
    # Pairwise convenience methods (one-row periods)
    # ------------------------------------------------------------------

    def beat(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has won against the given competitor.

        The result is applied as a one-bout rating period in which this
        competitor finished ahead of the opponent.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            scores (sequence of float, optional): The two scores in caller order.
                Validated only; the Plackett-Luce update is rank-based.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` is malformed or does not describe a win.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 1.0)
        self.__class__.apply_rating_period([(self, competitor, 1.0, scores)])

    def lost_to(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has lost to the given competitor.

        The result is applied as a one-bout rating period in which the
        opponent finished ahead of this competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.
            scores (sequence of float, optional): The two scores in caller order.
                Validated only; the Plackett-Luce update is rank-based.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` is malformed or does not describe a loss.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.0)
        self.__class__.apply_rating_period([(self, competitor, 0.0, scores)])

    def tied(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        The result is applied as a one-bout rating period in which both
        participants share the first rank.

        Args:
            competitor (BaseCompetitor): The opponent competitor that drew.
            scores (sequence of float, optional): The two scores in caller order.
                Must be equal. Validated only; the Plackett-Luce update is rank-based.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            ValueError: If ``scores`` is malformed or is not a draw.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.5)
        self.__class__.apply_rating_period([(self, competitor, 0.5, scores)])

    # ------------------------------------------------------------------
    # Serialization: parameters vs current state
    # ------------------------------------------------------------------

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_mu": self._initial_mu,
            "initial_sigma": self._initial_sigma,
            "model": self._model,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "mu": self._mu,
            "sigma": self._sigma,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        logger.debug("Importing parameters for OpenSkillCompetitor: %s", parameters)
        model = parameters.get("model", "plackett_luce")
        if model not in self._supported_models:
            raise InvalidParameterException(f"Unknown OpenSkill model variant: {model!r}")

        initial_mu = parameters.get("initial_mu", self._default_mu)
        initial_sigma = parameters.get("initial_sigma", self._default_sigma)
        if initial_sigma <= 0:
            raise InvalidParameterException("Initial sigma must be positive")

        self._initial_mu = initial_mu
        self._initial_sigma = initial_sigma
        self._model = model

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        logger.debug("Importing current state for OpenSkillCompetitor: %s", state)
        mu = state.get("mu", self._initial_mu)
        sigma = state.get("sigma", self._initial_sigma)
        if sigma <= 0:
            raise InvalidParameterException("Sigma must be positive")
        self._mu = mu
        self._sigma = sigma

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            OpenSkillCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_mu=parameters.get("initial_mu", cls._default_mu),
            initial_sigma=parameters.get("initial_sigma", cls._default_sigma),
            model=parameters.get("model", "plackett_luce"),
        )

    def reset(self) -> None:
        """Reset this competitor to its initial state."""
        logger.info("Resetting OpenSkillCompetitor to initial state (mu=%.2f, sigma=%.2f)", self._initial_mu, self._initial_sigma)
        self._mu = self._initial_mu
        self._sigma = self._initial_sigma
