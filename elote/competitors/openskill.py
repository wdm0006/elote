"""OpenSkill rating system competitor.

A native implementation of the Weng-Lin family of online ranking algorithms
(Weng & Lin, "A Bayesian Approximation Method for Online Ranking", JMLR 2011),
exposing all four models of the paper behind elote's unified competitor
interface via the ``model`` parameter:

- ``plackett_luce`` -- Algorithm 4 (default), the recommended general-purpose
  variant and the only one that generalizes to full orderings natively.
- ``bradley_terry_full`` -- Algorithm 1, logistic performance model, every
  participant compared against every other.
- ``bradley_terry_partial`` -- Algorithm 2, logistic model over a bounded
  pairing window; cheaper and more local than full pairing.
- ``thurstone`` -- Algorithm 3, Gaussian (Thurstone-Mosteller) performance
  model with an explicit draw margin ``epsilon``.

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
import sys
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


def _phi_minor(x: float) -> float:
    """Standard normal probability density function."""
    return _normal.pdf(x)


def _truncated_v(x: float, t: float) -> float:
    """The paper's ``V(x, t)`` (Weng-Lin 2011, Appendix E): the first
    derivative of the log of the normal tail probability.

    Used by the Thurstone-Mosteller update for a decisive pairwise result.
    The ``t`` argument is the draw margin expressed on the standardized
    scale. Mirrors the reference implementation's numerical guards.
    """
    xt = x - t
    denominator = _phi_major(xt)
    if denominator < sys.float_info.epsilon:
        return -xt
    return _phi_minor(xt) / denominator


def _truncated_w(x: float, t: float) -> float:
    """The paper's ``W(x, t)``: the second-moment companion of :func:`_truncated_v`.

    Used by the Thurstone-Mosteller update for a decisive pairwise result.
    """
    xt = x - t
    denominator = _phi_major(xt)
    if denominator < sys.float_info.epsilon:
        return 1.0 if x < 0 else 0.0
    v_value = _truncated_v(x, t)
    return v_value * (v_value + xt)


def _truncated_v_tie(x: float, t: float) -> float:
    """The paper's ``V-tilde(x, t)``: the tie analogue of :func:`_truncated_v`
    for a pairwise result inside the draw margin.
    """
    xx = abs(x)
    b = _phi_major(t - xx) - _phi_major(-t - xx)
    if b < 1e-5:
        if x < 0:
            return -x - t
        return -x + t
    a = _phi_minor(-t - xx) - _phi_minor(t - xx)
    return (-a if x < 0 else a) / b


def _truncated_w_tie(x: float, t: float) -> float:
    """The paper's ``W-tilde(x, t)``: the tie analogue of :func:`_truncated_w`
    for a pairwise result inside the draw margin.
    """
    xx = abs(x)
    b = _phi_major(t - xx) - _phi_major(-t - xx)
    if b < sys.float_info.epsilon:
        return 1.0
    v_tie = _truncated_v_tie(x, t)
    return ((t - xx) * _phi_minor(t - xx) + (t + xx) * _phi_minor(-t - xx)) / b + v_tie * v_tie


class OpenSkillCompetitor(BaseCompetitor):
    """OpenSkill rating system competitor (Weng-Lin family, Algorithms 1-4).

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
        _epsilon (float): Draw margin for the Thurstone-Mosteller variant,
            expressed on the raw skill scale. Default: 0.1.
        _partial_pairing_window (int): Pairing window (in sorted finishing
            positions on either side) for the ``bradley_terry_partial``
            variant. Default: 4.
        _default_mu (float): Default prior mean. Default: 25.0.
        _default_sigma (float): Default prior standard deviation. Default: 25/3.
        _supported_models (tuple of str): Model variants accepted by the
            ``model`` parameter: ``plackett_luce``, ``bradley_terry_full``,
            ``bradley_terry_partial`` and ``thurstone``.
    """

    _beta: ClassVar[float] = 25.0 / 6.0
    _tau: ClassVar[float] = 25.0 / 300.0
    _kappa: ClassVar[float] = 0.0001
    _epsilon: ClassVar[float] = 0.1
    _partial_pairing_window: ClassVar[int] = 4
    _default_mu: ClassVar[float] = 25.0
    _default_sigma: ClassVar[float] = 25.0 / 3.0
    _supported_models: ClassVar[Tuple[str, ...]] = (
        "plackett_luce",
        "bradley_terry_full",
        "bradley_terry_partial",
        "thurstone",
    )

    def __init__(self, initial_mu: Optional[float] = None, initial_sigma: Optional[float] = None, model: str = "plackett_luce"):
        """Initialize an OpenSkill competitor.

        Args:
            initial_mu (float, optional): The prior mean skill value.
                Default: _default_mu.
            initial_sigma (float, optional): The prior skill standard deviation.
                Default: _default_sigma.
            model (str, optional): Model variant from the Weng-Lin family.
                One of ``plackett_luce`` (Algorithm 4, default),
                ``bradley_terry_full`` (Algorithm 1), ``bradley_terry_partial``
                (Algorithm 2) or ``thurstone`` (Algorithm 3,
                Thurstone-Mosteller). All variants share the belief state,
                serialization and bout plumbing; only the closed-form update
                differs.

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
    def apply_bout(cls, participants: Sequence[Any], *, ranks: Optional[Sequence[float]] = None) -> None:
        """Apply one bout over an ordered participant set.

        This is the update shape the Weng-Lin math is defined over: every
        side's belief is updated once, in closed form, from the whole bout's
        outcome. Pairwise results are the two-side special case; genuine N-way
        bouts enter here.

        Each entry of ``participants`` is one side:

        - a single :class:`OpenSkillCompetitor` (a one-player team), or
        - a sequence (roster) of :class:`OpenSkillCompetitor` members, whose
          team strength aggregates the members' summed beliefs and whose update
          is distributed back to the members natively -- team play on the
          member level, with no wrapper object created.

        Args:
            participants (sequence): The bout's sides, in any order (``ranks``
                carries the outcome). Each side is an ``OpenSkillCompetitor`` or
                a sequence of them.
            ranks (sequence of float, optional): Finishing ranks per side, lower
                is better, equal ranks are ties. Defaults to ``0..n-1``, i.e.
                the sides given in finishing order.

        Raises:
            ValueError: If the bout has fewer than two sides, a side is empty, a
                competitor appears on more than one side, or ``ranks`` does not
                match the side list.
            MissMatchedCompetitorTypesException: If a participant is not an
                ``OpenSkillCompetitor``.
        """
        sides: List[List["OpenSkillCompetitor"]] = []
        for entry in participants:
            if isinstance(entry, cls):
                sides.append([entry])
            elif isinstance(entry, (list, tuple)):
                roster = list(entry)
                if not roster:
                    raise ValueError("a bout side must have at least one member")
                for member in roster:
                    if not isinstance(member, cls):
                        raise MissMatchedCompetitorTypesException(
                            f"{cls.__name__}.apply_bout only accepts {cls.__name__} competitors, "
                            f"got {type(member).__name__} in a roster"
                        )
                sides.append(roster)
            else:
                raise MissMatchedCompetitorTypesException(
                    f"{cls.__name__}.apply_bout only accepts {cls.__name__} competitors or rosters of them, "
                    f"got {type(entry).__name__}"
                )
        if len(sides) < 2:
            raise ValueError(f"a bout needs at least two sides, got {len(sides)}")

        members = [member for side in sides for member in side]
        if len({id(member) for member in members}) != len(members):
            raise ValueError("the same competitor cannot appear on two sides of one bout")
        cls._check_model_homogeneity(members)

        if ranks is None:
            bout_ranks: List[int] = list(range(len(sides)))
        else:
            if len(ranks) != len(sides):
                raise ValueError(f"ranks must have one entry per side, got {len(ranks)} for {len(sides)}")
            for rank in ranks:
                if isinstance(rank, bool) or not isinstance(rank, numbers.Real):
                    raise ValueError(f"ranks must contain only numbers, got {rank!r}")
            bout_ranks = [int(rank) for rank in ranks]

        cls._solve_bout(sides, bout_ranks)

    @classmethod
    def _check_model_homogeneity(cls, bout: List["OpenSkillCompetitor"]) -> None:
        """Refuse bouts that mix Weng-Lin variants.

        A bout's update is defined by exactly one model of the family, so a
        bout mixing e.g. ``plackett_luce`` and ``thurstone`` participants is
        ambiguous and is rejected before any belief is touched.

        Raises:
            ValueError: If the bout's participants do not share one variant.
        """
        models = {competitor._model for competitor in bout}
        if len(models) > 1:
            raise ValueError(f"a bout cannot mix Weng-Lin model variants, got {sorted(models)!r}")

    @staticmethod
    def _write_side_updates(
        participants: List[List["OpenSkillCompetitor"]],
        updates: List[List[Tuple[float, float]]],
    ) -> None:
        """Write one bout's per-member updates back through direct assignment.

        Never through the ``rating`` property: ``_minimum_rating`` clamping
        (built for 1000+-scale Elo systems) must not clip the mu-approx-25
        scale this family uses.
        """
        for side, side_updates in zip(participants, updates, strict=True):
            for member, (new_mu, new_sigma) in zip(side, side_updates, strict=True):
                member._mu = new_mu
                member._sigma = new_sigma

    @classmethod
    def _distribute_side_update(
        cls,
        side_index: int,
        mus: List[List[float]],
        inflated_sigmas: List[List[float]],
        team_sigma_squares: List[float],
        omega: float,
        delta: float,
    ) -> List[Tuple[float, float]]:
        """Distribute one side's team-level ``omega``/``delta`` to its members.

        Each member receives its share of the team update, proportional to its
        share of the side's inflated variance. For a one-member side that
        share is exactly 1, so the member receives the full team update and
        every formula reduces to the single-competitor case.
        """
        side_updates: List[Tuple[float, float]] = []
        for member_index, member_sigma in enumerate(inflated_sigmas[side_index]):
            member_variance = member_sigma**2
            share = member_variance / team_sigma_squares[side_index]
            new_mu = mus[side_index][member_index] + omega * share
            new_sigma = member_sigma * math.sqrt(max(1 - delta * share, cls._kappa))
            side_updates.append((new_mu, new_sigma))
        return side_updates

    @classmethod
    def _average_tied_side_mu_changes(
        cls,
        ranks: Sequence[int],
        mus: List[List[float]],
        updates: List[List[Tuple[float, float]]],
        inflated_sigmas: List[List[float]],
        team_sigma_squares: List[float],
    ) -> List[List[Tuple[float, float]]]:
        """Apply the documented tie rule to sides: every member of every side
        tied at a rank receives the average team mu change of its rank group,
        distributed by variance share, preserving each member's own prior.

        The reference ``openskill.py`` 6.2.0 implementation intends this rule
        but its mu adjustment is a no-op (upstream issue #201, fix in PR #203);
        see the module notes. Sigma is always per-member.
        """
        rank_groups: Dict[float, List[int]] = {}
        for i, rank in enumerate(ranks):
            rank_groups.setdefault(rank, []).append(i)
        averaged = [list(side) for side in updates]
        for indices in rank_groups.values():
            if len(indices) > 1:
                team_mu_changes = [
                    sum(updates[i][j][0] - mus[i][j] for j in range(len(updates[i]))) for i in indices
                ]
                average_change = sum(team_mu_changes) / len(indices)
                for i in indices:
                    for j in range(len(updates[i])):
                        share = inflated_sigmas[i][j] ** 2 / team_sigma_squares[i]
                        averaged[i][j] = (mus[i][j] + average_change * share, updates[i][j][1])
        return averaged

    @classmethod
    def _solve_bout(cls, participants: List[List["OpenSkillCompetitor"]], ranks: Sequence[int]) -> None:
        """Dispatch one bout to the closed-form update of its model variant.

        Every entry of ``participants`` is one side -- a list of member
        competitors. Every variant of the family shares the bout shape, the
        belief state and the writeback discipline; only the per-side update
        equations differ (Weng-Lin 2011, Algorithms 1-4).

        Args:
            participants (list of lists of OpenSkillCompetitor): The bout's
                sides, each a list of member competitors.
            ranks (sequence of int): Finishing ranks per side, lower is better,
                equal ranks are ties.
        """
        model = participants[0][0]._model
        if model == "plackett_luce":
            cls._solve_bout_plackett_luce(participants, ranks)
        elif model == "bradley_terry_full":
            cls._solve_bout_bradley_terry_full(participants, ranks)
        elif model == "bradley_terry_partial":
            cls._solve_bout_bradley_terry_partial(participants, ranks)
        elif model == "thurstone":
            cls._solve_bout_thurstone(participants, ranks)
        else:  # pragma: no cover - the constructor rejects unknown variants
            raise InvalidParameterException(f"unknown model variant: {model!r}")

    @classmethod
    def _solve_bout_plackett_luce(cls, participants: List[List["OpenSkillCompetitor"]], ranks: Sequence[int]) -> None:
        """Run the Plackett-Luce update (Weng-Lin 2011, Algorithm 4) over one bout's sides.

        Each entry of ``participants`` is one side -- a list of member
        competitors. Team strength aggregates the members' summed means and
        summed variances, and the team's update is distributed back to the
        members in proportion to each member's share of the team variance (for
        a one-member team that share is exactly 1, which reduces every formula
        to the single-competitor case). Every factor of the update reads a
        pre-update snapshot of the beliefs; the results are written back
        afterwards through direct ``_mu``/``_sigma`` assignment -- never through
        the ``rating`` property, so ``_minimum_rating`` clamping (built for
        1000+-scale Elo systems) can never clip the mu-approx-25 scale this
        family uses.

        Args:
            participants (list of lists of OpenSkillCompetitor): The bout's
                sides, each a list of member competitors.
            ranks (sequence of int): Finishing ranks per side, lower is better,
                equal ranks are ties.
        """
        k = len(participants)

        # Pre-update snapshot: the whole update is computed from the beliefs as
        # of the start of the bout.
        mus = [[member._mu for member in side] for side in participants]
        inflated_sigmas = [
            [math.sqrt(member._sigma**2 + cls._tau**2) for member in side] for side in participants
        ]

        # Additive dynamics: uncertainty is inflated by tau before the update,
        # so dynamics compound across bouts exactly as in the reference
        # implementation.
        #
        # Team strength: summed member means and summed member variances (the
        # one-member case is the member's own belief).
        team_mus = [sum(side_mus) for side_mus in mus]
        team_sigma_squares = [sum(member_sigma**2 for member_sigma in side_sigmas) for side_sigmas in inflated_sigmas]

        # Square root of the collective skill-plus-chance variance.
        c = math.sqrt(sum(sigma_squared + cls._beta**2 for sigma_squared in team_sigma_squares))

        # sum_q[q]: sum of exp(mu/c) over every side ranked at or below side q.
        sum_q = [0.0] * k
        for i in range(k):
            exp_mu_over_c = math.exp(team_mus[i] / c)
            for q in range(k):
                if ranks[i] >= ranks[q]:
                    sum_q[q] += exp_mu_over_c

        # a[q]: how many sides share side q's rank.
        a = [sum(1 for rank in ranks if rank == ranks[q]) for q in range(k)]

        updates: List[List[Tuple[float, float]]] = []
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

            updates.append(cls._distribute_side_update(i, mus, inflated_sigmas, team_sigma_squares, omega, delta))

        # Tied sides share an outcome, so their average mu change is applied to
        # every member of every tied side; sigma stays per-member.
        updates = cls._average_tied_side_mu_changes(ranks, mus, updates, inflated_sigmas, team_sigma_squares)

        cls._write_side_updates(participants, updates)

    @classmethod
    def _solve_bout_bradley_terry_full(cls, participants: List[List["OpenSkillCompetitor"]], ranks: Sequence[int]) -> None:
        """Run the Bradley-Terry full-pairing update (Weng-Lin 2011, Algorithm 1) over one bout's sides.

        Every side is compared against every other side: the pairwise logistic
        expectation and its variance contribution are summed over the whole
        bout. Writeback discipline matches the Plackett-Luce solver
        (pre-update snapshot, direct ``_mu``/``_sigma`` assignment).

        Args:
            participants (list of lists of OpenSkillCompetitor): The bout's
                sides, each a list of member competitors.
            ranks (sequence of int): Finishing ranks per side, lower is better,
                equal ranks are ties.
        """
        # Pre-update snapshot; additive dynamics inflate sigma by tau first.
        mus = [[member._mu for member in side] for side in participants]
        inflated_sigmas = [
            [math.sqrt(member._sigma**2 + cls._tau**2) for member in side] for side in participants
        ]
        team_mus = [sum(side_mus) for side_mus in mus]
        team_sigma_squares = [sum(member_sigma**2 for member_sigma in side_sigmas) for side_sigmas in inflated_sigmas]

        updates: List[List[Tuple[float, float]]] = []
        for i in range(len(participants)):
            omega = 0.0
            delta = 0.0
            for q, _ in enumerate(participants):
                if q == i:
                    continue
                c_iq = math.sqrt(team_sigma_squares[i] + team_sigma_squares[q] + 2 * cls._beta**2)
                sigma_squared_to_ciq = team_sigma_squares[i] / c_iq
                # Logistic win expectation of side i against q (paper eq. 41),
                # algebraically exp(mu_i/c)/[exp(mu_i/c)+exp(mu_q/c)].
                p_iq = 1.0 / (1.0 + math.exp((team_mus[q] - team_mus[i]) / c_iq))
                if ranks[q] > ranks[i]:
                    score = 1.0
                elif ranks[q] == ranks[i]:
                    score = 0.5
                else:
                    score = 0.0
                omega += sigma_squared_to_ciq * (score - p_iq)
                # Paper Algorithm 1: eta_q = gamma_q * (sigma_i/c_iq)^2 * p * (1-p).
                gamma = math.sqrt(team_sigma_squares[i]) / c_iq
                delta += (gamma * sigma_squared_to_ciq / c_iq) * p_iq * (1.0 - p_iq)

            updates.append(cls._distribute_side_update(i, mus, inflated_sigmas, team_sigma_squares, omega, delta))

        updates = cls._average_tied_side_mu_changes(ranks, mus, updates, inflated_sigmas, team_sigma_squares)
        cls._write_side_updates(participants, updates)

    @classmethod
    def _solve_bout_bradley_terry_partial(cls, participants: List[List["OpenSkillCompetitor"]], ranks: Sequence[int]) -> None:
        """Run the Bradley-Terry partial-pairing update (Weng-Lin 2011, Algorithm 2) over one bout's sides.

        Each side is compared only against neighbours within
        ``_partial_pairing_window`` finishing positions on either side of the
        rank-sorted bout, and the pairwise contributions are averaged over the
        comparisons instead of summed. This follows the reference
        implementation's bounded-window reading of Algorithm 2 (the paper
        itself pairs adjacent ranks and sums); it agrees with the paper's
        update for every bout no larger than the window.

        Args:
            participants (list of lists of OpenSkillCompetitor): The bout's
                sides, each a list of member competitors.
            ranks (sequence of int): Finishing ranks per side, lower is better,
                equal ranks are ties.
        """
        # Pre-update snapshot; additive dynamics inflate sigma by tau first.
        mus = [[member._mu for member in side] for side in participants]
        inflated_sigmas = [
            [math.sqrt(member._sigma**2 + cls._tau**2) for member in side] for side in participants
        ]
        team_mus = [sum(side_mus) for side_mus in mus]
        team_sigma_squares = [sum(member_sigma**2 for member_sigma in side_sigmas) for side_sigmas in inflated_sigmas]

        # Window positions follow the rank-sorted bout (stable within ties),
        # matching the reference implementation's partial-pairing pass.
        count = len(participants)
        order = sorted(range(count), key=lambda index: ranks[index])
        position_of = {index: position for position, index in enumerate(order)}
        window = cls._partial_pairing_window

        updates: List[List[Tuple[float, float]]] = []
        for i in range(count):
            start = max(0, position_of[i] - window)
            end = min(count, position_of[i] + window + 1)
            omega_sum = 0.0
            delta_sum = 0.0
            comparisons = 0
            for position in range(start, end):
                q = order[position]
                if q == i:
                    continue
                c_iq = math.sqrt(team_sigma_squares[i] + team_sigma_squares[q] + 2 * cls._beta**2)
                sigma_squared_to_ciq = team_sigma_squares[i] / c_iq
                p_iq = 1.0 / (1.0 + math.exp((team_mus[q] - team_mus[i]) / c_iq))
                if ranks[q] > ranks[i]:
                    score = 1.0
                elif ranks[q] == ranks[i]:
                    score = 0.5
                else:
                    score = 0.0
                omega_sum += sigma_squared_to_ciq * (score - p_iq)
                # Paper Algorithm 1: eta_q = gamma_q * (sigma_i/c_iq)^2 * p * (1-p).
                gamma = math.sqrt(team_sigma_squares[i]) / c_iq
                delta_sum += (gamma * sigma_squared_to_ciq / c_iq) * p_iq * (1.0 - p_iq)
                comparisons += 1

            if comparisons > 0:
                omega = omega_sum / comparisons
                delta = delta_sum / comparisons
            else:  # pragma: no cover - every bout has at least one comparison
                omega = 0.0
                delta = 0.0

            updates.append(cls._distribute_side_update(i, mus, inflated_sigmas, team_sigma_squares, omega, delta))

        updates = cls._average_tied_side_mu_changes(ranks, mus, updates, inflated_sigmas, team_sigma_squares)
        cls._write_side_updates(participants, updates)
    @classmethod
    def _solve_bout_thurstone(cls, participants: List[List["OpenSkillCompetitor"]], ranks: Sequence[int]) -> None:
        """Run the Thurstone-Mosteller update (Weng-Lin 2011, Algorithm 3) over one bout's sides.

        A Gaussian (Thurstone-Mosteller) performance model with an explicit
        draw margin: decisive pairwise results use the truncated-normal
        ``V``/``W`` functions, results inside the draw margin use the tie
        functions ``V-tilde``/``W-tilde`` (see the module helpers, which mirror
        the reference implementation's numerics).

        Args:
            participants (list of lists of OpenSkillCompetitor): The bout's
                sides, each a list of member competitors.
            ranks (sequence of int): Finishing ranks per side, lower is better,
                equal ranks are ties.
        """
        # Pre-update snapshot; additive dynamics inflate sigma by tau first.
        mus = [[member._mu for member in side] for side in participants]
        inflated_sigmas = [
            [math.sqrt(member._sigma**2 + cls._tau**2) for member in side] for side in participants
        ]
        team_mus = [sum(side_mus) for side_mus in mus]
        team_sigma_squares = [sum(member_sigma**2 for member_sigma in side_sigmas) for side_sigmas in inflated_sigmas]

        updates: List[List[Tuple[float, float]]] = []
        for i in range(len(participants)):
            omega = 0.0
            delta = 0.0
            for q, _ in enumerate(participants):
                if q == i:
                    continue
                c_iq = math.sqrt(team_sigma_squares[i] + team_sigma_squares[q] + 2 * cls._beta**2)
                sigma_squared_to_ciq = team_sigma_squares[i] / c_iq
                delta_mu = (team_mus[i] - team_mus[q]) / c_iq
                gamma = math.sqrt(team_sigma_squares[i]) / c_iq
                draw_margin_over_c = cls._epsilon / c_iq

                if ranks[q] > ranks[i]:
                    # i beat q: truncated-normal evidence of a decisive result.
                    omega += sigma_squared_to_ciq * _truncated_v(delta_mu, draw_margin_over_c)
                    delta += (gamma * sigma_squared_to_ciq / c_iq) * _truncated_w(delta_mu, draw_margin_over_c)
                elif ranks[q] < ranks[i]:
                    # i lost to q: symmetric, evaluated from the winner side.
                    omega -= sigma_squared_to_ciq * _truncated_v(-delta_mu, draw_margin_over_c)
                    delta += (gamma * sigma_squared_to_ciq / c_iq) * _truncated_w(-delta_mu, draw_margin_over_c)
                else:
                    # Tie inside the draw margin.
                    omega += sigma_squared_to_ciq * _truncated_v_tie(delta_mu, draw_margin_over_c)
                    delta += (gamma * sigma_squared_to_ciq / c_iq) * _truncated_w_tie(delta_mu, draw_margin_over_c)

            updates.append(cls._distribute_side_update(i, mus, inflated_sigmas, team_sigma_squares, omega, delta))

        updates = cls._average_tied_side_mu_changes(ranks, mus, updates, inflated_sigmas, team_sigma_squares)
        cls._write_side_updates(participants, updates)

    # ------------------------------------------------------------------
    # Period update (the batch hook the arenas call)
    # ------------------------------------------------------------------

    @classmethod
    def _validate_period(
        cls,
        results: Sequence[Tuple[BaseCompetitor, BaseCompetitor, float, Optional[Sequence[float]]]],
    ) -> List[Tuple[List[List["OpenSkillCompetitor"]], Tuple[int, ...]]]:
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
        bouts: List[Tuple[List[List["OpenSkillCompetitor"]], Tuple[int, ...]]] = []
        for competitor_a, competitor_b, outcome, scores in results:
            if outcome not in (1.0, 0.0, 0.5):
                raise ValueError(f"outcome must be one of 1.0, 0.0 or 0.5, got {outcome!r}")
            competitor_a.verify_competitor_types(competitor_b)
            if not isinstance(competitor_a, cls) or not isinstance(competitor_b, cls):
                raise MissMatchedCompetitorTypesException(
                    f"{cls.__name__}.apply_rating_period only accepts {cls.__name__} competitors"
                )
            if competitor_a._model != competitor_b._model:
                raise ValueError(
                    f"a rating period row cannot mix Weng-Lin model variants, "
                    f"got {competitor_a._model!r} and {competitor_b._model!r}"
                )
            validate_scores(scores, outcome)

            if outcome == 1.0:
                ranks = (0, 1)
            elif outcome == 0.0:
                ranks = (1, 0)
            else:
                ranks = (0, 0)
            bouts.append(([[competitor_a], [competitor_b]], ranks))
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
