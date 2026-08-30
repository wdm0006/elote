import math
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, TypeVar
from datetime import datetime

from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    InvalidRatingValueException,
    validate_scores,
)
from elote.logging import logger

T = TypeVar("T", bound="GlickoBoostCompetitor")

# One participant's games in a period, as (opponent index, colour, score).
_Schedule = List[List[Tuple[int, int, float]]]


class GlickoBoostCompetitor(BaseCompetitor):
    """Glicko-Boost rating system competitor.

    Glicko-Boost is Mark Glickman's extension of Glicko, described in
    `Glicko-Boost <https://www.glicko.net/glicko/glicko-boost.pdf>`_. Unlike Elo or
    Glicko it is defined over a whole rating period rather than over one game: the
    population is updated twice from the same pre-period ratings, players whose
    performance was exceptional have their pre-period RD boosted, and the pair of
    updates is then repeated. This class therefore overrides
    :meth:`~elote.competitors.base.BaseCompetitor.apply_rating_period` and is the
    first shipped system whose period update is not a replay of pairwise results.

    The six steps applied to one period are:

    1. Glicko updating of every player from the pre-period ratings and RDs, with a
       white-advantage term inside ``E()``.

    2. The same update from the pre-period ratings, but against the opponents' step 1
       ratings and RDs.

    3. An RD boost for players whose performance z-score exceeds ``k``.

    4. Step 1 again, using the boosted RDs.

    5. Step 2 again, using the step 4 results. These are the period's final ratings.

    6. An RD increase for the passage of time, applied when the competitor next takes
       part in a period (the way :class:`~elote.competitors.glicko.GlickoCompetitor`
       handles inactivity), so a competitor that sits out periods catches up on its
       next appearance.

    Colour is carried by argument order rather than by a new parameter: in a rating
    period row ``(a, b, outcome, scores)`` -- and in ``a.beat(b)`` -- ``a`` is white.
    Callers with no colour information leave ``_eta`` at its default of ``0.0``, which
    removes the white-advantage term entirely.

    ``beat``/``lost_to``/``tied`` apply the same algorithm to a one-game period, so a
    single result is never a different formula from a batch. Because of the two-pass
    structure a lone pairwise call is **not** identical to a Glicko update.

    Class Attributes:
        _q (float): ``ln(10)/400``, the Glicko scaling constant.
        _eta (float): Rating advantage for playing white. Default: ``0.0``; Glickman's is ``30.0``.
        _b1 (float): RD boost multiplicative factor. Default: 0.20139.
        _b2 (float): RD boost additive factor. Default: 17.5.
        _k (float): The z-score above which an RD is boosted. Default: 1.96.
        _alpha0 .. _alpha4 (float): RD-increase-over-time coefficients, at Glickman's values.
        _rd_unrated (float): The RD cap, ``RD_unr`` in the paper. Default: 250.0.
        _rating_period_days (float): Days in one rating period. Default: 30.0 (a month).
    """

    _q: ClassVar[float] = math.log(10) / 400
    _eta: ClassVar[float] = 0.0
    _b1: ClassVar[float] = 0.20139
    _b2: ClassVar[float] = 17.5
    _k: ClassVar[float] = 1.96
    _alpha0: ClassVar[float] = 5.83733
    _alpha1: ClassVar[float] = -1.75374e-04
    _alpha2: ClassVar[float] = -7.080124e-05
    _alpha3: ClassVar[float] = 0.001733792
    _alpha4: ClassVar[float] = 0.00026706
    _rd_unrated: ClassVar[float] = 250.0
    _rating_period_days: ClassVar[float] = 30.0

    def __init__(self, initial_rating: float = 1500, initial_rd: float = 250, initial_time: Optional[datetime] = None):
        """Initialize a Glicko-Boost competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 1500.
                Glickman's FIDE-specific default for an unrated player is 1946.25.
            initial_rd (float, optional): The initial rating deviation. Default: 250, the
                paper's ``RD_unr``.
            initial_time (datetime, optional): The initial timestamp for this competitor. When
                omitted the competitor has no recorded activity and adopts the time of its first
                period, so historical results can be replayed through it.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
            InvalidParameterException: If the initial RD is not positive.
        """
        super().__init__()
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")

        self._initial_rating = initial_rating
        self._initial_rd = initial_rd
        self._rating = initial_rating
        self.rd = initial_rd
        self._last_activity: Optional[datetime] = initial_time
        logger.debug(
            "Initialized GlickoBoostCompetitor with rating=%.1f, rd=%.1f, time=%s",
            self._initial_rating,
            self._initial_rd,
            self._last_activity,
        )

    def __repr__(self) -> str:
        return f"<GlickoBoostCompetitor: rating={self._rating}, rd={self.rd}>"

    def __str__(self) -> str:
        return f"<GlickoBoostCompetitor: rating={self._rating}, rd={self.rd}>"

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
            raise InvalidRatingValueException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = value

    @classmethod
    def _g(cls, rd: float) -> float:
        """Calculate Glicko's g-function, the weight an opponent's RD gives its result."""
        return 1 / math.sqrt(1 + 3 * cls._q**2 * rd**2 / math.pi**2)

    @classmethod
    def _e(cls, rating: float, colour: int, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate the expected score of one game, including the advantage to white.

        Args:
            rating (float): The player's rating.
            colour (int): ``1`` when the player had white, ``-1`` when black, ``0`` when unknown.
            opponent_rating (float): The opponent's rating.
            opponent_rd (float): The opponent's rating deviation.

        Returns:
            float: The expected score of the game.
        """
        exponent = -cls._g(opponent_rd) * (rating + colour * cls._eta - opponent_rating) / 400
        return 1 / (1 + 10**exponent)

    @classmethod
    def _glicko_update(
        cls,
        rating: float,
        rd: float,
        games: Sequence[Tuple[int, int, float]],
        opponents: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Apply one Glicko update with a white advantage (the paper's Section 2.1).

        Args:
            rating (float): The player's pre-update rating.
            rd (float): The player's pre-update rating deviation.
            games (sequence): ``(opponent index, colour, score)`` for each game played.
            opponents (sequence): ``(rating, rd)`` per participant, indexed by the game's
                opponent index.

        Returns:
            tuple of float: The updated rating and rating deviation.
        """
        numerator, information = cls._sufficient_statistics(rating, games, opponents)
        d_squared = 1 / (cls._q**2 * information)
        new_rd = math.sqrt(1 / (1 / rd**2 + 1 / d_squared))
        return rating + new_rd**2 * cls._q * numerator, new_rd

    @classmethod
    def _sufficient_statistics(
        cls,
        rating: float,
        games: Sequence[Tuple[int, int, float]],
        opponents: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Accumulate the two sums every step of the algorithm is built from.

        Returns:
            tuple of float: ``sum(g * (s - E))`` and ``sum(g**2 * E * (1 - E))``.
        """
        numerator = 0.0
        information = 0.0
        for opponent_index, colour, score in games:
            opponent_rating, opponent_rd = opponents[opponent_index]
            weight = cls._g(opponent_rd)
            expected = cls._e(rating, colour, opponent_rating, opponent_rd)
            numerator += weight * (score - expected)
            information += weight**2 * expected * (1 - expected)
        return numerator, information

    @classmethod
    def _performance_z_score(
        cls,
        rating: float,
        games: Sequence[Tuple[int, int, float]],
        opponents: Sequence[Tuple[float, float]],
    ) -> float:
        """Calculate the standardized excess of the actual score over the expected score.

        The player's own rating is its **pre-period** rating: the z-score measures how far
        the period's results ran ahead of what that rating predicted. The opponents are the
        step 2 population, as the paper's step 3 prescribes.
        """
        numerator, information = cls._sufficient_statistics(rating, games, opponents)
        return numerator / math.sqrt(information)

    @classmethod
    def _boosted_rd(cls, rd: float, z_score: float) -> float:
        """Apply the RD boost of the paper's Section 2.2 to one pre-period RD."""
        if z_score <= cls._k:
            return rd
        return min(cls._rd_unrated, (1 + (z_score - cls._k) * cls._b1) * rd + cls._b2)

    @classmethod
    def _inflated_rd(cls, rating: float, rd: float, periods: float = 1.0) -> float:
        """Increase an RD for the passage of ``periods`` rating periods (Section 2.3).

        The paper's formula adds one ``exp(...)`` term to ``RD**2`` per period; several
        elapsed periods add that term once each, which reduces to the published formula
        for a single period.
        """
        scaled_rating = rating / 1000
        increase = math.exp(
            cls._alpha0
            + cls._alpha1 * rd
            + cls._alpha2 * rd * scaled_rating
            + cls._alpha3 * scaled_rating
            + cls._alpha4 * scaled_rating**2
        )
        return min(cls._rd_unrated, math.sqrt(rd**2 + periods * increase))

    @classmethod
    def _solve_period(
        cls,
        ratings: Sequence[float],
        rds: Sequence[float],
        schedule: _Schedule,
    ) -> Dict[str, List[Any]]:
        """Run the five rating steps over one period and return every intermediate stage.

        Args:
            ratings (sequence of float): Pre-period ratings, one per participant.
            rds (sequence of float): Pre-period rating deviations, one per participant.
            schedule (list): Per participant, the ``(opponent index, colour, score)`` games.

        Returns:
            dict: ``step1``, ``step2``, ``step4`` and ``final`` as ``(rating, rd)`` lists,
            plus the ``z_score`` and boosted ``reset_rd`` lists behind them.
        """
        pre = [(rating, rd) for rating, rd in zip(ratings, rds, strict=True)]

        step1 = [cls._glicko_update(rating, rd, schedule[i], pre) for i, (rating, rd) in enumerate(pre)]
        step2 = [cls._glicko_update(rating, rd, schedule[i], step1) for i, (rating, rd) in enumerate(pre)]

        z_scores = [cls._performance_z_score(rating, schedule[i], step2) for i, (rating, _) in enumerate(pre)]
        reset_rd = [cls._boosted_rd(rd, z_scores[i]) for i, (_, rd) in enumerate(pre)]

        if reset_rd == list(rds):
            # No RD was boosted, so steps 4 and 5 would reproduce steps 1 and 2 exactly.
            return {
                "step1": step1,
                "step2": step2,
                "z_score": z_scores,
                "reset_rd": reset_rd,
                "step4": step1,
                "final": step2,
            }

        boosted = [(rating, reset_rd[i]) for i, (rating, _) in enumerate(pre)]
        step4 = [cls._glicko_update(rating, rd, schedule[i], boosted) for i, (rating, rd) in enumerate(boosted)]
        step5 = [cls._glicko_update(rating, rd, schedule[i], step4) for i, (rating, rd) in enumerate(boosted)]
        return {
            "step1": step1,
            "step2": step2,
            "z_score": z_scores,
            "reset_rd": reset_rd,
            "step4": step4,
            "final": step5,
        }

    @classmethod
    def _period_schedule(
        cls,
        results: Sequence[Tuple["BaseCompetitor", "BaseCompetitor", float, Optional[Sequence[float]]]],
    ) -> Tuple[List["GlickoBoostCompetitor"], _Schedule]:
        """Validate a period's rows and turn them into a participant list and schedule.

        Raises:
            ValueError: If an outcome is not ``1.0``, ``0.0`` or ``0.5``, or if a score
                payload is invalid or inconsistent with its outcome.
            MissMatchedCompetitorTypesException: If a row contains another rating system.
        """
        participants: List[GlickoBoostCompetitor] = []
        indices: Dict[int, int] = {}
        schedule: _Schedule = []

        def index_of(competitor: "GlickoBoostCompetitor") -> int:
            key = id(competitor)
            if key not in indices:
                indices[key] = len(participants)
                participants.append(competitor)
                schedule.append([])
            return indices[key]

        for competitor_a, competitor_b, outcome, scores in results:
            if outcome not in (1.0, 0.0, 0.5):
                raise ValueError(f"outcome must be one of 1.0, 0.0 or 0.5, got {outcome!r}")
            competitor_a.verify_competitor_types(competitor_b)
            if not isinstance(competitor_a, cls) or not isinstance(competitor_b, cls):
                raise ValueError(f"{cls.__name__}.apply_rating_period only accepts {cls.__name__} competitors")
            validate_scores(scores, outcome)

            # Argument order carries colour: the first competitor of a row played white.
            a_index, b_index = index_of(competitor_a), index_of(competitor_b)
            schedule[a_index].append((b_index, 1, outcome))
            schedule[b_index].append((a_index, -1, 1.0 - outcome))

        return participants, schedule

    @classmethod
    def apply_rating_period(
        cls,
        results: Sequence[Tuple["BaseCompetitor", "BaseCompetitor", float, Optional[Sequence[float]]]],
        *,
        period_end: Optional[Any] = None,
    ) -> None:
        """Apply results that share one rating period, using Glicko-Boost's own period update.

        This is where the whole algorithm lives: the pairwise methods route a single result
        through here as a one-game period, so every caller gets the same formulas.

        Args:
            results: ``(white, black, outcome, scores)`` tuples. Outcomes use ``1.0`` for a
                white win, ``0.0`` for a black win and ``0.5`` for a draw; scores follow the
                usual optional caller-order contract and are validated but not consumed.
            period_end: The shared activity time for the period. Elapsed rating periods since
                each participant's last activity inflate its RD before the update.

        Raises:
            ValueError: If an outcome or score payload is invalid.
            MissMatchedCompetitorTypesException: If a row contains another rating system.
        """
        participants, schedule = cls._period_schedule(results)
        if not participants:
            return

        for competitor in participants:
            competitor._advance_to(period_end)

        stages = cls._solve_period(
            [competitor.rating for competitor in participants],
            [competitor.rd for competitor in participants],
            schedule,
        )

        for competitor, (rating, rd) in zip(participants, stages["final"], strict=True):
            competitor._rating = max(cls._minimum_rating, rating)
            competitor.rd = rd
            if period_end is not None:
                competitor._last_activity = period_end
        logger.debug("Applied a Glicko-Boost rating period over %d competitors", len(participants))

    def _advance_to(self, current_time: Optional[datetime]) -> None:
        """Inflate this competitor's RD for the rating periods it has missed.

        A competitor with no recorded activity adopts the supplied time as its first activity
        instead of being inflated, so a historical replay stays possible.

        Raises:
            InvalidParameterException: If the time is before this competitor's last activity.
        """
        if self._last_activity is None:
            if current_time is not None:
                self._last_activity = current_time
            return
        if current_time is None:
            return
        if current_time < self._last_activity:
            raise InvalidParameterException("Period time cannot be before competitor's last activity time")

        days = (current_time - self._last_activity).total_seconds() / (24 * 3600)
        periods = days / self._rating_period_days
        if periods > 0:
            self.rd = self._inflated_rd(self._rating, self.rd, periods)

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        This uses the paper's own approximation, which combines the two rating deviations
        rather than using only the opponent's. With ``_eta`` at its default of ``0.0`` the
        result is exactly complementary: ``a.expected_score(b) + b.expected_score(a) == 1``.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        combined_rd = math.sqrt(self.rd**2 + competitor.rd**2)
        exponent = -self._g(combined_rd) * (self._rating + self._eta - competitor.rating) / 400
        return 1 / (1 + 10**exponent)

    def beat(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Update ratings after this competitor has won against the given competitor.

        The result is applied as a one-game rating period in which this competitor had white.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            match_time (datetime, optional): The time of the match, used as the period's end.
            scores (sequence of float, optional): The two scores in caller order. Validated only.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 1.0)
        self.__class__.apply_rating_period([(self, competitor, 1.0, scores)], period_end=match_time)

    def lost_to(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Update ratings after this competitor has lost to the given competitor.

        The result is applied as a one-game rating period in which this competitor had white,
        so the colour convention still follows the caller's argument order.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.
            match_time (datetime, optional): The time of the match, used as the period's end.
            scores (sequence of float, optional): The two scores in caller order.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.0)
        self.__class__.apply_rating_period([(self, competitor, 0.0, scores)], period_end=match_time)

    def tied(
        self,
        competitor: BaseCompetitor,
        match_time: Optional[datetime] = None,
        *,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Update ratings after this competitor has drawn with the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that drew.
            match_time (datetime, optional): The time of the match, used as the period's end.
            scores (sequence of float, optional): The two scores in caller order. Must be equal.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        self._validate_scores(scores, 0.5)
        self.__class__.apply_rating_period([(self, competitor, 0.5, scores)], period_end=match_time)

    def reset(self) -> None:
        """Reset this competitor to its initial rating, RD and activity."""
        logger.info(
            "Resetting GlickoBoostCompetitor to initial state (rating=%.1f, rd=%.1f)",
            self._initial_rating,
            self._initial_rd,
        )
        self._rating = self._initial_rating
        self.rd = self._initial_rd
        self._last_activity = None

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor."""
        return {"initial_rating": self._initial_rating, "initial_rd": self._initial_rd}

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor."""
        return {
            "rating": self._rating,
            "rd": self.rd,
            "last_activity": self._last_activity.isoformat() if self._last_activity is not None else None,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        initial_rating = parameters.get("initial_rating", 1500)
        if initial_rating < self._minimum_rating:
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self._initial_rating = initial_rating

        initial_rd = parameters.get("initial_rd", 250)
        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")
        self._initial_rd = initial_rd

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
            raise InvalidParameterException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = rating

        rd = state.get("rd", self._initial_rd)
        if rd <= 0:
            raise InvalidParameterException("RD must be positive")
        self.rd = rd

        # An explicit null means the competitor has never been active, while a missing key
        # belongs to a state document written before that was representable.
        if "last_activity" in state:
            last_activity = state["last_activity"]
            self._last_activity = datetime.fromisoformat(last_activity) if last_activity is not None else None
        else:
            self._last_activity = datetime.now()
            logger.warning(
                "Last activity time missing from state, using current time: %s", self._last_activity.isoformat()
            )

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from exported parameters."""
        return cls(
            initial_rating=parameters.get("initial_rating", 1500),
            initial_rd=parameters.get("initial_rd", 250),
        )
