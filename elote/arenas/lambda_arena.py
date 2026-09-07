import datetime
import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from tqdm import tqdm
from elote import EloCompetitor
from elote.arenas.base import BaseArena, Bout, History, MultiBout
from elote.competitors.base import BaseCompetitor, MissMatchedCompetitorTypesException
from elote.logging import logger


def _parse_sides(participants: Sequence[Any]) -> List[Tuple[Any, Optional[List[Any]]]]:
    """Split match_group participants into ``(side id, roster or None)`` pairs.

    An entry is a roster side exactly when it is a list/tuple pair of two items
    whose second item is itself a list or tuple; every other entry is a plain
    competitor id. Rosters are copied, and a side's id is only a label -- no
    competitor is created for it.

    Args:
        participants (sequence): The bout's sides, as given by the caller.

    Returns:
        list of (side id, roster or None): One entry per side, in given order.

    Raises:
        ValueError: If a roster side is empty or its roster is not a list/tuple.
    """
    sides: List[Tuple[Any, Optional[List[Any]]]] = []
    for entry in participants:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[1], (list, tuple))
        ):
            side_id, roster = entry
            roster_members = list(roster)
            if not roster_members:
                raise ValueError(f"side '{side_id}' has an empty roster")
            sides.append((side_id, roster_members))
        else:
            sides.append((entry, None))
    return sides


def _validate_group_ranks(ranks: Sequence[float], n_sides: int) -> List[int]:
    """Validate match_group ranks: one whole number per side.

    Args:
        ranks (sequence): The finishing ranks, lower is better, ties repeat.
        n_sides (int): The number of sides the ranks must describe.

    Returns:
        list of int: The validated ranks as ints.

    Raises:
        ValueError: If the count does not match the side list or any rank is not
            a whole number.
    """
    if len(ranks) != n_sides:
        raise ValueError(f"ranks must have one entry per participant, got {len(ranks)} for {n_sides}")
    validated: List[int] = []
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, numbers.Real):
            raise ValueError(f"ranks must contain only numbers, got {rank!r}")
        if not float(rank).is_integer():
            raise ValueError(f"ranks must be whole numbers, got {rank!r}")
        validated.append(int(rank))
    return validated


def _validate_group_scores(scores: Sequence[float], n_sides: int) -> List[float]:
    """Validate match_group scores: one finite, non-negative number per side.

    Args:
        scores (sequence): One score per side, in the order given.
        n_sides (int): The number of sides the scores must describe.

    Returns:
        list of float: The validated scores, normalized to built-in floats.

    Raises:
        ValueError: If the count does not match the side list or any score is
            negative, non-finite, or not a number.
    """
    if len(scores) != n_sides:
        raise ValueError(f"scores must have one entry per participant, got {len(scores)} for {n_sides}")
    validated: List[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise ValueError(f"scores must contain only numbers, got {score!r}")
        if not math.isfinite(float(score)):
            raise ValueError(f"scores must be finite numbers, got {score!r}")
        if score < 0:
            raise ValueError(f"scores must be non-negative, got {score!r}")
        validated.append(float(score))
    return validated


def _ranks_from_scores(scores: List[float]) -> List[int]:
    """Derive finishing ranks from per-side scores.

    Higher scores finish better; equal scores tie and share the earliest rank
    they reach (so 3.0, 3.0, 1.0 maps to ranks 0, 0, 2).

    Args:
        scores (list of float): The validated per-side scores.

    Returns:
        list of int: One rank per score, aligned with the input order.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    rank = 0
    for position, index in enumerate(order):
        if position > 0 and scores[index] != scores[order[position - 1]]:
            rank = position
        ranks[index] = rank
    return ranks


class LambdaArena(BaseArena):
    @staticmethod
    def _validate_outcome(outcome: Optional[float]) -> None:
        if outcome is not None and outcome not in (1.0, 0.0, 0.5):
            raise ValueError(f"outcome must be one of 1.0, 0.0 or 0.5, got {outcome!r}")

    def __init__(
        self,
        func: Callable[..., Optional[bool]],
        base_competitor: Type[BaseCompetitor] = EloCompetitor,
        base_competitor_kwargs: Optional[Dict[str, Any]] = None,
        initial_state: Optional[Dict[Any, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a LambdaArena with a comparison function.

        The LambdaArena uses a provided function to determine the outcome of matchups
        between competitors. This is particularly useful for comparing objects that
        aren't competitors themselves.

        Args:
            func (callable): A function that takes two arguments (a, b) and returns
                True if a beats b, False if b beats a, and None for a draw.
            base_competitor (class): The competitor class to use for ratings.
                Defaults to EloCompetitor.
            base_competitor_kwargs (dict, optional): Keyword arguments to pass to
                the base_competitor constructor.
            initial_state (dict, optional): Initial state for competitors, mapping
                competitor IDs either to exported competitor state documents (as
                produced by :meth:`export_state`) or to plain keyword arguments for
                the base_competitor constructor, such as ``{"initial_rating": 1200}``.
        """
        super().__init__()
        logger.debug("Initializing LambdaArena with competitor type: %s", base_competitor.__name__)
        self.func: Callable[..., Optional[bool]] = func
        self.competitors: Dict[Any, BaseCompetitor] = dict()
        self.base_competitor: Type[BaseCompetitor] = base_competitor
        # Define type hint once for the instance variable
        self.base_competitor_kwargs: Dict[str, Any]
        if base_competitor_kwargs is None:
            self.base_competitor_kwargs = dict()
        else:
            self.base_competitor_kwargs = base_competitor_kwargs

        # if some initial state is passed in, we can seed the population
        if initial_state is not None:
            for k, v in initial_state.items():
                # An exported competitor state document carries its own concrete type,
                # which from_state resolves through the subclass registry. Anything else
                # is treated as constructor keyword arguments.
                if isinstance(v, dict) and "type" in v:
                    self.competitors[k] = BaseCompetitor.from_state(v)
                else:
                    self.competitors[k] = self.base_competitor(**v)

        self.history: History = History()
        self.eval_history = History()
        self.validation_history = History()

    def clear_history(self) -> None:
        """Clear the history of bouts in this arena."""
        self.history = History()

    def set_competitor_class_var(self, name: str, value: Any) -> None:
        """Set a class variable on the base competitor class.

        This method allows for global configuration of all competitors
        managed by this arena.

        Args:
            name (str): The name of the class variable to set.
            value: The value to set for the class variable.
        """
        setattr(self.base_competitor, name, value)

    def tournament(self, matchups: List[Tuple[Any, ...]]) -> None:
        """Run a tournament with the given matchups.

        Process multiple matchups between competitors, updating ratings
        after each matchup.

        Args:
            matchups (list): A list of tuples, each unpacked into :meth:`matchup`. The
                first two entries are the competitors; further entries follow that
                method's signature, so a tuple may carry
                ``(a, b, attributes, match_time, outcome, scores)``.
        """
        for data in tqdm(matchups):
            self.matchup(*data)

    def matchup(
        self,
        a: Any,
        b: Any,
        attributes: Optional[Dict[str, Any]] = None,
        match_time: Optional[datetime.datetime] = None,
        outcome: Optional[float] = None,
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Process a single matchup between two competitors.

        This method handles a matchup between two competitors, creating them
        if they don't already exist in the arena. It uses the comparison function
        to determine the outcome and updates the ratings accordingly.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.
            attributes (dict, optional): Additional attributes to record with this bout.
            match_time (datetime, optional): The time when the match occurred.
            outcome (float, optional): A known result for this matchup, expressed from
                a's perspective as ``1.0`` (a wins), ``0.0`` (b wins) or ``0.5`` (draw).
                When supplied, the comparison function is not called.
            scores (sequence of float, optional): The two scores as ``(a_score, b_score)`` --
                always in the argument order of this call, regardless of who won. They are
                forwarded to the competitors' result methods in whatever order those methods
                require. Requires ``outcome`` to be supplied, since the scores must be checked
                against a known result.

        Raises:
            ValueError: If ``outcome`` is supplied and is not one of 1.0, 0.0 or 0.5, if
                ``scores`` is supplied without ``outcome``, or if ``scores`` is malformed,
                negative, non-finite, or disagrees with ``outcome``.
        """
        self._validate_outcome(outcome)
        if scores is not None and outcome is None:
            raise ValueError("scores requires an explicit outcome so the two can be checked for agreement")
        # Validated before any competitor is created or any history is recorded, so a bad
        # score payload leaves the arena completely unchanged.
        validated_scores = BaseCompetitor._validate_scores(scores, outcome) if outcome is not None else None

        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        predicted_outcome: float = self.expected_score(a, b)

        res: Optional[bool]
        if outcome is None:
            if attributes:
                res = self.func(a, b, attributes=attributes)
            else:
                res = self.func(a, b)
        elif outcome == 1.0:
            res = True
        elif outcome == 0.0:
            res = False
        else:
            res = None

        # Check if the competitor supports time-based ratings
        supports_time = hasattr(self.competitors[a], "_last_activity")

        # Scores are supplied in (a, b) order. The draw and a-wins branches call through
        # competitor a, so they keep that order; the b-wins branch reverses the call, so the
        # score pair has to be reversed with it.
        reversed_scores = None if validated_scores is None else (validated_scores[1], validated_scores[0])

        if res is None:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[a].tied(self.competitors[b], match_time=match_time, scores=validated_scores)
            else:
                self.competitors[a].tied(self.competitors[b], scores=validated_scores)
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="tie", attributes=attributes))
        elif res is True:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[a].beat(self.competitors[b], match_time=match_time, scores=validated_scores)
            else:
                self.competitors[a].beat(self.competitors[b], scores=validated_scores)
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="win", attributes=attributes))
        else:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[b].beat(self.competitors[a], match_time=match_time, scores=reversed_scores)
            else:
                self.competitors[b].beat(self.competitors[a], scores=reversed_scores)
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="loss", attributes=attributes))

    def match_group(
        self,
        participants: Sequence[Any],
        ranks: Optional[Sequence[float]] = None,
        scores: Optional[Sequence[float]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        match_time: Optional[datetime.datetime] = None,
    ) -> None:
        """Process a single bout between three or more sides (or two sides with rosters).

        This is the N-way analogue of :meth:`matchup`: it creates missing
        competitors, captures every pre-update prediction, runs one bout-level
        update over the whole participant set, and records the result as a
        :class:`~elote.arenas.base.MultiBout` on the same history the two-player
        entry points append to.

        The bout is updated natively -- one closed-form pass over the whole
        participant set -- not as a fan-out of pairwise results, which the
        Weng-Lin family (and any other bout-level model) treats differently.

        Args:
            participants (sequence): The sides, ordered by finish. Each entry is
                either a competitor id, or an ``(id, roster)`` pair where roster
                is a list or tuple of member ids for a team side. A side is a
                roster side exactly when its entry is a list/tuple pair whose
                second item is a list or tuple. When ``ranks`` is ``None`` the
                given order is taken as the finish order.
            ranks (sequence of float, optional): Finishing ranks per participant,
                lower is better, equal ranks are ties. Whole numbers are
                required. Defaults to ``0..n-1`` (the given order). When both
                ``ranks`` and ``scores`` are given, ``ranks`` defines the result
                and ``scores`` are recorded only.
            scores (sequence of float, optional): One score per participant.
                Validated and recorded with the bout; when ``ranks`` is omitted
                the scores define the finishing ranks (descending, ties share a
                rank). Not otherwise consumed by rank-based models.
            attributes (dict, optional): Additional attributes to record with this bout.
            match_time (datetime, optional): The time when the bout occurred.

        Raises:
            NotImplementedError: If the arena's rating system does not implement
                a bout-level update (``apply_bout``). The first shipped consumer
                is :class:`~elote.OpenSkillCompetitor`.
            ValueError: If the participant list is malformed (fewer than two
                sides, an empty roster, the same competitor on two sides), if
                ``ranks`` is not one whole number per participant, or if
                ``scores`` is not one finite, non-negative number per
                participant. Validation happens before any competitor is created
                or any history is recorded, so a bad bout leaves the arena
                unchanged.
        """
        # The bout-level hook is the only supported update path: fan-out of
        # pairwise results would double-count every bout.
        apply_bout = getattr(self.base_competitor, "apply_bout", None)
        if not callable(apply_bout):
            raise NotImplementedError(
                f"{self.base_competitor.__name__} does not implement a bout-level update "
                "(apply_bout); N-way bouts are not supported for this rating system"
            )

        sides = _parse_sides(participants)
        if len(sides) < 2:
            raise ValueError(f"a bout needs at least two sides, got {len(sides)}")

        # Validate everything -- ranks, scores, participant shape, and
        # competitor identity -- before creating any competitor or recording any
        # history, so a malformed bout leaves the arena unchanged.
        side_ids = [side_id for side_id, _ in sides]
        all_member_ids: List[Any] = []
        for side_id, roster in sides:
            if roster is None:
                all_member_ids.append(side_id)
            else:
                all_member_ids.extend(roster)
        if len(set(all_member_ids)) != len(all_member_ids):
            raise ValueError("the same competitor cannot appear on two sides of one bout")

        n_sides = len(sides)
        if ranks is not None:
            bout_ranks = _validate_group_ranks(ranks, n_sides)
        elif scores is not None:
            bout_ranks = _ranks_from_scores(_validate_group_scores(scores, n_sides))
        else:
            bout_ranks = list(range(n_sides))
        validated_scores = _validate_group_scores(scores, n_sides) if scores is not None else None

        # Competitors seeded through initial_state may belong to another rating
        # system; catch that before creating anything new.
        for member_id in all_member_ids:
            existing = self.competitors.get(member_id)
            if existing is not None and not isinstance(existing, self.base_competitor):
                raise MissMatchedCompetitorTypesException(
                    f"competitor '{member_id}' is a {type(existing).__name__}, but this arena rates "
                    f"{self.base_competitor.__name__} competitors"
                )

        # Create missing competitors, then collect each side's members.
        bout_sides: List[List[BaseCompetitor]] = []
        for side_id, roster in sides:
            if roster is None:
                bout_sides.append([self._get_or_create_competitor(side_id)])
            else:
                bout_sides.append([self._get_or_create_competitor(member_id) for member_id in roster])

        # Pre-update prediction: order the sides by pre-bout strength (strongest
        # first, ties keep the given order). Every strength is read before any
        # participant is updated. Team strength is the members' mean rating.
        strengths = [sum(member.rating for member in side) / len(side) for side in bout_sides]
        predicted_ranks = [side_ids[i] for i in sorted(range(n_sides), key=lambda i: -strengths[i])]

        apply_bout(bout_sides, ranks=bout_ranks)

        self.history.add_bout(
            MultiBout(
                participants=side_ids,
                ranks=bout_ranks,
                predicted_ranks=predicted_ranks,
                scores=validated_scores,
                attributes=attributes,
                match_time=match_time,
            )
        )

    def rating_period(
        self,
        matchups: Sequence[Tuple[Any, Any, float, Optional[Sequence[float]]]],
        *,
        period_end: Optional[datetime.datetime] = None,
    ) -> None:
        """Process a batch of results against one shared pre-period state.

        Every row is validated before the arena is changed. Predictions for all rows
        are then captured before the batch is applied, so an earlier result in the
        period cannot inform a later prediction from the same period.

        Args:
            matchups: ``(competitor_a, competitor_b, outcome, scores)`` tuples. Outcomes
                are ``1.0``, ``0.0``, or ``0.5`` from A's perspective. Scores are
                optional and use ``(a_score, b_score)`` caller order.

            period_end: The shared activity time for time-aware competitors.

        Raises:
            ValueError: If any outcome or score payload is invalid. Validation happens
                before competitors or bouts are added.
        """
        validated_matchups: List[Tuple[Any, Any, float, Optional[Tuple[float, float]]]] = []
        for a, b, outcome, scores in matchups:
            self._validate_outcome(outcome)
            validated_scores = BaseCompetitor._validate_scores(scores, outcome)
            validated_matchups.append((a, b, outcome, validated_scores))

        for a, b, _, _ in validated_matchups:
            if a not in self.competitors:
                self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
            if b not in self.competitors:
                self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        predictions = [self.expected_score(a, b) for a, b, _, _ in validated_matchups]
        results = [
            (self.competitors[a], self.competitors[b], outcome, scores)
            for a, b, outcome, scores in validated_matchups
        ]
        self.base_competitor.apply_rating_period(results, period_end=period_end)

        for (a, b, outcome, _), predicted_outcome in zip(validated_matchups, predictions, strict=True):
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome=outcome))

    def expected_score(self, a: Any, b: Any) -> float:
        """Calculate the expected score for a matchup between two competitors.

        This method returns the probability that competitor a will beat competitor b.
        It is a read: an identifier the arena has not seen is scored as an unrated
        competitor without being added to the population.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.

        Returns:
            float: The probability that a will beat b (between 0 and 1).
        """
        return self._competitor_for_prediction(a).expected_score(self._competitor_for_prediction(b))

    def _competitor_for_prediction(self, id_val: Any) -> BaseCompetitor:
        """Return the competitor for an identifier without adding it to the population."""
        if id_val in self.competitors:
            return self.competitors[id_val]
        logger.warning(
            "Competitor '%s' is not in this arena; predicting with an unrated competitor "
            "without adding it to the population.",
            id_val,
        )
        return self.base_competitor(**self.base_competitor_kwargs)

    def export_state(self) -> Dict[Any, Dict[str, Any]]:
        """Export the current state of this arena for serialization.

        Returns:
            dict: A dictionary containing the state of all competitors in this arena.
        """
        state: Dict[Any, Dict[str, Any]] = dict()
        for k, v in self.competitors.items():
            state[k] = v.export_state()
        return state

    # type: ignore[override]
    def leaderboard(self) -> List[Dict[str, Any]]:
        """Generate a leaderboard of all competitors.

        Returns:
            list: A list of dictionaries containing competitor IDs and their ratings,
                 sorted by rating in descending order.
        """
        # Restore original implementation returning list of dicts
        lb: List[Dict[str, Any]] = [{"competitor": k, "rating": v.rating} for k, v in self.competitors.items()]

        # Sort best-first (descending by rating) to match the docstring and method name
        return sorted(lb, key=lambda x: x.get("rating"), reverse=True)  # type: ignore[arg-type, return-value]

    def _get_or_create_competitor(self, id_val: str) -> BaseCompetitor:
        if id_val not in self.competitors:
            comp = self.base_competitor(**self.base_competitor_kwargs)
            self.competitors[id_val] = comp
            logger.debug("Created new competitor '%s' of type %s", id_val, self.base_competitor.__name__)
        return self.competitors[id_val]

    def process_history(self, bouts: List[Tuple[str, str, Optional[float]]], progress_bar: bool = True) -> None:
        iterator = tqdm(bouts) if progress_bar else bouts
        total_bouts = len(bouts)
        skipped_bouts = 0
        logger.info("Processing %d bouts for training history", total_bouts)
        for i, (a, b, outcome) in enumerate(iterator):
            self._validate_outcome(outcome)
            # Get or create competitors
            c_a = self._get_or_create_competitor(a)
            c_b = self._get_or_create_competitor(b)

            if outcome is not None:
                predicted_outcome = c_a.expected_score(c_b)
                if outcome == 1:
                    c_a.beat(c_b)
                elif outcome == 0:
                    c_b.beat(c_a)
                else:
                    c_a.tied(c_b)
                new_bout = Bout(a=a, b=b, outcome=outcome, predicted_outcome=predicted_outcome)
                self.history.add_bout(bout=new_bout)
            else:
                skipped_bouts += 1
                logger.debug("Skipping bout %d/%d: outcome value is None", i + 1, total_bouts)

        if skipped_bouts > 0:
            logger.info(
                "Skipped %d/%d bouts during training history processing due to None outcome", skipped_bouts, total_bouts
            )
        logger.info("Finished processing training history. %d competitors tracked.", len(self.competitors))

    def evaluate_performance(
        self, eval_bouts: List[Tuple[str, str, Optional[float]]], progress_bar: bool = True
    ) -> None:
        """Evaluate the performance of the competitors based on a list of evaluation bouts.

        Bouts naming a competitor the arena has not seen are skipped: evaluation never
        adds to the population, so unknown identifiers contribute no prediction.

        Args:
            eval_bouts (list): A list of (competitor_a, competitor_b, outcome) tuples.
            progress_bar (bool, optional): Whether to display a progress bar.
        """
        iterator = tqdm(eval_bouts) if progress_bar else eval_bouts
        total_bouts = len(eval_bouts)
        skipped_bouts = 0
        logger.info("Evaluating performance using %d bouts", total_bouts)
        for i, (a, b, outcome) in enumerate(iterator):
            self._validate_outcome(outcome)
            # Evaluation is a read: an identifier the arena never trained on has no
            # rating to evaluate, so the bout is skipped rather than scored against a
            # freshly defaulted competitor.
            unknown = a if a not in self.competitors else (b if b not in self.competitors else None)
            if unknown is not None:
                skipped_bouts += 1
                logger.warning(
                    "Skipping evaluation bout %d/%d: Competitor '%s' not found in training history.",
                    i + 1,
                    total_bouts,
                    unknown,
                )
                continue

            c_a = self.competitors[a]
            c_b = self.competitors[b]

            # Skip bouts where the actual outcome is None
            if outcome is None:
                skipped_bouts += 1
                continue

            # Calculate the expected outcome
            predicted_outcome = c_a.expected_score(c_b)
            new_bout = Bout(a=a, b=b, outcome=outcome, predicted_outcome=predicted_outcome)
            self.eval_history.add_bout(bout=new_bout)

        if skipped_bouts > 0:
            logger.info(
                "Skipped %d/%d bouts during evaluation due to missing competitors or None outcome.",
                skipped_bouts,
                total_bouts,
            )
        logger.info("Finished performance evaluation.")

    def validate(self, validation_bouts: List[Tuple[str, str, Optional[float]]], progress_bar: bool = True) -> None:
        """Run a validation set through the arena without updating ratings, only recording predictions.

        Bouts naming a competitor the arena has not seen are skipped: validation never
        adds to the population, so unknown identifiers contribute no prediction.

        Args:
            validation_bouts (list): A list of (competitor_a, competitor_b, outcome) tuples.
            progress_bar (bool, optional): Whether to display a progress bar.
        """
        iterator = tqdm(validation_bouts) if progress_bar else validation_bouts
        total_bouts = len(validation_bouts)
        skipped_bouts = 0
        logger.info("Validating model using %d bouts", total_bouts)
        for i, (a, b, outcome) in enumerate(iterator):
            self._validate_outcome(outcome)
            # Validation is a read: an identifier the arena never trained on has no
            # rating to validate, so the bout is skipped rather than scored against a
            # freshly defaulted competitor.
            unknown = a if a not in self.competitors else (b if b not in self.competitors else None)
            if unknown is not None:
                skipped_bouts += 1
                logger.warning(
                    "Skipping validation bout %d/%d: Competitor '%s' not found in training history.",
                    i + 1,
                    total_bouts,
                    unknown,
                )
                continue

            c_a = self.competitors[a]
            c_b = self.competitors[b]

            # Skip bouts where the actual outcome is None
            if outcome is None:
                skipped_bouts += 1
                continue

            # Calculate the expected outcome
            predicted_outcome = c_a.expected_score(c_b)
            new_bout = Bout(a=a, b=b, outcome=outcome, predicted_outcome=predicted_outcome)
            self.validation_history.add_bout(bout=new_bout)

        if skipped_bouts > 0:
            logger.info(
                "Skipped %d/%d bouts during validation due to missing competitors or None outcome.",
                skipped_bouts,
                total_bouts,
            )
        logger.info("Finished model validation.")

    def get_competitor_by_id(self, id_val: str) -> Optional[BaseCompetitor]:
        """Retrieve a competitor by their ID.

        Args:
            id_val (str): The ID of the competitor to retrieve.

        Returns:
            Optional[BaseCompetitor]: The retrieved competitor, or None if not found.
        """
        comp = self.competitors.get(id_val)
        if comp is None:
            logger.debug("Competitor with ID '%s' not found.", id_val)
        else:
            logger.debug("Retrieved competitor '%s'", id_val)
        return comp

    def get_all_competitors(self) -> List[BaseCompetitor]:
        """Retrieve a list of all competitors in the arena.

        Returns:
            list: A list of all competitors.
        """
        logger.debug("Retrieving all %d competitors.", len(self.competitors))
        return list(self.competitors.values())
