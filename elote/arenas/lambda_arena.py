import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from tqdm import tqdm
from elote import EloCompetitor
from elote.arenas.base import BaseArena, Bout, History
from elote.competitors.base import BaseCompetitor
from elote.logging import logger


class LambdaArena(BaseArena):
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
        if outcome is not None and outcome not in (1.0, 0.0, 0.5):
            raise ValueError(f"outcome must be one of 1.0, 0.0 or 0.5, got {outcome!r}")
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

    def expected_score(self, a: Any, b: Any) -> float:
        """Calculate the expected score for a matchup between two competitors.

        This method returns the probability that competitor a will beat competitor b.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.

        Returns:
            float: The probability that a will beat b (between 0 and 1).
        """
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        return self.competitors[a].expected_score(self.competitors[b])

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

        Args:
            eval_bouts (list): A list of (competitor_a, competitor_b, outcome) tuples.
            progress_bar (bool, optional): Whether to display a progress bar.
        """
        iterator = tqdm(eval_bouts) if progress_bar else eval_bouts
        total_bouts = len(eval_bouts)
        skipped_bouts = 0
        logger.info("Evaluating performance using %d bouts", total_bouts)
        for i, (a, b, outcome) in enumerate(iterator):
            # Get or create competitors
            try:
                c_a = self._get_or_create_competitor(a)
                c_b = self._get_or_create_competitor(b)
            except KeyError as e:
                # If a competitor is not found, skip this bout
                skipped_bouts += 1
                logger.warning(
                    "Skipping evaluation bout %d/%d: Competitor '%s' not found in training history.",
                    i + 1,
                    total_bouts,
                    e,
                )
                continue

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

        Args:
            validation_bouts (list): A list of (competitor_a, competitor_b, outcome) tuples.
            progress_bar (bool, optional): Whether to display a progress bar.
        """
        iterator = tqdm(validation_bouts) if progress_bar else validation_bouts
        total_bouts = len(validation_bouts)
        skipped_bouts = 0
        logger.info("Validating model using %d bouts", total_bouts)
        for i, (a, b, outcome) in enumerate(iterator):
            # Get or create competitors
            try:
                c_a = self._get_or_create_competitor(a)
                c_b = self._get_or_create_competitor(b)
            except KeyError as e:
                # If a competitor is not found, skip this bout
                skipped_bouts += 1
                logger.warning(
                    "Skipping validation bout %d/%d: Competitor '%s' not found in training history.",
                    i + 1,
                    total_bouts,
                    e,
                )
                continue

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
