import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tqdm import tqdm
from elote import EloCompetitor
from elote.arenas.base import BaseArena, Bout, History
from elote.competitors.base import BaseCompetitor


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
                competitor IDs to their initial parameters.
        """
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
                self.competitors[k] = self.base_competitor(**v)

        self.history: History = History()

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

    def tournament(self, matchups: List[Tuple[Any, Any]]) -> None:
        """Run a tournament with the given matchups.

        Process multiple matchups between competitors, updating ratings
        after each matchup.

        Args:
            matchups (list): A list of (competitor_a, competitor_b) tuples.
        """
        for data in tqdm(matchups):
            self.matchup(*data)

    def matchup(
        self,
        a: Any,
        b: Any,
        attributes: Optional[Dict[str, Any]] = None,
        match_time: Optional[datetime.datetime] = None,
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
        """
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        predicted_outcome: float = self.expected_score(a, b)

        if attributes:
            res = self.func(a, b, attributes=attributes)
        else:
            res = self.func(a, b)

        # Check if the competitor supports time-based ratings
        supports_time = hasattr(self.competitors[a], "_last_activity")

        if res is None:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[a].tied(self.competitors[b], match_time=match_time)
            else:
                self.competitors[a].tied(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="tie", attributes=attributes))
        elif res is True:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[a].beat(self.competitors[b], match_time=match_time)
            else:
                self.competitors[a].beat(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="win", attributes=attributes))
        else:
            if supports_time:
                # type: ignore[call-arg]
                self.competitors[b].beat(self.competitors[a], match_time=match_time)
            else:
                self.competitors[b].beat(self.competitors[a])
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

        # Restore original sorting key and add ignores for mypy errors
        return sorted(lb, key=lambda x: x.get("rating"))  # type: ignore[arg-type, return-value]
