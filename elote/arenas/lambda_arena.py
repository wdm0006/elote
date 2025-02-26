from tqdm import tqdm
from elote import EloCompetitor
from elote.arenas.base import BaseArena, Bout, History


class LambdaArena(BaseArena):
    def __init__(
        self,
        func,
        base_competitor=EloCompetitor,
        base_competitor_kwargs=None,
        initial_state=None,
    ):
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
        self.func = func
        self.competitors = dict()
        self.base_competitor = base_competitor
        if base_competitor_kwargs is None:
            self.base_competitor_kwargs = dict()
        else:
            self.base_competitor_kwargs = base_competitor_kwargs

        # if some initial state is passed in, we can seed the population
        if initial_state is not None:
            for k, v in initial_state.items():
                self.competitors[k] = self.base_competitor(**v)

        self.history = History()

    def clear_history(self):
        """Clear the history of bouts in this arena."""
        self.history = History()

    def set_competitor_class_var(self, name, value):
        """Set a class variable on the base competitor class.

        This method allows for global configuration of all competitors
        managed by this arena.

        Args:
            name (str): The name of the class variable to set.
            value: The value to set for the class variable.
        """
        setattr(self.base_competitor, name, value)

    def tournament(self, matchups):
        """Run a tournament with the given matchups.

        Process multiple matchups between competitors, updating ratings
        after each matchup.

        Args:
            matchups (list): A list of (competitor_a, competitor_b) tuples.

        Returns:
            list: A list of bout results.
        """
        for data in tqdm(matchups):
            self.matchup(*data)

    def matchup(self, a, b, attributes=None):
        """Process a single matchup between two competitors.

        This method handles a matchup between two competitors, creating them
        if they don't already exist in the arena. It uses the comparison function
        to determine the outcome and updates the ratings accordingly.

        Args:
            a: The first competitor or competitor identifier.
            b: The second competitor or competitor identifier.
            attributes (dict, optional): Additional attributes to record with this bout.

        Returns:
            The result of the matchup.
        """
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        predicted_outcome = self.expected_score(a, b)

        if attributes:
            res = self.func(a, b, attributes=attributes)
        else:
            res = self.func(a, b)

        if res is None:
            self.competitors[a].tied(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="tie", attributes=attributes))
        elif res is True:
            self.competitors[a].beat(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="win", attributes=attributes))
        else:
            self.competitors[b].beat(self.competitors[a])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome="loss", attributes=attributes))

    def expected_score(self, a, b):
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

    def export_state(self):
        """Export the current state of this arena for serialization.

        Returns:
            dict: A dictionary containing the state of all competitors in this arena.
        """
        state = dict()
        for k, v in self.competitors.items():
            state[k] = v.export_state()
        return state

    def leaderboard(self):
        """Generate a leaderboard of all competitors.

        Returns:
            list: A list of dictionaries containing competitor IDs and their ratings,
                 sorted by rating in descending order.
        """
        lb = [{"competitor": k, "rating": v.rating} for k, v in self.competitors.items()]

        return sorted(lb, key=lambda x: x.get("rating"))
