from typing import Dict, Any, ClassVar, Optional, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

T = TypeVar("T", bound="EloCompetitor")


class EloCompetitor(BaseCompetitor):
    """Elo rating system competitor.

    The Elo rating system is a method for calculating the relative skill levels of players
    in zero-sum games such as chess. It is named after its creator Arpad Elo, a Hungarian-American
    physics professor and chess master.

    In the Elo system, each player's rating changes based on the outcome of games and the
    rating of their opponents. The difference in ratings between two players determines
    the expected outcome of a match, and the actual outcome is used to update the ratings.

    Class Attributes:
        _base_rating (float): Base rating divisor used in the transformed rating calculation. Default: 400.
        _k_factor (float): Factor that determines how much ratings change after each match. Default: 32.
    """

    _base_rating: ClassVar[float] = 400
    _k_factor: ClassVar[float] = 32

    def __init__(self, initial_rating: float = 400, k_factor: Optional[float] = None):
        """Initialize an Elo competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 400.
            k_factor (float, optional): The K-factor to use for this competitor. If None,
                                       the class K-factor will be used. Default: None.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
            InvalidParameterException: If the k_factor is negative.
        """
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        if k_factor is not None and k_factor <= 0:
            raise InvalidParameterException("K-factor must be positive")

        self._initial_rating = initial_rating
        self._rating = initial_rating
        self._k_factor = k_factor if k_factor is not None else self._k_factor

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<EloCompetitor: rating={self._rating}, k_factor={self._k_factor}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<EloCompetitor: rating={self._rating}>"

    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        return {
            "initial_rating": self._initial_rating,
            "current_rating": self._rating,
            "k_factor": self._k_factor,
            "class_vars": {
                "base_rating": self._base_rating,
            },
        }

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            EloCompetitor: A new competitor with the same state as the exported one.

        Raises:
            KeyError: If the state dictionary is missing required keys.
        """
        # Configure class variables if provided
        if "class_vars" in state:
            class_vars = state["class_vars"]
            if "base_rating" in class_vars:
                cls._base_rating = class_vars["base_rating"]

        # Create a new competitor with the initial rating
        competitor = cls(initial_rating=state.get("initial_rating", 400), k_factor=state.get("k_factor", None))

        # Set the current rating if provided
        if "current_rating" in state:
            competitor._rating = state["current_rating"]

        return competitor

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating to the initial value.
        """
        self._rating = self._initial_rating

    @property
    def transformed_rating(self) -> float:
        """Get the transformed rating of this competitor.

        The transformed rating is used in the expected score calculation.

        Returns:
            float: The transformed rating.
        """
        return 10 ** (self._rating / self._base_rating)

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

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        In Elo rating, a player's expected score is their probability of winning plus half their probability of drawing.
        This is because in Elo rating a draw is not explicitly accounted for, but rather counted as half of a win and
        half of a loss. It can make the expected score a bit difficult to reason about, so be careful.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        return self.transformed_rating / (competitor.transformed_rating + self.transformed_rating)

    def beat(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on the match outcome where this competitor won.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)

        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self._rating = self._rating + self._k_factor * (1 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0 - lose_es)

    def tied(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on a drawn match outcome.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)

        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self._rating = self._rating + self._k_factor * (0.5 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0.5 - lose_es)
