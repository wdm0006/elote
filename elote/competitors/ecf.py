from typing import Dict, Any, ClassVar, Optional, Type, TypeVar
from collections import deque
import statistics

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

T = TypeVar("T", bound="ECFCompetitor")


class ECFCompetitor(BaseCompetitor):
    """English Chess Federation (ECF) rating system competitor.

    The ECF rating system is used by the English Chess Federation to rate chess players.
    It uses a moving average of performance ratings over a number of periods.

    Class Attributes:
        _delta (float): Maximum rating difference considered for updates. Default: 50.
        _n_periods (int): Number of periods to consider for the moving average. Default: 30.
    """

    _delta: ClassVar[float] = 50
    _n_periods: ClassVar[int] = 30

    def __init__(self, initial_rating: float = 100):
        """Initialize an ECF competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 100.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
        """
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        self.__initial_rating = initial_rating
        self.scores: Optional[deque] = None
        self.__cached_rating: Optional[float] = None  # Cache for the rating calculation

        # Cache for transformed rating calculation
        self._cached_transformed_elo_rating: Optional[float] = None
        self._cached_elo_conversion_for_transform: Optional[float] = None

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<ECFCompetitor: rating={self.rating}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<ECFCompetitor: rating={self.rating}>"

    def __initialize_ratings(self) -> None:
        """Initialize the ratings deque with the initial rating."""
        # Initialize with a single value instead of a full deque of None values
        self.scores = deque(maxlen=self._n_periods)
        self.scores.append(self.__initial_rating)
        self.__cached_rating = self.__initial_rating  # Initialize the cache

    @property
    def elo_conversion(self) -> float:
        """Convert the ECF rating to an approximate Elo rating.

        Returns:
            float: The approximate Elo rating.
        """
        return self.rating * 7.5 + 700

    @property
    def rating(self) -> float:
        """Get the current rating of this competitor.

        The rating is the average of all scores in the deque.

        Returns:
            float: The current rating.
        """
        if self.scores is None:
            self.__initialize_ratings()
            return self.__cached_rating

        # Return cached value if available
        if self.__cached_rating is not None:
            return self.__cached_rating

        # Calculate and cache the mean
        valid_scores = [_ for _ in self.scores if _ is not None]
        if valid_scores:
            self.__cached_rating = statistics.mean(valid_scores)
        else:
            self.__cached_rating = self.__initial_rating

        return self.__cached_rating

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor.

        This method is not directly supported by the ECF rating system,
        as ratings are calculated from the scores deque. This implementation
        adds the provided value to the scores deque.

        Args:
            value (float): The new rating value.

        Raises:
            InvalidRatingValueException: If the rating value is below the minimum rating.
        """
        if value < self._minimum_rating:
            raise InvalidRatingValueException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")

        self._update(value)

    def _update(self, rating: float) -> None:
        """Update the scores deque with a new rating.

        Args:
            rating (float): The new rating to add to the scores deque.
        """
        if self.scores is None:
            self.__initialize_ratings()

        # Invalidate the cache when updating scores
        self.__cached_rating = None
        self._cached_transformed_elo_rating = None
        self._cached_elo_conversion_for_transform = None

        # Add the new rating
        self.scores.append(rating)

    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        # Use the new standardized format
        return super().export_state()

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_rating": self.__initial_rating,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        scores_list = list(self.scores) if self.scores is not None else []
        return {
            "scores": scores_list,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_rating
        initial_rating = parameters.get("initial_rating", 100)
        if initial_rating < self._minimum_rating:
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self.__initial_rating = initial_rating

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidStateException: If any state variable is invalid.
        """
        # Restore the scores if provided
        scores = state.get("scores", [])
        if scores:
            self.scores = deque(scores, maxlen=self._n_periods)
            self.__cached_rating = None  # Force recalculation
        else:
            self.__initialize_ratings()

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            ECFCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_rating=parameters.get("initial_rating", 100),
        )

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            ECFCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Handle legacy state format
        if "type" not in state:
            # Configure class variables if provided
            if "class_vars" in state:
                class_vars = state["class_vars"]
                if "delta" in class_vars:
                    cls._delta = class_vars["delta"]
                if "n_periods" in class_vars:
                    cls._n_periods = class_vars["n_periods"]

            # Create a new competitor with the initial rating
            competitor = cls(initial_rating=state.get("initial_rating", 40))

            # Restore the scores if provided
            if "scores" in state and state["scores"]:
                competitor.scores = deque(state["scores"], maxlen=cls._n_periods)
                competitor.__cached_rating = None  # Force recalculation

            return competitor

        # Use the new standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's scores deque to contain only the initial rating.
        """
        self.scores = deque(maxlen=self._n_periods)
        self.scores.append(self.__initial_rating)
        self.__cached_rating = self.__initial_rating
        self._cached_transformed_elo_rating = None
        self._cached_elo_conversion_for_transform = None

    @property
    def transformed_elo_rating(self) -> float:
        """Get the transformed Elo rating of this competitor.

        The transformed rating is used in the expected score calculation.

        Returns:
            float: The transformed Elo rating.
        """
        # Check if we can use the cached value
        current_elo = self.elo_conversion
        if self._cached_transformed_elo_rating is not None and self._cached_elo_conversion_for_transform == current_elo:
            return self._cached_transformed_elo_rating

        # Calculate and cache the result
        self._cached_elo_conversion_for_transform = current_elo
        self._cached_transformed_elo_rating = 10 ** (current_elo / 400)
        return self._cached_transformed_elo_rating

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)

        # Calculate directly to avoid multiple property accesses
        my_transformed = self.transformed_elo_rating
        their_transformed = competitor.transformed_elo_rating
        return my_transformed / (their_transformed + my_transformed)

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

        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                competitors_rating = self_rating - self._delta
            else:
                competitors_rating = self_rating + self._delta

        # Update both competitors in one go
        self._update(competitors_rating + self._delta)
        competitor._update(self_rating - self._delta)

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

        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                competitors_rating = self_rating - self._delta
            else:
                competitors_rating = self_rating + self._delta

        # Update both competitors in one go
        self._update(competitors_rating)
        competitor._update(self_rating)
