import math
from typing import Dict, Any, ClassVar, Optional, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

T = TypeVar("T", bound="DWZCompetitor")


class DWZCompetitor(BaseCompetitor):
    """Deutsche Wertungszahl (DWZ) rating system competitor.

    The DWZ is the German chess rating system, similar to Elo but with some
    differences in how ratings are updated after matches.

    Class Attributes:
        _J (int): Development coefficient. Default: 10.
    """

    _J: ClassVar[int] = 10

    def __init__(self, initial_rating: float = 400):
        """Initialize a DWZ competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 400.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
        """
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        self._initial_rating = initial_rating
        self._count = 0
        self._initial_count = 0
        self._rating = initial_rating
        self._cached_E: Optional[float] = None
        self._cached_rating_for_E: Optional[float] = None
        self._cached_count_for_E: Optional[int] = None

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<DWZCompetitor: rating={self._rating}, count={self._count}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<DWZCompetitor: rating={self._rating}>"

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
        # Invalidate cache when rating changes
        self._cached_E = None
        self._cached_rating_for_E = None

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
            "initial_rating": self._initial_rating,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "rating": self._rating,
            "count": self._count,
            "initial_count": self._initial_count,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_rating
        initial_rating = parameters.get("initial_rating", 400)
        if initial_rating < self._minimum_rating:
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self._initial_rating = initial_rating

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidStateException: If any state variable is invalid.
        """
        # Validate and set rating
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
            raise InvalidParameterException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = rating

        # Set count and initial_count
        self._count = state.get("count", 0)
        self._initial_count = state.get("initial_count", 0)

        # Invalidate cache
        self._cached_E = None
        self._cached_rating_for_E = None
        self._cached_count_for_E = None

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            DWZCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_rating=parameters.get("initial_rating", 400),
        )

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            DWZCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Handle legacy state format
        if "type" not in state:
            # Configure class variables if provided
            if "class_vars" in state:
                class_vars = state["class_vars"]
                if "J" in class_vars:
                    cls._J = class_vars["J"]

            # Create a new competitor with the initial rating
            competitor = cls(initial_rating=state.get("initial_rating", 400))

            # Set the current rating and count if provided
            if "current_rating" in state:
                competitor._rating = state["current_rating"]
            if "count" in state:
                competitor._count = state["count"]
            if "initial_count" in state:
                competitor._initial_count = state["initial_count"]
            else:
                competitor._initial_count = competitor._count

            return competitor

        # Use the new standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating and count to their initial values.
        """
        self._rating = self._initial_rating
        self._count = self._initial_count
        self._cached_E = None
        self._cached_rating_for_E = None
        self._cached_count_for_E = None

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

        # Direct calculation to avoid property access overhead
        competitor_rating = competitor._rating
        return 1 / (1 + 10 ** ((competitor_rating - self._rating) / 400))

    @property
    def _E(self) -> int:
        """Calculate the development coefficient E.

        The development coefficient determines how much a player's rating changes
        after a match, based on their current rating and number of games played.

        Returns:
            int: The development coefficient.
        """
        # Check if we can use cached value
        if (
            self._cached_E is not None
            and self._cached_rating_for_E == self._rating
            and self._cached_count_for_E == self._count
        ):
            return self._cached_E

        # Calculate E
        E0 = (self._rating / 1000) ** 4 + self._J
        a = max([0.5, min([self._rating / 2000, 1])])

        if self._rating < 1300:
            B = math.exp((1300 - self._rating) / 150) - 1
        else:
            B = 0

        E = int(a * E0 + B)
        if B == 0:
            result = max([5, min([E, min([30, 5 * self._count])])])
        else:
            result = max([5, min([E, 150])])

        # Cache the result
        self._cached_E = result
        self._cached_rating_for_E = self._rating
        self._cached_count_for_E = self._count

        return result

    def _new_rating(self, competitor: "DWZCompetitor", W_a: float) -> float:
        """Calculate the new rating after a match.

        Args:
            competitor (DWZCompetitor): The opponent competitor.
            W_a (float): The actual outcome of the match (1 for win, 0.5 for draw, 0 for loss).

        Returns:
            float: The new rating.
        """
        # Calculate expected score directly to avoid multiple property accesses
        expected = self.expected_score(competitor)
        E_value = self._E  # Get E value once
        return self._rating + (800 / (E_value + self._count)) * (W_a - expected)

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

        # Calculate new ratings
        self_rating = self._new_rating(competitor, 1)
        competitor_rating = competitor._new_rating(self, 0)

        # Update ratings and counts in one go
        self._rating = self_rating
        self._count += 1
        self._cached_E = None  # Invalidate cache

        competitor.rating = competitor_rating
        competitor._count += 1

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

        # Calculate new ratings
        self_rating = self._new_rating(competitor, 0.5)
        competitor_rating = competitor._new_rating(self, 0.5)

        # Update ratings and counts in one go
        self._rating = self_rating
        self._count += 1
        self._cached_E = None  # Invalidate cache

        competitor.rating = competitor_rating
        competitor._count += 1
