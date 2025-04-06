import math
from typing import Dict, Any, ClassVar, Optional, Type, TypeVar, cast

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

T = TypeVar("T", bound="DWZCompetitor")


class DWZCompetitor(BaseCompetitor):
    """Deutsche Wertungszahl (DWZ) rating system competitor.

    The DWZ is the German chess rating system, similar to Elo but with some
    differences in how ratings are updated after matches, including factors
    based on player age and performance.

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
        competitor_dwz = cast(DWZCompetitor, competitor)

        # Direct calculation to avoid property access overhead
        competitor_rating = competitor_dwz._rating
        return 1.0 / (1.0 + 10 ** ((competitor_rating - self._rating) / 400.0))

    def _calculate_E(self, age: Optional[int], W_a: float, W_e: float) -> float:
        """Calculate the development coefficient E for a specific match context.

        Args:
            age (Optional[int]): The age of the player at the time of the match. Defaults to 26 if None.
            W_a (float): Achieved points in the match (1.0 for win, 0.5 for draw, 0.0 for loss).
            W_e (float): Expected points in the match.

        Returns:
            float: The calculated development coefficient E.
        """
        # Determine J based on age (default to adult > 25 if None)
        current_age = age if age is not None else 26
        if current_age <= 20:
            J = 5
        elif 21 <= current_age <= 25:
            J = 10
        else:  # age > 25 or default
            J = 15

        # Calculate E0
        Z_A = self._rating
        E0 = (Z_A / 1000.0) ** 4 + J

        # Calculate acceleration factor 'a'
        a = 1.0
        if current_age <= 20 and W_a > W_e:
            a = Z_A / 2000.0
            a = max(0.5, min(1.0, a))  # Clamp a between 0.5 and 1.0

        # Calculate braking value 'B'
        B = 0.0
        if Z_A < 1300 and W_a <= W_e:
            try:
                B = math.exp((1300.0 - Z_A) / 150.0) - 1.0
            except OverflowError:
                B = float("inf")  # Handle potential overflow for very low ratings

        # Calculate E
        E = a * E0 + B

        # Apply bounds based on tournament index 'i' (using game count + 1)
        i = self._count + 1  # Game count before this match + 1
        if B == 0.0:
            E_upper_bound = min(30.0, 5.0 * i)
        else:
            E_upper_bound = 150.0

        E = max(5.0, min(E, E_upper_bound))  # Clamp E: 5 <= E <= E_upper_bound

        return E

    def _new_rating(self, competitor: "DWZCompetitor", W_a: float, age: Optional[int] = None) -> float:
        """Calculate the new DWZ rating after a match.

        Args:
            competitor (DWZCompetitor): The opponent.
            W_a (float): The actual score achieved against the opponent (1.0, 0.5, or 0.0).
            age (Optional[int]): The age of this competitor at the time of the match. Defaults to 26.

        Returns:
            float: The new rating.
        """
        W_e = self.expected_score(competitor)
        E = self._calculate_E(age=age, W_a=W_a, W_e=W_e)

        # Formula uses n = number of games in evaluation period. Assuming game-by-game update, n=1.
        n = 1
        new_rating = self._rating + (800.0 / (E + n)) * (W_a - W_e)

        # Ensure rating doesn't fall below minimum
        return max(self._minimum_rating, new_rating)

    def beat(self, competitor: BaseCompetitor, age: Optional[int] = None) -> None:
        """Update ratings after this competitor wins against another.

        Args:
            competitor (BaseCompetitor): The opponent competitor.
            age (Optional[int]): The age of this competitor at the time of the match.
                                 Used for DWZ calculation. Defaults to 26 (adult).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_dwz = cast(DWZCompetitor, competitor)

        # Calculate new ratings using the age parameter
        my_new_rating = self._new_rating(competitor_dwz, 1.0, age=age)
        opponent_new_rating = competitor_dwz._new_rating(self, 0.0, age=None)  # Opponent age not needed here

        # Update ratings and counts simultaneously
        self.rating = my_new_rating
        competitor_dwz.rating = opponent_new_rating
        self._count += 1
        competitor_dwz._count += 1

    def tied(self, competitor: BaseCompetitor, age: Optional[int] = None) -> None:
        """Update ratings after this competitor ties with another.

        Args:
            competitor (BaseCompetitor): The opponent competitor.
            age (Optional[int]): The age of this competitor at the time of the match.
                                 Used for DWZ calculation. Defaults to 26 (adult).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_dwz = cast(DWZCompetitor, competitor)

        # Calculate new ratings using the age parameter
        my_new_rating = self._new_rating(competitor_dwz, 0.5, age=age)
        opponent_new_rating = competitor_dwz._new_rating(self, 0.5, age=None)  # Opponent age not needed here

        # Update ratings and counts simultaneously
        self.rating = my_new_rating
        competitor_dwz.rating = opponent_new_rating
        self._count += 1
        competitor_dwz._count += 1
