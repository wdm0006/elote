import math
from typing import Dict, Any, ClassVar, Tuple, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

T = TypeVar("T", bound="GlickoCompetitor")


class GlickoCompetitor(BaseCompetitor):
    """Glicko rating system competitor.

    The Glicko rating system is an improvement on the Elo rating system that takes
    into account the reliability of a rating. It was developed by Mark Glickman as
    an improvement to the Elo system.

    In addition to a rating, each competitor has a rating deviation (RD) that measures
    the reliability of the rating. A higher RD indicates a less reliable rating.

    Class Attributes:
        _c (float): Constant that determines how quickly the RD increases over time. Default: 1.
        _q (float): Scaling factor used in the rating calculation. Default: 0.0057565.
    """

    _c: ClassVar[float] = 1
    _q: ClassVar[float] = 0.0057565

    def __init__(self, initial_rating: float = 1500, initial_rd: float = 350):
        """Initialize a Glicko competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 1500.
            initial_rd (float, optional): The initial rating deviation of this competitor. Default: 350.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
            InvalidParameterException: If the initial RD is not positive.
        """
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

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<GlickoCompetitor: rating={self._rating}, rd={self.rd}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<GlickoCompetitor: rating={self._rating}, rd={self.rd}>"

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_rating": self._initial_rating,
            "initial_rd": self._initial_rd,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "rating": self._rating,
            "rd": self.rd,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_rating
        initial_rating = parameters.get("initial_rating", 1500)
        if initial_rating < self._minimum_rating:
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self._initial_rating = initial_rating

        # Validate and set initial_rd
        initial_rd = parameters.get("initial_rd", 350)
        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")
        self._initial_rd = initial_rd

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        # Validate and set rating
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
            raise InvalidParameterException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = rating

        # Validate and set rd
        rd = state.get("rd", self._initial_rd)
        if rd <= 0:
            raise InvalidParameterException("RD must be positive")
        self.rd = rd

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            GlickoCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_rating=parameters.get("initial_rating", 1500),
            initial_rd=parameters.get("initial_rd", 350),
        )

    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        # Use the new standardized format
        return super().export_state()

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            GlickoCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Handle legacy state format
        if "type" not in state:
            # Configure class variables if provided
            if "class_vars" in state:
                class_vars = state["class_vars"]
                if "c" in class_vars:
                    cls._c = class_vars["c"]
                if "q" in class_vars:
                    cls._q = class_vars["q"]

            # Create a new competitor with the initial rating and RD
            competitor = cls(initial_rating=state.get("initial_rating", 1500), initial_rd=state.get("initial_rd", 350))

            # Set the current rating and RD if provided
            if "current_rating" in state:
                competitor._rating = state["current_rating"]
            if "current_rd" in state:
                competitor.rd = state["current_rd"]

            return competitor

        # Use the new standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating and RD to their initial values.
        """
        self._rating = self._initial_rating
        self.rd = self._initial_rd

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

    @property
    def tranformed_rd(self) -> float:
        """Get the transformed rating deviation of this competitor.

        The transformed RD is used in the rating calculation and is capped at 350.

        Returns:
            float: The transformed rating deviation.
        """
        return min([350, math.sqrt(self.rd**2 + self._c**2)])

    @classmethod
    def _g(cls, x: float) -> float:
        """Calculate the g-function used in the Glicko rating system.

        Args:
            x (float): The input value.

        Returns:
            float: The g-function value.
        """
        return 1 / (math.sqrt(1 + 3 * cls._q**2 * (x**2) / math.pi**2))

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

        g_term = self._g(self.rd**2)
        E = 1 / (1 + 10 ** ((-1 * g_term * (self._rating - competitor.rating)) / 400))
        return E

    def beat(self, competitor: "GlickoCompetitor") -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on the match outcome where this competitor won.

        Args:
            competitor (GlickoCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        self._compute_match_result(competitor, s=1)

    def tied(self, competitor: "GlickoCompetitor") -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on a drawn match outcome.

        Args:
            competitor (GlickoCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        self._compute_match_result(competitor, s=0.5)

    def _compute_match_result(self, competitor: "GlickoCompetitor", s: float) -> None:
        """Compute the result of a match and update ratings.

        Args:
            competitor (GlickoCompetitor): The opponent competitor.
            s (float): The score of this competitor (1 for win, 0.5 for draw, 0 for loss).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        # first we update ourselves
        s_new_r, s_new_rd = self.update_competitor_rating(competitor, s)

        # then the competitor
        s = abs(s - 1)
        c_new_r, c_new_rd = competitor.update_competitor_rating(self, s)

        # assign everything
        self._rating = s_new_r
        self.rd = s_new_rd
        competitor.rating = c_new_r
        competitor.rd = c_new_rd

    def update_competitor_rating(self, competitor: "GlickoCompetitor", s: float) -> Tuple[float, float]:
        """Update the rating and RD of this competitor based on a match result.

        Args:
            competitor (GlickoCompetitor): The opponent competitor.
            s (float): The score of this competitor (1 for win, 0.5 for draw, 0 for loss).

        Returns:
            tuple: A tuple containing the new rating and RD.
        """
        E_term = self.expected_score(competitor)
        d_squared = (self._q**2 * (self._g(competitor.rd) ** 2 * E_term * (1 - E_term))) ** -1
        s_new_r = self._rating + (self._q / (1 / self.rd**2 + 1 / d_squared)) * self._g(competitor.rd) * (s - E_term)

        # Ensure the new rating doesn't go below the minimum rating
        s_new_r = max(self._minimum_rating, s_new_r)

        s_new_rd = math.sqrt((1 / self.rd**2 + 1 / d_squared) ** -1)
        return s_new_r, s_new_rd
