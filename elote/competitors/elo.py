from typing import Dict, Any, ClassVar, Optional, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException
from elote.logging import logger  # Import directly from the logging submodule

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
        super().__init__()  # Call base class constructor
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        if k_factor is not None and k_factor <= 0:
            raise InvalidParameterException("K-factor must be positive")

        self._initial_rating = initial_rating
        self._rating = initial_rating
        self._k_factor = k_factor if k_factor is not None else EloCompetitor._k_factor
        logger.debug(
            "Initialized EloCompetitor with initial rating %.1f, k_factor=%.1f", self._initial_rating, self._k_factor
        )

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

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_rating": self._initial_rating,
            "k_factor": self._k_factor if self._k_factor != self.__class__._k_factor else None,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "rating": self._rating,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_rating
        logger.debug("Importing parameters for EloCompetitor: %s", parameters)
        initial_rating = parameters.get("initial_rating", 400)
        if initial_rating < self._minimum_rating:
            logger.error("Invalid initial_rating in state: %.1f (minimum %.1f)", initial_rating, self._minimum_rating)
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self._initial_rating = initial_rating

        # Validate and set k_factor
        k_factor = parameters.get("k_factor", None)
        if k_factor is not None and k_factor <= 0:
            raise InvalidParameterException("K-factor must be positive")
        self._k_factor = k_factor if k_factor is not None else EloCompetitor._k_factor

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        # Validate and set rating
        logger.debug("Importing current state for EloCompetitor: %s", state)
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
            logger.error("Invalid rating in state: %.1f (minimum %.1f)", rating, self._minimum_rating)
            raise InvalidParameterException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = rating

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            EloCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_rating=parameters.get("initial_rating", 400),
            k_factor=parameters.get("k_factor", None),
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
            EloCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Validate and set initial_rating
        logger.debug("Creating EloCompetitor from state: %s", state)
        # Handle legacy state format
        if "type" not in state:
            logger.warning("Using legacy state format for EloCompetitor.from_state")
            # Configure class variables if provided
            if "class_vars" in state:
                logger.debug("Applying legacy class variables: %s", state["class_vars"])
                class_vars = state["class_vars"]
                if "base_rating" in class_vars:
                    cls._base_rating = class_vars["base_rating"]

            # Create a new competitor with the initial rating
            competitor = cls(initial_rating=state.get("initial_rating", 400), k_factor=state.get("k_factor", None))

            # Set the current rating if provided
            if "current_rating" in state:
                competitor._rating = state["current_rating"]

            return competitor

        # Use the new standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating to the initial rating.
        """
        logger.info("Resetting EloCompetitor to initial state (rating=%.1f)", self._initial_rating)
        self._rating = self._initial_rating

    @property
    def transformed_rating(self) -> float:
        """Get the transformed rating for this competitor.

        Returns:
            float: The transformed rating.
        """
        return float(10 ** (self.rating / 400))

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
        logger.debug("Setting rating for EloCompetitor to %.1f", value)
        if value < self._minimum_rating:
            logger.warning("Attempted to set rating %.1f below minimum %.1f", value, self._minimum_rating)
            raise InvalidRatingValueException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = value

    def expected_score(self, competitor: "BaseCompetitor") -> float:
        """Calculate the expected score against another competitor.

        Args:
            competitor (BaseCompetitor): The competitor to compare against.

        Returns:
            float: The expected score (probability of winning).
        """
        self.verify_competitor_types(competitor)
        return float(self.transformed_rating / (self.transformed_rating + competitor.transformed_rating))

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
        logger.debug("%s beat %s", self, competitor)
        # Revert to original logic
        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        self._rating = self._rating + self._k_factor * (1 - win_es)
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
        logger.debug("%s tied with %s", self, competitor)
        # Revert to original logic
        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        self._rating = self._rating + self._k_factor * (0.5 - win_es)
        competitor.rating = competitor.rating + self._k_factor * (0.5 - lose_es)
