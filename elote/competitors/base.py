import abc
from typing import Dict, Any, TypeVar, Type, ClassVar


class MissMatchedCompetitorTypesException(Exception):
    """Exception raised when attempting to compare or update competitors of different types.

    This exception is raised when operations are attempted between competitors
    that use different rating systems, which would lead to invalid results.
    """

    pass


class InvalidRatingValueException(Exception):
    """Exception raised when an invalid rating value is provided.

    This exception is raised when a rating value is outside the acceptable range
    for a particular rating system.
    """

    pass


class InvalidParameterException(Exception):
    """Exception raised when an invalid parameter is provided.

    This exception is raised when a parameter value is outside the acceptable range
    or of an incorrect type for a particular rating system.
    """

    pass


T = TypeVar("T", bound="BaseCompetitor")


class BaseCompetitor(abc.ABC):
    """Base abstract class for all rating system competitors.

    This class defines the interface that all rating system implementations must follow.
    Each competitor represents an entity with a rating that can be compared against
    other competitors of the same type.

    All rating system implementations should inherit from this class and implement
    the abstract methods. This ensures a consistent API across all rating systems.

    Class Attributes:
        _minimum_rating (float): The minimum allowed rating value. Default: 100.
                                This prevents ratings from becoming negative or
                                unreasonably low. Can be configured using the
                                configure_class method.

    Example:
        >>> from elote import EloCompetitor
        >>> player1 = EloCompetitor(initial_rating=1200)
        >>> player2 = EloCompetitor(initial_rating=1000)
        >>> player1.expected_score(player2)
        0.76
        >>> player1.beat(player2)  # Update ratings after player1 wins
        >>> player1.rating
        1205
        >>> player2.rating
        995
        >>> # Configure the minimum rating for all EloCompetitor instances
        >>> EloCompetitor.configure_class(minimum_rating=200)
    """

    # Default minimum rating to prevent negative or unreasonably low ratings
    _minimum_rating: ClassVar[float] = 100

    @property
    @abc.abstractmethod
    def rating(self) -> float:
        """Get the current rating value of this competitor.

        Returns:
            float: The current rating value.
        """
        pass

    @rating.setter
    @abc.abstractmethod
    def rating(self, value: float) -> None:
        """Set the current rating value of this competitor.

        Args:
            value (float): The new rating value.

        Raises:
            InvalidRatingValueException: If the rating value is invalid.
        """
        pass

    @abc.abstractmethod
    def expected_score(self, competitor: "BaseCompetitor") -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        pass

    @abc.abstractmethod
    def beat(self, competitor: "BaseCompetitor") -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on the match outcome where this competitor won.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        pass

    def lost_to(self, competitor: "BaseCompetitor") -> None:
        """Update ratings after this competitor has lost to the given competitor.

        This is a convenience method that calls beat() on the winning competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor.beat(self)

    @abc.abstractmethod
    def tied(self, competitor: "BaseCompetitor") -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on a drawn match outcome.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        pass

    @abc.abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            BaseCompetitor: A new competitor with the same state as the exported one.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating and any other state to the
        values it had when it was first created.
        """
        pass

    @classmethod
    def configure_class(cls, **kwargs) -> None:
        """Configure class-level parameters for this rating system.

        This method allows setting class-level parameters that affect all
        instances of this rating system.

        Args:
            **kwargs: Keyword arguments for class-level parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        for key, value in kwargs.items():
            if hasattr(cls, f"_{key}"):
                setattr(cls, f"_{key}", value)
            else:
                raise InvalidParameterException(f"Unknown class parameter: {key}")

    def configure(self, **kwargs) -> None:
        """Configure instance-level parameters for this competitor.

        This method allows setting instance-level parameters that affect only
        this competitor.

        Args:
            **kwargs: Keyword arguments for instance-level parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            else:
                raise InvalidParameterException(f"Unknown instance parameter: {key}")

    def verify_competitor_types(self, competitor: "BaseCompetitor") -> None:
        """Verify that the given competitor is of the same type as this one.

        This method ensures that operations between competitors are only performed
        when they use the same rating system.

        Args:
            competitor (BaseCompetitor): The competitor to verify.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        if not isinstance(competitor, self.__class__):
            raise MissMatchedCompetitorTypesException(
                f"Competitor types {type(competitor)} and {type(self)} cannot be co-mingled"
            )

    def __lt__(self, other: "BaseCompetitor") -> bool:
        """Compare if this competitor's rating is less than another's.

        Args:
            other (BaseCompetitor): The competitor to compare against.

        Returns:
            bool: True if this competitor's rating is less than the other's.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(other)
        return self.rating < other.rating

    def __gt__(self, other: "BaseCompetitor") -> bool:
        """Compare if this competitor's rating is greater than another's.

        Args:
            other (BaseCompetitor): The competitor to compare against.

        Returns:
            bool: True if this competitor's rating is greater than the other's.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(other)
        return self.rating > other.rating

    def __eq__(self, other: object) -> bool:
        """Compare if this competitor's rating is equal to another's.

        Args:
            other (object): The object to compare against.

        Returns:
            bool: True if other is a BaseCompetitor of the same type with the same rating.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        if not isinstance(other, BaseCompetitor):
            return NotImplemented
        try:
            self.verify_competitor_types(other)
            return self.rating == other.rating
        except MissMatchedCompetitorTypesException:
            return False

    def __le__(self, other: "BaseCompetitor") -> bool:
        """Compare if this competitor's rating is less than or equal to another's.

        Args:
            other (BaseCompetitor): The competitor to compare against.

        Returns:
            bool: True if this competitor's rating is less than or equal to the other's.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(other)
        return self.rating <= other.rating

    def __ge__(self, other: "BaseCompetitor") -> bool:
        """Compare if this competitor's rating is greater than or equal to another's.

        Args:
            other (BaseCompetitor): The competitor to compare against.

        Returns:
            bool: True if this competitor's rating is greater than or equal to the other's.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(other)
        return self.rating >= other.rating
