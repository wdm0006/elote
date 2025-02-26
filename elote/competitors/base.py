import abc


class MissMatchedCompetitorTypesException(Exception):
    """Exception raised when attempting to compare or update competitors of different types.

    This exception is raised when operations are attempted between competitors
    that use different rating systems, which would lead to invalid results.
    """

    pass


class BaseCompetitor:
    """Base abstract class for all rating system competitors.

    This class defines the interface that all rating system implementations must follow.
    Each competitor represents an entity with a rating that can be compared against
    other competitors of the same type.
    """

    @property
    @abc.abstractmethod
    def rating(self):
        """Get the current rating value of this competitor.

        Returns:
            float: The current rating value.
        """
        pass

    @abc.abstractmethod
    def expected_score(self, competitor):
        """Calculate the expected score (probability of winning) against another competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).
        """
        pass

    @abc.abstractmethod
    def beat(self, competitor):
        """Update ratings after this competitor has won against the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on the match outcome where this competitor won.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
        """
        pass

    def lost_to(self, competitor):
        """Update ratings after this competitor has lost to the given competitor.

        This is a convenience method that calls beat() on the winning competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that won.
        """
        competitor.beat(self)

    @abc.abstractmethod
    def tied(self, competitor):
        """Update ratings after this competitor has tied with the given competitor.

        This method updates the ratings of both this competitor and the opponent
        based on a drawn match outcome.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.
        """
        pass

    @abc.abstractmethod
    def export_state(self):
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        pass

    def verify_competitor_types(self, competitor):
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
                "Competitor types %s and %s cannot be co-mingled"
                % (
                    type(competitor),
                    type(self),
                )
            )
