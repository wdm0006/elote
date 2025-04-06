import abc
import json
import uuid
from typing import Dict, Any, TypeVar, Type, ClassVar, List, cast

from elote.logging import logger


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


class InvalidStateException(Exception):
    """Exception raised when an invalid state is provided for deserialization.

    This exception is raised when a state dictionary is missing required fields
    or contains invalid values.
    """

    pass


T = TypeVar("T", bound="BaseCompetitor")


# Registry to store competitor types
_competitor_registry: Dict[str, Type["BaseCompetitor"]] = {}


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
                                unreasonably low.
    """

    _minimum_rating: ClassVar[float] = 100

    @abc.abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize base competitor state."""
        # Placeholder for potential future base initialization
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses in the competitor registry.

        This method is called automatically when a subclass is created.
        It registers the subclass in the competitor registry for later retrieval.
        """
        super().__init_subclass__(**kwargs)
        _competitor_registry[cls.__name__] = cls
        logger.debug("Registered competitor class: %s", cls.__name__)

    @classmethod
    def get_competitor_class(cls, class_name: str) -> Type["BaseCompetitor"]:
        """Get a competitor class by name.

        Args:
            class_name (str): The name of the competitor class.

        Returns:
            Type[BaseCompetitor]: The competitor class.

        Raises:
            InvalidParameterException: If the class name is not registered.
        """
        if class_name not in _competitor_registry:
            logger.error("Attempted to get unknown competitor type: %s", class_name)
            raise InvalidParameterException(f"Unknown competitor type: {class_name}")
        logger.debug("Retrieved competitor class: %s", class_name)
        return _competitor_registry[class_name]

    @classmethod
    def list_competitor_types(cls) -> List[str]:
        """List all registered competitor types.

        Returns:
            List[str]: A list of registered competitor type names.
        """
        logger.debug("Listing registered competitor types: %s", list(_competitor_registry.keys()))
        return list(_competitor_registry.keys())

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
        logger.debug("Competitor %s lost to %s. Calling beat() on winner.", self, competitor)
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

    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        This method exports the competitor's state in a standardized format that can be
        used to recreate the competitor with the same state. The format includes:
        - type: The class name of the competitor
        - version: The version of the serialization format
        - created_at: The timestamp when the state was exported
        - id: A unique identifier for this state export
        - parameters: The parameters used to initialize the competitor
        - state: The current state variables of the competitor

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        import time

        logger.debug("Exporting state for competitor %s", self)

        # Get parameters and current state
        parameters = self._export_parameters()
        current_state = self._export_current_state()

        # Create a class_vars dictionary for backward compatibility
        class_vars = {}
        for attr in dir(self.__class__):
            # Skip private attributes, methods, and special attributes
            if attr.startswith("__") or callable(getattr(self.__class__, attr)) or attr.startswith("_abc_"):
                continue

            # Get the attribute value
            value = getattr(self.__class__, attr)

            # Skip non-JSON serializable values
            try:
                json.dumps(value)
                # For class variables with leading underscore, add them without the underscore
                if attr.startswith("_"):
                    class_vars[attr[1:]] = value
                else:
                    class_vars[attr] = value
            except (TypeError, OverflowError):
                pass

        # Create the standardized format
        state_dict = {
            "type": self.__class__.__name__,
            "version": 1,
            "created_at": int(time.time()),
            "id": str(uuid.uuid4()),
            "parameters": parameters,
            "state": current_state,
            "class_vars": class_vars,
        }

        # For backward compatibility, flatten parameters and state into the top-level dictionary
        for key, value in parameters.items():
            state_dict[key] = value
        for key, value in current_state.items():
            state_dict[key] = value

        logger.debug("Exported state: %s", state_dict)
        return state_dict

    @abc.abstractmethod
    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        This method should be implemented by subclasses to export the parameters
        that were used to initialize the competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        pass

    @abc.abstractmethod
    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        This method should be implemented by subclasses to export the current
        state variables of the competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        pass

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import the competitor's state from a dictionary.

        This method updates the competitor's state based on the provided dictionary,
        including parameters and current state variables.

        Args:
            state (dict): A dictionary containing the state of a competitor.

        Raises:
            InvalidStateException: If the state dictionary is invalid or incompatible.
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Validate the state dictionary
        self._validate_state_dict(state)
        logger.debug("Importing state into competitor %s", self)

        # Check that the competitor type matches
        if state["type"] != self.__class__.__name__:
            logger.error(
                "State import failed: Mismatched competitor types. Expected %s, got %s",
                self.__class__.__name__,
                state["type"],
            )
            raise InvalidStateException(
                f"Mismatched competitor types: expected {self.__class__.__name__}, got {state['type']}"
            )

        # Import the state
        logger.debug("Importing parameters: %s", state["parameters"])
        self._import_parameters(state["parameters"])
        logger.debug("Importing current state: %s", state["state"])
        self._import_current_state(state["state"])
        logger.info("Successfully imported state for competitor %s", self)

    def _validate_state_dict(self, state: Dict[str, Any]) -> None:
        """Validate the structure and content of a state dictionary.

        Args:
            state (dict): A state dictionary to validate.

        Raises:
            InvalidStateException: If the state dictionary is invalid.
        """
        required_fields = ["type", "version", "parameters", "state"]
        logger.debug("Validating state dictionary: %s", state)
        for field in required_fields:
            if field not in state:
                logger.error("State validation failed: Missing required field '%s'", field)
                raise InvalidStateException(f"Missing required field: {field}")

        if not isinstance(state["type"], str):
            logger.error("State validation failed: Field 'type' is not a string (%s)", type(state["type"]))
            raise InvalidStateException("Field 'type' must be a string")

        if not isinstance(state["version"], int):
            logger.error("State validation failed: Field 'version' is not an integer (%s)", type(state["version"]))
            raise InvalidStateException("Field 'version' must be an integer")

        if not isinstance(state["parameters"], dict):
            logger.error(
                "State validation failed: Field 'parameters' is not a dictionary (%s)", type(state["parameters"])
            )
            raise InvalidStateException("Field 'parameters' must be a dictionary")

        if not isinstance(state["state"], dict):
            logger.error("State validation failed: Field 'state' is not a dictionary (%s)", type(state["state"]))
            raise InvalidStateException("Field 'state' must be a dictionary")
        logger.debug("State dictionary validation passed.")

    @abc.abstractmethod
    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        This method should be implemented by subclasses to import parameters
        from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        pass

    @abc.abstractmethod
    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import the current state variables from a dictionary.

        This method should be implemented by subclasses to import current state
        variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidStateException: If any state variable is invalid.
        """
        pass

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a state dictionary.

        Args:
            state: A dictionary containing the state of a competitor, including its type and parameters.

        Returns:
            A new competitor of the same type as the exported one.

        Raises:
            InvalidStateException: If the state format is invalid or missing required fields.
        """
        # Validate required fields
        required_fields = ["type", "parameters", "state"]
        for field in required_fields:
            if field not in state:
                msg = f"State must contain a '{field}' field"
                logger.error(msg)
                raise InvalidStateException(msg)

        competitor_class = cls.get_competitor_class(state["type"])
        if competitor_class is None:
            msg = f"Unknown competitor type: {state['type']}"
            logger.error(msg)
            raise InvalidStateException(msg)

        logger.debug(f"Creating competitor of type {state['type']} from state")
        instance = competitor_class._create_from_parameters(state["parameters"])
        instance._import_current_state(state["state"])
        return cast(T, instance)

    @classmethod
    @abc.abstractmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        This method should be implemented by subclasses to create a new instance
        from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            BaseCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        pass

    def to_json(self) -> str:
        """Convert this competitor's state to a JSON string.

        Returns:
            str: A JSON string representing this competitor's state.
        """

        # Create a custom JSON encoder to handle non-serializable objects
        class CompetitorEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                # Handle types that aren't JSON serializable
                # For example, convert datetime objects to ISO format string
                try:
                    # Try to convert to a simple type
                    if hasattr(obj, "__dict__"):
                        return obj.__dict__
                    return str(obj)
                except Exception:
                    return str(obj)

        return json.dumps(self.export_state(), cls=CompetitorEncoder)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create a new competitor from a JSON string.

        Args:
            json_str (str): A JSON string representing a competitor's state.

        Returns:
            BaseCompetitor: A new competitor with the state from the JSON string.

        Raises:
            InvalidStateException: If the JSON string is invalid or incompatible.
            InvalidParameterException: If any parameter in the state is invalid.
        """
        try:
            state = json.loads(json_str)
            logger.debug("Successfully parsed JSON string to state dictionary.")
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON string: %s", e)
            raise InvalidStateException(f"Invalid JSON: {e}") from e

        return cls.from_state(state)

    def verify_competitor_types(self, competitor: "BaseCompetitor") -> None:
        """Verify that the competitor types match.

        Args:
            competitor (BaseCompetitor): The competitor to verify.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        if not isinstance(competitor, self.__class__):
            logger.warning("Type mismatch detected: %s vs %s", type(self), type(competitor))
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

    @classmethod
    def configure_class(cls, **kwargs: Any) -> None:
        """Configure class-level parameters for this rating system.

        This method allows setting class-level parameters that affect all
        instances of this rating system.

        Args:
            **kwargs: Keyword arguments for class-level parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        logger.debug("Configuring class %s with parameters: %s", cls.__name__, kwargs)
        for key, value in kwargs.items():
            if hasattr(cls, f"_{key}"):
                setattr(cls, f"_{key}", value)
                logger.debug("Set class parameter '_%s' to %s", key, value)
            else:
                logger.error("Attempted to configure unknown class parameter '_%s' for %s", key, cls.__name__)
                raise InvalidParameterException(f"Unknown class parameter: {key}")

    def configure(self, **kwargs: Any) -> None:
        """Configure instance-level parameters for this competitor.

        This method allows setting instance-level parameters that affect only
        this competitor.

        Args:
            **kwargs: Keyword arguments for instance-level parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        logger.debug("Configuring instance %s with parameters: %s", self, kwargs)
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
                logger.debug("Set instance parameter '_%s' to %s", key, value)
            else:
                logger.error("Attempted to configure unknown instance parameter '_%s' for %s", key, self)
                raise InvalidParameterException(f"Unknown instance parameter: {key}")

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the competitor's state to its initial configuration.

        This method should revert any changes made through updates (beat, lost_to, tied)
        and re-import the initial parameters and state.
        Note: Actual implementation might be needed depending on how initial state is stored.
        """
        # TODO: Implement actual reset logic if needed, possibly by restoring from initial state
        logger.info("Resetting competitor %s to initial state.", self)
        pass
