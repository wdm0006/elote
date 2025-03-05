from typing import Dict, Any, List, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidParameterException, InvalidStateException
from elote import (
    EloCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    ColleyMatrixCompetitor,
)

T = TypeVar("T", bound="BlendedCompetitor")

# Dictionary mapping competitor type names to their classes
competitor_types = {
    "EloCompetitor": EloCompetitor,
    "ECFCompetitor": ECFCompetitor,
    "DWZCompetitor": DWZCompetitor,
    "GlickoCompetitor": GlickoCompetitor,
    "Glicko2Competitor": Glicko2Competitor,
    "TrueSkillCompetitor": TrueSkillCompetitor,
    "ColleyMatrixCompetitor": ColleyMatrixCompetitor,
}


class BlendedCompetitor(BaseCompetitor):
    """Ensemble rating system that combines multiple rating algorithms.

    The BlendedCompetitor allows combining multiple rating systems to leverage
    their individual strengths while mitigating their weaknesses. By aggregating
    predictions from different rating algorithms, it can potentially provide more
    robust and accurate predictions than any single rating system alone.

    Supported blend modes:
        - "mean": Average the expected scores from all sub-competitors.
    """

    def __init__(self, competitors: List[Dict[str, Any]], blend_mode: str = "mean"):
        """Initialize a BlendedCompetitor with multiple rating systems.

        Args:
            competitors (List[Dict[str, Any]]): List of dictionaries specifying the
                                              sub-competitors to use. Each dictionary
                                              should have a "type" key with the name
                                              of the competitor class and a
                                              "competitor_kwargs" key with the
                                              arguments to pass to the constructor.
            blend_mode (str, optional): The method to use for blending the ratings.
                                      Currently only "mean" is supported. Default: "mean".

        Raises:
            InvalidParameterException: If the blend_mode is not supported or if any
                                      competitor specification is invalid.
        """
        if blend_mode not in ["mean"]:
            raise InvalidParameterException(f"Blend mode {blend_mode} not supported")

        self.sub_competitors: List[BaseCompetitor] = []
        self._initial_competitors = competitors.copy()  # Store for reset
        self.blend_mode = blend_mode

        # Create the sub-competitors
        for comp_spec in competitors:
            comp_type_name = comp_spec.get("type", "EloCompetitor")
            comp_kwargs = comp_spec.get("competitor_kwargs", {})

            # Get the competitor class
            comp_class = BaseCompetitor.get_competitor_class(comp_type_name)

            # Create the competitor
            self.sub_competitors.append(comp_class(**comp_kwargs))

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<BlendedCompetitor: mode={self.blend_mode}, sub_competitors={len(self.sub_competitors)}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<BlendedCompetitor: mode={self.blend_mode}>"

    @property
    def rating(self) -> float:
        """Get the combined rating of this competitor.

        For a BlendedCompetitor, the rating is the sum of all sub-competitor ratings.
        This is a simple way to represent the overall strength, but the expected_score
        method provides a more accurate way to compare competitors.

        Returns:
            float: The combined rating.
        """
        return sum([x.rating for x in self.sub_competitors])

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the rating of this competitor.

        This method is not directly supported by the BlendedCompetitor, as ratings
        are determined by the sub-competitors. This implementation raises an exception.

        Args:
            value (float): The new rating value.

        Raises:
            NotImplementedError: Always, as setting the rating directly is not supported.
        """
        raise NotImplementedError("Cannot directly set the rating of a BlendedCompetitor")

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "blend_mode": self.blend_mode,
            "competitors": self._initial_competitors,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "sub_competitors": [comp.export_state() for comp in self.sub_competitors],
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set blend_mode
        blend_mode = parameters.get("blend_mode", "mean")
        if blend_mode not in ["mean"]:
            raise InvalidParameterException(f"Blend mode {blend_mode} not supported")
        self.blend_mode = blend_mode

        # Store the initial competitors specification
        self._initial_competitors = parameters.get("competitors", [])

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidStateException: If any state variable is invalid.
        """
        # Get the sub-competitors state
        sub_competitors_state = state.get("sub_competitors", [])

        # Create new sub-competitors from their state
        self.sub_competitors = []
        for comp_state in sub_competitors_state:
            # Get the competitor type
            comp_type_name = comp_state.get("type")
            if not comp_type_name:
                raise InvalidStateException("Missing competitor type in sub-competitor state")

            # Create the competitor from its state
            comp_class = BaseCompetitor.get_competitor_class(comp_type_name)
            self.sub_competitors.append(comp_class.from_state(comp_state))

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            BlendedCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            competitors=parameters.get("competitors", []),
            blend_mode=parameters.get("blend_mode", "mean"),
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
            BlendedCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidStateException: If the state dictionary is invalid or incompatible.
            InvalidParameterException: If any competitor specification is invalid.
        """
        # Handle legacy state format
        if "type" not in state:
            blend_mode = state.get("blend_mode", "mean")
            competitors_state = state.get("competitors", [])

            # Create a new list of competitor specifications using from_state
            competitors = []
            for comp_state in competitors_state:
                comp_type_name = comp_state.get("type", "EloCompetitor")
                comp_type = competitor_types.get(comp_type_name)

                if comp_type is None:
                    raise InvalidParameterException(f"Unknown competitor type: {comp_type_name}")

                comp_kwargs = comp_state.get("competitor_kwargs", {})

                # Create a new competitor specification with just the initial parameters
                competitors.append(
                    {
                        "type": comp_type_name,
                        "competitor_kwargs": {"initial_rating": comp_kwargs.get("initial_rating", 400)},
                    }
                )

            # Create the blended competitor
            blended = cls(competitors=competitors, blend_mode=blend_mode)

            # Now update each sub-competitor with its full state
            for i, comp_state in enumerate(competitors_state):
                comp_type_name = comp_state.get("type", "EloCompetitor")
                comp_type = competitor_types.get(comp_type_name)
                comp_kwargs = comp_state.get("competitor_kwargs", {})

                # Replace the sub-competitor with one created from the full state
                blended.sub_competitors[i] = comp_type.from_state(comp_kwargs)

            return blended

        # Use the new standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets all sub-competitors to their initial states.
        """
        # Reset all sub-competitors
        for competitor in self.sub_competitors:
            competitor.reset()

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another competitor.

        For a BlendedCompetitor, the expected score is calculated by blending the
        expected scores from all sub-competitors according to the blend_mode.

        Args:
            competitor (BaseCompetitor): The opponent competitor to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            NotImplementedError: If the blend_mode is not supported.
        """
        self.verify_competitor_types(competitor)

        if self.blend_mode == "mean":
            es = []
            for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
                es.append(c.expected_score(other_c))
            return sum(es) / len(es)
        else:
            raise NotImplementedError(f"Blend mode {self.blend_mode} not supported")

    def beat(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method updates the ratings of all sub-competitors based on the match outcome.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)

        for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
            c.beat(other_c)

    def tied(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method updates the ratings of all sub-competitors based on the drawn match outcome.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)

        for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
            c.tied(other_c)  # Fixed: was using beat() instead of tied()
