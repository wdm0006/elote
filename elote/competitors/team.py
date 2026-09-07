from typing import Dict, Any, List, Optional, Sequence, Type, TypeVar

from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    InvalidStateException,
    MissMatchedCompetitorTypesException,
)
from elote.logging import logger

T = TypeVar("T", bound="TeamCompetitor")

_VALID_AGGREGATION_MODES = ("mean", "sum")


class TeamCompetitor(BaseCompetitor):
    """Composite competitor that rates a roster of members as a single side.

    A ``TeamCompetitor`` wraps a roster of :class:`~elote.competitors.base.BaseCompetitor`
    instances and exposes the usual competitor interface for the team as a whole. Bouts
    between two teams delegate to the members **positionally**: member ``i`` of one roster
    is paired against member ``i`` of the other, and each pair updates through the member
    rating system's own ``beat``/``tied`` math. The wrapper keeps no rating state of its
    own -- the team rating is derived from the members on every read, so rating history
    follows the member objects.

    Because both endpoints of such a bout are ``TeamCompetitor`` instances, team-vs-team
    play works in existing arenas (e.g. :class:`~elote.LambdaArena`) with no arena
    changes: seed the arena with the teams -- for example through ``initial_state`` with
    exported team state documents -- and run ``matchup`` as usual.

    Aggregation modes
    -----------------

    The ``aggregate`` flag selects how member ratings combine into the single number the
    :attr:`rating` property exposes -- the team's entry on any leaderboard:

    ``"mean"``
        Arithmetic mean of the member ratings. The team's rating lives on the same scale
        as the members' own rating system, so team entries and individual competitors can
        share one leaderboard and be compared directly. Adding or swapping a member moves
        the aggregate toward that member's rating; a weak member dilutes a strong roster.

    ``"sum"``
        Sum of the member ratings -- the roster's total rating mass. The value grows with
        roster size, so it is only meaningful between rosters of the same size: on a
        shared leaderboard a larger roster outranks a smaller one regardless of per-member
        strength. Choose ``sum`` when the combined mass is the quantity of interest, and
        compare sum-mode teams only against rosters of the same size.

    In both modes :meth:`expected_score` pairs members positionally and returns the mean
    of the members' own model probabilities, so predictions are valid probabilities either
    way; the flag governs the exposed rating aggregate only.

    A note on members: aggregation consumes the ``.rating`` scalar each member exposes.
    For Bayesian systems (TrueSkill, Whole-History Rating) that scalar is a derived
    ordinal, so an aggregate of ordinals is a convenience figure, not a fusion of the
    underlying belief distributions. Member-level team updates -- a single result
    distributed across a roster natively inside the rating model -- arrive with the
    OpenSkill N-way bout support.

    Serialization is nested: the exported parameters carry the roster as ``(type, init
    parameters)`` specifications, and the exported state carries each member's full
    exported state document. Membership identity is preserved by roster position and
    concrete type, so a restored team has the same members, in the same roles, with their
    rating histories intact. :meth:`reset` delegates to the members, restoring each to
    its own initial state.
    """

    def __init__(self, members: Sequence[BaseCompetitor], aggregate: str = "mean") -> None:
        """Initialize a TeamCompetitor from a roster of member competitors.

        Args:
            members (sequence of BaseCompetitor): The roster. Order defines the pairing
                used for bouts and the membership identity preserved by serialization;
                copy the roster rather than aliasing the caller's list.
            aggregate (str, optional): The aggregation mode for the exposed team rating.
                ``"mean"`` or ``"sum"``. Default: ``"mean"``.

        Raises:
            InvalidParameterException: If ``aggregate`` is not a supported mode, the
                roster is empty, or any roster entry is not a ``BaseCompetitor``.
        """
        super().__init__()  # Call base class constructor

        if aggregate not in _VALID_AGGREGATION_MODES:
            logger.error("Unsupported team aggregation mode specified: %s", aggregate)
            raise InvalidParameterException(
                f"Aggregation mode {aggregate} not supported; use one of {list(_VALID_AGGREGATION_MODES)}"
            )

        roster = list(members)
        if not roster:
            logger.error("TeamCompetitor requires at least one member competitor")
            raise InvalidParameterException("TeamCompetitor requires a non-empty roster of member competitors")

        for i, member in enumerate(roster):
            if not isinstance(member, BaseCompetitor):
                logger.error("Team member %d is not a BaseCompetitor: %r", i, member)
                raise InvalidParameterException(
                    f"Team member {i} must be a BaseCompetitor instance, got {type(member).__name__}"
                )

        self.members: List[BaseCompetitor] = roster
        self.aggregate = aggregate
        logger.debug("Initializing TeamCompetitor with mode '%s' and %d members", aggregate, len(roster))

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<TeamCompetitor: aggregate={self.aggregate}, members={len(self.members)}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<TeamCompetitor: aggregate={self.aggregate}>"

    @property
    def rating(self) -> float:
        """Get the aggregate rating of the team.

        The value is derived from the members on every read according to the
        ``aggregate`` mode: the mean of the member ratings, or their sum. See the class
        docstring for the semantics of each mode and its leaderboard implications.

        Returns:
            float: The aggregate team rating.
        """
        if self.aggregate == "mean":
            return sum(member.rating for member in self.members) / len(self.members)
        return float(sum(member.rating for member in self.members))

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the rating of this competitor.

        This method is not supported by TeamCompetitor, as the team rating is derived
        from the member competitors. Set the members' ratings instead.

        Args:
            value (float): The new rating value.

        Raises:
            NotImplementedError: Always, as setting the rating directly is not supported.
        """
        logger.warning("Attempted to set rating directly on TeamCompetitor, which is not supported.")
        raise NotImplementedError("Cannot directly set the rating of a TeamCompetitor; set the members' ratings")

    def _member_specs(self) -> List[Dict[str, Any]]:
        """Describe the roster as ``(type, init parameters)`` specifications."""
        return [{"type": type(member).__name__, "parameters": member._export_parameters()} for member in self.members]

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters, including the
                roster described as ``(type, init parameters)`` specifications.
        """
        return {
            "aggregate": self.aggregate,
            "members": self._member_specs(),
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables, including each
                member's full exported state document in roster order.
        """
        return {
            "members": [member.export_state() for member in self.members],
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        logger.debug("Importing parameters for TeamCompetitor: %s", parameters)
        aggregate = parameters.get("aggregate", "mean")
        if aggregate not in _VALID_AGGREGATION_MODES:
            logger.error("Invalid aggregate mode in state: %s", aggregate)
            raise InvalidParameterException(
                f"Aggregation mode {aggregate} not supported; use one of {list(_VALID_AGGREGATION_MODES)}"
            )
        self.aggregate = aggregate

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Rebuilds the roster from the nested member state documents, preserving member
        order (and therefore membership identity).

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidStateException: If any member state is invalid.
        """
        members_state = state.get("members", [])
        logger.debug("Importing current state for TeamCompetitor (%d members)", len(members_state))

        self.members = []
        for i, member_state in enumerate(members_state):
            member_type_name = member_state.get("type")
            if not member_type_name:
                logger.error("Missing type for team member %d in state during import.", i)
                raise InvalidStateException(f"Missing type for team member {i} in state")

            member_class = BaseCompetitor.get_competitor_class(member_type_name)
            try:
                self.members.append(member_class.from_state(member_state))
                logger.debug("Successfully imported state for team member %d (%s)", i, member_type_name)
            except Exception as e:
                logger.error("Failed to import state for team member %d ('%s'): %s", i, member_type_name, e)
                raise InvalidStateException(f"Failed to import state for team member {member_type_name}: {e}") from e

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters, including the roster
                described as ``(type, init parameters)`` specifications.

        Returns:
            TeamCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        logger.debug("Creating TeamCompetitor instance from parameters: %s", parameters)

        members: List[BaseCompetitor] = []
        for i, spec in enumerate(parameters.get("members", [])):
            member_type_name = spec.get("type")
            if not member_type_name:
                logger.error("Missing type for team member %d in parameters during creation.", i)
                raise InvalidParameterException(f"Missing type for team member {i} in parameters")

            member_class = BaseCompetitor.get_competitor_class(member_type_name)
            try:
                members.append(member_class._create_from_parameters(spec.get("parameters", {})))
            except Exception as e:
                logger.error("Failed to create team member %d ('%s'): %s", i, member_type_name, e)
                raise InvalidParameterException(f"Failed to initialize team member {member_type_name}: {e}") from e

        return cls(
            members=members,
            aggregate=parameters.get("aggregate", "mean"),
        )

    def verify_competitor_types(self, competitor: BaseCompetitor) -> None:
        """Verify that both competitors are teams with matching roster composition.

        Two teams can only bout when both are ``TeamCompetitor`` instances and their
        rosters have the same composition: the same member types in the same roster
        order, which also implies the same roster size. Member *states* may differ
        (that is what a bout updates); the composition may not.

        Args:
            competitor (BaseCompetitor): The competitor to verify.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor is not a
                ``TeamCompetitor``, or its roster composition differs.
        """
        super().verify_competitor_types(competitor)

        self_composition = [type(member).__name__ for member in self.members]
        competitor_composition = [type(member).__name__ for member in competitor.members]
        if self_composition != competitor_composition:
            logger.warning("Team roster composition mismatch: %s vs %s", self_composition, competitor_composition)
            raise MissMatchedCompetitorTypesException(
                f"TeamCompetitor rosters {self_composition} and {competitor_composition} cannot be co-mingled"
            )

    def expected_score(self, competitor: BaseCompetitor) -> float:
        """Calculate the expected score (probability of winning) against another team.

        Members are paired positionally and the result is the mean of the members' own
        model probabilities. This is independent of the ``aggregate`` mode: predictions
        are probabilities in both modes; the flag governs the exposed rating only.

        Args:
            competitor (BaseCompetitor): The opponent team to compare against.

        Returns:
            float: The probability of winning (between 0 and 1).

        Raises:
            MissMatchedCompetitorTypesException: If the roster compositions don't match.
        """
        self.verify_competitor_types(competitor)
        logger.debug("Calculating team expected score (mode='%s') between %s and %s", self.aggregate, self, competitor)

        expected_scores = [
            member.expected_score(opponent_member)
            for member, opponent_member in zip(self.members, competitor.members, strict=True)
        ]
        result = sum(expected_scores) / len(expected_scores)
        logger.debug("Team mean expected score: %.4f", result)
        return result

    def beat(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this team has won against the given team.

        Members are paired positionally: member ``i`` of this team beats member ``i`` of
        the opposing team through the members' own rating math.

        Args:
            competitor (BaseCompetitor): The opponent team that lost.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. Forwarded unchanged to every
                member pair.

        Raises:
            MissMatchedCompetitorTypesException: If the roster compositions don't match.
        """
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 1.0)
        logger.debug("%s beat %s. Updating %d member pairs.", self, competitor, len(self.members))

        for member, opponent_member in zip(self.members, competitor.members, strict=True):
            logger.debug("Updating member pair via beat: %s vs %s", member, opponent_member)
            member.beat(opponent_member, scores=validated)

    def tied(self, competitor: BaseCompetitor, *, scores: Optional[Sequence[float]] = None) -> None:
        """Update ratings after this team has tied with the given team.

        Members are paired positionally: member ``i`` of this team ties member ``i`` of
        the opposing team through the members' own rating math.

        Args:
            competitor (BaseCompetitor): The opponent team that tied.
            scores (sequence of float, optional): The two scores in caller order,
                ``(self_score, competitor_score)``. Must be equal. Forwarded unchanged to
                every member pair.

        Raises:
            MissMatchedCompetitorTypesException: If the roster compositions don't match.
        """
        self.verify_competitor_types(competitor)
        validated = self._validate_scores(scores, 0.5)
        logger.debug("%s tied with %s. Updating %d member pairs.", self, competitor, len(self.members))

        for member, opponent_member in zip(self.members, competitor.members, strict=True):
            logger.debug("Updating member pair via tied: %s vs %s", member, opponent_member)
            member.tied(opponent_member, scores=validated)

    def reset(self) -> None:
        """Reset this team to its initial state.

        Delegates to each member, restoring every member to its own initial rating. The
        roster composition itself never changes.
        """
        logger.info("Resetting TeamCompetitor members to their initial states.")
        for member in self.members:
            logger.debug("Resetting team member: %s", member)
            member.reset()
