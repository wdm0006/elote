import math
from typing import Dict, Any, ClassVar, Type, TypeVar, Optional, List, Tuple, cast
from datetime import datetime

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException
from elote.logging import logger  # Import directly from the logging submodule

T = TypeVar("T", bound="Glicko2Competitor")


class Glicko2Competitor(BaseCompetitor):
    """Glicko-2 rating system competitor.

    The Glicko-2 rating system is an improvement on the original Glicko system,
    developed by Mark Glickman. It introduces a volatility parameter that measures
    the degree of expected fluctuation in a player's rating.

    In Glicko-2, ratings are internally represented in a different scale than
    displayed to users. The internal scale uses a mean of 0 and a standard deviation
    of 1, while the displayed scale uses a mean of 1500 and a standard deviation of 173.7.

    Class Attributes:
        _tau (float): System constant that constrains the volatility over time. Default: 0.5.
                     Smaller values (e.g., 0.3 to 0.2) make volatility change more slowly.
                     Larger values (e.g., 0.6 to 1.0) allow volatility to change more quickly.
        _epsilon (float): Convergence tolerance for the volatility iteration. Default: 0.000001.
        _default_volatility (float): Default volatility for new competitors. Default: 0.06.
        _scale_factor (float): Scale factor for converting between Glicko-2 and original scales. Default: 173.7178.
        _rating_period_days (float): Number of days that constitute one rating period. Default: 1.0.
    """

    _tau: ClassVar[float] = 0.5
    _epsilon: ClassVar[float] = 0.000001
    _default_volatility: ClassVar[float] = 0.06
    _scale_factor: ClassVar[float] = 173.7178
    _rating_period_days: ClassVar[float] = 1.0

    def __init__(
        self,
        initial_rating: float = 1500,
        initial_rd: float = 350,
        initial_volatility: Optional[float] = None,
        initial_time: Optional[datetime] = None,
    ):
        """Initialize a Glicko-2 competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 1500.
            initial_rd (float, optional): The initial rating deviation of this competitor. Default: 350.
            initial_volatility (float, optional): The initial volatility of this competitor. Default: _default_volatility.
            initial_time (datetime, optional): The initial timestamp for this competitor. Default: current time.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
            InvalidParameterException: If the initial RD is not positive or if the initial volatility is not positive.
        """
        super().__init__()  # Call base class constructor
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")

        if initial_volatility is None:
            initial_volatility = self._default_volatility  # type: ignore[unreachable]
        elif initial_volatility <= 0:
            raise InvalidParameterException("Initial volatility must be positive")

        self._initial_rating = initial_rating
        self._initial_rd = initial_rd
        self._initial_volatility = initial_volatility

        # Store the Glicko-2 scale values (internal representation)
        self._mu = self._rating_to_mu(initial_rating)
        self._phi = self._rd_to_phi(initial_rd)
        self._sigma = initial_volatility

        # Store the original scale values (for display)
        self._rating = initial_rating
        self._rd = initial_rd

        # Store match results for rating period and track last activity
        self._match_results: List[Tuple["Glicko2Competitor", float, datetime]] = []
        self._last_activity = initial_time if initial_time is not None else datetime.now()

        logger.debug(
            "Initialized Glicko2Competitor: rating=%.1f, rd=%.1f, volatility=%.3f, mu=%.3f, phi=%.3f, time=%s",
            self._rating,
            self._rd,
            self._sigma,
            self._mu,
            self._phi,
            self._last_activity.isoformat(),
        )

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<Glicko2Competitor: rating={self._rating}, rd={self._rd}, volatility={self._sigma}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<Glicko2Competitor: rating={self._rating}, rd={self._rd}, volatility={self._sigma}>"

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_rating": self._initial_rating,
            "initial_rd": self._initial_rd,
            "initial_volatility": self._initial_volatility,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "rating": self._rating,
            "rd": self._rd,
            "volatility": self._sigma,
            "mu": self._mu,
            "phi": self._phi,
            "last_activity": self._last_activity.isoformat(),
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_rating
        logger.debug("Importing parameters for Glicko2Competitor: %s", parameters)
        initial_rating = parameters.get("initial_rating", 1500)
        if initial_rating < self._minimum_rating:
            logger.error("Invalid initial_rating in state: %.1f (minimum %.1f)", initial_rating, self._minimum_rating)
            raise InvalidParameterException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )
        self._initial_rating = initial_rating

        # Validate and set initial_rd
        initial_rd = parameters.get("initial_rd", 350)
        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")
        self._initial_rd = initial_rd

        # Validate and set initial_volatility
        initial_volatility = parameters.get("initial_volatility", self._default_volatility)
        if initial_volatility <= 0:
            raise InvalidParameterException("Initial volatility must be positive")
        self._initial_volatility = initial_volatility

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        # Validate and set rating
        logger.debug("Importing current state for Glicko2Competitor: %s", state)
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
            logger.error("Invalid rating in state: %.1f (minimum %.1f)", rating, self._minimum_rating)
            raise InvalidParameterException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = rating

        # Validate and set rd
        rd = state.get("rd", self._initial_rd)
        if rd <= 0:
            raise InvalidParameterException("RD must be positive")
        self._rd = rd

        # Validate and set volatility
        volatility = state.get("volatility", self._initial_volatility)
        if volatility <= 0:
            raise InvalidParameterException("Volatility must be positive")
        self._sigma = volatility

        # Set internal Glicko-2 scale values
        self._mu = state.get("mu", self._rating_to_mu(rating))
        self._phi = state.get("phi", self._rd_to_phi(rd))

        # Set last activity time
        if "last_activity" in state:
            self._last_activity = datetime.fromisoformat(state["last_activity"])
        else:
            self._last_activity = datetime.now()
            logger.warning(
                "Last activity time missing from state, using current time: %s", self._last_activity.isoformat()
            )

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            Glicko2Competitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_rating=parameters.get("initial_rating", 1500),
            initial_rd=parameters.get("initial_rd", 350),
            initial_volatility=parameters.get("initial_volatility", cls._default_volatility),
        )

    def export_state(self) -> Dict[str, Any]:
        """Export the current state of this competitor for serialization.

        Returns:
            dict: A dictionary containing all necessary information to recreate
                 this competitor's current state.
        """
        # Use the standardized format
        return super().export_state()

    @classmethod
    def from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Create a new competitor from a previously exported state.

        Args:
            state (dict): A dictionary containing the state of a competitor,
                         as returned by export_state().

        Returns:
            Glicko2Competitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        logger.debug("Creating Glicko2Competitor from state: %s", state)
        # Handle legacy state format
        if "type" not in state:
            logger.warning("Using legacy state format for Glicko2Competitor.from_state")
            # Configure class variables if provided
            if "class_vars" in state:
                logger.debug("Applying legacy class variables: %s", state["class_vars"])
                class_vars = state["class_vars"]
                if "tau" in class_vars:
                    cls._tau = class_vars["tau"]
                if "epsilon" in class_vars:
                    cls._epsilon = class_vars["epsilon"]
                if "default_volatility" in class_vars:
                    cls._default_volatility = class_vars["default_volatility"]
                if "scale_factor" in class_vars:
                    cls._scale_factor = class_vars["scale_factor"]

            # Create a new competitor with the initial parameters
            competitor = cls(
                initial_rating=state.get("initial_rating", 1500),
                initial_rd=state.get("initial_rd", 350),
                initial_volatility=state.get("initial_volatility", cls._default_volatility),
            )

            # Set the current state if provided
            if "current_rating" in state:
                competitor._rating = state["current_rating"]
            if "current_rd" in state:
                competitor._rd = state["current_rd"]
            if "current_volatility" in state:
                competitor._sigma = state["current_volatility"]
            if "current_mu" in state:
                competitor._mu = state["current_mu"]
            if "current_phi" in state:
                competitor._phi = state["current_phi"]

            return competitor

        # Use the standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's rating, RD, and volatility to their initial values.
        """
        logger.info(
            "Resetting Glicko2Competitor to initial state (rating=%.1f, rd=%.1f, sigma=%.3f)",
            self._initial_rating,
            self._initial_rd,
            self._initial_volatility,
        )
        self._rating = self._initial_rating
        self._rd = self._initial_rd
        self._sigma = self._initial_volatility
        self._mu = self._rating_to_mu(self._initial_rating)
        self._phi = self._rd_to_phi(self._initial_rd)
        self._match_results = []
        self._last_activity = datetime.now()

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
        logger.debug("Setting rating for Glicko2Competitor to %.1f", value)
        if value < self._minimum_rating:
            logger.warning("Attempted to set rating %.1f below minimum %.1f", value, self._minimum_rating)
            raise InvalidRatingValueException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
        self._rating = value
        self._mu = self._rating_to_mu(value)

    @property
    def rd(self) -> float:
        """Get the current rating deviation of this competitor.

        Returns:
            float: The current rating deviation.
        """
        return self._rd

    @rd.setter
    def rd(self, value: float) -> None:
        """Set the current rating deviation of this competitor.

        Args:
            value (float): The new rating deviation value.

        Raises:
            InvalidParameterException: If the rating deviation is not positive.
        """
        logger.debug("Setting RD for Glicko2Competitor to %.1f", value)
        if value <= 0:
            logger.error("Attempted to set non-positive RD: %.1f", value)
            raise InvalidParameterException("RD must be positive")
        self._rd = value
        self._phi = self._rd_to_phi(value)

    @property
    def volatility(self) -> float:
        """Get the current volatility of this competitor.

        Returns:
            float: The current volatility.
        """
        return self._sigma

    @volatility.setter
    def volatility(self, value: float) -> None:
        """Set the current volatility of this competitor.

        Args:
            value (float): The new volatility value.

        Raises:
            InvalidParameterException: If the volatility is not positive.
        """
        logger.debug("Setting volatility for Glicko2Competitor to %.3f", value)
        if value <= 0:
            logger.error("Attempted to set non-positive volatility: %.3f", value)
            raise InvalidParameterException("Volatility must be positive")
        self._sigma = value

    def _rating_to_mu(self, rating: float) -> float:
        """Convert a rating from the original scale to the Glicko-2 scale.

        Args:
            rating (float): The rating on the original scale.

        Returns:
            float: The rating on the Glicko-2 scale.
        """
        return (rating - 1500) / self._scale_factor

    def _mu_to_rating(self, mu: float) -> float:
        """Convert a rating from the Glicko-2 scale to the original scale.

        Args:
            mu (float): The rating on the Glicko-2 scale.

        Returns:
            float: The rating on the original scale.
        """
        return mu * self._scale_factor + 1500

    def _rd_to_phi(self, rd: float) -> float:
        """Convert a rating deviation from the original scale to the Glicko-2 scale.

        Args:
            rd (float): The rating deviation on the original scale.

        Returns:
            float: The rating deviation on the Glicko-2 scale.
        """
        return rd / self._scale_factor

    def _phi_to_rd(self, phi: float) -> float:
        """Convert a rating deviation from the Glicko-2 scale to the original scale.

        Args:
            phi (float): The rating deviation on the Glicko-2 scale.

        Returns:
            float: The rating deviation on the original scale.
        """
        return phi * self._scale_factor

    def _g(self, phi: float) -> float:
        """Calculate the g-function used in the Glicko-2 rating system.

        Args:
            phi (float): The rating deviation on the Glicko-2 scale.

        Returns:
            float: The g-function value.
        """
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Calculate the expected score function.

        Args:
            mu (float): The rating of this competitor on the Glicko-2 scale.
            mu_j (float): The rating of the opponent on the Glicko-2 scale.
            phi_j (float): The rating deviation of the opponent on the Glicko-2 scale.

        Returns:
            float: The expected score (probability of winning).
        """
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

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
        competitor_glicko2 = cast(Glicko2Competitor, competitor)
        logger.debug("Calculating expected score between %s and %s", self, competitor_glicko2)
        self._phi_j = self._rd_to_phi(competitor.rd)
        return self._E(self._mu, competitor_glicko2._mu, self._phi_j)

    def update_rd_for_inactivity(self, current_time: Optional[datetime] = None) -> None:
        """Update the rating deviation based on time elapsed since last activity.

        This implements Glickman's formula for increasing uncertainty in ratings
        over time when a player is inactive. In Glicko-2, the phi (internal RD)
        increases based on the current volatility (sigma) parameter and the number
        of rating periods that have passed.

        Args:
            current_time (datetime, optional): The current time to calculate inactivity against.
                If None, uses the current system time.
        """
        if current_time is None:
            current_time = datetime.now()  # type: ignore[unreachable]

        # Calculate number of rating periods (can be fractional)
        days_inactive = (current_time - self._last_activity).total_seconds() / (24 * 3600)
        rating_periods = days_inactive / self._rating_period_days

        if rating_periods > 0:
            # In Glicko-2, phi increases based on the current volatility (sigma)
            # over the inactive period
            self._phi = math.sqrt(self._phi**2 + rating_periods * self._sigma**2)
            # Convert back to original scale RD
            old_rd = self._rd
            self._rd = self._phi_to_rd(self._phi)
            # Cap RD at 350 in the original scale
            if self._rd > 350:
                logger.debug("RD capped at 350 (was %.1f)", self._rd)
                self._rd = 350
                self._phi = self._rd_to_phi(350)
            logger.debug(
                "Updated RD for %s due to inactivity (%.1f periods): %.1f -> %.1f",
                self,
                rating_periods,
                old_rd,
                self._rd,
            )

    def update_ratings(self) -> None:
        """Update ratings based on recorded match results.

        This method implements the Glicko-2 rating system update algorithm.
        It processes all recorded match results and updates the rating, RD, and volatility.
        If no matches were played, it updates the RD based on inactivity.
        After updating, the match results are cleared.
        """
        if not self._match_results:
            logger.debug("No matches for %s, updating RD for inactivity.", self)
            # Update RD for inactivity up to current time
            self.update_rd_for_inactivity()
            return

        # Get the most recent match time
        current_time = max(match_time for _, _, match_time in self._match_results)

        # Update RD for inactivity before processing matches
        logger.debug("Updating RD for %s before processing %d matches.", self, len(self._match_results))
        self.update_rd_for_inactivity(current_time)

        if not self._match_results:
            return

        # Step 1: Initialize values
        mu = self._mu
        phi = self._phi
        sigma = self._sigma
        logger.debug("Starting rating update for %s: mu=%.3f, phi=%.3f, sigma=%.3f", self, mu, phi, sigma)

        # Step 2: For each opponent j, compute g(phi_j) and E(mu, mu_j, phi_j)
        v_inv = 0.0
        delta_sum = 0.0
        for opponent, score, _ in self._match_results:
            self.verify_competitor_types(opponent)
            opponent_glicko2 = cast(Glicko2Competitor, opponent)
            phi_j = self._rd_to_phi(opponent.rd)
            g_j = self._g(phi_j)
            E_j = self._E(mu, opponent_glicko2._mu, phi_j)
            v_inv += g_j**2 * E_j * (1 - E_j)
            delta_sum += g_j * (score - E_j)

        # Step 3: Compute v and delta
        v = 1 / v_inv
        delta = v * delta_sum
        logger.debug("Calculated v=%.3e and delta=%.3f", v, delta)

        # Step 4: Determine new volatility sigma'
        def f(x: float) -> float:
            """Function to find the root of to determine new volatility."""
            e_x = math.exp(x)
            a = e_x * (delta**2 - phi**2 - v - e_x)
            b = 2 * ((phi**2 + v + e_x) ** 2)
            c = x - math.log(sigma**2)
            return a / b - c / self._tau**2

        # Find new volatility using iteration
        A = math.log(sigma**2)
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(A - k * self._tau) < 0:
                k += 1
            B = A - k * self._tau

        fa = f(A)
        fb = f(B)

        iter_count = 0
        # Iterate until convergence
        while abs(B - A) > self._epsilon:
            iter_count += 1
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb < 0:
                A = B
                fa = fb
            else:
                fa = fa / 2
            B = C
            fb = fc

        sigma_prime = math.exp(A / 2)

        # Step 5: Update phi* and phi
        logger.debug("Volatility iteration converged after %d steps. New sigma_prime=%.5f", iter_count, sigma_prime)
        phi_star = math.sqrt(phi**2 + sigma_prime**2)
        phi_prime = 1 / math.sqrt(1 / phi_star**2 + 1 / v)

        # Step 6: Update mu
        mu_prime = mu + phi_prime**2 * delta_sum

        # Step 7: Convert back to original scale
        self._rating = self._mu_to_rating(mu_prime)
        self._rd = self._phi_to_rd(phi_prime)
        self._sigma = sigma_prime

        # Ensure rating doesn't go below minimum
        old_rating = self._rating
        self._rating = max(self._minimum_rating, self._rating)
        if self._rating != old_rating:
            logger.warning("Rating for %s adjusted to minimum %.1f (was %.1f)", self, self._minimum_rating, old_rating)

        # Cap RD at 350
        old_rd = self._rd
        if self._rd > 350:
            self._rd = 350
            self._phi = self._rd_to_phi(350)
            logger.debug("RD for %s capped at 350 (was %.1f)", self, old_rd)

        # Update internal values
        self._mu = mu_prime
        self._phi = phi_prime

        # Update last activity time and clear match results
        self._last_activity = current_time
        self._match_results = []
        logger.info(
            "Rating update complete for %s: new_rating=%.1f, new_rd=%.1f, new_sigma=%.5f",
            self,
            self._rating,
            self._rd,
            self._sigma,
        )

    def beat(self, competitor: BaseCompetitor, match_time: Optional[datetime] = None) -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method records the match result for later processing during the rating period update.
        The actual rating update happens when update_ratings() is called.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.
            match_time (datetime, optional): The time when the match occurred. Default: current time.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            InvalidParameterException: If the match time is before either competitor's last activity.
        """
        self.verify_competitor_types(competitor)
        logger.debug("%s beat %s (time=%s). Recording result.", self, competitor, match_time)
        competitor_glicko2 = cast(Glicko2Competitor, competitor)

        # Get the match time
        current_time = match_time if match_time is not None else datetime.now()

        # Validate match time is not before last activity
        if current_time < self._last_activity:
            logger.error("Match time %s is before self last activity %s", current_time, self._last_activity)
            raise InvalidParameterException("Match time cannot be before competitor's last activity time")
        if current_time < competitor_glicko2._last_activity:
            logger.error(
                "Match time %s is before opponent last activity %s", current_time, competitor_glicko2._last_activity
            )
            raise InvalidParameterException("Match time cannot be before opponent's last activity time")

        # Update RDs for both competitors based on inactivity before recording match
        self.update_rd_for_inactivity(current_time)
        competitor_glicko2.update_rd_for_inactivity(current_time)

        # Get the match time
        current_time = match_time if match_time is not None else datetime.now()

        # Validate match time is not before last activity
        if current_time < self._last_activity:
            logger.error("Match time %s is before self last activity %s", current_time, self._last_activity)
            raise InvalidParameterException("Match time cannot be before competitor's last activity time")
        if current_time < competitor_glicko2._last_activity:
            logger.error(
                "Match time %s is before opponent last activity %s", current_time, competitor_glicko2._last_activity
            )
            raise InvalidParameterException("Match time cannot be before opponent's last activity time")

        # Update RDs for both competitors based on inactivity before recording match
        self.update_rd_for_inactivity(current_time)
        competitor_glicko2.update_rd_for_inactivity(current_time)

        # Record the match result for this competitor
        self._match_results.append((competitor_glicko2, 1.0, current_time))

        # Record the match result for the opponent
        competitor_glicko2._match_results.append((self, 0.0, current_time))

        # Update ratings immediately
        self.update_ratings()
        competitor_glicko2.update_ratings()

    def tied(self, competitor: BaseCompetitor, match_time: Optional[datetime] = None) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method records the match result for later processing during the rating period update.
        The actual rating update happens when update_ratings() is called.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.
            match_time (datetime, optional): The time when the match occurred. Default: current time.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
            InvalidParameterException: If the match time is before either competitor's last activity.
        """
        self.verify_competitor_types(competitor)
        logger.debug("%s tied with %s (time=%s). Recording result.", self, competitor, match_time)
        competitor_glicko2 = cast(Glicko2Competitor, competitor)

        # Get the match time
        current_time = match_time if match_time is not None else datetime.now()

        # Validate match time is not before last activity
        if current_time < self._last_activity:
            logger.error("Match time %s is before self last activity %s", current_time, self._last_activity)
            raise InvalidParameterException("Match time cannot be before competitor's last activity time")
        if current_time < competitor_glicko2._last_activity:
            logger.error(
                "Match time %s is before opponent last activity %s", current_time, competitor_glicko2._last_activity
            )
            raise InvalidParameterException("Match time cannot be before opponent's last activity time")

        # Update RDs for both competitors based on inactivity before recording match
        self.update_rd_for_inactivity(current_time)
        competitor_glicko2.update_rd_for_inactivity(current_time)

        # Get the match time
        current_time = match_time if match_time is not None else datetime.now()

        # Validate match time is not before last activity
        if current_time < self._last_activity:
            logger.error("Match time %s is before self last activity %s", current_time, self._last_activity)
            raise InvalidParameterException("Match time cannot be before competitor's last activity time")
        if current_time < competitor_glicko2._last_activity:
            logger.error(
                "Match time %s is before opponent last activity %s", current_time, competitor_glicko2._last_activity
            )
            raise InvalidParameterException("Match time cannot be before opponent's last activity time")

        # Update RDs for both competitors based on inactivity before recording match
        self.update_rd_for_inactivity(current_time)
        competitor_glicko2.update_rd_for_inactivity(current_time)

        # Record the match result for this competitor
        self._match_results.append((competitor_glicko2, 0.5, current_time))

        # Record the match result for the opponent
        competitor_glicko2._match_results.append((self, 0.5, current_time))

        # Update ratings immediately
        self.update_ratings()
        competitor_glicko2.update_ratings()
