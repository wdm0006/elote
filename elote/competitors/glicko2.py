import math
from typing import Dict, Any, ClassVar, Type, TypeVar

from elote.competitors.base import BaseCompetitor, InvalidRatingValueException, InvalidParameterException

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
    """

    _tau: ClassVar[float] = 0.5
    _epsilon: ClassVar[float] = 0.000001
    _default_volatility: ClassVar[float] = 0.06
    _scale_factor: ClassVar[float] = 173.7178

    def __init__(self, initial_rating: float = 1500, initial_rd: float = 350, initial_volatility: float = None):
        """Initialize a Glicko-2 competitor.

        Args:
            initial_rating (float, optional): The initial rating of this competitor. Default: 1500.
            initial_rd (float, optional): The initial rating deviation of this competitor. Default: 350.
            initial_volatility (float, optional): The initial volatility of this competitor. Default: _default_volatility.

        Raises:
            InvalidRatingValueException: If the initial rating is below the minimum rating.
            InvalidParameterException: If the initial RD is not positive or if the initial volatility is not positive.
        """
        if initial_rating < self._minimum_rating:
            raise InvalidRatingValueException(
                f"Initial rating cannot be below the minimum rating of {self._minimum_rating}"
            )

        if initial_rd <= 0:
            raise InvalidParameterException("Initial RD must be positive")

        if initial_volatility is None:
            initial_volatility = self._default_volatility
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

        # Store match results for rating period
        self._match_results = []

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
        rating = state.get("rating", self._initial_rating)
        if rating < self._minimum_rating:
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
        # Handle legacy state format
        if "type" not in state:
            # Configure class variables if provided
            if "class_vars" in state:
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
        self._rating = self._initial_rating
        self._rd = self._initial_rd
        self._sigma = self._initial_volatility
        self._mu = self._rating_to_mu(self._initial_rating)
        self._phi = self._rd_to_phi(self._initial_rd)
        self._match_results = []

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
        if value <= 0:
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
        if value <= 0:
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
        competitor_glicko2 = competitor  # type: Glicko2Competitor
        return self._E(self._mu, competitor_glicko2._mu, competitor_glicko2._phi)

    def beat(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has won against the given competitor.

        This method records the match result for later processing during the rating period update.
        The actual rating update happens when update_ratings() is called.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_glicko2 = competitor  # type: Glicko2Competitor

        # Record the match result for this competitor
        self._match_results.append((competitor_glicko2, 1.0))

        # Record the match result for the opponent
        competitor_glicko2._match_results.append((self, 0.0))

        # Update ratings immediately
        self.update_ratings()
        competitor_glicko2.update_ratings()

    def tied(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        This method records the match result for later processing during the rating period update.
        The actual rating update happens when update_ratings() is called.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_glicko2 = competitor  # type: Glicko2Competitor

        # Record the match result for this competitor
        self._match_results.append((competitor_glicko2, 0.5))

        # Record the match result for the opponent
        competitor_glicko2._match_results.append((self, 0.5))

        # Update ratings immediately
        self.update_ratings()
        competitor_glicko2.update_ratings()

    def update_ratings(self) -> None:
        """Update ratings based on recorded match results.

        This method implements the Glicko-2 rating system update algorithm.
        It processes all recorded match results and updates the rating, RD, and volatility.
        After updating, the match results are cleared.
        """
        if not self._match_results:
            # If no matches were played, increase the RD
            self._phi = math.sqrt(self._phi**2 + self._sigma**2)
            self._rd = self._phi_to_rd(self._phi)
            return

        # Step 1: Initialize values
        mu = self._mu
        phi = self._phi
        sigma = self._sigma
        v_inv = 0.0  # Inverse of the variance of the change in rating

        # Step 2: Calculate the estimated variance of the player's rating based on game outcomes
        for opponent, _score in self._match_results:
            opponent_phi = opponent._phi
            opponent_mu = opponent._mu
            g_phi_j = self._g(opponent_phi)
            E_mu_mu_j_phi_j = self._E(mu, opponent_mu, opponent_phi)
            v_inv += g_phi_j**2 * E_mu_mu_j_phi_j * (1 - E_mu_mu_j_phi_j)

        v = 1.0 / v_inv if v_inv > 0 else float("inf")

        # Step 3: Calculate the quantity delta, the estimated improvement in rating
        delta = 0.0
        for opponent, _score in self._match_results:
            opponent_phi = opponent._phi
            opponent_mu = opponent._mu
            g_phi_j = self._g(opponent_phi)
            E_mu_mu_j_phi_j = self._E(mu, opponent_mu, opponent_phi)
            delta += g_phi_j * (_score - E_mu_mu_j_phi_j)
        delta *= v

        # Step 4: Calculate the new volatility sigma'
        # This uses the iterative algorithm described in the Glicko-2 paper
        a = math.log(sigma**2)
        tau = self._tau
        epsilon = self._epsilon

        # Compute the function f(x) and its derivative f'(x)
        def f(x):
            exp_x = math.exp(x)
            phi_squared = phi**2
            delta_squared = delta**2
            return exp_x * (delta_squared - phi_squared - v - exp_x) / (2 * (phi_squared + v + exp_x) ** 2) - (
                x - a
            ) / (tau**2)

        # Find the value x that satisfies f(x) = 0 using the Illinois algorithm
        # (a variant of the regula falsi method)
        # First, bracket the root
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau

        A = a
        f_A = f(A)
        f_B = f(B)

        # If the signs of f(A) and f(B) are the same, adjust B
        if f_A * f_B > 0:
            if f_A < 0:
                B = A + tau
            else:
                B = A - tau
            f_B = f(B)

        # Iterate until convergence
        while abs(B - A) > epsilon:
            # Calculate the new approximation C using the Illinois method
            # Add a check to prevent division by zero
            if abs(f_B - f_A) < 1e-10:  # Use a small epsilon value to check for near-zero
                # If the difference is too small, use bisection method instead
                C = (A + B) / 2
            else:
                C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)

            # Update the brackets
            if f_C * f_A < 0:
                B = A
                f_B = f_A
            else:
                f_B *= 0.5  # Illinois method adjustment

            A = C
            f_A = f_C

        # The new volatility sigma' is e^(A/2)
        sigma_prime = math.exp(A / 2)

        # Step 5: Calculate the new phi* (pre-update phi)
        phi_star = math.sqrt(phi**2 + sigma_prime**2)

        # Step 6: Calculate the new phi' (post-update phi)
        phi_prime = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)

        # Step 7: Calculate the new mu' (post-update mu)
        mu_prime = mu + phi_prime**2 * delta / v

        # Step 8: Convert back to the original scale
        self._mu = mu_prime
        self._phi = phi_prime
        self._sigma = sigma_prime
        self._rating = self._mu_to_rating(mu_prime)
        self._rd = self._phi_to_rd(phi_prime)

        # Clear the match results
        self._match_results = []
