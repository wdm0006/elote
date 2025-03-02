import math
from scipy import special
from typing import Dict, Any, ClassVar, Tuple, Type, TypeVar, List

from elote.competitors.base import BaseCompetitor, InvalidParameterException

T = TypeVar("T", bound="TrueSkillCompetitor")


class TrueSkillCompetitor(BaseCompetitor):
    """TrueSkill rating system competitor.

    TrueSkill is a Bayesian skill rating system developed by Microsoft Research.
    It generalizes the Elo and Glicko rating systems to handle team-based games
    and multiplayer scenarios. TrueSkill models each player's skill as a Gaussian
    distribution with a mean (mu) and standard deviation (sigma).

    The mean represents the player's estimated skill, while the standard deviation
    represents the system's uncertainty about that estimate. As more games are played,
    the uncertainty typically decreases.

    Class Attributes:
        _beta (float): The skill factor that controls how much the game outcome depends
                      on skill vs. chance. Default: 4.166.
        _tau (float): The additive dynamics factor that increases uncertainty over time.
                     Default: 0.083.
        _draw_probability (float): The probability of a draw. Default: 0.10 (10%).
        _default_mu (float): The default mean skill value for new players. Default: 25.0.
        _default_sigma (float): The default standard deviation for new players. Default: 8.333.
    """

    _beta: ClassVar[float] = 4.166
    _tau: ClassVar[float] = 0.083
    _draw_probability: ClassVar[float] = 0.10
    _default_mu: ClassVar[float] = 25.0
    _default_sigma: ClassVar[float] = 8.333

    def __init__(self, initial_mu: float = None, initial_sigma: float = None):
        """Initialize a TrueSkill competitor.

        Args:
            initial_mu (float, optional): The initial mean skill value. Default: _default_mu.
            initial_sigma (float, optional): The initial standard deviation. Default: _default_sigma.

        Raises:
            InvalidParameterException: If the initial sigma is not positive.
        """
        # Set default values if not provided
        if initial_mu is None:
            initial_mu = self._default_mu
        if initial_sigma is None:
            initial_sigma = self._default_sigma

        if initial_sigma <= 0:
            raise InvalidParameterException("Initial sigma must be positive")

        # Store initial values for reset
        self._initial_mu = initial_mu
        self._initial_sigma = initial_sigma

        # Current skill parameters
        self._mu = initial_mu
        self._sigma = initial_sigma

    def __repr__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<TrueSkillCompetitor: mu={self._mu:.2f}, sigma={self._sigma:.2f}, rating={self.rating:.2f}>"

    def __str__(self) -> str:
        """Return a string representation of this competitor.

        Returns:
            str: A string representation of this competitor.
        """
        return f"<TrueSkillCompetitor: rating={self.rating:.2f}>"

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor.

        Returns:
            dict: A dictionary containing the initialization parameters.
        """
        return {
            "initial_mu": self._initial_mu,
            "initial_sigma": self._initial_sigma,
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor.

        Returns:
            dict: A dictionary containing the current state variables.
        """
        return {
            "mu": self._mu,
            "sigma": self._sigma,
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary.

        Args:
            parameters (dict): A dictionary containing parameters.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        # Validate and set initial_mu
        initial_mu = parameters.get("initial_mu", self._default_mu)
        self._initial_mu = initial_mu

        # Validate and set initial_sigma
        initial_sigma = parameters.get("initial_sigma", self._default_sigma)
        if initial_sigma <= 0:
            raise InvalidParameterException("Initial sigma must be positive")
        self._initial_sigma = initial_sigma

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary.

        Args:
            state (dict): A dictionary containing state variables.

        Raises:
            InvalidParameterException: If any state variable is invalid.
        """
        # Validate and set mu
        mu = state.get("mu", self._initial_mu)
        self._mu = mu

        # Validate and set sigma
        sigma = state.get("sigma", self._initial_sigma)
        if sigma <= 0:
            raise InvalidParameterException("Sigma must be positive")
        self._sigma = sigma

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters.

        Args:
            parameters (dict): A dictionary containing parameters.

        Returns:
            TrueSkillCompetitor: A new competitor instance.

        Raises:
            InvalidParameterException: If any parameter is invalid.
        """
        return cls(
            initial_mu=parameters.get("initial_mu", cls._default_mu),
            initial_sigma=parameters.get("initial_sigma", cls._default_sigma),
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
            TrueSkillCompetitor: A new competitor with the same state as the exported one.

        Raises:
            InvalidParameterException: If any parameter in the state is invalid.
        """
        # Handle legacy state format
        if "type" not in state:
            # Configure class variables if provided
            if "class_vars" in state:
                class_vars = state["class_vars"]
                if "beta" in class_vars:
                    cls._beta = class_vars["beta"]
                if "tau" in class_vars:
                    cls._tau = class_vars["tau"]
                if "draw_probability" in class_vars:
                    cls._draw_probability = class_vars["draw_probability"]
                if "default_mu" in class_vars:
                    cls._default_mu = class_vars["default_mu"]
                if "default_sigma" in class_vars:
                    cls._default_sigma = class_vars["default_sigma"]

            # Create a new competitor with the initial parameters
            competitor = cls(
                initial_mu=state.get("initial_mu", cls._default_mu),
                initial_sigma=state.get("initial_sigma", cls._default_sigma),
            )

            # Set the current state if provided
            if "current_mu" in state:
                competitor._mu = state["current_mu"]
            if "current_sigma" in state:
                competitor._sigma = state["current_sigma"]

            return competitor

        # Use the standardized format
        return super().from_state(state)

    def reset(self) -> None:
        """Reset this competitor to its initial state.

        This method resets the competitor's mu and sigma to their initial values.
        """
        self._mu = self._initial_mu
        self._sigma = self._initial_sigma

    @property
    def rating(self) -> float:
        """Get the current conservative rating of this competitor.

        TrueSkill uses a conservative rating estimate (mu - 3*sigma) to ensure
        that a player's displayed rating has a 99% chance of being below their
        actual skill level. This encourages players to keep playing to reduce
        uncertainty and increase their displayed rating.

        Returns:
            float: The current conservative rating.
        """
        return max(self._mu - 3 * self._sigma, self._minimum_rating)

    @rating.setter
    def rating(self, value: float) -> None:
        """Set the current rating of this competitor.

        This method is not directly supported by TrueSkill, as ratings are derived
        from mu and sigma. This implementation raises an exception.

        Args:
            value (float): The new rating value.

        Raises:
            NotImplementedError: Always, as setting the rating directly is not supported.
        """
        raise NotImplementedError("Cannot directly set the rating of a TrueSkillCompetitor. Set mu and sigma instead.")

    @property
    def mu(self) -> float:
        """Get the current mean skill value of this competitor.

        Returns:
            float: The current mean skill value.
        """
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        """Set the current mean skill value of this competitor.

        Args:
            value (float): The new mean skill value.
        """
        self._mu = value

    @property
    def sigma(self) -> float:
        """Get the current standard deviation of this competitor.

        Returns:
            float: The current standard deviation.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Set the current standard deviation of this competitor.

        Args:
            value (float): The new standard deviation value.

        Raises:
            InvalidParameterException: If the standard deviation is not positive.
        """
        if value <= 0:
            raise InvalidParameterException("Sigma must be positive")
        self._sigma = value

    def _v(self, beta_squared: float, sigma_squared: float) -> float:
        """Calculate the skill variance factor.

        Args:
            beta_squared (float): The squared beta parameter.
            sigma_squared (float): The squared sigma parameter.

        Returns:
            float: The skill variance factor.
        """
        return math.sqrt(beta_squared + sigma_squared)

    def _w(self, v: float) -> float:
        """Calculate the win probability factor.

        Args:
            v (float): The skill variance factor.

        Returns:
            float: The win probability factor.
        """
        return v * math.sqrt(2)

    def _gaussian_cdf(self, x: float) -> float:
        """Calculate the cumulative distribution function of the standard Gaussian.

        Args:
            x (float): The input value.

        Returns:
            float: The CDF value.
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _gaussian_pdf(self, x: float) -> float:
        """Calculate the probability density function of the standard Gaussian.

        Args:
            x (float): The input value.

        Returns:
            float: The PDF value.
        """
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x**2) / 2)

    def _v_win(self, t: float) -> float:
        """Calculate the v function for win probability.

        Args:
            t (float): The input value.

        Returns:
            float: The v function value.
        """
        pdf = self._gaussian_pdf(t)
        cdf = self._gaussian_cdf(t)
        return pdf / cdf if cdf > 0 else 0

    def _w_win(self, t: float) -> float:
        """Calculate the w function for win probability.

        Args:
            t (float): The input value.

        Returns:
            float: The w function value.
        """
        v = self._v_win(t)
        return v * (v + t)

    def _v_draw(self, t: float, epsilon: float) -> float:
        """Calculate the v function for draw probability.

        Args:
            t (float): The input value.
            epsilon (float): The draw margin.

        Returns:
            float: The v function value.
        """
        pdf_1 = self._gaussian_pdf(t - epsilon)
        pdf_2 = self._gaussian_pdf(t + epsilon)
        cdf_1 = self._gaussian_cdf(t - epsilon)
        cdf_2 = self._gaussian_cdf(t + epsilon)
        denominator = cdf_2 - cdf_1
        return (pdf_1 - pdf_2) / denominator if denominator > 0 else 0

    def _w_draw(self, t: float, epsilon: float) -> float:
        """Calculate the w function for draw probability.

        Args:
            t (float): The input value.
            epsilon (float): The draw margin.

        Returns:
            float: The w function value.
        """
        v = self._v_draw(t, epsilon)
        pdf_1 = self._gaussian_pdf(t - epsilon)
        pdf_2 = self._gaussian_pdf(t + epsilon)
        cdf_1 = self._gaussian_cdf(t - epsilon)
        cdf_2 = self._gaussian_cdf(t + epsilon)
        denominator = cdf_2 - cdf_1
        return v**2 + ((t - epsilon) * pdf_1 - (t + epsilon) * pdf_2) / denominator if denominator > 0 else 0

    @staticmethod
    def _calculate_draw_margin(beta: float, draw_probability: float) -> float:
        """
        Calculate the draw margin based on beta and draw probability.

        Args:
            beta: The beta parameter
            draw_probability: The probability of a draw

        Returns:
            The draw margin
        """
        return math.sqrt(2) * beta * special.erfinv(1 - 2 * draw_probability)

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
        competitor_ts = competitor  # type: TrueSkillCompetitor

        # Calculate the skill difference
        mu_diff = self._mu - competitor_ts._mu

        # Calculate the total variance
        beta_squared = self._beta**2
        sigma_squared_sum = self._sigma**2 + competitor_ts._sigma**2
        v = math.sqrt(beta_squared + sigma_squared_sum)

        # Calculate the win probability
        t = mu_diff / v
        win_prob = self._gaussian_cdf(t)

        # Calculate the draw probability
        draw_margin = self._calculate_draw_margin(self._beta, self._draw_probability)
        draw_prob = self._gaussian_cdf((mu_diff + draw_margin) / v) - self._gaussian_cdf((mu_diff - draw_margin) / v)

        # Adjust win probability to account for draws
        adjusted_win_prob = win_prob - draw_prob / 2

        return adjusted_win_prob

    def beat(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has won against the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that lost.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_ts = competitor  # type: TrueSkillCompetitor

        # Calculate the skill difference
        mu_diff = self._mu - competitor_ts._mu

        # Calculate the total variance
        beta_squared = self._beta**2
        sigma_squared_1 = self._sigma**2
        sigma_squared_2 = competitor_ts._sigma**2
        sigma_squared_sum = sigma_squared_1 + sigma_squared_2
        v = math.sqrt(beta_squared + sigma_squared_sum)

        # Calculate the performance update
        t = mu_diff / v
        v_win_value = self._v_win(t)
        w_win_value = self._w_win(t)

        # Update the winner's mu and sigma
        sigma_squared_to_v_squared = sigma_squared_1 / v**2
        self._mu = self._mu + sigma_squared_to_v_squared * v_win_value
        self._sigma = math.sqrt(sigma_squared_1 * (1 - sigma_squared_to_v_squared * w_win_value))

        # Update the loser's mu and sigma
        sigma_squared_to_v_squared = sigma_squared_2 / v**2
        competitor_ts._mu = competitor_ts._mu - sigma_squared_to_v_squared * v_win_value
        competitor_ts._sigma = math.sqrt(sigma_squared_2 * (1 - sigma_squared_to_v_squared * w_win_value))

        # Apply the dynamic factor to increase uncertainty over time
        self._sigma = math.sqrt(self._sigma**2 + self._tau**2)
        competitor_ts._sigma = math.sqrt(competitor_ts._sigma**2 + self._tau**2)

    def tied(self, competitor: BaseCompetitor) -> None:
        """Update ratings after this competitor has tied with the given competitor.

        Args:
            competitor (BaseCompetitor): The opponent competitor that tied.

        Raises:
            MissMatchedCompetitorTypesException: If the competitor types don't match.
        """
        self.verify_competitor_types(competitor)
        competitor_ts = competitor  # type: TrueSkillCompetitor

        # Calculate the skill difference
        mu_diff = self._mu - competitor_ts._mu

        # Calculate the total variance
        beta_squared = self._beta**2
        sigma_squared_1 = self._sigma**2
        sigma_squared_2 = competitor_ts._sigma**2
        sigma_squared_sum = sigma_squared_1 + sigma_squared_2
        v = math.sqrt(beta_squared + sigma_squared_sum)

        # Calculate the draw margin
        draw_margin = self._calculate_draw_margin(self._beta, self._draw_probability)

        # Calculate the performance update for a draw
        t = mu_diff / v
        v_draw_value = self._v_draw(t, draw_margin / v)
        w_draw_value = self._w_draw(t, draw_margin / v)

        # Update the first player's mu and sigma
        sigma_squared_to_v_squared = sigma_squared_1 / v**2
        self._mu = self._mu + sigma_squared_to_v_squared * v_draw_value
        self._sigma = math.sqrt(sigma_squared_1 * (1 - sigma_squared_to_v_squared * w_draw_value))

        # Update the second player's mu and sigma
        sigma_squared_to_v_squared = sigma_squared_2 / v**2
        competitor_ts._mu = competitor_ts._mu - sigma_squared_to_v_squared * v_draw_value
        competitor_ts._sigma = math.sqrt(sigma_squared_2 * (1 - sigma_squared_to_v_squared * w_draw_value))

        # Apply the dynamic factor to increase uncertainty over time
        self._sigma = math.sqrt(self._sigma**2 + self._tau**2)
        competitor_ts._sigma = math.sqrt(competitor_ts._sigma**2 + self._tau**2)

    @classmethod
    def match_quality(cls, player1: "TrueSkillCompetitor", player2: "TrueSkillCompetitor") -> float:
        """Calculate the match quality between two players.

        Match quality is a value between 0 and 1 that represents how evenly matched
        two players are. A value of 1 indicates a perfectly even match, while a value
        of 0 indicates a completely one-sided match.

        Args:
            player1 (TrueSkillCompetitor): The first player.
            player2 (TrueSkillCompetitor): The second player.

        Returns:
            float: The match quality (between 0 and 1).
        """
        # Calculate the skill difference
        mu_diff = player1._mu - player2._mu

        # Calculate the total variance
        beta_squared = cls._beta**2
        sigma_squared_sum = player1._sigma**2 + player2._sigma**2
        v = math.sqrt(2 * beta_squared + sigma_squared_sum)

        # Calculate the match quality
        exp_term = -(mu_diff**2) / (2 * v**2)
        sqrt_term = math.sqrt(2 * beta_squared) / v

        return sqrt_term * math.exp(exp_term)

    @classmethod
    def create_team(cls, players: List["TrueSkillCompetitor"]) -> Tuple[float, float]:
        """Create a virtual team competitor from a list of players.

        This method combines the skills of multiple players into a single team skill.
        The team's mu is the sum of the players' mus, and the team's sigma is the
        square root of the sum of the players' sigma squared.

        Args:
            players (List[TrueSkillCompetitor]): The list of players in the team.

        Returns:
            Tuple[float, float]: The team's mu and sigma.
        """
        team_mu = sum(player._mu for player in players)
        team_sigma = math.sqrt(sum(player._sigma**2 for player in players))
        return team_mu, team_sigma

    @classmethod
    def update_team(
        cls,
        team_players: List["TrueSkillCompetitor"],
        team_mu_diff: float,
        team_sigma_squared: float,
        v: float,
        result_func: callable,
    ) -> None:
        """Update the ratings of players in a team.

        Args:
            team_players (List[TrueSkillCompetitor]): The list of players in the team.
            team_mu_diff (float): The difference in team mu.
            team_sigma_squared (float): The team's sigma squared.
            v (float): The skill variance factor.
            result_func (callable): The function to calculate the performance update.
        """
        # Calculate the performance update
        t = team_mu_diff / v
        v_value = result_func(t)
        w_value = result_func(t, derivative=True)

        # Update each player's mu and sigma
        for player in team_players:
            sigma_squared = player._sigma**2
            sigma_squared_to_v_squared = sigma_squared / v**2
            player._mu = player._mu + sigma_squared_to_v_squared * v_value
            player._sigma = math.sqrt(sigma_squared * (1 - sigma_squared_to_v_squared * w_value))

            # Apply the dynamic factor to increase uncertainty over time
            player._sigma = math.sqrt(player._sigma**2 + cls._tau**2)
