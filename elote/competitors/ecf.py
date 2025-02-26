from elote.competitors.base import BaseCompetitor
from collections import deque
import statistics


class ECFCompetitor(BaseCompetitor):
    _delta = 50
    _n_periods = 30

    def __init__(self, initial_rating: float = 40):
        """

        class vars:

         * _delta: default 50
         * _n_periods: default 30

        :param initial_rating: the initial rating to use for a new competitor who has no history.  Default 40
        """
        self.__initial_rating = initial_rating
        self.scores = None
        self.__cached_rating = None  # Cache for the rating calculation

    def __repr__(self):
        return "<ECFCompetitor: %s>" % (self.__hash__())

    def __str__(self):
        return "<ECFCompetitor>"

    def __initialize_ratings(self):
        # Initialize with a single value instead of a full deque of None values
        self.scores = deque(maxlen=self._n_periods)
        self.scores.append(self.__initial_rating)
        self.__cached_rating = self.__initial_rating  # Initialize the cache

    @property
    def elo_conversion(self):
        return self.rating * 7.5 + 700

    @property
    def rating(self):
        """

        :return:
        """
        if self.scores is None:
            self.__initialize_ratings()
            return self.__cached_rating

        # Return cached value if available
        if self.__cached_rating is not None:
            return self.__cached_rating

        # Calculate and cache the mean
        valid_scores = [_ for _ in self.scores if _ is not None]
        if valid_scores:
            self.__cached_rating = statistics.mean(valid_scores)
        else:
            self.__cached_rating = self.__initial_rating

        return self.__cached_rating

    def _update(self, rating: float):
        if self.scores is None:
            self.__initialize_ratings()

        # Invalidate the cache when updating scores
        self.__cached_rating = None

        # Add the new rating
        self.scores.append(rating)

        # No need to manually popleft since we're using maxlen

    def export_state(self):
        """
        Exports all information needed to re-create this competitor from scratch later on.

        :return: dictionary of kwargs and class-args to re-instantiate this object
        """
        return {
            "initial_rating": self.rating,
            "class_vars": {"_delta": self._delta, "_n_periods": self._n_periods},
        }

    # Cache the transformed rating calculation
    _cached_transformed_elo_rating = None
    _cached_elo_conversion_for_transform = None

    @property
    def transformed_elo_rating(self):
        # Check if we can use the cached value
        current_elo = self.elo_conversion
        if self._cached_transformed_elo_rating is not None and self._cached_elo_conversion_for_transform == current_elo:
            return self._cached_transformed_elo_rating

        # Calculate and cache the result
        self._cached_elo_conversion_for_transform = current_elo
        self._cached_transformed_elo_rating = 10 ** (current_elo / 400)
        return self._cached_transformed_elo_rating

    def expected_score(self, competitor: BaseCompetitor):
        """
        The expected outcome of a match between this competitor and one passed in. Scaled between 0-1, where 1 is a strong
        likelihood of this competitor winning and 0 is a strong likelihood of this competitor losing.

        :param competitor: another EloCompetitor to compare this competitor to.
        :return: likelihood to beat the passed competitor, as a float 0-1.
        """
        self.verify_competitor_types(competitor)

        # Calculate directly to avoid multiple property accesses
        my_transformed = self.transformed_elo_rating
        their_transformed = competitor.transformed_elo_rating
        return my_transformed / (their_transformed + my_transformed)

    def beat(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that lost a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that lost their bout
        :type competitor: ECFCompetitor
        """
        self.verify_competitor_types(competitor)

        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                competitors_rating = self_rating - self._delta
            else:
                competitors_rating = self_rating + self._delta

        # Update both competitors in one go
        self._update(competitors_rating + self._delta)
        competitor._update(self_rating - self._delta)

    def tied(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that tied in a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that tied with this one
        :type competitor: ECFCompetitor
        """
        self.verify_competitor_types(competitor)

        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                competitors_rating = self_rating - self._delta
            else:
                competitors_rating = self_rating + self._delta

        # Update both competitors in one go
        self._update(competitors_rating)
        competitor._update(self_rating)
