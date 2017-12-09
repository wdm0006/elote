from elote.competitors.base import BaseCompetitor
from collections import deque
import statistics


class ECFCompetitor(BaseCompetitor):
    _delta = 50
    _n_periods = 30

    def __init__(self, initial_rating=40):
        self.__initial_rating = initial_rating
        self.scores = None

    def __repr__(self):
        return '<ECFCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<ECFCompetitor>'

    def __initialize_ratings(self):
        self.scores = deque([None for _ in range(self._n_periods - 1)])
        self.scores.append(self.__initial_rating)

    @property
    def elo_conversion(self):
        return self.rating * 7.5 + 700

    @property
    def rating(self):
        if self.scores is None:
            self.__initialize_ratings()

        return statistics.mean([_ for _ in self.scores if _ is not None])

    def _update(self, rating):
        if self.scores is None:
            self.__initialize_ratings()

        self.scores.append(rating)
        _ = self.scores.popleft()

    def export_state(self):
        return {
            "initial_rating": self.rating
        }

    @property
    def transformed_elo_rating(self):
        return 10 ** (self.elo_conversion / 400)

    def expected_score(self, competitor):
        return self.transformed_elo_rating / (competitor.transformed_elo_rating + self.transformed_elo_rating)

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """
        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                sign = -1
            else:
                sign = 1
            competitors_rating = self_rating + (sign * self._delta)

        self._update(competitors_rating + self._delta)
        competitor._update(self_rating - self._delta)

    def tied(self, competitor):
        if self.scores is None:
            self.__initialize_ratings()

        # store the at-scoring-time ratings for both competitors
        self_rating = self.rating
        competitors_rating = competitor.rating

        # limit the competitors based on the class delta value
        if abs(self_rating - competitors_rating) > self._delta:
            if self_rating > competitors_rating:
                sign = -1
            else:
                sign = 1
            competitors_rating = self_rating + (sign * self._delta)

        self._update(competitors_rating)
        competitor._update(self_rating)
