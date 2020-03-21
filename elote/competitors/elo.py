from elote.competitors.base import BaseCompetitor


class EloCompetitor(BaseCompetitor):
    _base_rating = 400
    _k_factor = 32

    def __init__(self, initial_rating=400):
        self.rating = initial_rating

    def __repr__(self):
        return '<EloCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<EloCompetitor>'

    def export_state(self):
        return {
            "initial_rating": self.rating
        }

    @property
    def transformed_rating(self):
        return 10 ** (self.rating / self._base_rating)

    def expected_score(self, competitor):
        return self.transformed_rating / (competitor.transformed_rating + self.transformed_rating)

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """
        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self.rating = self.rating + self._k_factor * (1 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0 - lose_es)

    def tied(self, competitor):
        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self.rating = self.rating + self._k_factor * (0.5 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0.5 - lose_es)
