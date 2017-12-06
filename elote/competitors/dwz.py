from elote.competitors.base import BaseCompetitor
import math


class DWZCompetitor(BaseCompetitor):
    _J = 10

    def __init__(self, initial_rating=400):
        self._count = 0
        self.rating = initial_rating

    def __repr__(self):
        return '<DWZCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<DWZCompetitor>'

    def export_state(self):
        return {
            "initial_rating": self.rating
        }

    def expected_score(self, competitor):
        return 1 / (1 + 10 ** ((competitor.rating - self.rating) / 400 ))

    @property
    def _E(self):
        E0 = (self.rating / 1000) ** 4 + self._J
        a = max([0.5, min([self.rating / 2000, 1])])

        if self.rating < 1300:
            B = math.exp((1300 - self.rating) / 150) - 1
        else:
            B = 0

        E = int(a * E0 + B)
        if B == 0:
            return max([5, min([E, min([30, 5 * self._count])])])
        else:
            return max([5, min([E, 150])])

    def _new_rating(self, competitor, W_a):
        return self.rating + (800 / (self._E + self._count)) * (W_a - self.expected_score(competitor))

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """

        self_rating = self._new_rating(competitor, 1)
        competitor_rating = competitor._new_rating(self, 0)

        self.rating = self_rating
        self._count += 1

        competitor.rating = competitor_rating
        competitor._count += 1

    def tied(self, competitor):
        self_rating = self._new_rating(competitor, 0.5)
        competitor_rating = competitor._new_rating(self, 0.5)

        self.rating = self_rating
        self._count += 1

        competitor.rating = competitor_rating
        competitor._count += 1