import math
from elote.competitors.base import BaseCompetitor


class GlickoCompetitor(BaseCompetitor):
    """ from http://www.glicko.net/glicko/glicko.pdf"""
    _c = 1
    _q = 0.0057565

    def __init__(self, initial_rating=1500, initial_rd=350):
        self.rating = initial_rating
        self.rd = initial_rd

    def __repr__(self):
        return '<GlickoCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<GlickoCompetitor>'

    def export_state(self):
        return {
            "initial_rating": self.rating,
            "initial_rd": self.rd
        }

    @property
    def tranformed_rd(self):
        return min([350, math.sqrt(self.rd ** 2 + self._c ** 2)])

    @classmethod
    def _g(cls, x):
        return 1 / (math.sqrt(1 + 3 * cls._q ** 2 * (x ** 2) / math.pi ** 2))

    def expected_score(self, competitor):
        g_term = self._g(self.rd ** 2)
        E = 1 / (1 + 10 ** ((-1 * g_term * (self.rating - competitor.rating))/400))
        return E

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """

        # first we update ourselves
        s = 1
        E_term = self.expected_score(competitor)
        d_squared = (self._q ** 2 * (self._g(competitor.rd) ** 2 * E_term * (1 - E_term))) ** -1
        s_new_r = self.rating + (self._q / (1 / self.rd ** 2 + 1 / d_squared)) * self._g(competitor.rd) * (s - E_term)
        s_new_rd = math.sqrt((1 / self.rd ** 2 + 1 / d_squared) ** -1)

        # then the competitor
        s = 0
        E_term = competitor.expected_score(self)
        d_squared = (self._q ** 2 * (self._g(self.rd) ** 2 * E_term * (1 - E_term))) ** -1
        c_new_r = competitor.rating + (self._q / (1 / competitor.rd ** 2 + 1 / d_squared)) * self._g(self.rd) * (
            s - E_term)
        c_new_rd = math.sqrt((1 / competitor.rd ** 2 + 1 / d_squared) ** -1)

        # assign everything
        self.rating = s_new_r
        self.rd = s_new_rd
        competitor.rating = c_new_r
        competitor.rd = c_new_rd

    def tied(self, competitor):
        # first we update ourselves
        s = 0.5
        E_term = self.expected_score(competitor)
        d_squared = (self._q ** 2 * (self._g(competitor.rd) ** 2 * E_term * (1 - E_term))) ** -1
        s_new_r = self.rating + (self._q / (1 / self.rd ** 2 + 1 / d_squared)) * self._g(competitor.rd) * (s - E_term)
        s_new_rd = math.sqrt((1 / self.rd ** 2 + 1 / d_squared) ** -1)

        # then the competitor
        s = 0.5
        E_term = competitor.expected_score(self)
        d_squared = (self._q ** 2 * (self._g(self.rd) ** 2 * E_term * (1 - E_term))) ** -1
        c_new_r = competitor.rating + (self._q / (1 / competitor.rd ** 2 + 1 / d_squared)) * self._g(self.rd) * (
            s - E_term)
        c_new_rd = math.sqrt((1 / competitor.rd ** 2 + 1 / d_squared) ** -1)

        # assign everything
        self.rating = s_new_r
        self.rd = s_new_rd
        competitor.rating = c_new_r
        competitor.rd = c_new_rd
