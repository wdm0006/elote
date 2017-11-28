import math

class EloCompetitor:
    _base_rating = 400
    _k_factor = 32

    def __init__(self, initial_rating=400):
        self.rating = initial_rating

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
        competitor._rating = competitor.rating + self._k_factor * (0 - lose_es)

    def lost_to(self, competitor):
        competitor.beat(self)

    def tied(self, competitor):
        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self.rating = self.rating + self._k_factor * (0.5 - win_es)

        # update the loser's rating
        competitor._rating = competitor.rating + self._k_factor * (0.5 - lose_es)

class GlickoCompetitor:
    """ from http://www.glicko.net/glicko/glicko.pdf"""
    _c = 1
    _q = 0.0057565

    def __init__(self, initial_rating=1500, initial_rd=350):
        self.rating = initial_rating
        self.rd = initial_rd

    @property
    def tranformed_rd():
        return min([350, math.sqrt(self.rd ** 2 + self._c ** 2)])

    @classmethod
    def _g(cls, x):
        return 1 / (math.sqrt(1 + 3 * cls._q **2 * (x ** 2) / math.pi ** 2))

    def expected_score(self, competitor):
        g_term = self._g(math.sqrt(self.rd ** 2 + competitor.rd ** 2) * (self.rating - competitor.rating) / 400)
        E = 1 / (1 + 10 ** (-1 * g_term))
        return E

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """

        # first we update ourselves
        s = 1
        E_term = self.expected_score(competitor)
        d_squared = (self._q ** 2 * (self._g(competitor.rd) ** 2 * E_term * (1 - E_term))) ** -1
        s_new_r = self.rating + (self._q / (1 / self.rd **2 + 1 / d_squared)) * self._g(competitor.rd) * (s - E_term)
        s_new_rd = math.sqrt((1 / self.rd ** 2 + 1 / d_squared) ** -1)

        # then the competitor
        s = 0
        E_term = competitor.expected_score(self)
        d_squared = (self._q ** 2 * (self._g(self.rd) ** 2 * E_term * (1 - E_term))) ** -1
        c_new_r = competitor.rating + (self._q / (1 / competitor.rd **2 + 1 / d_squared)) * self._g(self.rd) * (s - E_term)
        c_new_rd = math.sqrt((1 / competitor.rd ** 2 + 1 / d_squared) ** -1)

        # assign everything
        self.rating = s_new_r
        self.rd = s_new_rd
        competitor.rating = c_new_r
        competitor.rd = c_new_rd

    def lost_to(self, competitor):
        competitor.beat(self)

    def tied(self, competitor):
        # first we update ourselves
        s = 0.5
        E_term = self.expected_score(competitor)
        d_squared = (self._q ** 2 * (self._g(competitor.rd) ** 2 * E_term * (1 - E_term))) ** -1
        s_new_r = self.rating + (self._q / (1 / self.rd **2 + 1 / d_squared)) * self._g(competitor.rd) * (s - E_term)
        s_new_rd = math.sqrt((1 / self.rd ** 2 + 1 / d_squared) ** -1)

        # then the competitor
        s = 0.5
        E_term = competitor.expected_score(self)
        d_squared = (self._q ** 2 * (self._g(self.rd) ** 2 * E_term * (1 - E_term))) ** -1
        c_new_r = competitor.rating + (self._q / (1 / competitor.rd **2 + 1 / d_squared)) * self._g(self.rd) * (s - E_term)
        c_new_rd = math.sqrt((1 / competitor.rd ** 2 + 1 / d_squared) ** -1)

        # assign everything
        self.rating = s_new_r
        self.rd = s_new_rd
        competitor.rating = c_new_r
        competitor.rd = c_new_rd

