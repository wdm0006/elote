import math
from elote.competitors.base import BaseCompetitor


class GlickoCompetitor(BaseCompetitor):
    _c = 1
    _q = 0.0057565

    def __init__(self, initial_rating: float = 1500, initial_rd: float = 350):
        """from http://www.glicko.net/glicko/glicko.pdf

        class vars:

         * _c: default 1
         * _q: default 0.0057565

        :param initial_rating: the initial rating to use for a new competitor who has no history.  Default 1500
        :param initial_rd: initial value of rd to use for new competitors with no history. Default 350
        """
        self.rating = initial_rating
        self.rd = initial_rd

    def __repr__(self):
        return '<GlickoCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<GlickoCompetitor>'

    def export_state(self):
        """
        Exports all information needed to re-create this competitor from scratch later on.

        :return: dictionary of kwargs and class-args to re-instantiate this object
        """
        return {
            "initial_rating": self.rating,
            "initial_rd": self.rd,
            "class_vars": {
                "_c": self._c,
                "_q": self._q
            }
        }

    @property
    def tranformed_rd(self):
        return min([350, math.sqrt(self.rd ** 2 + self._c ** 2)])

    @classmethod
    def _g(cls, x):
        return 1 / (math.sqrt(1 + 3 * cls._q ** 2 * (x ** 2) / math.pi ** 2))

    def expected_score(self, competitor: BaseCompetitor):
        """
        The expected outcome of a match between this competitor and one passed in. Scaled between 0-1, where 1 is a strong
        likelihood of this competitor winning and 0 is a strong likelihood of this competitor losing.

        :param competitor: another EloCompetitor to compare this competitor to.
        :return: likelihood to beat the passed competitor, as a float 0-1.
        """

        self.verify_competitor_types(competitor)

        g_term = self._g(self.rd ** 2)
        E = 1 / (1 + 10 ** ((-1 * g_term * (self.rating - competitor.rating))/400))
        return E

    def beat(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that lost a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that lost thr bout
        :type competitor: GlickoCompetitor
        """

        self.verify_competitor_types(competitor)

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

    def tied(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that tied in a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that tied with this one
        :type competitor: GlickoCompetitor
        """

        self.verify_competitor_types(competitor)

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
