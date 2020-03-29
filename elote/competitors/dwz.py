from elote.competitors.base import BaseCompetitor
import math


class DWZCompetitor(BaseCompetitor):
    _J = 10

    def __init__(self, initial_rating: float = 400):
        """

        class vars:

         * _J: default 10

        :param initial_rating: the initial rating to use for a new competitor who has no history.  Default 400
        """
        self._count = 0
        self._rating = initial_rating

    def __repr__(self):
        return '<DWZCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<DWZCompetitor>'

    @property
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, value):
        self._rating = value

    def export_state(self):
        """
        Exports all information needed to re-create this competitor from scratch later on.

        :return: dictionary of kwargs and class-args to re-instantiate this object
        """
        return {
            "initial_rating": self._rating,
            "class_vars": {
                "_J": self._J
            }
        }


    def expected_score(self, competitor: BaseCompetitor):
        """
        The expected outcome of a match between this competitor and one passed in. Scaled between 0-1, where 1 is a strong
        likelihood of this competitor winning and 0 is a strong likelihood of this competitor losing.

        :param competitor: another EloCompetitor to compare this competitor to.
        :return: likelihood to beat the passed competitor, as a float 0-1.
        """
        self.verify_competitor_types(competitor)

        return 1 / (1 + 10 ** ((competitor.rating - self._rating) / 400))

    @property
    def _E(self):
        E0 = (self._rating / 1000) ** 4 + self._J
        a = max([0.5, min([self._rating / 2000, 1])])

        if self._rating < 1300:
            B = math.exp((1300 - self._rating) / 150) - 1
        else:
            B = 0

        E = int(a * E0 + B)
        if B == 0:
            return max([5, min([E, min([30, 5 * self._count])])])
        else:
            return max([5, min([E, 150])])

    def _new_rating(self, competitor, W_a):
        return self._rating + (800 / (self._E + self._count)) * (W_a - self.expected_score(competitor))

    def beat(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that lost a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that lost thr bout
        :type competitor: DWZCompetitor
        """

        self.verify_competitor_types(competitor)

        self_rating = self._new_rating(competitor, 1)
        competitor_rating = competitor._new_rating(self, 0)

        self._rating = self_rating
        self._count += 1

        competitor.rating = competitor_rating
        competitor._count += 1

    def tied(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that tied in a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that tied with this one
        :type competitor: DWZCompetitor
        """
        self.verify_competitor_types(competitor)

        self_rating = self._new_rating(competitor, 0.5)
        competitor_rating = competitor._new_rating(self, 0.5)

        self._rating = self_rating
        self._count += 1

        competitor.rating = competitor_rating
        competitor._count += 1