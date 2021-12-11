from elote.competitors.base import BaseCompetitor


class EloCompetitor(BaseCompetitor):
    _base_rating = 400
    _k_factor = 32

    def __init__(self, initial_rating: float = 400, k_factor: float = 32):
        """
        **Overview**

        Elo rating is a rating system based on pairwise comparisons. Ratings are given to competitors based on their
        comparisons (bouts) with peers, in which they can win, lose or tie. The change in a players rating is scaled by
        the rating difference between them and their competitor.

        [1] Elo, Arpad (1978). The Rating of Chessplayers, Past and Present. Arco. ISBN 0-668-04721-6.

        **Basic Usage**

        .. code-block:: python

            from elote import EloCompetitor
            good = EloCompetitor(initial_rating=400)
            better = EloCompetitor(initial_rating=500)
            better.beat(good)
            print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))


        **Class Variables**

        Class variables are configured for all competitors in a population, not on a per-competitor basis. See the
        documentation on ``Arenas`` to see how to set these safely.

         * _base_rating: defaults to 400.
         * _k_factor: tunes the speed of response to new information, higher is faster response. Default=32

        **Configuration Options**

        :param initial_rating: the initial rating to use for a new competitor who has no history.  Default 400
        :type initial_rating: int
        """
        self._rating = initial_rating
        self._k_factor = k_factor

    def __repr__(self):
        return '<EloCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<EloCompetitor>'

    def export_state(self):
        """
        Exports all information needed to re-create this competitor from scratch later on.

        :return: dictionary of kwargs and class-args to re-instantiate this object
        """
        return {
            "initial_rating": self._rating,
            "class_vars": {
                "_k_factor": self._k_factor,
                "_base_rating": self._base_rating
            }
        }

    @property
    def transformed_rating(self):
        return 10 ** (self._rating / self._base_rating)

    @property
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, value):
        self._rating = value

    def expected_score(self, competitor: BaseCompetitor):
        """
        In Elo rating, a player's expected score is their probability of winning plus half their probability of drawing.
        This is because in Elo rating a draw is not explicitly accounted for, but rather counted as half of a win and
        half of a loss. It can make the expected score a bit difficult to reason about, so be careful.

        :param competitor: another EloCompetitor to compare this competitor to.
        :return: likelihood to beat the passed competitor, as a float 0-1.
        """

        self.verify_competitor_types(competitor)

        return self.transformed_rating / (competitor.transformed_rating + self.transformed_rating)

    def beat(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that lost a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that lost their bout
        :type competitor: EloCompetitor
        """

        self.verify_competitor_types(competitor)

        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self._rating = self._rating + self._k_factor * (1 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0 - lose_es)

    def tied(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that tied in a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that tied with this one
        :type competitor: EloCompetitor
        """

        self.verify_competitor_types(competitor)

        win_es = self.expected_score(competitor)
        lose_es = competitor.expected_score(self)

        # update the winner's rating
        self._rating = self._rating + self._k_factor * (0.5 - win_es)

        # update the loser's rating
        competitor.rating = competitor.rating + self._k_factor * (0.5 - lose_es)
