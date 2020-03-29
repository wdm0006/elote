from tqdm import tqdm
from elote import EloCompetitor
from elote.arenas.base import BaseArena, Bout, History


class LambdaArena(BaseArena):
    def __init__(self, func, base_competitor=EloCompetitor, base_competitor_kwargs=None, initial_state=None):
        """

        :param func:
        :param base_competitor:
        :param base_competitor_kwargs:
        :param initial_state:
        """
        self.func = func
        self.competitors = dict()
        self.base_competitor = base_competitor
        if base_competitor_kwargs is None:
            self.base_competitor_kwargs = dict()
        else:
            self.base_competitor_kwargs = base_competitor_kwargs

        # if some initial state is passed in, we can seed the population
        if initial_state is not None:
            for k, v in initial_state.items():
                self.competitors[k] = self.base_competitor(**v)

        self.history = History()

    def clear_history(self):
        self.history = History()

    def set_competitor_class_var(self, name, value):
        """

        :param name:
        :param value:
        :return:
        """
        setattr(self.base_competitor, name, value)

    def tournament(self, matchups):
        """

        :param matchups:
        :return:
        """
        for data in tqdm(matchups):
            self.matchup(*data)

    def matchup(self, a, b, attributes=None):
        """

        :param a:
        :param b:
        :return:
        """
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        predicted_outcome = self.expected_score(a, b)

        if attributes:
            res = self.func(a, b, attributes=attributes)
        else:
            res = self.func(a, b)

        if res is None:
            self.competitors[a].tied(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome='tie', attributes=attributes))
        elif res is True:
            self.competitors[a].beat(self.competitors[b])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome='win', attributes=attributes))
        else:
            self.competitors[b].beat(self.competitors[a])
            self.history.add_bout(Bout(a, b, predicted_outcome, outcome='loss', attributes=attributes))

    def expected_score(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        return self.competitors[a].expected_score(self.competitors[b])

    def export_state(self):
        """

        :return:
        """
        out = dict()
        for k, v in self.competitors.items():
            out[k] = v.export_state()
        return out

    def leaderboard(self):
        """

        :return:
        """
        lb = [
            {
                "competitor": k,
                "rating": v.rating
            }
            for k, v in self.competitors.items()
        ]

        return sorted(lb, key=lambda x: x.get('rating'))