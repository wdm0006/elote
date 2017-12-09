from elote import EloCompetitor
from elote.arenas.base import BaseArena


class LambdaArena(BaseArena):
    def __init__(self, func, base_competitor=EloCompetitor, base_competitor_kwargs=None, initial_state=None):
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

    def set_competitor_class_var(self, name, value):
        setattr(self.base_competitor, name, value)

    def tournament(self, matchups):
        for a, b in matchups:
            self.matchup(a, b)

    def matchup(self, a, b):
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        res = self.func(a, b)
        if res is None:
            self.competitors[a].tied(self.competitors[b])
        elif res is True:
            self.competitors[a].beat(self.competitors[b])
        else:
            self.competitors[b].beat(self.competitors[a])

    def expected_score(self, a, b):
        """expected liklihood of a beating b"""
        if a not in self.competitors:
            self.competitors[a] = self.base_competitor(**self.base_competitor_kwargs)
        if b not in self.competitors:
            self.competitors[b] = self.base_competitor(**self.base_competitor_kwargs)

        return self.competitors[a].expected_score(self.competitors[b])

    def export_state(self):
        out = dict()
        for k, v in self.competitors.items():
            out[k] = v.export_state()
        return out

    def leaderboard(self):
        lb = [
            {
                "competitor": k,
                "rating": v.rating
            }
            for k, v in self.competitors.items()
        ]

        return sorted(lb, key=lambda x: x.get('rating'))