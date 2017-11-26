from elote.competitors import EloCompetitor


class LambdaArena:
    def __init__(self, func, base_competitor=EloCompetitor, base_competitor_kwargs=None):
        self.func = func
        self.competitors = dict()
        self.base_competitor = base_competitor
        if base_competitor_kwargs is None:
            self.base_competitor_kwargs = dict()
        else:
            self.base_competitor_kwargs = base_competitor_kwargs

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

    def leaderboard(self):
        lb = [
            {
                "competitor": k,
                "rating": v.rating
            }
            for k, v in self.competitors.items()
        ]

        return sorted(lb, key=lambda x: x.get('rating'))