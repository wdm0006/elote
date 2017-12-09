from elote.competitors.base import BaseCompetitor
from elote import EloCompetitor, ECFCompetitor, DWZCompetitor, GlickoCompetitor

competitor_types = {
    "EloCompetitor": EloCompetitor,
    "ECFCompetitor": ECFCompetitor,
    "DWZCompetitor": DWZCompetitor,
    "GlickoCompetitor": GlickoCompetitor
}


class BlendedCompetitor(BaseCompetitor):
    def __init__(self, competitors, blend_mode="mean"):
        self._sub_competitors = []
        for competitor in competitors:
            comp_type = competitor_types.get(competitor.get('type', 'EloCompetitor'))
            comp_kwargs = competitor.get('competitor_kwargs', {})
            self._sub_competitors.append(comp_type(**comp_kwargs))

        self.blend_mode = blend_mode

    def __repr__(self):
        return '<BlendedCompetitor: %s>' % (self.__hash__())

    def __str__(self):
        return '<BlendedCompetitor>'

    def export_state(self):
        return {
            "blend_mode": self.blend_mode,
            "competitors": [
                {
                    "type": x.__name__,
                    "competitor_kwargs": x.export_state()
                }
                for x in self._sub_competitors
            ]
        }

    def expected_score(self, competitor):
        if self.blend_mode == 'mean':
            return sum([x.expected_score(competitor) for x in self._sub_competitors]) / len(self._sub_competitors)
        else:
            raise NotImplementedError('Blend mode %s not supported' % (self.blend_mode, ))

    def beat(self, competitor):
        """
        takes in a competitor object that lost, updates both's scores.
        """
        for c in self._sub_competitors:
            c.beat(competitor)

    def tied(self, competitor):
        for c in self._sub_competitors:
            c.tied(competitor)
