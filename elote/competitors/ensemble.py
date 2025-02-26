from elote.competitors.base import BaseCompetitor
from elote import EloCompetitor, ECFCompetitor, DWZCompetitor, GlickoCompetitor

competitor_types = {
    "EloCompetitor": EloCompetitor,
    "ECFCompetitor": ECFCompetitor,
    "DWZCompetitor": DWZCompetitor,
    "GlickoCompetitor": GlickoCompetitor,
}


class BlendedCompetitor(BaseCompetitor):
    def __init__(self, competitors: list, blend_mode: str = "mean"):
        """

        :param competitors:
        :param blend_mode:
        """
        self.sub_competitors = []
        for competitor in competitors:
            comp_type = competitor_types.get(competitor.get("type", "EloCompetitor"))
            comp_kwargs = competitor.get("competitor_kwargs", {})
            self.sub_competitors.append(comp_type(**comp_kwargs))

        self.blend_mode = blend_mode

    def __repr__(self):
        return "<BlendedCompetitor: %s>" % (self.__hash__())

    def __str__(self):
        return "<BlendedCompetitor>"

    @property
    def rating(self):
        return sum([x.rating for x in self.sub_competitors])

    def export_state(self):
        """
        Exports all information needed to re-create this competitor from scratch later on.

        :return: dictionary of kwargs and class-args to re-instantiate this object
        """
        return {
            "blend_mode": self.blend_mode,
            "competitors": [{"type": x.__name__, "competitor_kwargs": x.export_state()} for x in self.sub_competitors],
        }

    def expected_score(self, competitor: BaseCompetitor):
        """
        The expected outcome of a match between this competitor and one passed in. Scaled between 0-1, where 1 is a strong
        likelihood of this competitor winning and 0 is a strong likelihood of this competitor losing.

        :param competitor: another EloCompetitor to compare this competitor to.
        :return: likelihood to beat the passed competitor, as a float 0-1.
        """

        self.verify_competitor_types(competitor)

        if self.blend_mode == "mean":
            es = list()
            for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
                es.append(c.expected_score(other_c))
            return sum(es) / len(es)
        else:
            raise NotImplementedError("Blend mode %s not supported" % (self.blend_mode,))

    def beat(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that lost a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that lost their bout
        :type competitor: BlendedCompetitor
        """

        self.verify_competitor_types(competitor)

        for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
            c.beat(other_c)

    def tied(self, competitor: BaseCompetitor):
        """
        Takes in a competitor object that tied in a match to this competitor, updates the ratings for both.

        :param competitor: the competitor that tied with this one
        :type competitor: BlendedCompetitor
        """

        self.verify_competitor_types(competitor)

        for c, other_c in zip(self.sub_competitors, competitor.sub_competitors):
            c.beat(other_c)
