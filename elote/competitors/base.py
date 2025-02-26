import abc


class MissMatchedCompetitorTypesException(Exception):
    pass


class BaseCompetitor:
    @property
    @abc.abstractmethod
    def rating(self):
        pass

    @abc.abstractmethod
    def expected_score(self, competitor):
        pass

    @abc.abstractmethod
    def beat(self, competitor):
        pass

    def lost_to(self, competitor):
        competitor.beat(self)

    @abc.abstractmethod
    def tied(self, competitor):
        pass

    @abc.abstractmethod
    def export_state(self):
        pass

    def verify_competitor_types(self, competitor):
        if not isinstance(competitor, self.__class__):
            raise MissMatchedCompetitorTypesException(
                "Competitor types %s and %s cannot be co-mingled"
                % (
                    type(competitor),
                    type(self),
                )
            )
