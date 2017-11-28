import abc


class BaseCompetitor:
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
