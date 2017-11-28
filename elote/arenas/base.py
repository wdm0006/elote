import abc


class BaseArena:
    @abc.abstractmethod
    def set_competitor_class_var(self, name, value):
        pass

    @abc.abstractmethod
    def tournament(self, matchups):
        pass

    @abc.abstractmethod
    def matchup(self, a, b):
        pass

    @abc.abstractmethod
    def leaderboard(self):
        pass

    @abc.abstractmethod
    def export_state(self):
        pass