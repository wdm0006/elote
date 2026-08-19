import copy
import pickle
import unittest

from elote import (
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    KeenerCompetitor,
    LambdaArena,
    MasseyCompetitor,
    WholeHistoryRatingCompetitor,
)


# The global-fit systems are the ones that keep their opponent graph in dicts keyed by
# competitor objects, so they are the ones whose __eq__/__hash__ pair has to stay
# state-independent for copy and pickle to be able to rebuild a cyclic graph.
GLOBAL_FIT_COMPETITORS = (
    ColleyMatrixCompetitor,
    MasseyCompetitor,
    KeenerCompetitor,
    BradleyTerryCompetitor,
    WholeHistoryRatingCompetitor,
)


def _graph(competitor_cls):
    """Build a three-competitor graph containing a win, another win and a tie."""
    a, b, c = competitor_cls(), competitor_cls(), competitor_cls()
    a.beat(b)
    b.beat(c)
    a.tied(c)
    return a, b, c


def _ratings(competitors):
    return [competitor.rating for competitor in competitors]


class TestGlobalFitCompetitorCopy(unittest.TestCase):
    """Global-fit competitors must survive copy.deepcopy and pickle round trips."""

    def test_deepcopy_reproduces_ratings(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                original = _graph(competitor_cls)
                expected = _ratings(original)
                copied = copy.deepcopy(original)
                self.assertEqual(_ratings(copied), expected)

    def test_pickle_reproduces_ratings(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                original = _graph(competitor_cls)
                expected = _ratings(original)
                copied = pickle.loads(pickle.dumps(original))
                self.assertEqual(_ratings(copied), expected)

    def test_copied_graph_is_independent(self):
        """A further result on the copy moves the copy and leaves the original alone."""
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            for label, clone in (
                ("deepcopy", copy.deepcopy),
                ("pickle", lambda graph: pickle.loads(pickle.dumps(graph))),
            ):
                with self.subTest(competitor=competitor_cls.__name__, clone=label):
                    original = _graph(competitor_cls)
                    expected = _ratings(original)
                    copied_a, copied_b, _ = clone(original)

                    copied_b.beat(copied_a)

                    self.assertNotEqual(copied_a.rating, expected[0])
                    self.assertEqual(_ratings(original), expected)


class TestGlobalFitArenaCopy(unittest.TestCase):
    """An arena built on a global-fit competitor must be deep-copyable."""

    MATCHUPS = (("a", "b"), ("b", "c"), ("c", "a"), ("a", "c"), ("b", "a"))

    def _trained_arena(self, competitor_cls):
        arena = LambdaArena(lambda a, b: a > b, base_competitor=competitor_cls)
        for a, b in self.MATCHUPS:
            arena.matchup(a, b)
        return arena

    def test_deepcopy_preserves_leaderboard(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                arena = self._trained_arena(competitor_cls)
                expected = arena.leaderboard()
                self.assertEqual(copy.deepcopy(arena).leaderboard(), expected)

    def test_deepcopied_arena_is_independent(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                arena = self._trained_arena(competitor_cls)
                expected = arena.leaderboard()

                copied = copy.deepcopy(arena)
                for _ in range(3):
                    copied.matchup("c", "a", outcome=1.0)

                self.assertNotEqual(copied.leaderboard(), expected)
                self.assertEqual(arena.leaderboard(), expected)


class TestGlobalFitCompetitorIdentity(unittest.TestCase):
    """Equality and hashing stay identity-based, independent of rating state."""

    def test_distinct_competitors_with_equal_ratings_are_not_equal(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                first, second = competitor_cls(), competitor_cls()
                self.assertEqual(first.rating, second.rating)
                self.assertNotEqual(first, second)
                self.assertEqual(first, first)
                self.assertEqual(len({first, second}), 2)

    def test_hash_is_stable_across_results(self):
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                a, b, _ = _graph(competitor_cls)
                before = hash(a)
                a.beat(b)
                self.assertEqual(hash(a), before)

    def test_copy_is_not_equal_to_its_original(self):
        """A copy is a different object, so it must not compare equal to the original."""
        for competitor_cls in GLOBAL_FIT_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                a, _, _ = _graph(competitor_cls)
                copied = copy.deepcopy(a)
                self.assertNotEqual(copied, a)
                self.assertEqual(len({a, copied}), 2)


if __name__ == "__main__":
    unittest.main()
