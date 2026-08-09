"""Behavioural tests for KeenerCompetitor.

Numeric reference values live in tests/test_KeenerCompetitor_known_values.py.
"""

import json
import unittest

from elote import KeenerCompetitor, LambdaArena
from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    InvalidRatingValueException,
    MissMatchedCompetitorTypesException,
)


def _play(schedule, count=3):
    """Apply a schedule of ``(i, j, score_i, score_j)`` tuples to ``count`` fresh competitors."""
    competitors = [KeenerCompetitor() for _ in range(count)]
    for i, j, score_i, score_j in schedule:
        if score_i > score_j:
            competitors[i].beat(competitors[j], scores=(score_i, score_j))
        elif score_i < score_j:
            competitors[i].lost_to(competitors[j], scores=(score_i, score_j))
        else:
            competitors[i].tied(competitors[j], scores=(score_i, score_j))
    return competitors


class TestKeenerBasics(unittest.TestCase):
    def test_defaults(self):
        competitor = KeenerCompetitor()
        self.assertEqual(competitor.rating, 1.0)
        self.assertEqual(competitor.num_games, 0)
        self.assertEqual(competitor._points_for, 0.0)
        self.assertEqual(competitor._points_against, 0.0)

    def test_custom_initial_rating(self):
        self.assertEqual(KeenerCompetitor(initial_rating=2.5).rating, 2.5)

    def test_non_positive_initial_rating_rejected(self):
        for bad in (0.0, -1.0):
            with self.subTest(initial_rating=bad):
                with self.assertRaises(InvalidRatingValueException):
                    KeenerCompetitor(initial_rating=bad)

    def test_rating_setter_rejects_values_below_the_floor(self):
        competitor = KeenerCompetitor()
        with self.assertRaises(InvalidRatingValueException):
            competitor.rating = -0.5

    def test_type_mismatch_is_rejected(self):
        from elote import EloCompetitor

        competitor = KeenerCompetitor()
        with self.assertRaises(MissMatchedCompetitorTypesException):
            competitor.expected_score(EloCompetitor())
        with self.assertRaises(MissMatchedCompetitorTypesException):
            competitor.beat(EloCompetitor())

    def test_registered_for_state_reconstruction(self):
        self.assertIn("KeenerCompetitor", BaseCompetitor.list_competitor_types())
        self.assertIs(BaseCompetitor.get_competitor_class("KeenerCompetitor"), KeenerCompetitor)

    def test_configure_class_validates_parameters(self):
        with self.assertRaises(InvalidParameterException):
            KeenerCompetitor.configure_class(expected_score_scale=0)
        with self.assertRaises(InvalidParameterException):
            KeenerCompetitor.configure_class(perturbation=0)


class TestKeenerExpectedScore(unittest.TestCase):
    def test_two_fresh_competitors_predict_one_half(self):
        self.assertEqual(KeenerCompetitor().expected_score(KeenerCompetitor()), 0.5)

    def test_predictions_are_exactly_complementary(self):
        a, b, c = _play([(0, 1, 40, 3), (1, 2, 21, 17), (0, 2, 35, 0)])
        for first, second in ((a, b), (b, a), (a, c), (c, b)):
            with self.subTest(pair=(first.rating, second.rating)):
                self.assertEqual(first.expected_score(second) + second.expected_score(first), 1.0)

    def test_predictions_stay_in_the_unit_interval_after_lopsided_results(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        for _ in range(12):
            a.beat(b, scores=(99.0, 0.0))
            for first, second in ((a, b), (b, a)):
                score = first.expected_score(second)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_stronger_competitor_is_favoured(self):
        a, b, _ = _play([(0, 1, 40, 3), (1, 2, 21, 17), (0, 2, 35, 0)])
        self.assertGreater(a.rating, b.rating)
        self.assertGreater(a.expected_score(b), 0.5)


class TestKeenerResults(unittest.TestCase):
    def test_win_updates_both_sides(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(28.0, 7.0))

        self.assertEqual((a._wins, a._losses, a._ties), (1, 0, 0))
        self.assertEqual((b._wins, b._losses, b._ties), (0, 1, 0))
        self.assertEqual((a._points_for, a._points_against), (28.0, 7.0))
        self.assertEqual((b._points_for, b._points_against), (7.0, 28.0))
        self.assertGreater(a.rating, b.rating)

    def test_lost_to_records_the_same_game_as_beat(self):
        winner_first = KeenerCompetitor(), KeenerCompetitor()
        winner_first[0].beat(winner_first[1], scores=(28.0, 7.0))

        loser_first = KeenerCompetitor(), KeenerCompetitor()
        loser_first[1].lost_to(loser_first[0], scores=(7.0, 28.0))

        self.assertEqual(loser_first[0].rating, winner_first[0].rating)
        self.assertEqual(loser_first[1].rating, winner_first[1].rating)
        self.assertEqual(loser_first[0]._points_for, 28.0)
        self.assertEqual(loser_first[1]._points_for, 7.0)

    def test_draw_is_symmetric(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.tied(b, scores=(17.0, 17.0))

        self.assertEqual(a.rating, b.rating)
        self.assertEqual((a._ties, b._ties), (1, 1))
        self.assertEqual((a._points_for, b._points_for), (17.0, 17.0))

    def test_omitted_scores_fall_back_to_unit_scores(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b)
        self.assertEqual((a._points_for, a._points_against), (1.0, 0.0))

        c, d = KeenerCompetitor(), KeenerCompetitor()
        c.tied(d)
        self.assertEqual((c._points_for, d._points_for), (0.5, 0.5))

    def test_unit_scores_reproduce_the_omitted_score_path(self):
        plain = KeenerCompetitor(), KeenerCompetitor()
        plain[0].beat(plain[1])
        plain[0].tied(plain[1])

        scored = KeenerCompetitor(), KeenerCompetitor()
        scored[0].beat(scored[1], scores=(1.0, 0.0))
        scored[0].tied(scored[1], scores=(0.5, 0.5))

        self.assertEqual(plain[0].rating, scored[0].rating)
        self.assertEqual(plain[1].rating, scored[1].rating)

    def test_margins_matter(self):
        """A blowout and a one-point win over the same schedule must not fit the same ratings."""
        blowout = _play([(0, 1, 50, 0), (1, 2, 3, 2)])
        narrow = _play([(0, 1, 3, 2), (1, 2, 3, 2)])
        self.assertNotAlmostEqual(blowout[0].rating, narrow[0].rating, places=6)

    def test_ratings_are_positive_and_average_one(self):
        competitors = _play([(0, 1, 40, 3), (1, 2, 21, 17), (0, 2, 35, 0)])
        for competitor in competitors:
            self.assertGreater(competitor.rating, 0.0)
        self.assertAlmostEqual(sum(c.rating for c in competitors) / len(competitors), 1.0, places=9)

    def test_disconnected_groups_are_rated_independently(self):
        a, b, c, d = (KeenerCompetitor() for _ in range(4))
        a.beat(b, scores=(30.0, 10.0))
        first_pair = (a.rating, b.rating)

        c.beat(d, scores=(4.0, 3.0))
        # The second group shares no opponent with the first, so the first is untouched.
        self.assertEqual((a.rating, b.rating), first_pair)
        self.assertNotEqual(c.rating, a.rating)

    def test_reset_restores_a_fresh_competitor(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(28.0, 7.0))
        a.reset()

        fresh = KeenerCompetitor()
        self.assertEqual(a.rating, fresh.rating)
        self.assertEqual(a.num_games, 0)
        self.assertEqual((a._points_for, a._points_against), (0.0, 0.0))
        self.assertEqual(a._scores_for, {})


class TestKeenerOrderIndependence(unittest.TestCase):
    """The fit must depend on the set of results, not the order they arrived in.

    The fixture below was selected programmatically rather than by hand: a search over
    random scored schedules kept only one whose RAW solver output differs between the two
    replay orders (by 5.6e-16 here) while the canonicalized output agrees. A hand-picked
    schedule usually solves bit-identically in both orders, so the test would pass with the
    canonicalization deleted and guard nothing.
    """

    SCHEDULE = [(2, 1, 28, 32), (2, 0, 11, 32), (1, 0, 6, 28), (1, 0, 5, 34)]
    REPLAY = [(1, 0, 5, 34), (2, 0, 11, 32), (1, 0, 6, 28), (2, 1, 28, 32)]

    def test_replaying_in_a_different_order_gives_identical_ratings(self):
        first = [c.rating for c in _play(self.SCHEDULE)]
        second = [c.rating for c in _play(self.REPLAY)]
        self.assertEqual(first, second)

    def test_the_fixture_is_order_sensitive_without_canonicalization(self):
        """Guards the guard: with rounding effectively off, the two orders must disagree."""
        original = KeenerCompetitor._round_decimals
        try:
            KeenerCompetitor._round_decimals = 17
            first = [c.rating for c in _play(self.SCHEDULE)]
            second = [c.rating for c in _play(self.REPLAY)]
            self.assertNotEqual(first, second)
        finally:
            KeenerCompetitor._round_decimals = original


class TestKeenerSerialization(unittest.TestCase):
    def test_state_round_trip_restores_ratings_and_score_aggregates(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(35.0, 3.0))
        a.tied(b, scores=(10.0, 10.0))

        restored = KeenerCompetitor.from_state(a.export_state())

        self.assertEqual(restored.rating, a.rating)
        self.assertEqual((restored._wins, restored._losses, restored._ties), (1, 0, 1))
        self.assertEqual(restored._points_for, 45.0)
        self.assertEqual(restored._points_against, 13.0)

    def test_json_round_trip(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(21.0, 14.0))

        restored = KeenerCompetitor.from_json(a.to_json())
        self.assertEqual(restored.rating, a.rating)
        self.assertEqual(restored._points_for, 21.0)

    def test_exported_state_is_json_serializable(self):
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(21.0, 14.0))
        self.assertIsInstance(json.dumps(a.export_state()), str)

    def test_match_graph_is_not_restored(self):
        """Documented limitation, shared with Colley, Massey and Bradley-Terry.

        The opponent graph holds live object references, so it cannot be serialized. A
        restored competitor keeps its rating and score totals but has no opponents, so
        continued play re-fits it against only the new results.
        """
        a, b = KeenerCompetitor(), KeenerCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        restored_a = KeenerCompetitor.from_state(a.export_state())
        restored_b = KeenerCompetitor.from_state(b.export_state())
        self.assertEqual(restored_a._scores_for, {})
        self.assertEqual(restored_a._opponents, {})

        # Continuing play on the restored pair does not reproduce the original pair's fit.
        restored_a.beat(restored_b, scores=(35.0, 3.0))
        a.beat(b, scores=(35.0, 3.0))
        self.assertNotAlmostEqual(restored_a.rating, a.rating, places=6)


class TestKeenerArena(unittest.TestCase):
    def test_arena_round_trip_preserves_the_leaderboard(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=KeenerCompetitor)
        arena.matchup("a", "b", outcome=1.0, scores=(28.0, 7.0))
        arena.matchup("b", "c", outcome=1.0, scores=(14.0, 7.0))
        arena.matchup("a", "c", outcome=0.0, scores=(3.0, 35.0))

        restored = LambdaArena(
            lambda a, b: True,
            base_competitor=KeenerCompetitor,
            initial_state=json.loads(json.dumps(arena.export_state())),
        )
        self.assertEqual(restored.leaderboard(), arena.leaderboard())

    def test_arena_forwards_scores_in_competitor_order(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=KeenerCompetitor)
        arena.matchup("a", "b", outcome=0.0, scores=(3.0, 35.0))

        self.assertEqual(arena.competitors["b"]._points_for, 35.0)
        self.assertEqual(arena.competitors["a"]._points_for, 3.0)
        self.assertEqual(arena.competitors["b"]._wins, 1)

    def test_arena_without_scores_uses_unit_fallback(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=KeenerCompetitor)
        arena.matchup("a", "b")
        self.assertEqual(arena.competitors["a"]._points_for, 1.0)
        self.assertEqual(len(arena.history.bouts), 1)


if __name__ == "__main__":
    unittest.main()
