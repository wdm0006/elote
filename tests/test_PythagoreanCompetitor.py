"""Behavioural tests for PythagoreanCompetitor.

Numeric reference values live in tests/test_PythagoreanCompetitor_known_values.py.
"""

import json
import random
import unittest

from elote import LambdaArena, PythagoreanCompetitor
from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    MissMatchedCompetitorTypesException,
)


def _play(schedule, count=3):
    """Apply a schedule of ``(i, j, score_i, score_j)`` tuples to ``count`` fresh competitors."""
    competitors = [PythagoreanCompetitor() for _ in range(count)]
    for i, j, score_i, score_j in schedule:
        if score_i > score_j:
            competitors[i].beat(competitors[j], scores=(score_i, score_j))
        elif score_i < score_j:
            competitors[i].lost_to(competitors[j], scores=(score_i, score_j))
        else:
            competitors[i].tied(competitors[j], scores=(score_i, score_j))
    return competitors


class TestPythagoreanBasics(unittest.TestCase):
    def test_defaults(self):
        competitor = PythagoreanCompetitor()
        self.assertEqual(competitor.rating, 0.5)
        self.assertEqual(competitor.rating, PythagoreanCompetitor._default_initial_rating)
        self.assertEqual(competitor.num_games, 0)
        self.assertEqual((competitor._points_for, competitor._points_against), (0.0, 0.0))

    def test_custom_exponent(self):
        competitor = PythagoreanCompetitor(exponent=2.0)
        self.assertEqual(competitor._exponent, 2.0)
        # A fresh competitor sits at 0.5 whatever the exponent.
        self.assertEqual(competitor.rating, 0.5)

    def test_non_positive_exponent_rejected(self):
        for bad in (0.0, -1.0):
            with self.subTest(exponent=bad):
                with self.assertRaises(InvalidParameterException):
                    PythagoreanCompetitor(exponent=bad)

    def test_constructor_does_not_accept_an_initial_rating(self):
        """The rating is derived from points, so there is no starting rating to hand over.

        benchmark_competitors forces initial_rating=1500 on any incremental class that
        accepts one, which is meaningless on a [0, 1] scale; not accepting the argument is
        what keeps this class out of that path.
        """
        with self.assertRaises(TypeError):
            PythagoreanCompetitor(initial_rating=1500)

    def test_rating_cannot_be_set_directly(self):
        competitor = PythagoreanCompetitor()
        with self.assertRaises(NotImplementedError):
            competitor.rating = 0.9

    def test_type_mismatch_is_rejected(self):
        from elote import EloCompetitor

        competitor = PythagoreanCompetitor()
        with self.assertRaises(MissMatchedCompetitorTypesException):
            competitor.expected_score(EloCompetitor())
        with self.assertRaises(MissMatchedCompetitorTypesException):
            competitor.beat(EloCompetitor())

    def test_registered_for_state_reconstruction(self):
        self.assertIn("PythagoreanCompetitor", BaseCompetitor.list_competitor_types())
        self.assertIs(BaseCompetitor.get_competitor_class("PythagoreanCompetitor"), PythagoreanCompetitor)

    def test_configure_class_validates_parameters(self):
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor.configure_class(exponent=0)
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor.configure_class(prior_points=-1)


class TestPythagoreanExpectedScore(unittest.TestCase):
    def test_two_fresh_competitors_predict_one_half(self):
        self.assertEqual(PythagoreanCompetitor().expected_score(PythagoreanCompetitor()), 0.5)

    def test_predictions_are_exactly_complementary(self):
        a, b, c = _play([(0, 1, 40, 3), (1, 2, 21, 17), (0, 2, 35, 0)])
        for first, second in ((a, b), (b, a), (a, c), (c, b)):
            with self.subTest(pair=(first.rating, second.rating)):
                self.assertEqual(first.expected_score(second) + second.expected_score(first), 1.0)

    def test_complementarity_holds_for_a_fixture_that_defeats_the_naive_form(self):
        """Guards the derivation of the reverse direction, not just log5 itself.

        The fixture was selected programmatically: evaluating log5 independently in each
        direction, instead of deriving one from the other, loses exact complementarity for
        about 13% of random point-total pairs (26,894 of 200,000 sampled), and none of the
        hand-written fixtures in this class happen to be among them.
        """
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        foils = [PythagoreanCompetitor() for _ in range(3)]
        a.beat(foils[0], scores=(65.0, 60.0))
        a.beat(foils[1], scores=(64.0, 61.0))
        b.lost_to(foils[2], scores=(73.0, 389.0))

        self.assertEqual((a._points_for, a._points_against), (129.0, 121.0))
        self.assertEqual((b._points_for, b._points_against), (73.0, 389.0))
        self.assertEqual(a.expected_score(b) + b.expected_score(a), 1.0)

    def test_equal_records_predict_exactly_one_half(self):
        """Two competitors with identical totals are an even matchup, exactly."""
        a, b, c, d = (PythagoreanCompetitor() for _ in range(4))
        a.beat(c, scores=(31.0, 17.0))
        b.beat(d, scores=(31.0, 17.0))
        self.assertEqual(a.rating, b.rating)
        self.assertEqual(a.expected_score(b), 0.5)
        self.assertEqual(b.expected_score(a), 0.5)

    def test_predictions_stay_in_the_unit_interval_after_lopsided_results(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        for _ in range(50):
            a.beat(b, scores=(99.0, 0.0))
            for first, second in ((a, b), (b, a)):
                score = first.expected_score(second)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
            self.assertEqual(a.expected_score(b) + b.expected_score(a), 1.0)

    def test_an_unbeaten_competitor_still_has_a_defined_prediction(self):
        """PA = 0 would make the raw rating exactly 1 and log5 a 0/0; the prior prevents it."""
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        for _ in range(10):
            a.beat(b, scores=(21.0, 0.0))

        self.assertLess(a.rating, 1.0)
        self.assertGreater(b.rating, 0.0)
        self.assertLess(a.expected_score(b), 1.0)
        self.assertGreater(a.expected_score(b), 0.99)

    def test_stronger_competitor_is_favoured(self):
        a, b, _ = _play([(0, 1, 40, 3), (1, 2, 21, 17), (0, 2, 35, 0)])
        self.assertGreater(a.rating, b.rating)
        self.assertGreater(a.expected_score(b), 0.5)


class TestPythagoreanResults(unittest.TestCase):
    def test_win_updates_both_sides(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(28.0, 7.0))

        self.assertEqual((a._points_for, a._points_against), (28.0, 7.0))
        self.assertEqual((b._points_for, b._points_against), (7.0, 28.0))
        self.assertEqual((a.num_games, b.num_games), (1, 1))
        self.assertGreater(a.rating, 0.5)
        self.assertLess(b.rating, 0.5)

    def test_lost_to_records_the_same_game_as_beat(self):
        winner_first = PythagoreanCompetitor(), PythagoreanCompetitor()
        winner_first[0].beat(winner_first[1], scores=(28.0, 7.0))

        loser_first = PythagoreanCompetitor(), PythagoreanCompetitor()
        loser_first[1].lost_to(loser_first[0], scores=(7.0, 28.0))

        self.assertEqual(loser_first[0].rating, winner_first[0].rating)
        self.assertEqual(loser_first[1].rating, winner_first[1].rating)
        self.assertEqual(loser_first[0]._points_for, 28.0)
        self.assertEqual(loser_first[1]._points_for, 7.0)

    def test_draw_is_symmetric(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.tied(b, scores=(17.0, 17.0))

        self.assertEqual(a.rating, b.rating)
        self.assertEqual(a.rating, 0.5)
        self.assertEqual((a.num_games, b.num_games), (1, 1))

    def test_omitted_scores_fall_back_to_unit_scores(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b)
        self.assertEqual((a._points_for, a._points_against), (1.0, 0.0))

        c, d = PythagoreanCompetitor(), PythagoreanCompetitor()
        c.tied(d)
        self.assertEqual((c._points_for, d._points_for), (0.5, 0.5))

    def test_unit_scores_reproduce_the_omitted_score_path(self):
        plain = PythagoreanCompetitor(), PythagoreanCompetitor()
        plain[0].beat(plain[1])
        plain[0].tied(plain[1])

        scored = PythagoreanCompetitor(), PythagoreanCompetitor()
        scored[0].beat(scored[1], scores=(1.0, 0.0))
        scored[0].tied(scored[1], scores=(0.5, 0.5))

        self.assertEqual(plain[0].rating, scored[0].rating)
        self.assertEqual(plain[1].rating, scored[1].rating)

    def test_real_margins_move_the_rating_in_the_documented_direction(self):
        """A bigger margin over the same result must rate higher, and a unit win sits between."""
        blowout = PythagoreanCompetitor(), PythagoreanCompetitor()
        blowout[0].beat(blowout[1], scores=(50.0, 3.0))

        narrow = PythagoreanCompetitor(), PythagoreanCompetitor()
        narrow[0].beat(narrow[1], scores=(24.0, 21.0))

        unit = PythagoreanCompetitor(), PythagoreanCompetitor()
        unit[0].beat(unit[1])

        self.assertGreater(blowout[0].rating, unit[0].rating)
        self.assertGreater(unit[0].rating, narrow[0].rating)
        self.assertGreater(narrow[0].rating, 0.5)
        # The losers mirror the winners.
        self.assertLess(blowout[1].rating, unit[1].rating)
        self.assertLess(unit[1].rating, narrow[1].rating)

    def test_points_conceded_lower_the_rating(self):
        """Two runs with identical points scored are separated only by points allowed."""
        stingy = _play([(0, 1, 30, 10), (0, 2, 30, 10)])
        leaky = _play([(0, 1, 30, 28), (0, 2, 30, 28)])

        self.assertEqual(stingy[0]._points_for, leaky[0]._points_for)
        self.assertGreater(stingy[0].rating, leaky[0].rating)

    def test_the_opponent_does_not_affect_the_rating(self):
        """The defining limitation: no strength-of-schedule adjustment at all."""
        weak_opponent = PythagoreanCompetitor(), PythagoreanCompetitor()
        weak_opponent[0].beat(weak_opponent[1], scores=(28.0, 7.0))

        strong_opponent = [PythagoreanCompetitor() for _ in range(3)]
        strong_opponent[1].beat(strong_opponent[2], scores=(70.0, 0.0))
        strong_opponent[0].beat(strong_opponent[1], scores=(28.0, 7.0))

        self.assertEqual(weak_opponent[0].rating, strong_opponent[0].rating)

    def test_exponent_controls_how_sharply_points_translate_to_rating(self):
        low = PythagoreanCompetitor(exponent=1.0), PythagoreanCompetitor(exponent=1.0)
        low[0].beat(low[1], scores=(28.0, 14.0))

        high = PythagoreanCompetitor(exponent=8.0), PythagoreanCompetitor(exponent=8.0)
        high[0].beat(high[1], scores=(28.0, 14.0))

        self.assertGreater(high[0].rating, low[0].rating)

    def test_reset_restores_a_fresh_competitor(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(28.0, 7.0))
        a.reset()

        self.assertEqual(a.rating, PythagoreanCompetitor().rating)
        self.assertEqual(a.num_games, 0)
        self.assertEqual((a._points_for, a._points_against), (0.0, 0.0))


class TestPythagoreanInArena(unittest.TestCase):
    def test_long_arena_run_keeps_every_rating_in_the_unit_interval(self):
        """A long run must not raise and must not push any rating outside [0, 1].

        Sustained lopsided play is what drives the accumulators far apart, which is exactly
        where a saturating rating or an undefined log5 would show up.
        """
        arena = LambdaArena(lambda a, b: True, base_competitor=PythagoreanCompetitor)
        rng = random.Random(1979)
        names = ["a", "b", "c", "d", "e", "f"]
        strength = {name: 14 + 4 * index for index, name in enumerate(names)}

        for _ in range(2500):
            first, second = rng.sample(names, 2)
            first_score = max(0, int(rng.gauss(strength[first], 7)))
            second_score = max(0, int(rng.gauss(strength[second], 7)))
            if first_score > second_score:
                outcome = 1.0
            elif first_score < second_score:
                outcome = 0.0
            else:
                outcome = 0.5
            arena.matchup(first, second, outcome=outcome, scores=(first_score, second_score))

            for competitor in arena.competitors.values():
                self.assertGreaterEqual(competitor.rating, 0.0)
                self.assertLessEqual(competitor.rating, 1.0)

        self.assertEqual(len(arena.competitors), len(names))
        self.assertEqual(len(arena.history.bouts), 2500)
        # The seeded strengths are ordered, so the leaderboard should broadly recover them.
        leaderboard = [entry["competitor"] for entry in arena.leaderboard()]
        self.assertEqual(leaderboard[0], "f")
        self.assertEqual(leaderboard[-1], "a")

    def test_arena_forwards_scores_in_caller_order(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=PythagoreanCompetitor)
        arena.matchup("a", "b", outcome=0.0, scores=(3.0, 35.0))

        self.assertEqual(arena.competitors["a"]._points_for, 3.0)
        self.assertEqual(arena.competitors["b"]._points_for, 35.0)


class TestPythagoreanSerialization(unittest.TestCase):
    def test_state_round_trip_restores_the_rating_and_the_totals(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(35.0, 3.0))
        a.tied(b, scores=(10.0, 10.0))

        restored = PythagoreanCompetitor.from_state(a.export_state())

        self.assertEqual(restored.rating, a.rating)
        self.assertEqual(restored._points_for, 45.0)
        self.assertEqual(restored._points_against, 13.0)
        self.assertEqual(restored.num_games, 2)

    def test_json_round_trip(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(21.0, 14.0))

        restored = PythagoreanCompetitor.from_json(a.to_json())
        self.assertEqual(restored.rating, a.rating)
        self.assertEqual(restored._points_for, 21.0)

    def test_exported_state_is_json_serializable(self):
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(21.0, 14.0))
        self.assertIsInstance(json.dumps(a.export_state()), str)

    def test_a_custom_exponent_survives_the_round_trip(self):
        a = PythagoreanCompetitor(exponent=2.0)
        b = PythagoreanCompetitor(exponent=2.0)
        a.beat(b, scores=(28.0, 14.0))

        restored = PythagoreanCompetitor.from_state(a.export_state())
        self.assertEqual(restored._exponent, 2.0)
        self.assertEqual(restored.rating, a.rating)

    def test_continued_play_after_a_restore_matches_the_original(self):
        """There is no opponent graph to lose, so a restore is exact, including further play."""
        a, b = PythagoreanCompetitor(), PythagoreanCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        restored_a = PythagoreanCompetitor.from_state(json.loads(a.to_json()))
        restored_b = PythagoreanCompetitor.from_state(json.loads(b.to_json()))

        restored_a.beat(restored_b, scores=(21.0, 14.0))
        a.beat(b, scores=(21.0, 14.0))

        self.assertEqual(restored_a.rating, a.rating)
        self.assertEqual(restored_b.rating, b.rating)

    def test_negative_totals_are_rejected_on_import(self):
        state = PythagoreanCompetitor().export_state()
        state["state"]["points_for"] = -1.0
        with self.assertRaises(InvalidParameterException):
            PythagoreanCompetitor.from_state(state)


if __name__ == "__main__":
    unittest.main()
