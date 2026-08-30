"""Behavioural tests for GlickoBoostCompetitor.

The numeric checks against Glickman's published tables live in
``test_GlickoBoostCompetitor_known_values.py``; this module covers the interface: the
period override, the colour convention, the probability contract, serialization and the
arena integration.
"""

import datetime
import json
import unittest
from unittest.mock import patch

from elote import EloCompetitor, GlickoBoostCompetitor, LambdaArena
from elote.competitors.base import (
    BaseCompetitor,
    InvalidParameterException,
    InvalidRatingValueException,
    MissMatchedCompetitorTypesException,
)


PERIOD_END = datetime.datetime(2026, 1, 31, tzinfo=datetime.timezone.utc)
NEXT_PERIOD_END = datetime.datetime(2026, 3, 2, tzinfo=datetime.timezone.utc)


def _population(names="ABCD", **kwargs):
    return {name: GlickoBoostCompetitor(**kwargs) for name in names}


def _rows(population, schedule):
    return [(population[a], population[b], outcome, None) for a, b, outcome in schedule]


SCHEDULE = (("A", "B", 1.0), ("B", "C", 0.5), ("C", "D", 0.0), ("A", "D", 1.0), ("B", "D", 1.0))


class TestPeriodOverride(unittest.TestCase):
    """The period math has to live in apply_rating_period, and be reached once per period."""

    def test_apply_rating_period_overrides_the_base_implementation(self):
        self.assertIsNot(
            GlickoBoostCompetitor.apply_rating_period.__func__,
            BaseCompetitor.apply_rating_period.__func__,
        )

    def test_a_multi_game_period_reaches_the_override_once_and_makes_no_pairwise_call(self):
        population = _population()
        rows = _rows(population, SCHEDULE)
        original = GlickoBoostCompetitor.apply_rating_period
        calls = []

        def spy(results, *, period_end=None):
            calls.append(list(results))
            return original(results, period_end=period_end)

        with (
            patch.object(GlickoBoostCompetitor, "apply_rating_period", staticmethod(spy)),
            patch.object(GlickoBoostCompetitor, "beat", autospec=True) as beat,
            patch.object(GlickoBoostCompetitor, "lost_to", autospec=True) as lost_to,
            patch.object(GlickoBoostCompetitor, "tied", autospec=True) as tied,
        ):
            GlickoBoostCompetitor.apply_rating_period(rows, period_end=PERIOD_END)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), len(SCHEDULE))
        beat.assert_not_called()
        lost_to.assert_not_called()
        tied.assert_not_called()
        self.assertNotEqual(population["A"].rating, population["D"].rating)

    def test_arena_rating_period_reaches_the_override_once(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=GlickoBoostCompetitor)
        rows = [("A", "B", 1.0, None), ("B", "C", 0.5, None)]

        with patch.object(
            GlickoBoostCompetitor,
            "apply_rating_period",
            wraps=GlickoBoostCompetitor.apply_rating_period,
        ) as apply:
            arena.rating_period(rows, period_end=PERIOD_END)

        apply.assert_called_once()
        self.assertEqual(len(apply.call_args.args[0]), len(rows))
        self.assertGreater(arena.competitors["A"].rating, arena.competitors["B"].rating)

    def test_a_period_is_not_the_same_as_replaying_it_pairwise(self):
        """The two-pass structure means a period is genuinely not a stream of pairwise updates."""
        period_population = _population()
        pairwise_population = _population()

        GlickoBoostCompetitor.apply_rating_period(_rows(period_population, SCHEDULE), period_end=PERIOD_END)
        for a, b, outcome in SCHEDULE:
            first, second = pairwise_population[a], pairwise_population[b]
            if outcome == 1.0:
                first.beat(second, match_time=PERIOD_END)
            elif outcome == 0.0:
                first.lost_to(second, match_time=PERIOD_END)
            else:
                first.tied(second, match_time=PERIOD_END)

        self.assertNotAlmostEqual(period_population["A"].rating, pairwise_population["A"].rating, places=3)

    def test_period_validation_rejects_bad_rows(self):
        population = _population()
        with self.assertRaisesRegex(ValueError, "outcome must be one of"):
            GlickoBoostCompetitor.apply_rating_period([(population["A"], population["B"], 0.25, None)])
        with self.assertRaisesRegex(ValueError, "do not describe a win"):
            GlickoBoostCompetitor.apply_rating_period([(population["A"], population["B"], 1.0, (0, 1))])
        with self.assertRaises(MissMatchedCompetitorTypesException):
            GlickoBoostCompetitor.apply_rating_period([(population["A"], EloCompetitor(), 1.0, None)])
        self.assertEqual(population["A"].rating, 1500)


class TestColourConvention(unittest.TestCase):
    """Argument order is the colour channel: the first competitor in a row played white."""

    def setUp(self):
        self._original_eta = GlickoBoostCompetitor._eta

    def tearDown(self):
        GlickoBoostCompetitor.configure_class(eta=self._original_eta)

    @staticmethod
    def _ratings_for(schedule):
        population = _population()
        GlickoBoostCompetitor.apply_rating_period(_rows(population, schedule), period_end=PERIOD_END)
        return {name: competitor.rating for name, competitor in population.items()}

    @staticmethod
    def _swapped(schedule):
        return tuple((b, a, 1.0 - outcome) for a, b, outcome in schedule)

    def test_swapping_colours_changes_ratings_with_a_white_advantage(self):
        GlickoBoostCompetitor.configure_class(eta=30.0)
        as_written = self._ratings_for(SCHEDULE)
        swapped = self._ratings_for(self._swapped(SCHEDULE))

        self.assertNotEqual(as_written, swapped)
        for name in as_written:
            self.assertNotAlmostEqual(as_written[name], swapped[name], places=3)

    def test_swapping_colours_changes_nothing_without_a_white_advantage(self):
        GlickoBoostCompetitor.configure_class(eta=0.0)
        as_written = self._ratings_for(SCHEDULE)
        swapped = self._ratings_for(self._swapped(SCHEDULE))

        for name in as_written:
            self.assertAlmostEqual(as_written[name], swapped[name], places=9)

    def test_eta_defaults_to_no_colour_information(self):
        self.assertEqual(GlickoBoostCompetitor._eta, 0.0)


class TestExpectedScore(unittest.TestCase):
    """The probability contract every shipped system holds to."""

    def test_two_fresh_competitors_predict_a_coin_flip(self):
        self.assertEqual(GlickoBoostCompetitor().expected_score(GlickoBoostCompetitor()), 0.5)

    def test_expected_score_is_bounded_and_complementary_after_lopsided_results(self):
        a, b = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        for step in range(8):
            a.beat(b)
            for first, second in ((a, b), (b, a)):
                score = first.expected_score(second)
                self.assertGreaterEqual(score, 0.0, f"step {step} produced {score}")
                self.assertLessEqual(score, 1.0, f"step {step} produced {score}")
            self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0, places=12)

    def test_expected_score_rises_with_the_rating_gap(self):
        base = GlickoBoostCompetitor(initial_rating=1500)
        scores = [base.expected_score(GlickoBoostCompetitor(initial_rating=rating)) for rating in (1700, 1500, 1300)]
        self.assertEqual(scores, sorted(scores))

    def test_expected_score_rejects_another_rating_system(self):
        with self.assertRaises(MissMatchedCompetitorTypesException):
            GlickoBoostCompetitor().expected_score(EloCompetitor())


class TestPairwiseFallback(unittest.TestCase):
    """beat/lost_to/tied are one-game rating periods of the same algorithm."""

    def test_beat_moves_both_competitors(self):
        winner, loser = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        winner.beat(loser)

        self.assertGreater(winner.rating, 1500)
        self.assertLess(loser.rating, 1500)
        self.assertLess(winner.rd, 250)
        self.assertLess(loser.rd, 250)

    def test_lost_to_is_the_mirror_of_beat(self):
        winner, loser = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        other_winner, other_loser = GlickoBoostCompetitor(), GlickoBoostCompetitor()

        winner.beat(loser)
        other_loser.lost_to(other_winner)

        self.assertAlmostEqual(other_winner.rating, winner.rating, places=9)
        self.assertAlmostEqual(other_loser.rating, loser.rating, places=9)

    def test_tied_leaves_equal_competitors_equal(self):
        a, b = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        a.tied(b)

        self.assertAlmostEqual(a.rating, b.rating, places=9)
        self.assertLess(a.rd, 250)

    def test_score_payloads_are_validated(self):
        a, b = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        with self.assertRaisesRegex(ValueError, "do not describe a win"):
            a.beat(b, scores=(1, 2))
        with self.assertRaisesRegex(ValueError, "do not describe a draw"):
            a.tied(b, scores=(1, 2))
        a.beat(b, scores=(2, 1))
        self.assertGreater(a.rating, b.rating)


class TestRatingDeviationOverTime(unittest.TestCase):
    """Step 6: the RD grows again for the periods a competitor sits out."""

    def test_a_missed_period_inflates_the_rd_before_the_next_update(self):
        a, b = GlickoBoostCompetitor(initial_rd=100), GlickoBoostCompetitor(initial_rd=100)
        a.beat(b, match_time=PERIOD_END)
        rating_before, rd_before = a.rating, a.rd

        expected_periods = (NEXT_PERIOD_END - PERIOD_END).total_seconds() / (24 * 3600)
        expected_periods /= GlickoBoostCompetitor._rating_period_days
        expected_rd = GlickoBoostCompetitor._inflated_rd(rating_before, rd_before, expected_periods)

        a._advance_to(NEXT_PERIOD_END)
        self.assertAlmostEqual(a.rd, expected_rd, places=9)
        self.assertGreater(a.rd, rd_before)

    def test_results_inside_one_period_do_not_inflate_the_rd(self):
        a, b = GlickoBoostCompetitor(initial_rd=100), GlickoBoostCompetitor(initial_rd=100)
        a.beat(b, match_time=PERIOD_END)
        rd_after_first = a.rd

        a._advance_to(PERIOD_END)
        self.assertEqual(a.rd, rd_after_first)

        a.beat(b, match_time=PERIOD_END)
        self.assertLess(a.rd, rd_after_first)

    def test_rd_never_exceeds_the_unrated_cap(self):
        competitor = GlickoBoostCompetitor(initial_rd=249)
        competitor._last_activity = PERIOD_END
        competitor._advance_to(PERIOD_END + datetime.timedelta(days=3650))

        self.assertLessEqual(competitor.rd, GlickoBoostCompetitor._rd_unrated)

    def test_a_period_before_the_last_activity_is_rejected(self):
        a, b = GlickoBoostCompetitor(), GlickoBoostCompetitor()
        a.beat(b, match_time=NEXT_PERIOD_END)
        with self.assertRaises(InvalidParameterException):
            a.beat(b, match_time=PERIOD_END)


class TestStateRoundTrip(unittest.TestCase):
    """Serialization has to carry rating, RD and last activity."""

    def _trained(self):
        a, b = GlickoBoostCompetitor(initial_rating=1600, initial_rd=200), GlickoBoostCompetitor()
        a.beat(b, match_time=PERIOD_END)
        return a

    def test_export_state_round_trip(self):
        competitor = self._trained()
        restored = GlickoBoostCompetitor.from_state(competitor.export_state())

        self.assertEqual(restored.rating, competitor.rating)
        self.assertEqual(restored.rd, competitor.rd)
        self.assertEqual(restored._last_activity, competitor._last_activity)
        self.assertEqual(restored._initial_rating, 1600)
        self.assertEqual(restored._initial_rd, 200)

    def test_json_round_trip_preserves_expected_score_exactly(self):
        competitor = self._trained()
        opponent = GlickoBoostCompetitor()
        restored = GlickoBoostCompetitor.from_json(json.dumps(json.loads(competitor.to_json())))

        self.assertEqual(restored.expected_score(opponent), competitor.expected_score(opponent))

    def test_registry_resolves_the_class_by_name(self):
        self.assertIn("GlickoBoostCompetitor", BaseCompetitor.list_competitor_types())
        self.assertIs(BaseCompetitor.get_competitor_class("GlickoBoostCompetitor"), GlickoBoostCompetitor)

    def test_reset_restores_the_initial_state(self):
        competitor = self._trained()
        competitor.reset()

        self.assertEqual(competitor.rating, 1600)
        self.assertEqual(competitor.rd, 200)
        self.assertIsNone(competitor._last_activity)

    def test_invalid_construction_is_rejected(self):
        with self.assertRaises(InvalidRatingValueException):
            GlickoBoostCompetitor(initial_rating=1)
        with self.assertRaises(InvalidParameterException):
            GlickoBoostCompetitor(initial_rd=0)


class TestArenaIntegration(unittest.TestCase):
    """The arena has to be able to drive and restore a Glicko-Boost population."""

    def test_arena_round_trip_keeps_the_leaderboard(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=GlickoBoostCompetitor)
        arena.rating_period([("A", "B", 1.0, None), ("B", "C", 0.5, None)], period_end=PERIOD_END)
        before = arena.leaderboard()

        restored = LambdaArena(
            lambda a, b: True,
            base_competitor=GlickoBoostCompetitor,
            initial_state=arena.export_state(),
        )

        self.assertEqual(restored.leaderboard(), before)

    def test_streaming_matchups_train_a_population(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=GlickoBoostCompetitor)
        for _ in range(5):
            arena.matchup("A", "B", outcome=1.0)

        self.assertGreater(arena.competitors["A"].rating, arena.competitors["B"].rating)
        self.assertEqual(len(arena.history.bouts), 5)


if __name__ == "__main__":
    unittest.main()
