"""Edge tests for LambdaArena and evaluation utilities.

Covers the warmup/outcome validation, callback semantics, history
reprocessing, leaderboard/state surfaces, and evaluation parameter/grouping
helpers flagged by the mutation campaign.
"""

import unittest
from datetime import datetime

import elote.evaluation as evaluation
from elote.arenas.lambda_arena import LambdaArena
from elote.competitors.base import InvalidParameterException
from elote.competitors.elo import EloCompetitor


def always_a_wins(a, b):
    """Comparison callable: receives the two bout ids, True = a wins."""
    return True


class LambdaArenaConstructionTest(unittest.TestCase):
    def test_noninteger_warmup_rejected(self):
        """warmup must be an integer count of early bouts to discard."""
        with self.assertRaises(TypeError):
            LambdaArena(always_a_wins, base_competitor=EloCompetitor, warmup=1.5)
        with self.assertRaises(TypeError):
            LambdaArena(always_a_wins, base_competitor=EloCompetitor, warmup=-0.1)


class LambdaArenaTournamentTest(unittest.TestCase):
    def test_callback_decision_drives_update(self):
        """A True callback verdict must apply the win to the first id."""
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])

        ratings = {row["competitor"]: row["rating"] for row in arena.leaderboard()}
        self.assertAlmostEqual(ratings["x"], 416.0, places=6)
        self.assertAlmostEqual(ratings["y"], 384.0, places=6)

    def test_explicit_outcome_bypasses_callback(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.matchup("a", "b", outcome=0.0)

        ratings = {row["competitor"]: row["rating"] for row in arena.leaderboard()}
        self.assertAlmostEqual(ratings["a"], 384.0, places=6)
        self.assertAlmostEqual(ratings["b"], 416.0, places=6)

    def test_explicit_draw_bisects_ratings(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.matchup("a", "b", outcome=0.5)

        ratings = {row["competitor"]: row["rating"] for row in arena.leaderboard()}
        self.assertAlmostEqual(ratings["a"], 400.0, places=6)
        self.assertAlmostEqual(ratings["b"], 400.0, places=6)

    def test_invalid_outcome_rejected_without_state_change(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        with self.assertRaises(ValueError):
            arena.matchup("a", "b", outcome=0.7)
        with self.assertRaises(ValueError):
            arena.matchup("a", "b", scores=(1.0, 0.0))  # scores need outcome

        self.assertEqual(arena.leaderboard(), [])


class LambdaArenaHistoryTest(unittest.TestCase):
    def test_process_history_handles_draws_and_none(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.process_history([("a", "b", 0.5), ("c", "d", None), ("a", "b", 1.0)], progress_bar=False)

        for competitor_id in ("a", "b", "c", "d"):
            self.assertIsNotNone(arena.get_competitor_by_id(competitor_id))

    def test_process_history_validates_outcomes(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        with self.assertRaises(ValueError):
            arena.process_history([("e", "f", 0.7)], progress_bar=False)

    def test_get_competitor_by_id_for_unseen_id_is_none(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])
        self.assertIsNone(arena.get_competitor_by_id("never-played"))


class LambdaArenaSurfaceTest(unittest.TestCase):
    def test_leaderboard_sorted_descending(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])

        board = arena.leaderboard()
        ratings = [row["rating"] for row in board]
        self.assertEqual(ratings, sorted(ratings, reverse=True))

    def test_expected_score_unseen_competitor_does_not_raise(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])

        probability = arena.expected_score("x", "never-played")
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_export_state_covers_every_competitor(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])

        state = arena.export_state()
        self.assertEqual(set(state), {"x", "y"})

    def test_clear_history_resets_arena(self):
        arena = LambdaArena(always_a_wins, base_competitor=EloCompetitor)
        arena.tournament([("x", "y")])
        arena.clear_history()

        # clear_history resets the bout log only; earned ratings remain.
        self.assertEqual(arena.history.bouts, [])
        self.assertEqual(len(arena.leaderboard()), 2)


class EvaluationHelpersTest(unittest.TestCase):
    def test_validate_competitor_params_accepts_known_vars(self):
        evaluation._validate_competitor_params(EloCompetitor, ["k_factor"])

    def test_validate_competitor_params_rejects_unknown_vars(self):
        with self.assertRaises(InvalidParameterException):
            evaluation._validate_competitor_params(EloCompetitor, ["k_factor", "nonexistent"])

    def test_group_by_period_buckets_by_iso_week(self):
        bouts = [
            ("a", "b", 1.0, datetime(2020, 1, 3)),  # Fri, ISO 2020-W01
            ("b", "c", 0.0, datetime(2020, 1, 5)),  # Sun, ISO 2020-W01
            ("a", "c", 1.0, datetime(2020, 1, 8)),  # Wed, ISO 2020-W02
        ]
        grouped = evaluation.group_by_period(bouts)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped[0]), 2)
        self.assertEqual(grouped[1], [("a", "c", 1.0, datetime(2020, 1, 8))])


if __name__ == "__main__":
    unittest.main()
