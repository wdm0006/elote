"""Tests for the optional score payload on the common result API.

The score channel is cross-cutting: every competitor class must accept it, malformed
payloads must be rejected before anything mutates, and the arena must forward the pair in
the right competitor order. Systems that actually consume a score (Massey and Keener) are
covered here for the plumbing and in their own known-value file for the numbers.
"""

import unittest

from elote import (
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    DWZCompetitor,
    ECFCompetitor,
    EloCompetitor,
    Glicko2Competitor,
    GlickoCompetitor,
    KeenerCompetitor,
    LambdaArena,
    MasseyCompetitor,
    TrueSkillCompetitor,
)


def _always_true(a, b, attributes=None):
    """Comparison function for arenas that are always driven by an explicit outcome."""
    return True


COMPETITOR_CLASSES = (
    EloCompetitor,
    GlickoCompetitor,
    Glicko2Competitor,
    TrueSkillCompetitor,
    ECFCompetitor,
    DWZCompetitor,
    ColleyMatrixCompetitor,
    BradleyTerryCompetitor,
    MasseyCompetitor,
    KeenerCompetitor,
)

# A win, a draw and a loss, with the unit scores each result implies.
UNIT_SCHEDULE = (
    ("beat", (1.0, 0.0)),
    ("tied", (0.5, 0.5)),
    ("lost_to", (0.0, 1.0)),
)


class TestScoreValidation(unittest.TestCase):
    """Malformed score payloads are rejected before any state changes."""

    BAD_PAYLOADS = (
        ("not a sequence", 5),
        ("string", "3-1"),
        ("too short", (3.0,)),
        ("too long", (3.0, 1.0, 0.0)),
        ("non numeric", (3.0, "1")),
        ("boolean", (True, False)),
        ("negative", (3.0, -1.0)),
        ("infinite", (float("inf"), 1.0)),
        ("nan", (float("nan"), 1.0)),
    )

    def test_malformed_scores_rejected_for_every_competitor(self):
        for competitor_class in COMPETITOR_CLASSES:
            for label, payload in self.BAD_PAYLOADS:
                with self.subTest(competitor=competitor_class.__name__, payload=label):
                    a = competitor_class()
                    b = competitor_class()
                    with self.assertRaises(ValueError):
                        a.beat(b, scores=payload)

    def test_scores_must_agree_with_the_declared_outcome(self):
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                a = competitor_class()
                b = competitor_class()
                # beat() requires the caller's score to be the larger one.
                with self.assertRaises(ValueError):
                    a.beat(b, scores=(1.0, 3.0))
                with self.assertRaises(ValueError):
                    a.beat(b, scores=(2.0, 2.0))
                # lost_to() requires the opposite.
                with self.assertRaises(ValueError):
                    a.lost_to(b, scores=(3.0, 1.0))
                # tied() requires equality.
                with self.assertRaises(ValueError):
                    a.tied(b, scores=(3.0, 1.0))

    def test_rejected_scores_leave_both_competitors_untouched(self):
        """A ValueError must fire before either side of the pair is mutated."""
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                a = competitor_class()
                b = competitor_class()
                before = (a.export_state()["state"], b.export_state()["state"])
                with self.assertRaises(ValueError):
                    a.beat(b, scores=(-1.0, -2.0))
                self.assertEqual((a.export_state()["state"], b.export_state()["state"]), before)


class TestUnitScoreEquivalence(unittest.TestCase):
    """Supplying the unit scores a result implies must reproduce the no-score behaviour."""

    def test_unit_scores_match_the_omitted_score_path(self):
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                plain_a, plain_b = competitor_class(), competitor_class()
                scored_a, scored_b = competitor_class(), competitor_class()

                for method, unit_scores in UNIT_SCHEDULE:
                    getattr(plain_a, method)(plain_b)
                    getattr(scored_a, method)(scored_b, scores=unit_scores)

                # Not exact equality: Glicko-1/Glicko-2 inflate RD from wall-clock elapsed
                # time, so the two runs differ by ~1e-10 for reasons unrelated to scores.
                self.assertAlmostEqual(plain_a.rating, scored_a.rating, places=6)
                self.assertAlmostEqual(plain_b.rating, scored_b.rating, places=6)


class TestMasseyConsumesScores(unittest.TestCase):
    """Massey is the first system to actually read the score payload."""

    def test_win_contributes_the_real_margin(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        self.assertEqual(a._point_differential, 32.0)
        self.assertEqual(b._point_differential, -32.0)
        self.assertEqual((a._wins, b._losses), (1, 1))

    def test_lost_to_reverses_the_score_pair(self):
        winner_first_a, winner_first_b = MasseyCompetitor(), MasseyCompetitor()
        winner_first_a.beat(winner_first_b, scores=(35.0, 3.0))

        loser_first_a, loser_first_b = MasseyCompetitor(), MasseyCompetitor()
        # Same game, called from the loser: scores stay in caller order.
        loser_first_b.lost_to(loser_first_a, scores=(3.0, 35.0))

        self.assertEqual(loser_first_a._point_differential, winner_first_a._point_differential)
        self.assertEqual(loser_first_b._point_differential, winner_first_b._point_differential)
        self.assertEqual(loser_first_a.rating, winner_first_a.rating)
        self.assertEqual(loser_first_b.rating, winner_first_b.rating)

    def test_equal_score_draw_contributes_no_margin_but_counts_the_game(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.tied(b, scores=(17.0, 17.0))

        self.assertEqual(a._point_differential, 0.0)
        self.assertEqual(b._point_differential, 0.0)
        self.assertEqual((a.num_games, b.num_games), (1, 1))

    def test_real_margins_change_the_fit(self):
        """A blowout and a narrow win over the same schedule must not fit the same ratings."""
        blowout = [MasseyCompetitor() for _ in range(3)]
        blowout[0].beat(blowout[1], scores=(50.0, 0.0))
        blowout[1].beat(blowout[2], scores=(3.0, 2.0))

        narrow = [MasseyCompetitor() for _ in range(3)]
        narrow[0].beat(narrow[1], scores=(3.0, 2.0))
        narrow[1].beat(narrow[2], scores=(3.0, 2.0))

        self.assertNotAlmostEqual(blowout[0].rating, narrow[0].rating, places=6)

    def test_score_derived_differential_survives_state_round_trip(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(35.0, 3.0))

        restored = MasseyCompetitor.from_state(a.export_state())
        self.assertEqual(restored._point_differential, 32.0)
        self.assertEqual(restored.rating, a.rating)
        self.assertEqual((restored._wins, restored._losses, restored._ties), (1, 0, 0))


class TestArenaScoreForwarding(unittest.TestCase):
    """LambdaArena forwards the (a, b)-ordered pair in whichever order the call needs."""

    def _arena(self):
        return LambdaArena(_always_true, base_competitor=MasseyCompetitor)

    def test_a_wins_branch(self):
        arena = self._arena()
        arena.matchup("a", "b", outcome=1.0, scores=(35.0, 3.0))
        self.assertEqual(arena.competitors["a"]._point_differential, 32.0)
        self.assertEqual(arena.competitors["b"]._point_differential, -32.0)

    def test_b_wins_branch_reverses_the_pair(self):
        arena = self._arena()
        # Scores stay in (a, b) order even though b won and the call is reversed internally.
        arena.matchup("a", "b", outcome=0.0, scores=(3.0, 35.0))
        self.assertEqual(arena.competitors["b"]._point_differential, 32.0)
        self.assertEqual(arena.competitors["a"]._point_differential, -32.0)
        self.assertEqual(arena.competitors["b"]._wins, 1)
        self.assertEqual(arena.competitors["a"]._losses, 1)

    def test_draw_branch(self):
        arena = self._arena()
        arena.matchup("a", "b", outcome=0.5, scores=(17.0, 17.0))
        self.assertEqual(arena.competitors["a"]._point_differential, 0.0)
        self.assertEqual(arena.competitors["a"]._ties, 1)
        self.assertEqual(arena.competitors["b"]._ties, 1)

    def test_bout_is_still_recorded(self):
        arena = self._arena()
        arena.matchup("a", "b", outcome=0.0, scores=(3.0, 35.0))
        self.assertEqual(len(arena.history.bouts), 1)
        self.assertEqual(arena.history.bouts[0].outcome, "loss")

    def test_scores_require_an_explicit_outcome(self):
        arena = self._arena()
        with self.assertRaises(ValueError):
            arena.matchup("a", "b", scores=(35.0, 3.0))
        self.assertEqual(arena.competitors, {})
        self.assertEqual(len(arena.history.bouts), 0)

    def test_invalid_scores_create_no_competitors_and_no_bout(self):
        for label, payload in (
            ("inconsistent with outcome", (3.0, 35.0)),
            ("negative", (35.0, -3.0)),
            ("non finite", (float("inf"), 3.0)),
            ("wrong length", (35.0,)),
        ):
            with self.subTest(payload=label):
                arena = self._arena()
                with self.assertRaises(ValueError):
                    arena.matchup("a", "b", outcome=1.0, scores=payload)
                self.assertEqual(arena.competitors, {})
                self.assertEqual(len(arena.history.bouts), 0)

    def test_tournament_carries_a_score_payload(self):
        arena = self._arena()
        arena.tournament([("a", "b", None, None, 1.0, (35.0, 3.0))])
        self.assertEqual(arena.competitors["a"]._point_differential, 32.0)

    def test_every_competitor_class_accepts_arena_scores(self):
        for competitor_class in COMPETITOR_CLASSES:
            with self.subTest(competitor=competitor_class.__name__):
                arena = LambdaArena(_always_true, base_competitor=competitor_class)
                arena.matchup("a", "b", outcome=1.0, scores=(35.0, 3.0))
                arena.matchup("a", "b", outcome=0.5, scores=(7.0, 7.0))
                arena.matchup("a", "b", outcome=0.0, scores=(3.0, 35.0))
                self.assertEqual(len(arena.history.bouts), 3)


if __name__ == "__main__":
    unittest.main()
