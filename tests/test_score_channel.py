"""Tests for the optional score payload on the common result API.

The score channel is cross-cutting: every competitor class must accept it, malformed
payloads must be rejected before anything mutates, and the arena must forward the pair in
the right competitor order. Systems that actually consume a score (Massey and Keener) are
covered here for the plumbing and in their own known-value file for the numbers.
"""

import datetime
import random
import unittest

from elote import (
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    DataSplit,
    DWZCompetitor,
    ECFCompetitor,
    EloCompetitor,
    Glicko2Competitor,
    GlickoCompetitor,
    KeenerCompetitor,
    LambdaArena,
    MasseyCompetitor,
    PythagoreanCompetitor,
    WholeHistoryRatingCompetitor,
    TrueSkillCompetitor,
    benchmark_competitors,
    train_arena_with_dataset,
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
    PythagoreanCompetitor,
    WholeHistoryRatingCompetitor,
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


_SEASON_START = datetime.datetime(2024, 9, 1)

# A hand-built schedule covering all three branches of the training helper: rows where a
# won, rows where b won, and a drawn row. Scores live under the CollegeFootballDataset keys.
SCORED_ROWS = [
    ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": 31, "away_score": 17}),
    ("Bengals", "Steelers", 0.0, _SEASON_START + datetime.timedelta(days=1), {"home_score": 3, "away_score": 45}),
    ("Ravens", "Steelers", 0.0, _SEASON_START + datetime.timedelta(days=2), {"home_score": 10, "away_score": 24}),
    ("Steelers", "Browns", 1.0, _SEASON_START + datetime.timedelta(days=3), {"home_score": 28, "away_score": 7}),
    ("Browns", "Ravens", 0.5, _SEASON_START + datetime.timedelta(days=4), {"home_score": 20, "away_score": 20}),
]

SCORE_KEYS = ("home_score", "away_score")


def _leaderboard(arena):
    return {entry["competitor"]: entry["rating"] for entry in arena.leaderboard()}


def _trained(competitor_class, rows, **kwargs):
    arena = LambdaArena(_always_true, base_competitor=competitor_class)
    train_arena_with_dataset(arena, rows, **kwargs)
    return arena


def _scored_split(num_matchups=180, test_ratio=0.3, seed=11):
    """Build a DataSplit whose attributes carry real, outcome-consistent point scores."""
    rng = random.Random(seed)
    names = [f"team_{i}" for i in range(8)]
    strengths = {name: rng.uniform(-10.0, 10.0) for name in names}

    rows = []
    for i in range(num_matchups):
        a, b = rng.sample(names, 2)
        a_score = max(0, round(24 + strengths[a] - strengths[b] + rng.gauss(0, 6)))
        b_score = max(0, round(24 + strengths[b] - strengths[a] + rng.gauss(0, 6)))
        outcome = 1.0 if a_score > b_score else (0.0 if a_score < b_score else 0.5)
        rows.append(
            (a, b, outcome, _SEASON_START + datetime.timedelta(days=i), {"home_score": a_score, "away_score": b_score})
        )

    split_at = int(len(rows) * (1 - test_ratio))
    return DataSplit(train=rows[:split_at], test=rows[split_at:])


class TestDatasetScoreForwarding(unittest.TestCase):
    """train_arena_with_dataset only reaches the margin path when score_keys is supplied."""

    def test_score_keys_reproduce_hand_driven_matchups(self):
        """Training with score_keys must equal driving the same games through the arena."""
        by_helper = _trained(MasseyCompetitor, SCORED_ROWS, score_keys=SCORE_KEYS)

        by_hand = LambdaArena(_always_true, base_competitor=MasseyCompetitor)
        by_hand.matchup("Ravens", "Bengals", attributes=SCORED_ROWS[0][4], outcome=1.0, scores=(31, 17))
        by_hand.matchup("Steelers", "Bengals", attributes=SCORED_ROWS[1][4], outcome=1.0, scores=(45, 3))
        by_hand.matchup("Steelers", "Ravens", attributes=SCORED_ROWS[2][4], outcome=1.0, scores=(24, 10))
        by_hand.matchup("Steelers", "Browns", attributes=SCORED_ROWS[3][4], outcome=1.0, scores=(28, 7))
        by_hand.matchup("Browns", "Ravens", attributes=SCORED_ROWS[4][4], outcome=0.5, scores=(20, 20))

        self.assertEqual(_leaderboard(by_helper), _leaderboard(by_hand))
        # Exact values, so a change in either path is visible rather than merely "equal".
        self.assertEqual(
            _leaderboard(by_helper),
            {"Steelers": 19.25, "Ravens": 0.0, "Browns": -0.875, "Bengals": -18.375},
        )

    def test_score_keys_change_the_massey_fit(self):
        """The same rows without score_keys stay on unit margins, so the fit differs."""
        self.assertEqual(
            _leaderboard(_trained(MasseyCompetitor, SCORED_ROWS)),
            {"Steelers": 0.75, "Ravens": 0.0, "Browns": -0.125, "Bengals": -0.625},
        )

    def test_default_reproduces_the_current_output_for_keener(self):
        """Pinned literals for both paths of the other score-consuming system."""
        self.assertEqual(
            _leaderboard(_trained(KeenerCompetitor, SCORED_ROWS)),
            {"Steelers": 1.5088623362, "Ravens": 0.9649085365, "Browns": 0.923455846, "Bengals": 0.6027732813},
        )
        self.assertEqual(
            _leaderboard(_trained(KeenerCompetitor, SCORED_ROWS, score_keys=SCORE_KEYS)),
            {"Steelers": 1.7295652954, "Ravens": 0.9585153377, "Browns": 0.9169460231, "Bengals": 0.3949733438},
        )

    def test_scores_are_inert_for_a_system_that_ignores_them(self):
        """Elo validates the payload and ignores it, so its ratings must not move."""
        self.assertEqual(
            _leaderboard(_trained(EloCompetitor, SCORED_ROWS, score_keys=SCORE_KEYS)),
            _leaderboard(_trained(EloCompetitor, SCORED_ROWS)),
        )

    def test_bout_shape_is_unchanged(self):
        """Forwarding scores must not alter the recorded bouts."""
        with_scores = _trained(MasseyCompetitor, SCORED_ROWS, score_keys=SCORE_KEYS)
        without = _trained(MasseyCompetitor, SCORED_ROWS)

        self.assertEqual(
            [(bout.a, bout.b, bout.outcome) for bout in with_scores.history.bouts],
            [(bout.a, bout.b, bout.outcome) for bout in without.history.bouts],
        )

    def test_b_wins_rows_reverse_the_score_pair(self):
        """A 45-0 result must land identically however the row expresses it.

        The b-wins branch calls ``matchup(b, a, ...)``, so the pair has to be reversed with
        it. This case is deliberately not symmetric under swapping the two competitors.
        """
        as_a_loss = _trained(
            MasseyCompetitor,
            [("Browns", "Steelers", 0.0, _SEASON_START, {"home_score": 0, "away_score": 45})],
            score_keys=SCORE_KEYS,
        )
        as_b_win = _trained(
            MasseyCompetitor,
            [("Steelers", "Browns", 1.0, _SEASON_START, {"home_score": 45, "away_score": 0})],
            score_keys=SCORE_KEYS,
        )

        self.assertEqual(_leaderboard(as_a_loss), {"Steelers": 22.5, "Browns": -22.5})
        self.assertEqual(_leaderboard(as_a_loss), _leaderboard(as_b_win))

    def test_rows_without_usable_scores_train_without_them(self):
        """Real feeds have gaps, so a row short of two numbers is trained score-free."""
        gappy = [
            ("Ravens", "Bengals", 1.0, _SEASON_START, None),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": 31}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"away_score": 17}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": 31, "away_score": None}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": None, "away_score": None}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": "31", "away_score": "17"}),
            ("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": 31, "away_score": float("nan")}),
        ]

        with_keys = _trained(MasseyCompetitor, gappy, score_keys=SCORE_KEYS)
        without_keys = _trained(MasseyCompetitor, gappy)

        self.assertEqual(len(with_keys.history.bouts), len(gappy))
        self.assertEqual(_leaderboard(with_keys), _leaderboard(without_keys))

    def test_mixed_rows_use_scores_only_where_they_are_present(self):
        """One gappy row must not disable the margin path for the rest of the schedule."""
        rows = list(SCORED_ROWS)
        rows[0] = (rows[0][0], rows[0][1], rows[0][2], rows[0][3], {"home_score": 31})

        mixed = _trained(MasseyCompetitor, rows, score_keys=SCORE_KEYS)
        self.assertNotEqual(_leaderboard(mixed), _leaderboard(_trained(MasseyCompetitor, rows)))
        self.assertNotEqual(
            _leaderboard(mixed), _leaderboard(_trained(MasseyCompetitor, SCORED_ROWS, score_keys=SCORE_KEYS))
        )

    def test_malformed_payloads_still_raise(self):
        """A present-but-wrong score pair is a data error, not a gap."""
        rows = [("Ravens", "Bengals", 1.0, _SEASON_START, {"home_score": 3, "away_score": 45})]
        with self.assertRaises(ValueError):
            _trained(MasseyCompetitor, rows, score_keys=SCORE_KEYS)


class TestBenchmarkScoreForwarding(unittest.TestCase):
    """benchmark_competitors can put the score-based systems on real margins."""

    def test_score_keys_change_the_benchmarked_metrics(self):
        split = _scored_split()
        configs = [
            {"class": MasseyCompetitor, "name": "Massey"},
            {"class": KeenerCompetitor, "name": "Keener"},
        ]

        plain = benchmark_competitors(configs, split, _always_true, optimize_thresholds=False)
        scored = benchmark_competitors(configs, split, _always_true, optimize_thresholds=False, score_keys=SCORE_KEYS)

        self.assertEqual([r["name"] for r in scored], ["Massey", "Keener"])
        for plain_result, scored_result in zip(plain, scored, strict=True):
            with self.subTest(competitor=scored_result["name"]):
                self.assertGreater(sum(scored_result["confusion_matrix"].values()), 0)
                self.assertNotEqual(plain_result["accuracy"], scored_result["accuracy"])

    def test_default_leaves_the_benchmark_on_unit_margins(self):
        """Without score_keys the trained arena must match a score-free training run."""
        split = _scored_split(num_matchups=60)
        result = benchmark_competitors(
            [{"class": MasseyCompetitor, "name": "Massey"}], split, _always_true, optimize_thresholds=False
        )[0]

        self.assertEqual(
            _leaderboard(result["arena"]),
            _leaderboard(_trained(MasseyCompetitor, split.train)),
        )


if __name__ == "__main__":
    unittest.main()
