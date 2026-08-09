import datetime
import json
import random
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


class TestArenas(unittest.TestCase):
    def test_lambda_arena_initialization(self):
        """Test that the LambdaArena initializes correctly with different parameters."""
        # Test with default parameters
        arena = LambdaArena(lambda a, b: a > b)
        self.assertEqual(len(arena.competitors), 0)
        self.assertEqual(arena.base_competitor, EloCompetitor)
        self.assertEqual(arena.base_competitor_kwargs, {})

        # Test with custom competitor class parameters
        arena = LambdaArena(lambda a, b: a > b, base_competitor_kwargs={"initial_rating": 1000})
        self.assertEqual(arena.base_competitor_kwargs, {"initial_rating": 1000})

        # Test with initial state
        initial_state = {"A": {"initial_rating": 1200}, "B": {"initial_rating": 800}}
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)
        self.assertEqual(len(arena.competitors), 2)
        self.assertIn("A", arena.competitors)
        self.assertIn("B", arena.competitors)
        self.assertEqual(arena.competitors["A"].rating, 1200)
        self.assertEqual(arena.competitors["B"].rating, 800)

    def test_lambda_arena_matchup(self):
        """Test that matchups work correctly in the LambdaArena."""

        # Create a simple comparison function
        def compare(a, b, attributes=None):
            if attributes and "force_winner" in attributes:
                return attributes["force_winner"] == a
            return a > b

        arena = LambdaArena(compare)

        # Test a simple matchup where a > b
        arena.matchup(10, 5)
        self.assertEqual(len(arena.competitors), 2)
        self.assertIn(10, arena.competitors)
        self.assertIn(5, arena.competitors)

        # The winner's rating should be higher than the initial rating
        initial_rating = EloCompetitor().rating
        self.assertGreater(arena.competitors[10].rating, initial_rating)

        # Test a matchup with attributes
        arena.matchup("X", "Y", attributes={"force_winner": "Y"})
        self.assertEqual(len(arena.competitors), 4)
        self.assertIn("X", arena.competitors)
        self.assertIn("Y", arena.competitors)

        # Y should have won despite X normally being greater alphabetically
        self.assertGreater(arena.competitors["Y"].rating, initial_rating)

    def test_lambda_arena_matchup_outcome_override(self):
        """A supplied outcome bypasses the comparison function and is recorded as a bout."""
        calls = []

        def compare(a, b, attributes=None):
            calls.append((a, b))
            return True

        arena = LambdaArena(compare)
        attributes = {"round": 1}

        arena.matchup("A", "B", attributes=attributes, outcome=0.5)
        arena.matchup("C", "D", outcome=1.0)
        arena.matchup("E", "F", outcome=0.0)

        self.assertEqual(calls, [])
        self.assertEqual(len(arena.history.bouts), 3)

        draw_bout, win_bout, loss_bout = arena.history.bouts
        self.assertEqual(draw_bout._normalized_outcome(), "draw")
        self.assertEqual(draw_bout.attributes, attributes)
        self.assertEqual(win_bout._normalized_outcome(), "a")
        self.assertEqual(loss_bout._normalized_outcome(), "b")

        # The ratings follow the supplied outcome, not the comparison function.
        initial_rating = EloCompetitor().rating
        self.assertEqual(arena.competitors["A"].rating, initial_rating)
        self.assertGreater(arena.competitors["C"].rating, initial_rating)
        self.assertGreater(arena.competitors["F"].rating, initial_rating)

    def test_lambda_arena_matchup_rejects_unknown_outcome(self):
        """Only 1.0, 0.0 and 0.5 are valid explicit outcomes."""
        arena = LambdaArena(lambda a, b: True)

        with self.assertRaisesRegex(ValueError, "outcome must be one of"):
            arena.matchup("A", "B", outcome=0.25)

        self.assertEqual(arena.history.bouts, [])
        self.assertEqual(arena.competitors, {})

    def test_lambda_arena_tournament(self):
        """Test that tournaments work correctly in the LambdaArena."""
        arena = LambdaArena(lambda a, b: a > b)

        # Create a tournament with multiple matchups
        matchups = [(10, 5), (15, 8), (5, 3), (8, 10)]

        arena.tournament(matchups)

        # Check that all competitors are in the arena
        self.assertEqual(len(arena.competitors), 5)
        for competitor in [3, 5, 8, 10, 15]:
            self.assertIn(competitor, arena.competitors)

        # Check that the history has recorded all bouts
        self.assertEqual(len(arena.history.bouts), 4)

    def test_lambda_arena_expected_score(self):
        """Test that expected scores are calculated correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors with different ratings
        initial_state = {"A": {"initial_rating": 1200}, "B": {"initial_rating": 800}}
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)

        # A should have a higher expected score against B
        self.assertGreater(arena.expected_score("A", "B"), 0.5)
        self.assertLess(arena.expected_score("B", "A"), 0.5)

        # Test with new competitors that aren't in the arena yet
        score_c_d = arena.expected_score("C", "D")
        self.assertAlmostEqual(score_c_d, 0.5, places=2)

        # Now they should be in the arena with default ratings
        self.assertIn("C", arena.competitors)
        self.assertIn("D", arena.competitors)

    def test_lambda_arena_export_state(self):
        """Test that the arena state can be exported correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors and run some matchups
        arena.matchup("A", "B")
        arena.matchup("B", "C")
        arena.matchup("A", "C")

        # Export the state
        state = arena.export_state()

        # Check that all competitors are in the state
        self.assertEqual(len(state), 3)
        self.assertIn("A", state)
        self.assertIn("B", state)
        self.assertIn("C", state)

        # Check that the state contains the correct information
        for competitor in ["A", "B", "C"]:
            self.assertIn("initial_rating", state[competitor])
            self.assertIn("class_vars", state[competitor])

    def test_lambda_arena_leaderboard(self):
        """Test that the leaderboard is generated correctly."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some competitors with different ratings
        initial_state = {
            "A": {"initial_rating": 1200},
            "B": {"initial_rating": 1000},
            "C": {"initial_rating": 800},
        }
        arena = LambdaArena(lambda a, b: a > b, initial_state=initial_state)

        # Get the leaderboard
        leaderboard = arena.leaderboard()

        # Check that the leaderboard is sorted by rating (descending, best-first)
        self.assertEqual(len(leaderboard), 3)
        self.assertEqual(leaderboard[0]["competitor"], "A")
        self.assertEqual(leaderboard[1]["competitor"], "B")
        self.assertEqual(leaderboard[2]["competitor"], "C")

    def test_lambda_arena_clear_history(self):
        """Test that the history can be cleared."""
        arena = LambdaArena(lambda a, b: a > b)

        # Add some matchups
        arena.matchup("A", "B")
        arena.matchup("B", "C")

        # Check that the history has recorded the bouts
        self.assertEqual(len(arena.history.bouts), 2)

        # Clear the history
        arena.clear_history()

        # Check that the history is empty
        self.assertEqual(len(arena.history.bouts), 0)

    def test_lambda_arena_set_competitor_class_var(self):
        """Test that competitor class variables can be set."""
        arena = LambdaArena(lambda a, b: a > b)

        # Set a class variable
        arena.set_competitor_class_var("_k_factor", 16)

        # Check that the class variable was set
        self.assertEqual(EloCompetitor._k_factor, 16)

        # Reset the class variable for other tests
        arena.set_competitor_class_var("_k_factor", 32)

    def _assert_history_recorded(self, history, expected_len):
        """Assert that a history has the expected number of bouts with sane predictions."""
        self.assertEqual(len(history.bouts), expected_len)
        for bout in history.bouts:
            self.assertIsNotNone(bout.predicted_outcome)
            self.assertGreaterEqual(bout.predicted_outcome, 0.0)
            self.assertLessEqual(bout.predicted_outcome, 1.0)

    def test_process_history_non_colley_competitors(self):
        """process_history should run end-to-end for non-Colley competitors.

        Regression test for a TypeError ('unhashable type') that occurred because
        the method passed competitor *objects* to expected_score, which expects IDs.
        """
        for competitor_cls in (EloCompetitor, GlickoCompetitor):
            with self.subTest(competitor=competitor_cls.__name__):
                arena = LambdaArena(func=lambda x, y: True, base_competitor=competitor_cls)
                bouts = [("x", "y", 1), ("y", "z", 0), ("x", "z", 0.5)]
                arena.process_history(bouts, progress_bar=False)

                # All bouts have non-None outcomes, so all should be recorded.
                self._assert_history_recorded(arena.history, len(bouts))
                for cid in ("x", "y", "z"):
                    self.assertIn(cid, arena.competitors)

    def test_process_history_skips_none_outcome(self):
        """Bouts with a None outcome should be skipped, not recorded."""
        arena = LambdaArena(func=lambda x, y: True)
        bouts = [("x", "y", 1), ("y", "z", None)]
        arena.process_history(bouts, progress_bar=False)
        self.assertEqual(len(arena.history.bouts), 1)

    def test_evaluate_performance_non_colley_competitors(self):
        """evaluate_performance should run end-to-end for non-Colley competitors."""
        for competitor_cls in (EloCompetitor, GlickoCompetitor):
            with self.subTest(competitor=competitor_cls.__name__):
                arena = LambdaArena(func=lambda x, y: True, base_competitor=competitor_cls)
                arena.process_history([("x", "y", 1), ("y", "z", 0)], progress_bar=False)

                eval_bouts = [("x", "y", 1), ("y", "z", 0)]
                arena.evaluate_performance(eval_bouts, progress_bar=False)

                # evaluate_performance must not modify the training history.
                self.assertEqual(len(arena.history.bouts), 2)
                self._assert_history_recorded(arena.eval_history, len(eval_bouts))

    def test_validate_non_colley_competitors(self):
        """validate should run end-to-end for non-Colley competitors without updating ratings."""
        for competitor_cls in (EloCompetitor, GlickoCompetitor):
            with self.subTest(competitor=competitor_cls.__name__):
                arena = LambdaArena(func=lambda x, y: True, base_competitor=competitor_cls)
                arena.process_history([("x", "y", 1), ("y", "z", 0)], progress_bar=False)

                ratings_before = {cid: c.rating for cid, c in arena.competitors.items()}

                validation_bouts = [("x", "y", 1), ("y", "z", 0)]
                arena.validate(validation_bouts, progress_bar=False)

                self._assert_history_recorded(arena.validation_history, len(validation_bouts))

                # validate only records predictions; it must not update ratings.
                ratings_after = {cid: c.rating for cid, c in arena.competitors.items()}
                self.assertEqual(ratings_before, ratings_after)


class TestArenaStateRoundTrip(unittest.TestCase):
    """LambdaArena must be restorable from its own export_state() output."""

    # Every shipped competitor class must survive the round trip.
    ALL_COMPETITORS = (
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

    # Colley, Massey, Bradley-Terry and Keener reset their opponent match graph on import by design
    # (documented in their own docstrings), so only the incremental systems can be
    # expected to keep rating identically after a restore.
    INCREMENTAL_COMPETITORS = (
        EloCompetitor,
        GlickoCompetitor,
        Glicko2Competitor,
        TrueSkillCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    )

    # Elo aborts on its rating floor when a competitor loses repeatedly from the
    # default start of 400, so give it the same headroom the examples use.
    COMPETITOR_KWARGS = {EloCompetitor: {"initial_rating": 1200}}

    @staticmethod
    def _matchups(seed, count):
        rng = random.Random(seed)
        pairs = []
        while len(pairs) < count:
            a, b = str(rng.randint(1, 8)), str(rng.randint(1, 8))
            if a != b:
                pairs.append((a, b))
        return pairs

    @classmethod
    def _trained_arena(cls, competitor_cls):
        arena = LambdaArena(
            lambda a, b: a > b,
            base_competitor=competitor_cls,
            base_competitor_kwargs=cls.COMPETITOR_KWARGS.get(competitor_cls),
        )
        for a, b in cls._matchups(seed=1, count=60):
            arena.matchup(a, b)
        return arena

    @staticmethod
    def _ratings(arena):
        return {entry["competitor"]: entry["rating"] for entry in arena.leaderboard()}

    def test_round_trip_through_json_reproduces_leaderboard(self):
        """A JSON round trip of export_state() must reproduce the leaderboard exactly."""
        for competitor_cls in self.ALL_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                arena = self._trained_arena(competitor_cls)

                # Going through JSON proves the exported state is serializable, which
                # is the whole point of the persist-and-resume journey.
                state = json.loads(json.dumps(arena.export_state()))
                restored = LambdaArena(lambda a, b: a > b, base_competitor=competitor_cls, initial_state=state)

                self.assertEqual(arena.leaderboard(), restored.leaderboard())

    def test_round_trip_resolves_concrete_type_from_state(self):
        """The recorded type in the state decides the class, not base_competitor."""
        arena = self._trained_arena(GlickoCompetitor)
        state = json.loads(json.dumps(arena.export_state()))

        # base_competitor is left at its default (EloCompetitor) on purpose.
        restored = LambdaArena(lambda a, b: a > b, initial_state=state)

        self.assertEqual(len(restored.competitors), len(arena.competitors))
        for competitor in restored.competitors.values():
            self.assertIsInstance(competitor, GlickoCompetitor)
        self.assertEqual(arena.leaderboard(), restored.leaderboard())

    def test_restored_arena_keeps_rating_identically(self):
        """Playing on further must give the same ratings as never having restored."""
        # A fixed match time keeps the time-aware systems deterministic; without it
        # the two arenas inflate rating deviations by slightly different elapsed times.
        match_time = datetime.datetime.now() + datetime.timedelta(days=1)

        for competitor_cls in self.INCREMENTAL_COMPETITORS:
            with self.subTest(competitor=competitor_cls.__name__):
                arena = self._trained_arena(competitor_cls)
                state = json.loads(json.dumps(arena.export_state()))
                restored = LambdaArena(lambda a, b: a > b, base_competitor=competitor_cls, initial_state=state)

                for a, b in self._matchups(seed=2, count=25):
                    arena.matchup(a, b, match_time=match_time)
                    restored.matchup(a, b, match_time=match_time)

                self.assertEqual(self._ratings(arena), self._ratings(restored))

    def test_plain_constructor_kwargs_are_still_supported(self):
        """The pre-existing kwargs form of initial_state must keep working."""
        arena = LambdaArena(
            lambda a, b: a > b,
            initial_state={"A": {"initial_rating": 1200}, "B": {"initial_rating": 800}},
        )

        self.assertIsInstance(arena.competitors["A"], EloCompetitor)
        self.assertEqual(arena.competitors["A"].rating, 1200)
        self.assertEqual(arena.competitors["B"].rating, 800)
