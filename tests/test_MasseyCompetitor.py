import json
import math
import random
import unittest

from elote import EloCompetitor, LambdaArena, MasseyCompetitor
from elote.competitors.base import (
    InvalidParameterException,
    InvalidRatingValueException,
    MissMatchedCompetitorTypesException,
)


class TestMassey(unittest.TestCase):
    """Behavioural tests for MasseyCompetitor."""

    def test_improvement(self):
        """Beating fresh opponents drives a competitor's rating up."""
        anchor = MasseyCompetitor()
        previous = anchor.rating
        for _ in range(5):
            anchor.beat(MasseyCompetitor())
            self.assertGreater(anchor.rating, previous)
            previous = anchor.rating

    def test_decay(self):
        """Losing to fresh opponents drives a competitor's rating down."""
        anchor = MasseyCompetitor()
        previous = anchor.rating
        for _ in range(5):
            MasseyCompetitor().beat(anchor)
            self.assertLess(anchor.rating, previous)
            previous = anchor.rating

    def test_lost_to_matches_beat(self):
        """a.lost_to(b) must produce the same fit as b.beat(a)."""
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.lost_to(b)

        c, d = MasseyCompetitor(), MasseyCompetitor()
        d.beat(c)

        self.assertEqual(a.rating, c.rating)
        self.assertEqual(b.rating, d.rating)

    def test_draw_leaves_both_competitors_level(self):
        """A drawn game contributes no margin, so both ratings stay at zero."""
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.tied(b)

        self.assertEqual(a.rating, 0.0)
        self.assertEqual(b.rating, 0.0)
        self.assertEqual(a.num_games, 1)
        self.assertEqual(b.num_games, 1)

    def test_type_mismatch_is_rejected(self):
        """Massey competitors cannot be mixed with other rating systems."""
        massey = MasseyCompetitor()
        for call in ("expected_score", "beat", "tied", "lost_to"):
            with self.subTest(call=call):
                with self.assertRaises(MissMatchedCompetitorTypesException):
                    getattr(massey, call)(EloCompetitor())


class TestMasseyNegativeRatings(unittest.TestCase):
    """Massey ratings are zero mean, so negative ratings must be legal."""

    def test_minimum_rating_is_unbounded_below(self):
        """The class must override the inherited floor of 100."""
        self.assertEqual(MasseyCompetitor._minimum_rating, float("-inf"))

    def test_losing_run_produces_a_negative_rating(self):
        """A competitor that only loses ends below zero without raising.

        This is the guard on the ``_minimum_rating`` override: ``_recalculate_ratings``
        assigns through the ``rating`` property, so with the inherited floor of 100 the
        very first fit would raise InvalidRatingValueException.
        """
        loser = MasseyCompetitor()
        winners = [MasseyCompetitor() for _ in range(4)]

        winners[0].beat(loser)
        loser.lost_to(winners[1])
        winners[2].beat(loser)
        loser.lost_to(winners[3])

        self.assertLess(loser.rating, 0.0)
        # And the winners, who only ever played the loser, sit above it.
        for winner in winners:
            self.assertGreater(winner.rating, loser.rating)

    def test_explicit_assignment_below_zero_is_allowed(self):
        """The rating setter must accept negative values."""
        competitor = MasseyCompetitor()
        competitor.rating = -12.5
        self.assertEqual(competitor.rating, -12.5)

    def test_negative_initial_rating_is_allowed(self):
        """A negative starting rating is a legal Massey configuration."""
        self.assertEqual(MasseyCompetitor(initial_rating=-3.0).rating, -3.0)


class TestMasseyInvariants(unittest.TestCase):
    """Structural properties of the Massey fit."""

    def test_ratings_of_a_connected_component_sum_to_zero(self):
        """The pinned constraint row forces zero-mean ratings."""
        competitors = [MasseyCompetitor() for _ in range(6)]
        schedule = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 4), (3, 5)]
        for winner, loser in schedule:
            competitors[winner].beat(competitors[loser])
        competitors[2].tied(competitors[5])

        self.assertAlmostEqual(sum(c.rating for c in competitors), 0.0, places=12)

    def test_identical_records_produce_one_single_rating(self):
        """A five-player circular schedule leaves everyone 1-1 and equally rated."""
        competitors = [MasseyCompetitor() for _ in range(5)]
        for i in range(5):
            competitors[i].beat(competitors[(i + 1) % 5])

        ratings = {c.rating for c in competitors}
        self.assertEqual(len(ratings), 1, f"expected one tied rating, got {sorted(ratings)}")
        self.assertAlmostEqual(next(iter(ratings)), 0.0, places=12)

    # A five-competitor schedule chosen by search, not by eye: the raw solver output for
    # this fixture differs in its last bits between the two orderings below, so the test
    # fails if _recalculate_ratings stops canonicalizing the solution.
    ORDER_INDEPENDENCE_SCHEDULE = [
        ("d", "a", False),
        ("e", "d", False),
        ("c", "d", False),
        ("b", "e", False),
        ("a", "c", False),
        ("e", "b", False),
        ("a", "c", False),
        ("a", "c", False),
        ("e", "b", False),
        ("d", "e", False),
        ("c", "a", False),
        ("e", "a", True),
    ]

    def test_ratings_are_order_independent(self):
        """The same results applied in two different orders give identical ratings."""
        schedule = self.ORDER_INDEPENDENCE_SCHEDULE

        def fit(order, results):
            competitors = {name: MasseyCompetitor() for name in order}
            for winner, loser, drawn in results:
                if drawn:
                    competitors[winner].tied(competitors[loser])
                else:
                    competitors[winner].beat(competitors[loser])
            return {name: competitor.rating for name, competitor in competitors.items()}

        forwards = fit("abcde", schedule)
        backwards = fit("edcba", list(reversed(schedule)))

        self.assertEqual(forwards, backwards)
        # Guard against the fixture silently degenerating into an all-zero schedule.
        self.assertGreater(max(abs(rating) for rating in forwards.values()), 0.1)

    def test_fallback_rating_calculation_is_zero_mean(self):
        """The defensive singular-matrix fallback rates by average margin, zero mean.

        The constrained system is non-singular for any connected component, so this branch
        is unreachable in normal play; it is exercised directly so it is not dead code.
        """
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        b.beat(c)
        a.beat(c)

        a._fallback_rating_calculation([a, b, c])

        # a is 2-0 (+2/2), b is 1-1 (0/2), c is 0-2 (-2/2); mean 0, so ratings are +1, 0, -1.
        self.assertAlmostEqual(a.rating, 1.0, places=12)
        self.assertAlmostEqual(b.rating, 0.0, places=12)
        self.assertAlmostEqual(c.rating, -1.0, places=12)

    def test_disconnected_groups_are_rated_independently(self):
        """A result in one component leaves an unrelated component untouched."""
        a, b = MasseyCompetitor(), MasseyCompetitor()
        c, d = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        ratings_before = (a.rating, b.rating)

        c.beat(d)

        self.assertEqual((a.rating, b.rating), ratings_before)


class TestMasseyExpectedScore(unittest.TestCase):
    """Probability contract specifics for MasseyCompetitor."""

    def test_fresh_competitors_give_exactly_one_half(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        self.assertEqual(a.expected_score(b), 0.5)

    def test_expected_scores_are_exactly_complementary(self):
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.beat(c)
        b.tied(c)

        for first, second in ((a, b), (b, c), (a, c)):
            with self.subTest(pair=(first.rating, second.rating)):
                self.assertEqual(first.expected_score(second) + second.expected_score(first), 1.0)

    def test_scale_is_class_configurable(self):
        """A larger scale sharpens the predicted probability."""
        strong = MasseyCompetitor(initial_rating=0.5)
        weak = MasseyCompetitor(initial_rating=-0.5)
        default = strong.expected_score(weak)

        try:
            MasseyCompetitor.configure_class(expected_score_scale=8.0)
            self.assertGreater(strong.expected_score(weak), default)
        finally:
            MasseyCompetitor.configure_class(expected_score_scale=2.0)

        self.assertEqual(strong.expected_score(weak), default)

    def test_non_positive_scale_is_rejected(self):
        with self.assertRaises(InvalidParameterException):
            MasseyCompetitor.configure_class(expected_score_scale=0.0)
        self.assertEqual(MasseyCompetitor._expected_score_scale, 2.0)


class TestMasseySerialization(unittest.TestCase):
    """State export/import for MasseyCompetitor."""

    def test_state_round_trip_preserves_rating_and_record(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.beat(b)
        b.tied(a)

        restored = MasseyCompetitor.from_state(json.loads(json.dumps(a.export_state())))

        self.assertEqual(restored.rating, a.rating)
        self.assertEqual(restored._wins, a._wins)
        self.assertEqual(restored._losses, a._losses)
        self.assertEqual(restored._ties, a._ties)
        self.assertEqual(restored._point_differential, a._point_differential)

    def test_export_reports_the_concrete_type(self):
        state = MasseyCompetitor().export_state()
        self.assertEqual(state["type"], "MasseyCompetitor")
        self.assertIn("initial_rating", state)

    def test_class_is_registered(self):
        self.assertIn("MasseyCompetitor", MasseyCompetitor.list_competitor_types())
        self.assertIs(MasseyCompetitor.get_competitor_class("MasseyCompetitor"), MasseyCompetitor)

    def test_reset_restores_a_fresh_competitor(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b)
        a.reset()

        self.assertEqual(a.rating, 0.0)
        self.assertEqual(a.num_games, 0)
        self.assertEqual(a._point_differential, 0.0)
        self.assertEqual(a._opponents, {})

    def test_arena_state_round_trip_reproduces_the_leaderboard(self):
        """A JSON round trip through LambdaArena reproduces the leaderboard exactly."""
        arena = LambdaArena(lambda a, b: a > b, base_competitor=MasseyCompetitor)
        for a, b in (("1", "2"), ("2", "3"), ("3", "1"), ("1", "4"), ("4", "2"), ("3", "4")):
            arena.matchup(a, b)

        state = json.loads(json.dumps(arena.export_state()))
        restored = LambdaArena(lambda a, b: a > b, initial_state=state)

        self.assertEqual(arena.leaderboard(), restored.leaderboard())
        for competitor in restored.competitors.values():
            self.assertIsInstance(competitor, MasseyCompetitor)


class TestMasseyRatingSetterValidation(unittest.TestCase):
    """The floor is unbounded by default but the setter still honours it if configured."""

    def test_setter_raises_when_a_floor_is_configured(self):
        competitor = MasseyCompetitor()
        try:
            MasseyCompetitor.configure_class(minimum_rating=0.0)
            with self.assertRaises(InvalidRatingValueException):
                competitor.rating = -1.0
        finally:
            MasseyCompetitor.configure_class(minimum_rating=float("-inf"))

        competitor.rating = -1.0
        self.assertEqual(competitor.rating, -1.0)


if __name__ == "__main__":
    unittest.main()


def _football_shaped_schedule(seed=17, competitors=20, matchups=300):
    """Build a seeded schedule of point-scored games on a points-per-game scale.

    Each competitor gets a latent strength drawn from ``N(0, 10)``; a game between ``a`` and
    ``b`` scores each side from ``N(28 +/- (s_a - s_b) / 2, 10)``, rounded and floored at zero,
    so margins are tens of points rather than the unit margins Massey falls back to when no
    ``scores`` payload is supplied.

    Returns:
        list: ``(a, b, a_score, b_score)`` tuples, all decisive.
    """
    rng = random.Random(seed)
    names = ["c%d" % i for i in range(competitors)]
    strength = {name: rng.gauss(0, 10) for name in names}

    schedule = []
    for _ in range(matchups):
        a, b = rng.sample(names, 2)
        edge = strength[a] - strength[b]
        a_score = max(0, round(rng.gauss(28 + edge / 2, 10)))
        b_score = max(0, round(rng.gauss(28 - edge / 2, 10)))
        if a_score == b_score:
            a_score += 1
        schedule.append((a, b, float(a_score), float(b_score)))
    return schedule


def _trained_arena_predictions(train_fraction=0.7, **kwargs):
    """Train a score-fed Massey arena on part of a schedule and predict the rest.

    Returns:
        list: ``(predicted_probability_for_a, actual_outcome)`` pairs for the held-out games.
    """
    schedule = _football_shaped_schedule(**kwargs)
    split = int(len(schedule) * train_fraction)

    arena = LambdaArena(None, base_competitor=MasseyCompetitor)
    for a, b, a_score, b_score in schedule[:split]:
        arena.matchup(a, b, outcome=1.0 if a_score > b_score else 0.0, scores=(a_score, b_score))

    return [
        (arena.expected_score(a, b), 1.0 if a_score > b_score else 0.0) for a, b, a_score, b_score in schedule[split:]
    ]


class TestMasseyScoredArenaProbabilities(unittest.TestCase):
    """Predicted probabilities stay usable once Massey is fed real point margins.

    With real scores the fitted ratings are points per game, so the rating difference is
    routinely tens of points. Without the rating-scale normalization in ``expected_score`` the
    logistic saturates and returns exactly 0.0 / 1.0, which makes log loss unbounded. These
    tests assert on probability values, not on accuracy -- accuracy is structurally blind to
    the defect, since the normalization is a monotone recalibration that reorders nothing.
    """

    def setUp(self):
        self.predictions = _trained_arena_predictions()
        self.probabilities = [p for p, _ in self.predictions]

    def test_no_prediction_is_degenerate(self):
        """No held-out prediction is exactly 0.0 or 1.0."""
        self.assertEqual([p for p in self.probabilities if p in (0.0, 1.0)], [])

    def test_no_prediction_lands_in_the_extreme_bins(self):
        """No held-out prediction is <= 1e-3 or >= 1 - 1e-3."""
        extreme = [p for p in self.probabilities if p <= 1e-3 or p >= 1 - 1e-3]
        self.assertEqual(extreme, [], "%d of %d predictions saturated" % (len(extreme), len(self.probabilities)))
        self.assertGreater(min(self.probabilities), 0.008)
        self.assertLess(max(self.probabilities), 0.992)

    def test_log_loss_and_brier_are_pinned(self):
        """Value assertions on this arena's calibration, so a future remapping is visible."""
        n = len(self.predictions)
        self.assertEqual(n, 90)

        log_loss = sum(-(y * math.log(p) + (1 - y) * math.log(1 - p)) for p, y in self.predictions) / n
        brier = sum((p - y) ** 2 for p, y in self.predictions) / n

        self.assertAlmostEqual(log_loss, 0.4938, places=4)
        self.assertAlmostEqual(brier, 0.1610, places=4)

    def test_accuracy_is_unchanged_by_the_normalization(self):
        """Documentation, not a guard: accuracy cannot see the defect at all."""
        correct = sum(1 for p, y in self.predictions if (p > 0.5) == (y > 0.5))
        self.assertAlmostEqual(correct / len(self.predictions), 0.7555555555555555, places=12)


class TestMasseyRatingScale(unittest.TestCase):
    """The fitted rating scale that makes the expected-score logistic dimensionless."""

    def test_defaults_to_one_for_an_unfitted_competitor(self):
        self.assertEqual(MasseyCompetitor()._rating_scale, 1.0)
        self.assertEqual(MasseyCompetitor(initial_rating=7.5)._rating_scale, 1.0)

    def test_expected_score_is_invariant_to_the_unit_of_the_scores(self):
        """The same schedule in tenths of a point gives the same probabilities."""
        points = [("a", "b", 31.0, 17.0), ("b", "c", 24.0, 20.0), ("a", "c", 45.0, 10.0)]

        def fit(factor):
            competitors = {name: MasseyCompetitor() for name in "abc"}
            for first, second, first_score, second_score in points:
                competitors[first].beat(competitors[second], scores=(first_score * factor, second_score * factor))
            return competitors

        unit, scaled = fit(1.0), fit(10.0)
        for first, second in (("a", "b"), ("b", "c"), ("a", "c")):
            with self.subTest(pair=(first, second)):
                self.assertAlmostEqual(
                    unit[first].expected_score(unit[second]),
                    scaled[first].expected_score(scaled[second]),
                    places=12,
                )

    def test_scale_survives_the_state_round_trip(self):
        a, b, c = MasseyCompetitor(), MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(31.0, 17.0))
        b.beat(c, scores=(24.0, 20.0))

        self.assertGreater(a._rating_scale, 1.0)

        restored = MasseyCompetitor.from_state(json.loads(json.dumps(a.export_state())))
        self.assertEqual(restored._rating_scale, a._rating_scale)
        self.assertEqual(restored.expected_score(c), a.expected_score(c))

    def test_reset_restores_the_unfitted_scale(self):
        a, b = MasseyCompetitor(), MasseyCompetitor()
        a.beat(b, scores=(31.0, 17.0))
        self.assertNotEqual(a._rating_scale, 1.0)

        a.reset()
        self.assertEqual(a._rating_scale, 1.0)
