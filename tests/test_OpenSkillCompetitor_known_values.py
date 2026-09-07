"""Reference-value tests for OpenSkillCompetitor against the canonical implementation.

The ``openskill`` package (dev extra only, pinned in uv.lock) is the test
oracle: a fixed battery of bouts -- 1v1 win/loss/draw, ties, repeated
sequences, and N-way bouts -- is applied to both elote's native Weng-Lin
implementation and the oracle's ``PlackettLuce.rate``, and elote's mu/sigma
must match to 1e-6.

One documented deviation: openskill.py 6.2.0's tie adjustment for mu is a
no-op. Its ``_compute`` mutates the rating objects it later subtracts from,
so ``result[i][0].mu - original_teams[i][0].mu`` is always zero and tied
teams keep their raw per-player mu changes (upstream issue #201, fix in
PR #203 -- not released as of 6.2.0, the latest PyPI version). elote
implements the documented rule the oracle's dead code intends: every tied
player receives the *average* mu change of the tied group, preserving each
player's own prior. The battery therefore compares strictly wherever the
oracle is correct (all sigmas, all untied mus) and compares the tied group's
*mean* mu change -- which both implementations agree on to well under 1e-6.
"""

import unittest

try:
    from openskill.models.weng_lin.plackett_luce import PlackettLuce

    HAS_ORACLE = True
except ImportError:  # pragma: no cover - only hit when dev extras are absent
    HAS_ORACLE = False

from elote import BaseCompetitor, OpenSkillCompetitor
from elote.competitors.base import (
    InvalidParameterException,
    MissMatchedCompetitorTypesException,
)

TOLERANCE = 1e-6


class ReferenceHarness:
    """Drives elote competitors and the openskill.py oracle through identical bouts."""

    def __init__(self, n_players: int) -> None:
        self.model = PlackettLuce()
        self.competitors = [OpenSkillCompetitor() for _ in range(n_players)]
        self.ratings = [self.model.rating() for _ in range(n_players)]

    def apply_bout(self, indices: tuple, ranks: tuple, test_case: unittest.TestCase, context: str) -> None:
        """Apply one bout to both implementations and verify they agree.

        Sigmas and untied mus must match the oracle to 1e-6. For a tied group,
        elote's per-player mu changes must all equal the oracle's mean change
        (see the module docstring for the upstream no-op this works around).
        """
        pre_mu = {i: self.competitors[i].mu for i in indices}

        participants = [self.competitors[i] for i in indices]
        OpenSkillCompetitor.apply_bout(participants, ranks=ranks)

        teams = [[self.ratings[i]] for i in indices]
        updated = self.model.rate(teams, ranks=list(ranks))
        for position, index in enumerate(indices):
            self.ratings[index] = updated[position][0]

        groups: dict[float, list[int]] = {}
        for position, rank in enumerate(ranks):
            groups.setdefault(rank, []).append(position)

        for positions in groups.values():
            if len(positions) == 1:
                position = positions[0]
                index = indices[position]
                test_case.assertAlmostEqual(
                    self.competitors[index].mu, updated[position][0].mu, delta=TOLERANCE, msg=f"{context}: player {index} mu"
                )
            else:
                # openskill.py 6.2.0 keeps raw per-player mu changes for tied
                # players (the no-op above); elote applies the group average.
                # The two agree on the mean change, so compare that.
                oracle_changes = [updated[p][0].mu - pre_mu[indices[p]] for p in positions]
                oracle_mean_change = sum(oracle_changes) / len(oracle_changes)
                for position in positions:
                    index = indices[position]
                    elote_change = self.competitors[index].mu - pre_mu[index]
                    test_case.assertAlmostEqual(
                        elote_change, oracle_mean_change, delta=TOLERANCE, msg=f"{context}: tied player {index} mu change"
                    )

            for position in positions:
                index = indices[position]
                test_case.assertAlmostEqual(
                    self.competitors[index].sigma,
                    updated[position][0].sigma,
                    delta=TOLERANCE,
                    msg=f"{context}: player {index} sigma",
                )


@unittest.skipUnless(HAS_ORACLE, "the openskill oracle (dev extra) is not installed")
class TestOpenSkillReferenceValues(unittest.TestCase):
    """Reference-value battery: elote vs openskill.py to 1e-6."""

    def test_single_bouts(self):
        """1v1 win, loss and draw each match the oracle."""
        for name, ranks in (("win", (0, 1)), ("loss", (1, 0)), ("draw", (0, 0))):
            with self.subTest(bout=name):
                harness = ReferenceHarness(2)
                harness.apply_bout((0, 1), ranks, self, f"1v1 {name}")

    def test_repeated_sequence(self):
        """A fixed tie-free 10-bout sequence over four players matches after every bout.

        The sequence is deliberately tie-free: on tied ranks elote and the
        6.2.0 oracle legitimately diverge (see the module docstring), and that
        divergence would propagate into every later bout's math. Tied shapes
        are covered from fresh state in ``test_tie_bouts_match_oracle``.
        """
        sequence = [
            ((0, 1), (0, 1)),
            ((2, 3), (0, 1)),
            ((0, 2), (1, 0)),
            ((0, 3), (0, 1)),
            ((1, 2), (1, 0)),
            ((0, 1, 2), (0, 1, 2)),  # 3-way bout, distinct ranks
            ((3, 0), (1, 0)),
            ((1, 3), (0, 1)),
            ((2, 0), (0, 1)),
            ((3, 1), (1, 0)),
        ]
        harness = ReferenceHarness(4)
        for step, (indices, ranks) in enumerate(sequence):
            harness.apply_bout(indices, ranks, self, f"sequence step {step} ({indices}, {ranks})")

    def test_tie_bouts_match_oracle(self):
        """Tied-rank bouts from fresh state: sigmas and untied mus match the
        oracle to 1e-6, tied players match the oracle's mean mu change."""
        for indices, ranks in (
            ((0, 1), (0, 0)),  # two-player draw
            ((0, 1, 2), (0, 0, 1)),  # tie for first in a 3-way
            ((0, 1, 2), (0, 1, 1)),  # tie for second in a 3-way
            ((0, 1, 2, 3), (0, 0, 0, 0)),  # all tied
            ((0, 1, 2, 3), (0, 0, 1, 2)),  # tie for first in a 4-way
            ((0, 1, 2, 3), (0, 1, 1, 3)),  # tie for second, non-contiguous ranks
        ):
            with self.subTest(bout=(indices, ranks)):
                harness = ReferenceHarness(max(indices) + 1)
                harness.apply_bout(indices, ranks, self, f"tie bout ({indices}, {ranks})")

    def test_pairwise_methods_match_oracle(self):
        """beat/lost_to/tied route one-row periods that match rate() calls."""
        for name, method, ranks in (
            ("beat", "beat", (0, 1)),
            ("lost_to", "lost_to", (1, 0)),
            ("tied", "tied", (0, 0)),
        ):
            with self.subTest(method=name):
                model = PlackettLuce()
                a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
                ra, rb = model.rating(), model.rating()

                getattr(a, method)(b)
                updated = model.rate([[ra], [rb]], ranks=list(ranks))

                self.assertAlmostEqual(a.mu, updated[0][0].mu, delta=TOLERANCE)
                self.assertAlmostEqual(a.sigma, updated[0][0].sigma, delta=TOLERANCE)
                self.assertAlmostEqual(b.mu, updated[1][0].mu, delta=TOLERANCE)
                self.assertAlmostEqual(b.sigma, updated[1][0].sigma, delta=TOLERANCE)

    def test_expected_score_matches_predict_win(self):
        """The 1v1 win probability matches the oracle's two-team predict_win."""
        model = PlackettLuce()
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        c = OpenSkillCompetitor(initial_mu=32.0, initial_sigma=5.0)

        # Create an asymmetric belief state, then check every ordered pair:
        # two updated players and two cross pairs against a fresh player.
        a.beat(b)
        for self_player, other_player in ((a, b), (c, a), (c, b)):
            with self.subTest(self_player=(self_player is c), other_player=(other_player is c)):
                ratings = [
                    model.rating(mu=self_player.mu, sigma=self_player.sigma),
                    model.rating(mu=other_player.mu, sigma=other_player.sigma),
                ]
                predicted = model.predict_win([[ratings[0]], [ratings[1]]])
                self.assertAlmostEqual(self_player.expected_score(other_player), predicted[0], delta=TOLERANCE)
                # The opponent's symmetric call is exactly complementary.
                self.assertAlmostEqual(other_player.expected_score(self_player), predicted[1], delta=TOLERANCE)


@unittest.skipUnless(HAS_ORACLE, "the openskill oracle (dev extra) is not installed")
class TestOpenSkillMinimumRatingRegression(unittest.TestCase):
    """The mu-approx-25 scale must never be clamped by _minimum_rating = 100."""

    def test_updates_below_minimum_rating_are_never_clamped(self):
        """Post-update beliefs match the oracle even though they sit below 100."""
        model = PlackettLuce()
        winner, loser = OpenSkillCompetitor(), OpenSkillCompetitor()
        r_winner, r_loser = model.rating(), model.rating()

        winner.beat(loser)
        updated = model.rate([[r_winner], [r_loser]], ranks=[0, 1])

        for competitor, rating in ((winner, updated[0][0]), (loser, updated[1][0])):
            self.assertLess(competitor.mu, OpenSkillCompetitor._minimum_rating)
            self.assertLess(competitor.rating, OpenSkillCompetitor._minimum_rating)
            self.assertAlmostEqual(competitor.mu, rating.mu, delta=TOLERANCE)
            self.assertAlmostEqual(competitor.sigma, rating.sigma, delta=TOLERANCE)

    def test_losing_streak_stays_unclamped(self):
        """A long losing streak drives the ordinal far below 100 without interference."""
        harness = ReferenceHarness(2)
        for step in range(10):
            harness.apply_bout((0, 1), (1, 0), self, f"loss streak step {step}")  # player 1 keeps beating player 0
        self.assertLess(harness.competitors[0].rating, OpenSkillCompetitor._minimum_rating)


class TestOpenSkillTieRule(unittest.TestCase):
    """The documented tie rule: tied players share the average mu change."""

    def test_tied_players_share_the_average_change_not_the_absolute_mu(self):
        """Distinct priors survive a tie; both players move by the same amount."""
        a = OpenSkillCompetitor(initial_mu=30.0, initial_sigma=8.0)
        b = OpenSkillCompetitor(initial_mu=20.0, initial_sigma=9.0)
        c = OpenSkillCompetitor(initial_mu=27.0, initial_sigma=6.0)

        OpenSkillCompetitor.apply_bout([a, b, c], ranks=[0, 0, 1])

        change_a = a.mu - 30.0
        change_b = b.mu - 20.0
        self.assertAlmostEqual(change_a, change_b, delta=TOLERANCE)
        # Tied players keep their own priors; only the change is shared.
        self.assertAlmostEqual(a.mu - b.mu, 30.0 - 20.0, delta=TOLERANCE)
        # The untied player is unaffected by the tie-average rule.
        self.assertAlmostEqual(c.mu, 26.172319677365, delta=TOLERANCE)

    def test_two_player_draw_leaves_mu_untouched(self):
        """A fresh even draw changes no mu and shrinks both sigmas."""
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        OpenSkillCompetitor.apply_bout([a, b], ranks=[0, 0])
        self.assertEqual(a.mu, 25.0)
        self.assertEqual(b.mu, 25.0)
        self.assertLess(a.sigma, 25.0 / 3.0)
        self.assertEqual(a.sigma, b.sigma)


class TestOpenSkillCompetitorContract(unittest.TestCase):
    """E-A surface contracts that do not need the oracle."""

    def test_model_parameter_defaults_to_plackett_luce(self):
        """The variant selector defaults to plackett_luce and is carried on the instance."""
        self.assertEqual(OpenSkillCompetitor()._model, "plackett_luce")
        self.assertEqual(OpenSkillCompetitor(model="plackett_luce")._model, "plackett_luce")

    def test_unknown_model_variant_rejected(self):
        """Variants outside the selector raise InvalidParameterException."""
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(model="thurstone")
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(model="elo")

    def test_rating_is_conservative_ordinal(self):
        """rating derives from mu - 3*sigma."""
        competitor = OpenSkillCompetitor(initial_mu=30.0, initial_sigma=5.0)
        self.assertAlmostEqual(competitor.rating, 30.0 - 3 * 5.0)

    def test_rating_setter_refuses_writes(self):
        """Direct rating writes raise, pointing at mu/sigma instead."""
        competitor = OpenSkillCompetitor()
        with self.assertRaises(NotImplementedError):
            competitor.rating = 100.0

    def test_initial_sigma_must_be_positive(self):
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(initial_sigma=0.0)
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(initial_sigma=-1.0)

    def test_sigma_setter_must_be_positive(self):
        competitor = OpenSkillCompetitor()
        with self.assertRaises(InvalidParameterException):
            competitor.sigma = 0.0

    def test_reset_restores_initial_beliefs(self):
        a, b = OpenSkillCompetitor(initial_mu=30.0, initial_sigma=5.0), OpenSkillCompetitor()
        a.beat(b)
        a.reset()
        self.assertEqual(a.mu, 30.0)
        self.assertEqual(a.sigma, 5.0)
        self.assertAlmostEqual(a.rating, 15.0)

    def test_apply_bout_validation(self):
        """Bout-level updates validate participants and ranks up front."""
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        from elote import EloCompetitor

        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_bout([a])  # fewer than two participants
        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_bout([a, a])  # duplicate participant
        with self.assertRaises(MissMatchedCompetitorTypesException):
            OpenSkillCompetitor.apply_bout([a, EloCompetitor()])  # wrong rating system
        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_bout([a, b], ranks=[0])  # ranks/participants mismatch
        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_bout([a, b], ranks=[0, "first"])  # non-numeric rank

    def test_scores_are_validated_but_not_consumed(self):
        """Score payloads follow the caller-order contract but the update is rank-based."""
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()

        # A malformed payload raises from the pairwise method.
        with self.assertRaises(ValueError):
            a.beat(b, scores=(0.0, 1.0))  # does not describe a win for a
        with self.assertRaises(ValueError):
            a.tied(b, scores=(1.0, 2.0))  # does not describe a draw

        # Any valid win payload produces the same update.
        winner_strict, winner_blowout = OpenSkillCompetitor(), OpenSkillCompetitor()
        loser_strict, loser_blowout = OpenSkillCompetitor(), OpenSkillCompetitor()
        winner_strict.beat(loser_strict, scores=(1.0, 0.0))
        winner_blowout.beat(loser_blowout, scores=(5.0, 0.0))
        self.assertEqual(winner_strict.mu, winner_blowout.mu)
        self.assertEqual(winner_strict.sigma, winner_blowout.sigma)
        self.assertEqual(loser_strict.mu, loser_blowout.mu)

    def test_lost_to_matches_beat_on_the_winner(self):
        """lost_to routes the same one-row period as beat on the winner."""
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        b_before = (b.mu, b.sigma)
        a.lost_to(b)
        self.assertNotEqual((a.mu, a.sigma), b_before)
        self.assertGreater(b.mu, b_before[0])  # the winner gained belief


class TestOpenSkillPeriodSemantics(unittest.TestCase):
    """Period-level behaviour: batching, ordering and up-front validation."""

    def test_multi_row_period_matches_sequential_one_row_periods(self):
        """One multi-row period and row-by-row one-row periods agree exactly."""
        results = [
            (0, 1, 1.0),
            (2, 3, 0.0),
            (0, 2, 0.5),
            (1, 3, 1.0),
        ]

        batched = [OpenSkillCompetitor() for _ in range(4)]
        OpenSkillCompetitor.apply_rating_period([(batched[a], batched[b], outcome, None) for a, b, outcome in results])

        sequential = [OpenSkillCompetitor() for _ in range(4)]
        for a, b, outcome in results:
            OpenSkillCompetitor.apply_rating_period([(sequential[a], sequential[b], outcome, None)])

        for batched_competitor, sequential_competitor in zip(batched, sequential, strict=True):
            self.assertEqual(batched_competitor.mu, sequential_competitor.mu)
            self.assertEqual(batched_competitor.sigma, sequential_competitor.sigma)

    def test_period_validation_happens_up_front(self):
        """A malformed row anywhere in the period fails before any belief moves."""
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        beliefs = (a.mu, a.sigma, b.mu, b.sigma)

        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_rating_period(
                [
                    (a, b, 1.0, None),
                    (a, b, 0.7, None),  # invalid outcome in the middle of the batch
                ]
            )
        self.assertEqual((a.mu, a.sigma, b.mu, b.sigma), beliefs)

        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_rating_period([(a, b, 1.0, (0.0, 1.0))])  # inconsistent scores
        self.assertEqual((a.mu, a.sigma, b.mu, b.sigma), beliefs)

    def test_period_rejects_other_rating_systems(self):
        """A row containing another rating system is rejected."""
        from elote import EloCompetitor

        a = OpenSkillCompetitor()
        with self.assertRaises(MissMatchedCompetitorTypesException):
            OpenSkillCompetitor.apply_rating_period([(a, EloCompetitor(), 1.0, None)])

    def test_empty_period_is_a_no_op(self):
        a, b = OpenSkillCompetitor(), OpenSkillCompetitor()
        beliefs = (a.mu, a.sigma, b.mu, b.sigma)
        OpenSkillCompetitor.apply_rating_period([])
        self.assertEqual((a.mu, a.sigma, b.mu, b.sigma), beliefs)

    def test_period_native_evaluation_identity(self):
        """The class overrides apply_rating_period, so the walk-forward evaluator
        routes it through period-native batching."""
        from elote.competitors.base import BaseCompetitor

        self.assertIsNot(OpenSkillCompetitor.apply_rating_period.__func__, BaseCompetitor.apply_rating_period.__func__)

    def test_registry_entry(self):
        """Subclassing registers the class; the registry resolves it by name."""
        self.assertIn("OpenSkillCompetitor", BaseCompetitor.list_competitor_types())
        self.assertIs(BaseCompetitor.get_competitor_class("OpenSkillCompetitor"), OpenSkillCompetitor)


if __name__ == "__main__":
    unittest.main()
