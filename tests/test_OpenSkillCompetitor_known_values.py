"""Reference-value tests for OpenSkillCompetitor against the canonical implementation.

The ``openskill`` package (dev extra only, pinned in uv.lock) is the test
oracle: a fixed battery of bouts -- 1v1 win/loss/draw, ties, repeated
sequences, and N-way bouts -- is applied to both elote's native Weng-Lin
implementation and the oracle's ``rate`` for every model variant, and elote's
mu/sigma must match to 1e-6. The battery covers all four variants behind the
``model`` selector: ``plackett_luce`` (Algorithm 4), ``bradley_terry_full``
(Algorithm 1), ``bradley_terry_partial`` (Algorithm 2) and ``thurstone``
(Algorithm 3).

Bouts here are bouts between single-member teams -- every participant is its
own team, the only bout shape elote's surface accepts (multi-member rosters
are deferred, see the research notes on roster-level bout input). N-way bouts
are therefore genuine multi-team bouts from the oracle's perspective.

One documented deviation: openskill.py 6.2.0's tie adjustment for mu is a
no-op. Its ``_compute`` mutates the rating objects it later subtracts from,
so ``result[i][0].mu - original_teams[i][0].mu`` is always zero and tied
teams keep their raw per-player mu changes (upstream issue #201, fix in
PR #203 -- not released as of 6.2.0, the latest PyPI version). The same dead
block sits in the Plackett-Luce, Bradley-Terry full and Thurstone-Mosteller
full models; the partial-pairing model has no averaging block at all. Elote
implements the documented rule the oracle's dead code intends: every tied
player receives the *average* mu change of the tied group, preserving each
player's own prior. The battery therefore compares strictly wherever the
oracle is correct (all sigmas, all untied mus) and compares the tied group's
*mean* mu change -- which both implementations agree on to well under 1e-6.
"""

import unittest

try:
    from openskill.models.weng_lin.bradley_terry_full import BradleyTerryFull
    from openskill.models.weng_lin.bradley_terry_part import BradleyTerryPart
    from openskill.models.weng_lin.plackett_luce import PlackettLuce
    from openskill.models.weng_lin.thurstone_mosteller_full import ThurstoneMostellerFull

    HAS_ORACLE = True
except ImportError:  # pragma: no cover - only hit when dev extras are absent
    HAS_ORACLE = False

from elote import BaseCompetitor, OpenSkillCompetitor
from elote.competitors.base import (
    InvalidParameterException,
    MissMatchedCompetitorTypesException,
)

TOLERANCE = 1e-6

ORACLE_MODELS = {
    "plackett_luce": "PlackettLuce",
    "bradley_terry_full": "BradleyTerryFull",
    "bradley_terry_partial": "BradleyTerryPart",
    "thurstone": "ThurstoneMostellerFull",
}
if HAS_ORACLE:
    ORACLE_MODELS = {
        "plackett_luce": PlackettLuce,
        "bradley_terry_full": BradleyTerryFull,
        "bradley_terry_partial": BradleyTerryPart,
        "thurstone": ThurstoneMostellerFull,
    }

VARIANTS = tuple(ORACLE_MODELS)


class ReferenceHarness:
    """Drives elote competitors and the openskill.py oracle through identical bouts."""

    def __init__(self, n_players: int, model: str = "plackett_luce") -> None:
        self.model_name = model
        self.model = ORACLE_MODELS[model]()
        self.competitors = [OpenSkillCompetitor(model=model) for _ in range(n_players)]
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
    """Reference-value battery: elote vs openskill.py to 1e-6, per variant."""

    def test_single_bouts(self):
        """1v1 win, loss and draw each match the oracle, for every variant."""
        for variant in VARIANTS:
            for name, ranks in (("win", (0, 1)), ("loss", (1, 0)), ("draw", (0, 0))):
                with self.subTest(model=variant, bout=name):
                    harness = ReferenceHarness(2, model=variant)
                    harness.apply_bout((0, 1), ranks, self, f"1v1 {variant} {name}")

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
        for variant in VARIANTS:
            harness = ReferenceHarness(4, model=variant)
            for step, (indices, ranks) in enumerate(sequence):
                harness.apply_bout(indices, ranks, self, f"{variant} sequence step {step} ({indices}, {ranks})")

    def test_tie_bouts_match_oracle(self):
        """Tied-rank bouts from fresh state: sigmas and untied mus match the
        oracle to 1e-6, tied players match the oracle's mean mu change."""
        for variant in VARIANTS:
            for indices, ranks in (
                ((0, 1), (0, 0)),  # two-player draw
                ((0, 1, 2), (0, 0, 1)),  # tie for first in a 3-way
                ((0, 1, 2), (0, 1, 1)),  # tie for second in a 3-way
                ((0, 1, 2, 3), (0, 0, 0, 0)),  # all tied
                ((0, 1, 2, 3), (0, 0, 1, 2)),  # tie for first in a 4-way
                ((0, 1, 2, 3), (0, 1, 1, 3)),  # tie for second, non-contiguous ranks
            ):
                with self.subTest(model=variant, bout=(indices, ranks)):
                    harness = ReferenceHarness(max(indices) + 1, model=variant)
                    harness.apply_bout(indices, ranks, self, f"{variant} tie bout ({indices}, {ranks})")

    def test_multi_team_bouts_match_oracle(self):
        """Wide multi-team fields match the oracle, including the partial-pairing window.

        Twelve single-member teams with distinct ranks exercise
        ``bradley_terry_partial``'s bounded pairing window (the default window
        of 4 clips for fields larger than 2*4+1 participants), and a second
        field mixes ties across the whole rank range. With a field this wide
        the oracle's partial pairing differs from its full pairing, so this is
        where the window/average reading of Algorithm 2 is actually pinned.
        """
        for variant in VARIANTS:
            with self.subTest(model=variant, bout="12-way distinct ranks"):
                harness = ReferenceHarness(12, model=variant)
                harness.apply_bout(tuple(range(12)), tuple(range(12)), self, f"{variant} 12-way field")
            with self.subTest(model=variant, bout="12-way with ties"):
                harness = ReferenceHarness(12, model=variant)
                harness.apply_bout(
                    tuple(range(12)),
                    (0, 1, 1, 3, 3, 3, 6, 7, 8, 8, 10, 11),
                    self,
                    f"{variant} 12-way ties",
                )

    def test_pairwise_methods_match_oracle(self):
        """beat/lost_to/tied route one-row periods that match rate() calls, per variant."""
        for variant in VARIANTS:
            for name, method, ranks in (
                ("beat", "beat", (0, 1)),
                ("lost_to", "lost_to", (1, 0)),
                ("tied", "tied", (0, 0)),
            ):
                with self.subTest(model=variant, method=name):
                    model = ORACLE_MODELS[variant]()
                    a, b = OpenSkillCompetitor(model=variant), OpenSkillCompetitor(model=variant)
                    ra, rb = model.rating(), model.rating()

                    getattr(a, method)(b)
                    updated = model.rate([[ra], [rb]], ranks=list(ranks))

                    self.assertAlmostEqual(a.mu, updated[0][0].mu, delta=TOLERANCE)
                    self.assertAlmostEqual(a.sigma, updated[0][0].sigma, delta=TOLERANCE)
                    self.assertAlmostEqual(b.mu, updated[1][0].mu, delta=TOLERANCE)
                    self.assertAlmostEqual(b.sigma, updated[1][0].sigma, delta=TOLERANCE)

    def test_expected_score_matches_predict_win(self):
        """The 1v1 win probability matches the oracle's two-team predict_win, per variant."""
        for variant in VARIANTS:
            with self.subTest(model=variant):
                model = ORACLE_MODELS[variant]()
                a, b = OpenSkillCompetitor(model=variant), OpenSkillCompetitor(model=variant)
                c = OpenSkillCompetitor(initial_mu=32.0, initial_sigma=5.0, model=variant)

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
        for variant in VARIANTS:
            with self.subTest(model=variant):
                model = ORACLE_MODELS[variant]()
                winner, loser = OpenSkillCompetitor(model=variant), OpenSkillCompetitor(model=variant)
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
        for variant in VARIANTS:
            with self.subTest(model=variant):
                harness = ReferenceHarness(2, model=variant)
                for step in range(10):
                    harness.apply_bout((0, 1), (1, 0), self, f"{variant} loss streak step {step}")  # player 1 keeps beating player 0
                self.assertLess(harness.competitors[0].rating, OpenSkillCompetitor._minimum_rating)


@unittest.skipUnless(HAS_ORACLE, "the openskill oracle (dev extra) is not installed")
class TestOpenSkillTieRule(unittest.TestCase):
    """The documented tie rule: tied players share the average mu change."""

    def test_tied_players_share_the_average_change_not_the_absolute_mu(self):
        """Distinct priors survive a tie; both players move by the same amount."""
        for variant in VARIANTS:
            with self.subTest(model=variant):
                a = OpenSkillCompetitor(initial_mu=30.0, initial_sigma=8.0, model=variant)
                b = OpenSkillCompetitor(initial_mu=20.0, initial_sigma=9.0, model=variant)
                c = OpenSkillCompetitor(initial_mu=27.0, initial_sigma=6.0, model=variant)

                OpenSkillCompetitor.apply_bout([a, b, c], ranks=[0, 0, 1])

                change_a = a.mu - 30.0
                change_b = b.mu - 20.0
                self.assertAlmostEqual(change_a, change_b, delta=TOLERANCE)
                # Tied players keep their own priors; only the change is shared.
                self.assertAlmostEqual(a.mu - b.mu, 30.0 - 20.0, delta=TOLERANCE)

    def test_two_player_draw_leaves_mu_untouched(self):
        """A fresh even draw changes no mu and shrinks both sigmas, for every variant."""
        for variant in VARIANTS:
            with self.subTest(model=variant):
                a, b = OpenSkillCompetitor(model=variant), OpenSkillCompetitor(model=variant)
                OpenSkillCompetitor.apply_bout([a, b], ranks=[0, 0])
                self.assertEqual(a.mu, 25.0)
                self.assertEqual(b.mu, 25.0)
                self.assertLess(a.sigma, 25.0 / 3.0)
                self.assertEqual(a.sigma, b.sigma)

    def test_decisive_result_moves_beliefs_in_the_right_direction(self):
        """beat raises the winner's mu, lowers the loser's, and shrinks both sigmas."""
        for variant in VARIANTS:
            with self.subTest(model=variant):
                winner, loser = OpenSkillCompetitor(model=variant), OpenSkillCompetitor(model=variant)
                winner.beat(loser)
                self.assertGreater(winner.mu, 25.0)
                self.assertLess(loser.mu, 25.0)
                self.assertLess(winner.sigma, 25.0 / 3.0)
                self.assertLess(loser.sigma, 25.0 / 3.0)


class TestOpenSkillCompetitorContract(unittest.TestCase):
    """E-A surface contracts that do not need the oracle."""

    def test_model_parameter_defaults_to_plackett_luce(self):
        """The variant selector defaults to plackett_luce and is carried on the instance."""
        self.assertEqual(OpenSkillCompetitor()._model, "plackett_luce")
        self.assertEqual(OpenSkillCompetitor(model="plackett_luce")._model, "plackett_luce")

    def test_every_variant_is_accepted_and_carried(self):
        """All four family members construct and report their variant."""
        for variant in ("plackett_luce", "bradley_terry_full", "bradley_terry_partial", "thurstone"):
            with self.subTest(model=variant):
                self.assertEqual(OpenSkillCompetitor(model=variant)._model, variant)

    def test_unknown_model_variant_rejected(self):
        """Variants outside the selector raise InvalidParameterException."""
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(model="bradley_terry")  # the full/partial suffix is required
        with self.assertRaises(InvalidParameterException):
            OpenSkillCompetitor(model="elo")

    def test_mixed_variant_bout_rejected(self):
        """A bout mixing Weng-Lin variants is ambiguous and rejected before any update."""
        a = OpenSkillCompetitor(model="plackett_luce")
        b = OpenSkillCompetitor(model="thurstone")
        beliefs = (a.mu, a.sigma, b.mu, b.sigma)
        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_bout([a, b], ranks=[0, 1])
        self.assertEqual((a.mu, a.sigma, b.mu, b.sigma), beliefs)

    def test_mixed_variant_period_row_rejected(self):
        """A period row mixing variants fails up-front, before any belief moves."""
        a = OpenSkillCompetitor(model="bradley_terry_full")
        b = OpenSkillCompetitor(model="bradley_terry_partial")
        beliefs = (a.mu, a.sigma, b.mu, b.sigma)
        with self.assertRaises(ValueError):
            OpenSkillCompetitor.apply_rating_period([(a, b, 1.0, None)])
        self.assertEqual((a.mu, a.sigma, b.mu, b.sigma), beliefs)

    def test_serialization_round_trip_preserves_the_variant(self):
        """export_state/from_state round-trips the model selection."""
        for variant in ("plackett_luce", "bradley_terry_full", "bradley_terry_partial", "thurstone"):
            with self.subTest(model=variant):
                competitor = OpenSkillCompetitor(initial_mu=28.0, initial_sigma=6.0, model=variant)
                state = competitor.export_state()
                restored = OpenSkillCompetitor.from_state(state)
                self.assertIsInstance(restored, OpenSkillCompetitor)
                self.assertEqual(restored._model, variant)
                self.assertEqual((restored.mu, restored.sigma), (competitor.mu, competitor.sigma))

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
