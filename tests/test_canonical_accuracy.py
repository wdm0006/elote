"""Canonical accuracy regression suite.

Pins every rating system to the canonical reference values from the accuracy audit
(research findings ``art_dQPENpEX`` §4): Glickman (1999) and Glickman (2012a) worked
examples, Herbrich's two-player TrueSkill factor update, the Colley (2002), Massey
(1997) and Keener (1993) constructions, James' log5, the ECF and DWZ rules, and the
structural properties of the iterative systems (Bradley-Terry MM, WHR, Blended).

These anchors were independently confirmed to match elote's output at
float-appropriate tolerances (1e-3 where the paper rounds its printed values,
1e-6..1e-9 for closed forms). Their job is regression protection: any edit that
drifts the update or expected-score math moves a pinned value and fails here.
"""

import datetime

import pytest
from elote import (
    BlendedCompetitor,
    BradleyTerryCompetitor,
    ColleyMatrixCompetitor,
    DWZCompetitor,
    ECFCompetitor,
    EloCompetitor,
    Glicko2Competitor,
    GlickoBoostCompetitor,
    GlickoCompetitor,
    KeenerCompetitor,
    MasseyCompetitor,
    PythagoreanCompetitor,
    TrueSkillCompetitor,
    WholeHistoryRatingCompetitor,
)


class TestEloCanonical:
    """Elo logistic expected score and K=32 update (Elo 1978 / standard formulation)."""

    def test_expected_score_200_point_gap(self) -> None:
        # E = 1 / (1 + 10 ** (ΔR / 400)); audit anchor E = 0.759747 (tol 1e-6).
        higher = EloCompetitor(initial_rating=1200, k_factor=32)
        lower = EloCompetitor(initial_rating=1000, k_factor=32)

        assert higher.expected_score(lower) == pytest.approx(0.759747, abs=1e-6)
        assert lower.expected_score(higher) == pytest.approx(1 - 0.759747, abs=1e-6)

    def test_beat_k32_1200_vs_1000(self) -> None:
        # 1200 + 32 * (1 - 0.7597469266...) = 1207.6880983; audit prints 1207.688/992.312.
        winner = EloCompetitor(initial_rating=1200, k_factor=32)
        loser = EloCompetitor(initial_rating=1000, k_factor=32)

        winner.beat(loser)

        assert winner.rating == pytest.approx(1207.6880983472654, abs=1e-6)
        assert loser.rating == pytest.approx(992.3119016527346, abs=1e-6)

    def test_beat_symmetric_pair_exchanges_k_half(self) -> None:
        # Equal ratings give E = 0.5, so a K=32 win moves each side exactly K/2.
        winner = EloCompetitor(initial_rating=1500, k_factor=32)
        loser = EloCompetitor(initial_rating=1500, k_factor=32)

        winner.beat(loser)

        assert winner.rating == pytest.approx(1516.0, abs=1e-6)
        assert loser.rating == pytest.approx(1484.0, abs=1e-6)


class TestGlickoCanonical:
    """Glickman (1999) worked example: 1500/200 beats 1400/30, q = ln(10)/400."""

    def test_expected_score_worked_example(self) -> None:
        winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)

        # Paper prints E = 0.6395 at 4dp.
        assert winner.expected_score(loser) == pytest.approx(0.6395, abs=1e-3)

    def test_beat_worked_example_both_sides(self) -> None:
        # Paper values are rounded to 2dp -> abs tolerance 1e-3 (audit protocol).
        winner = GlickoCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoCompetitor(initial_rating=1400, initial_rd=30)

        winner.beat(loser)

        assert winner.rating == pytest.approx(1563.432, abs=1e-3)
        assert winner.rd == pytest.approx(175.220, abs=1e-3)
        assert loser.rating == pytest.approx(1398.343, abs=1e-3)
        assert loser.rd == pytest.approx(29.925, abs=1e-3)


class TestGlicko2Canonical:
    """Glickman (2012a) five-step worked example: 1500/200/0.06 beats 1400/30."""

    def test_beat_worked_example_including_opponent(self) -> None:
        # Default volatility is the paper's sigma = 0.06. The paper prints r' = 1563.56,
        # RD' = 175.40; the audit pins the same values at 1e-4 on the full-precision
        # output, including the opponent side of the update.
        winner = Glicko2Competitor(initial_rating=1500, initial_rd=200)
        loser = Glicko2Competitor(initial_rating=1400, initial_rd=30)

        winner.beat(loser)

        assert winner.rating == pytest.approx(1563.5641943063383, abs=1e-4)
        assert winner.rd == pytest.approx(175.402655938555, abs=1e-4)
        assert winner.volatility == pytest.approx(0.059998657304847616, abs=1e-4)

        assert loser.rating == pytest.approx(1398.1435582337338, abs=1e-4)
        assert loser.rd == pytest.approx(31.67021528115062, abs=1e-4)

    def test_expected_score_matches_glicko_worked_example(self) -> None:
        # The g(RD)-scaled logistic agrees with the Glicko example at 4dp.
        winner = Glicko2Competitor(initial_rating=1500, initial_rd=200)
        loser = Glicko2Competitor(initial_rating=1400, initial_rd=30)

        assert winner.expected_score(loser) == pytest.approx(0.6395, abs=1e-3)


class TestTrueSkillCanonical:
    """Herbrich et al. (2006) two-player factor-graph update, beta = 4.166."""

    def test_beat_draw_free_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # draw_probability = 0 zeroes the draw margin (epsilon = 0).
        monkeypatch.setattr(TrueSkillCompetitor, "_draw_probability", 0.0)

        winner = TrueSkillCompetitor()
        loser = TrueSkillCompetitor()

        winner.beat(loser)

        assert winner.mu == pytest.approx(29.2054, abs=1e-3)
        assert winner.sigma == pytest.approx(7.1945, abs=1e-3)
        assert loser.mu == pytest.approx(20.7946, abs=1e-3)
        assert loser.sigma == pytest.approx(7.1945, abs=1e-3)

    def test_beat_default_draw_probability(self) -> None:
        # The shipped default draw_probability = 0.10 sets epsilon = sqrt(2) * beta *
        # Phi^-1((p + 1) / 2), which pulls both means toward each other.
        winner = TrueSkillCompetitor()
        loser = TrueSkillCompetitor()

        winner.beat(loser)

        assert winner.mu == pytest.approx(29.3957, abs=1e-3)
        assert winner.sigma == pytest.approx(7.1711, abs=1e-3)
        assert loser.mu == pytest.approx(20.6043, abs=1e-3)
        assert loser.sigma == pytest.approx(7.1711, abs=1e-3)

    def test_rating_is_conservative_mu_minus_3_sigma(self) -> None:
        # Audit note: .rating is the conservative mu - 3 * sigma; mu/sigma come from
        # .mu/.sigma. Pin that convention so it cannot change silently.
        winner = TrueSkillCompetitor()
        loser = TrueSkillCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(winner.mu - 3 * winner.sigma, abs=1e-9)
        assert loser.rating == pytest.approx(loser.mu - 3 * loser.sigma, abs=1e-9)


class TestColleyCanonical:
    """Colley (2002) matrix method: symmetric 2-game schedule, ratings on (0, 1)."""

    def test_single_game_ratings(self) -> None:
        # One game between equals: (0.5 + n/2) / (2 + n) with n = 1 -> 0.625 / 0.375.
        winner = ColleyMatrixCompetitor()
        loser = ColleyMatrixCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(0.625, abs=1e-9)
        assert loser.rating == pytest.approx(0.375, abs=1e-9)

    def test_two_game_sweep_ratings(self) -> None:
        # 2-0 series: (0.5 + 2/2) / (2 + 2) = 2/3 and 1/3 (audit prints 0.667/0.333).
        winner = ColleyMatrixCompetitor()
        loser = ColleyMatrixCompetitor()

        winner.beat(loser)
        winner.beat(loser)

        assert winner.rating == pytest.approx(2 / 3, abs=1e-9)
        assert loser.rating == pytest.approx(1 / 3, abs=1e-9)


class TestMasseyCanonical:
    """Massey (1997) least squares: rating difference equals observed point margin."""

    def test_two_point_win_gives_two_point_rating_gap(self) -> None:
        winner = MasseyCompetitor()
        loser = MasseyCompetitor()

        winner.beat(loser, scores=(2, 0))

        assert winner.rating - loser.rating == pytest.approx(2.0, abs=1e-9)
        assert winner.rating == pytest.approx(1.0, abs=1e-9)
        assert loser.rating == pytest.approx(-1.0, abs=1e-9)


class TestKeenerCanonical:
    """Keener (1993) skew-function Perron rating with elote's documented defaults
    (games_prior = 2.0, perturbation = 1e-4, keener.py); ratings are mean-1 (sum 2)."""

    def test_single_game_ratings(self) -> None:
        winner = KeenerCompetitor()
        loser = KeenerCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(1.317604, abs=1e-6)
        assert loser.rating == pytest.approx(0.682396, abs=1e-6)

    def test_ratings_are_mean_one(self) -> None:
        winner = KeenerCompetitor()
        loser = KeenerCompetitor()

        winner.beat(loser)

        assert winner.rating + loser.rating == pytest.approx(2.0, abs=1e-6)


class TestPythagoreanCanonical:
    """James log5 with the documented Pythagorean exponent k = 2.37 and a symmetric
    one-point prior; a single 3-1 win fixes both points shares exactly."""

    def test_single_3_1_win_ratings(self) -> None:
        # Winner share: 1 / (1 + ((1+1)/(3+1)) ** 2.37) = 0.837910; loser is its complement.
        winner = PythagoreanCompetitor()
        loser = PythagoreanCompetitor()

        winner.beat(loser, scores=(3, 1))

        assert winner.rating == pytest.approx(0.8379099807553965, abs=1e-9)
        assert loser.rating == pytest.approx(0.16209001924460362, abs=1e-9)

    def test_log5_expected_score(self) -> None:
        # log5: x(1-y) / (x(1-y) + y(1-x)) on the two points shares.
        winner = PythagoreanCompetitor()
        loser = PythagoreanCompetitor()

        winner.beat(loser, scores=(3, 1))

        assert winner.expected_score(loser) == pytest.approx(0.9639286249635484, abs=1e-9)


class TestECFCanonical:
    """ECF linear-points rule: delta = 50 per win, handicap-free window mean."""

    def test_single_game_exchange(self) -> None:
        # Both start at 100; a first win moves each side exactly delta = 50.
        winner = ECFCompetitor()
        loser = ECFCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(125, abs=1e-9)
        assert loser.rating == pytest.approx(75, abs=1e-9)


class TestDWZCanonical:
    """Ingo/DWZ update with the E0 boundary blend and braking terms:
    400 vs 400, both players unranked (i = 1, so the winner's E clamps to the
    5 * i floor-side bound while the loser's braking value B caps E at 150)."""

    def test_beat_400_vs_400(self) -> None:
        # Winner: 400 + (800 / (5 + 1)) * 0.5 = 466.667.
        # Loser: braking B = e^((1300-400)/150) - 1 caps E at 150:
        # 400 + (800 / (150 + 1)) * (-0.5) = 397.351.
        winner = DWZCompetitor()
        loser = DWZCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(466.6666666666667, abs=1e-6)
        assert loser.rating == pytest.approx(397.35099337748346, abs=1e-6)


class TestGlickoBoostCanonical:
    """GlickoBoost (combined-RD expected score): 1500/200 vs 1400/200."""

    def test_expected_score_combined_rd(self) -> None:
        # g(sqrt(200^2 + 200^2)) scaling of the 100-point gap -> 0.605485.
        winner = GlickoBoostCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoBoostCompetitor(initial_rating=1400, initial_rd=200)

        assert winner.expected_score(loser) == pytest.approx(0.6054850321469811, abs=1e-6)

    def test_beat_symmetric_update_and_rd(self) -> None:
        # Complementary expected scores give a symmetric ±50.98 rating exchange and
        # identical post-update RDs of 181.58 (audit anchor, tol 1e-6).
        winner = GlickoBoostCompetitor(initial_rating=1500, initial_rd=200)
        loser = GlickoBoostCompetitor(initial_rating=1400, initial_rd=200)

        winner.beat(loser)

        assert winner.rating == pytest.approx(1550.980513989161, abs=1e-6)
        assert winner.rd == pytest.approx(181.58269232705803, abs=1e-6)
        assert loser.rating == pytest.approx(1349.019486010839, abs=1e-6)
        assert loser.rd == pytest.approx(181.58269232705803, abs=1e-6)


class TestBradleyTerryCanonical:
    """Bradley-Terry MM (mean-retrace) maximum likelihood, iteration-capped at 10k
    (tol 1e-3: a single 1-0 game diverges, so the cap location is part of the model)."""

    def test_single_game_ratings(self) -> None:
        winner = BradleyTerryCompetitor()
        loser = BradleyTerryCompetitor()

        winner.beat(loser)

        assert winner.rating == pytest.approx(1739.4008690243395, abs=1e-3)
        assert loser.rating == pytest.approx(1260.5991309756605, abs=1e-3)

    def test_expected_score_after_single_game(self) -> None:
        # Logistic on the fitted Elo-scale ratings; audit anchor E = 0.9403.
        winner = BradleyTerryCompetitor()
        loser = BradleyTerryCompetitor()

        winner.beat(loser)

        assert winner.expected_score(loser) == pytest.approx(0.9403, abs=1e-3)
        assert loser.expected_score(winner) == pytest.approx(1 - 0.9403, abs=1e-3)


class TestWHRCanonical:
    """Whole-History Rating (Coulom 2008): per-day Bradley-Terry factors with a
    Gaussian prior; structural anchors plus the single-game audit value."""

    def test_single_game_symmetric_revision(self) -> None:
        # One win at high solver precision: +0.859 / -0.859 around the 1500 prior.
        winner = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        loser = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        start = datetime.datetime(2020, 1, 1)

        winner.beat(loser, start)

        assert winner.rating == pytest.approx(1500.8591987718726, abs=1e-3)
        assert loser.rating == pytest.approx(1499.1408012281274, abs=1e-3)

    def test_structural_properties(self) -> None:
        # winner > loser, revisions mirror around the prior, and more results pull the
        # winner further up (WHR's whole-history refit must be monotone here).
        first_winner = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        first_loser = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        start = datetime.datetime(2020, 1, 1)

        first_winner.beat(first_loser, start)

        assert first_winner.rating > first_loser.rating
        assert first_winner.rating - 1500 == pytest.approx(1500 - first_loser.rating, abs=1e-6)

        sweep_winner = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        sweep_loser = WholeHistoryRatingCompetitor(precision=1e-10, max_iterations=100)
        sweep_winner.beat(sweep_loser, start)
        for day in range(1, 4):
            sweep_winner.beat(sweep_loser, start + datetime.timedelta(days=day))

        assert sweep_winner.rating > first_winner.rating
        assert sweep_loser.rating < first_loser.rating


class TestBlendedCanonical:
    """BlendedCompetitor mean mode: blend of member expected scores, member updates on beat."""

    @staticmethod
    def _specs() -> list:
        return [
            {"type": "EloCompetitor", "competitor_kwargs": {}},
            {"type": "GlickoCompetitor", "competitor_kwargs": {}},
        ]

    def test_equal_pair_expected_score_is_half(self) -> None:
        # Two identically-configured blends: every member agrees at 0.5.
        first = BlendedCompetitor(competitors=self._specs(), blend_mode="mean")
        second = BlendedCompetitor(competitors=self._specs(), blend_mode="mean")

        assert first.expected_score(second) == pytest.approx(0.5, abs=1e-9)

    def test_beat_updates_members_and_flips_expected_score(self) -> None:
        winner = BlendedCompetitor(competitors=self._specs(), blend_mode="mean")
        loser = BlendedCompetitor(competitors=self._specs(), blend_mode="mean")
        member_ratings_before = [member.rating for member in winner.sub_competitors]

        winner.beat(loser)

        assert winner.expected_score(loser) > 0.5
        assert winner.rating > loser.rating
        for member, rating_before in zip(winner.sub_competitors, member_ratings_before, strict=True):
            assert member.rating > rating_before
