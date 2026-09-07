"""Tests for ``LambdaArena.match_group`` -- N-way and roster bouts.

Covers free-for-alls of three or more players, ties, team rosters with
per-member updates, the pre-update prediction record, up-front validation that
leaves the arena unchanged, exclusion of N-way bouts from the two-sided
analytics, and the non-equivalence of a native N-way bout versus its pairwise
fan-out decomposition.

The two-player paths are pinned byte-identically by the golden battery in
``tests/test_two_player_goldens.py``. The N-way math itself is pinned against
the ``openskill`` oracle (dev extra) here.
"""

import datetime
import unittest

import pytest

from elote import EloCompetitor, LambdaArena, OpenSkillCompetitor
from elote.arenas.base import Bout, MultiBout
from elote.competitors.base import MissMatchedCompetitorTypesException

try:
    from openskill.models.weng_lin.plackett_luce import PlackettLuce

    HAS_ORACLE = True
except ImportError:  # pragma: no cover - only hit when dev extras are absent
    HAS_ORACLE = False

TOLERANCE = 1e-6


def make_arena(**kwargs):
    """An OpenSkill arena whose comparison function never runs (explicit outcomes)."""
    return LambdaArena(lambda a, b: None, base_competitor=OpenSkillCompetitor, **kwargs)


class TestMatchGroupBasics:
    """Record-keeping and competitor creation."""

    def test_creates_missing_competitors(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus"])
        assert set(arena.competitors) == {"ada", "grace", "linus"}

    def test_multibout_recorded_with_fields(self):
        arena = make_arena()
        match_time = datetime.datetime(2026, 9, 7, 12, 0, tzinfo=datetime.timezone.utc)
        arena.match_group(
            ["ada", "grace", "linus"],
            attributes={"round": 1},
            match_time=match_time,
        )
        assert len(arena.history.bouts) == 1
        bout = arena.history.bouts[0]
        assert isinstance(bout, MultiBout)
        assert bout.participants == ["ada", "grace", "linus"]
        assert bout.ranks == [0, 1, 2]
        assert bout.predicted_ranks == ["ada", "grace", "linus"]
        assert bout.scores is None
        assert bout.attributes == {"round": 1}
        assert bout.match_time == match_time

    def test_two_player_bout_records_multibout(self):
        """match_group with two sides is valid: two sides with a ranked result."""
        arena = make_arena()
        arena.match_group(["ada", "linus"])
        assert isinstance(arena.history.bouts[0], MultiBout)

    def test_default_ranks_follow_participant_order(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus", "kurt"])
        assert arena.history.bouts[0].ranks == [0, 1, 2, 3]

    def test_explicit_ranks_with_ties(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus", "kurt"], ranks=[0, 1, 1, 3])
        assert arena.history.bouts[0].ranks == [0, 1, 1, 3]

    def test_scores_derive_ranks_with_shared_earliest_rank(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus"], scores=[3.0, 3.0, 1.0])
        assert arena.history.bouts[0].ranks == [0, 0, 2]
        assert arena.history.bouts[0].scores == [3.0, 3.0, 1.0]

    def test_ranks_win_over_scores(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus"], ranks=[0, 1, 2], scores=[10, 9, 8])
        bout = arena.history.bouts[0]
        assert bout.ranks == [0, 1, 2]
        assert bout.scores == [10.0, 9.0, 8.0]

    def test_tied_players_from_fresh_state_get_equal_updates(self):
        arena = make_arena()
        arena.match_group(["ada", "grace", "linus", "kurt"], ranks=[0, 1, 1, 3])
        grace, linus = arena.competitors["grace"], arena.competitors["linus"]
        assert grace.mu == linus.mu
        assert grace.sigma == linus.sigma

    def test_predicted_ranks_captured_pre_update(self):
        arena = make_arena()
        arena.matchup("ada", "linus", outcome=1.0)  # ada stronger than the defaults now
        arena.match_group(["ada", "grace", "linus"])
        assert arena.history.bouts[-1].predicted_ranks == ["ada", "grace", "linus"]

    def test_roster_sides_record_labels(self):
        arena = make_arena()
        arena.match_group([("red", ["ada", "grace"]), ("blue", ["linus", "kurt"])], ranks=[0, 0])
        bout = arena.history.bouts[0]
        assert bout.participants == ["red", "blue"]
        assert bout.ranks == [0, 0]
        assert set(arena.competitors) == {"ada", "grace", "linus", "kurt"}


class TestMatchGroupOpenSkillUpdates:
    """The bout-level OpenSkill update and its member-level team path."""

    def test_n_way_bout_is_not_the_pairwise_fanout(self):
        """A native 3-way bout must not equal its pairwise decomposition.

        The fan-out replays the same result as a beats b, a beats c, b beats c
        through the documented two-player path. The Weng-Lin update consumes the
        whole bout at once, so the two must differ.
        """
        n_way = make_arena()
        n_way.match_group(["a", "b", "c"], ranks=[0, 1, 2])

        fanout = make_arena()
        fanout.matchup("a", "b", outcome=1.0)
        fanout.matchup("a", "c", outcome=1.0)
        fanout.matchup("b", "c", outcome=1.0)

        for player in ("a", "b", "c"):
            assert n_way.competitors[player].mu != fanout.competitors[player].mu, player
            assert n_way.competitors[player].sigma != fanout.competitors[player].sigma, player

    def test_two_sided_group_bout_matches_matchup_path(self):
        """With two single sides a win/loss group bout behaves like matchup."""
        group = make_arena()
        group.match_group(["a", "b"], ranks=[0, 1])

        pairwise = make_arena()
        pairwise.matchup("a", "b", outcome=1.0)

        for player in ("a", "b"):
            assert group.competitors[player].mu == pytest.approx(pairwise.competitors[player].mu)
            assert group.competitors[player].sigma == pytest.approx(pairwise.competitors[player].sigma)

    def test_roster_bout_updates_every_member(self):
        arena = make_arena()
        arena.match_group([("red", ["ada", "grace"]), ("blue", ["linus", "kurt"])], ranks=[0, 1])
        for player in ("ada", "grace", "linus", "kurt"):
            member = arena.competitors[player]
            assert member.mu != OpenSkillCompetitor().mu, player
            assert member.sigma != OpenSkillCompetitor().sigma, player
        # Equal members of equal rosters receive equal updates.
        assert arena.competitors["ada"].mu == arena.competitors["grace"].mu
        assert arena.competitors["linus"].mu == arena.competitors["kurt"].mu
        # No wrapper competitor object is created for the team itself.
        assert "red" not in arena.competitors
        assert "blue" not in arena.competitors

    def test_roster_bout_matches_direct_apply_bout(self):
        """The arena path is the competitor-level bout update, end to end."""
        arena = make_arena()
        arena.match_group([("red", ["ada", "grace"]), ("blue", ["linus", "kurt"])], ranks=[0, 1])

        direct = {name: OpenSkillCompetitor() for name in ("ada", "grace", "linus", "kurt")}
        OpenSkillCompetitor.apply_bout(
            [[direct["ada"], direct["grace"]], [direct["linus"], direct["kurt"]]],
            ranks=[0, 1],
        )
        for name, member in direct.items():
            assert arena.competitors[name].mu == pytest.approx(member.mu, abs=TOLERANCE)
            assert arena.competitors[name].sigma == pytest.approx(member.sigma, abs=TOLERANCE)

    @unittest.skipUnless(HAS_ORACLE, "the openskill oracle (dev extra) is not installed")
    def test_roster_bout_matches_oracle(self):
        """A 1 + 2 + 1 side bout matches PlackettLuce.rate on nested teams."""
        import openskill  # noqa: F401  (import guard mirrors the known-values battery)

        model = PlackettLuce()
        elote_players = [OpenSkillCompetitor() for _ in range(4)]
        oracle_players = [model.rating() for _ in range(4)]

        sides = [[elote_players[0]], [elote_players[1], elote_players[2]], [elote_players[3]]]
        OpenSkillCompetitor.apply_bout(sides, ranks=[0, 1, 2])

        oracle_teams = [[oracle_players[0]], [oracle_players[1], oracle_players[2]], [oracle_players[3]]]
        updated = model.rate(oracle_teams, ranks=[0, 1, 2])

        for side_index, side in enumerate(sides):
            for member_index, member in enumerate(side):
                oracle_rating = updated[side_index][member_index]
                assert member.mu == pytest.approx(oracle_rating.mu, abs=TOLERANCE)
                assert member.sigma == pytest.approx(oracle_rating.sigma, abs=TOLERANCE)


class TestMatchGroupValidation:
    """Up-front validation: a malformed bout leaves the arena unchanged."""

    def assert_arena_unchanged(self, arena):
        assert arena.competitors == {}
        assert arena.history.bouts == []

    def test_fewer_than_two_sides(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="at least two sides"):
            arena.match_group(["ada"])
        self.assert_arena_unchanged(arena)

    def test_rank_count_mismatch(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="one entry per participant"):
            arena.match_group(["ada", "grace", "linus"], ranks=[0, 1])
        self.assert_arena_unchanged(arena)

    def test_fractional_ranks_rejected(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="whole numbers"):
            arena.match_group(["ada", "grace"], ranks=[0.5, 1])
        self.assert_arena_unchanged(arena)

    def test_non_numeric_ranks_rejected(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="only numbers"):
            arena.match_group(["ada", "grace"], ranks=["first", 1])
        self.assert_arena_unchanged(arena)

    def test_boolean_ranks_rejected(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="only numbers"):
            arena.match_group(["ada", "grace"], ranks=[True, 1])

    def test_score_count_mismatch(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="one entry per participant"):
            arena.match_group(["ada", "grace", "linus"], scores=[1.0, 2.0])
        self.assert_arena_unchanged(arena)

    def test_negative_scores_rejected(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="non-negative"):
            arena.match_group(["ada", "grace"], scores=[1.0, -0.5])
        self.assert_arena_unchanged(arena)

    def test_non_finite_scores_rejected(self):
        arena = make_arena()
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                arena.match_group(["ada", "grace"], scores=[1.0, bad])
        self.assert_arena_unchanged(arena)

    def test_duplicate_competitor_across_sides(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="two sides"):
            arena.match_group(["ada", "ada", "grace"])
        self.assert_arena_unchanged(arena)

    def test_duplicate_member_across_rosters(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="two sides"):
            arena.match_group([("red", ["ada", "grace"]), ("blue", ["grace", "kurt"])], ranks=[0, 1])
        self.assert_arena_unchanged(arena)

    def test_empty_roster_rejected(self):
        arena = make_arena()
        with pytest.raises(ValueError, match="empty roster"):
            arena.match_group([("red", []), ("blue", ["linus"])])
        self.assert_arena_unchanged(arena)

    def test_mismatched_seeded_competitor_rejected(self):
        arena = LambdaArena(lambda a, b: None, base_competitor=OpenSkillCompetitor)
        arena.competitors["elo_seed"] = EloCompetitor()
        with pytest.raises(MissMatchedCompetitorTypesException):
            arena.match_group(["elo_seed", "ada"])
        assert set(arena.competitors) == {"elo_seed"}
        assert arena.history.bouts == []

    def test_elo_arena_not_implemented(self):
        arena = LambdaArena(lambda a, b: True, base_competitor=EloCompetitor)
        with pytest.raises(NotImplementedError, match="apply_bout"):
            arena.match_group(["ada", "grace"])
        assert arena.competitors == {}


class TestNWayAnalyticsExclusion:
    """N-way bouts are recorded but excluded from the two-sided analytics."""

    def make_history(self):
        arena = make_arena()
        arena.matchup("ada", "linus", outcome=1.0)
        arena.match_group(["ada", "grace", "linus"])
        return arena

    def test_history_holds_both_types(self):
        arena = self.make_history()
        assert len(arena.history.bouts) == 2
        assert isinstance(arena.history.bouts[0], Bout)
        assert isinstance(arena.history.bouts[1], MultiBout)
        assert len(arena.history._pairwise_bouts()) == 1

    def test_report_covers_pairwise_only(self):
        arena = self.make_history()
        assert len(arena.history.report_results()) == 1

    def test_confusion_matrix_counts_pairwise_only(self):
        arena = self.make_history()
        matrix = arena.history.confusion_matrix()
        assert matrix["tp"] + matrix["fp"] + matrix["tn"] + matrix["fn"] == 1

    def test_metrics_draws_counts_pairwise_only(self):
        arena = self.make_history()
        metrics = arena.history.calculate_metrics_with_draws()
        matrix = metrics["confusion_matrix"]
        assert matrix["tp"] + matrix["fp"] + matrix["tn"] + matrix["fn"] == 1

    def test_calibration_data_pairwise_only(self):
        arena = self.make_history()
        y_true, y_prob = arena.history.get_calibration_data()
        assert len(y_true) == 1
        assert len(y_prob) == 1

    def test_random_search_runs_with_mixed_history(self):
        arena = self.make_history()
        accuracy, thresholds = arena.history.random_search(trials=10, seed=7)
        assert 0.0 <= accuracy <= 1.0
        assert len(thresholds) == 2

    def test_accuracy_by_prior_bouts_skips_n_way(self):
        arena = self.make_history()
        evaluation = make_arena()
        evaluation.matchup("ada", "linus", outcome=1.0)
        evaluation.match_group(["ada", "grace", "linus"])
        # The evaluation history supplies the bouts to bin; the arena supplies
        # the prior-bout counts. Only the pairwise evaluation bout is binned.
        results = evaluation.history.accuracy_by_prior_bouts(arena)
        total = sum(bin_entry["total"] for bin_entry in results["binned"].values())
        assert total == 1


class TestScoreDerivedRankMath:
    """Unit checks for the pure rank-derivation helper."""

    def test_descending_scores(self):
        from elote.arenas.lambda_arena import _ranks_from_scores

        assert _ranks_from_scores([1.0, 2.0, 3.0]) == [2, 1, 0]

    def test_ties_share_earliest_rank(self):
        from elote.arenas.lambda_arena import _ranks_from_scores

        assert _ranks_from_scores([3.0, 3.0, 1.0]) == [0, 0, 2]

    def test_all_tied(self):
        from elote.arenas.lambda_arena import _ranks_from_scores

        assert _ranks_from_scores([2.0, 2.0, 2.0]) == [0, 0, 0]


class TestMultiBoutRecord:
    """The standalone record type."""

    def test_repr(self):
        bout = MultiBout(participants=["a", "b", "c"], ranks=[0, 1, 1], predicted_ranks=["a", "b", "c"])
        assert repr(bout) == "<MultiBout: 3 participants>"

    def test_ranks_none_when_no_result_given(self):
        bout = MultiBout(participants=["a", "b"], ranks=None, predicted_ranks=["a", "b"])
        assert bout.ranks is None

    def test_input_lists_are_copied(self):
        participants = ["a", "b", "c"]
        ranks = [0, 1, 2]
        bout = MultiBout(participants=participants, ranks=ranks, predicted_ranks=list(participants))
        participants.append("d")
        ranks[0] = 9
        assert bout.participants == ["a", "b", "c"]
        assert bout.ranks == [0, 1, 2]
