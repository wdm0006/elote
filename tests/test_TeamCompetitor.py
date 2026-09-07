"""TeamCompetitor composite: aggregation modes, parity checks, serialization, reset, and arena bouts."""

import pytest
from elote import EloCompetitor, GlickoCompetitor, LambdaArena, TeamCompetitor
from elote.competitors.base import InvalidParameterException, MissMatchedCompetitorTypesException


def make_team(ratings=(1200, 1400), aggregate="mean", member_class=EloCompetitor):
    """Build a TeamCompetitor with one member per initial rating."""
    members = [member_class(initial_rating=r) for r in ratings]
    return TeamCompetitor(members, aggregate=aggregate)


class TestAggregation:
    def test_default_mode_is_mean(self):
        team = make_team((1200, 1400))
        assert team.aggregate == "mean"
        assert team.rating == pytest.approx(1300)

    def test_mean_is_average_of_member_ratings(self):
        team = make_team((1200, 1400, 1600), aggregate="mean")
        assert team.rating == pytest.approx(1400)

    def test_sum_is_total_of_member_ratings(self):
        team = make_team((1200, 1400), aggregate="sum")
        assert team.rating == pytest.approx(2600)

    def test_aggregate_follows_member_updates(self):
        team = make_team((1200, 1400))
        team.beat(make_team((1200, 1400)))
        expected = sum(member.rating for member in team.members) / 2
        assert team.rating == pytest.approx(expected)

    def test_rating_setter_is_rejected(self):
        team = make_team()
        with pytest.raises(NotImplementedError):
            team.rating = 1500

    def test_expected_score_is_symmetric_for_identical_rosters(self):
        a = make_team((1200, 1400))
        b = make_team((1200, 1400))
        assert a.expected_score(b) == pytest.approx(0.5)

    def test_expected_score_is_a_probability_in_both_modes(self):
        for aggregate in ("mean", "sum"):
            a = make_team((1600, 1200), aggregate=aggregate)
            b = make_team((1200, 1600), aggregate=aggregate)
            score = a.expected_score(b)
            assert 0.0 <= score <= 1.0


class TestConstructionValidation:
    def test_unknown_mode_raises(self):
        with pytest.raises(InvalidParameterException):
            TeamCompetitor([EloCompetitor()], aggregate="median")

    def test_empty_roster_raises(self):
        with pytest.raises(InvalidParameterException):
            TeamCompetitor([])

    def test_non_competitor_member_raises(self):
        with pytest.raises(InvalidParameterException):
            TeamCompetitor([EloCompetitor(), "not a competitor"])

    def test_roster_is_copied_not_aliased(self):
        roster = [EloCompetitor()]
        team = TeamCompetitor(roster)
        roster.append(EloCompetitor())
        assert len(team.members) == 1


class TestParityChecks:
    def test_rejects_non_team_opponent(self):
        team = make_team()
        with pytest.raises(MissMatchedCompetitorTypesException):
            team.verify_competitor_types(EloCompetitor())

    def test_rejects_different_member_types(self):
        a = make_team(member_class=EloCompetitor)
        b = make_team(member_class=GlickoCompetitor)
        with pytest.raises(MissMatchedCompetitorTypesException):
            a.verify_competitor_types(b)

    def test_rejects_different_roster_sizes(self):
        a = make_team((1200, 1400))
        b = make_team((1200,))
        with pytest.raises(MissMatchedCompetitorTypesException):
            a.verify_competitor_types(b)

    def test_rejects_different_roster_order(self):
        a = TeamCompetitor([EloCompetitor(), GlickoCompetitor()])
        b = TeamCompetitor([GlickoCompetitor(), EloCompetitor()])
        with pytest.raises(MissMatchedCompetitorTypesException):
            a.verify_competitor_types(b)

    def test_beat_rejects_mismatch_before_updating_members(self):
        a = make_team((1200, 1400))
        b = make_team((1200,))
        with pytest.raises(MissMatchedCompetitorTypesException):
            a.beat(b)
        assert a.members[0].rating == 1200
        assert a.members[1].rating == 1400


class TestMemberUpdates:
    def test_beat_updates_all_member_pairs(self):
        a = make_team((1200, 1400))
        b = make_team((1200, 1400))
        a.beat(b)
        assert a.members[0].rating > 1200
        assert a.members[1].rating > 1400
        assert b.members[0].rating < 1200
        assert b.members[1].rating < 1400

    def test_pairing_is_positional(self):
        # a's strong member is paired with b's weak member and vice versa; the
        # upset (a's 1200 beating b's 1600) must gain more than the expected
        # win (a's 1600 beating b's 1200). Cross-pairing would reverse this.
        a = TeamCompetitor([EloCompetitor(initial_rating=1600), EloCompetitor(initial_rating=1200)])
        b = TeamCompetitor([EloCompetitor(initial_rating=1200), EloCompetitor(initial_rating=1600)])
        a.beat(b)
        gain_of_strong = a.members[0].rating - 1600
        gain_of_weak = a.members[1].rating - 1200
        assert gain_of_weak > gain_of_strong

    def test_tied_updates_members(self):
        # crossed rosters: each position pairs members of different ratings, and
        # an Elo draw lifts the lower-rated member of each pair
        a = TeamCompetitor([EloCompetitor(initial_rating=1600), EloCompetitor(initial_rating=1200)])
        b = TeamCompetitor([EloCompetitor(initial_rating=1200), EloCompetitor(initial_rating=1600)])
        a.tied(b)
        assert a.members[1].rating > 1200
        assert b.members[0].rating > 1200

    def test_scores_forwarded_to_members(self):
        a = make_team()
        b = make_team()
        a.beat(b, scores=(30, 10))
        assert a.members[0].rating > 1200

    def test_invalid_scores_rejected(self):
        a = make_team()
        b = make_team()
        with pytest.raises(ValueError):
            a.beat(b, scores=(10, 30))  # scores disagree with the declared win


class TestSerialization:
    def test_export_contains_nested_member_state(self):
        team = make_team((1200, 1400), aggregate="sum")
        state = team.export_state()
        assert state["type"] == "TeamCompetitor"
        assert state["aggregate"] == "sum"
        assert [member["type"] for member in state["members"]] == ["EloCompetitor", "EloCompetitor"]
        assert "rating" in state["members"][0]["state"]

    def test_from_state_round_trip_preserves_member_state(self):
        team = make_team((1200, 1400))
        team.beat(make_team((1200, 1400)))
        state = team.export_state()

        restored = TeamCompetitor.from_state(state)
        assert restored.aggregate == "mean"
        assert restored.rating == pytest.approx(team.rating)
        for original, clone in zip(team.members, restored.members, strict=True):
            assert type(clone) is type(original)
            assert clone.rating == pytest.approx(original.rating)

    def test_round_trip_preserves_membership_order(self):
        team = TeamCompetitor([EloCompetitor(initial_rating=1600), EloCompetitor(initial_rating=1200)])
        restored = TeamCompetitor.from_state(team.export_state())
        assert restored.members[0].rating == pytest.approx(1600)
        assert restored.members[1].rating == pytest.approx(1200)

    def test_round_trip_preserves_sum_mode(self):
        team = make_team((1200, 1400), aggregate="sum")
        restored = TeamCompetitor.from_state(team.export_state())
        assert restored.aggregate == "sum"
        assert restored.rating == pytest.approx(2600)

    def test_mixed_member_round_trip(self):
        team = TeamCompetitor([EloCompetitor(initial_rating=1200), GlickoCompetitor(initial_rating=1450)])
        opponents = TeamCompetitor([EloCompetitor(initial_rating=1200), GlickoCompetitor(initial_rating=1450)])
        team.tied(opponents)

        restored = TeamCompetitor.from_state(team.export_state())
        assert [type(member).__name__ for member in restored.members] == ["EloCompetitor", "GlickoCompetitor"]
        assert restored.members[1].rating == pytest.approx(team.members[1].rating)

    def test_import_state_into_existing_team(self):
        team = make_team((1200, 1400))
        team.beat(make_team((1200, 1400)))
        state = team.export_state()

        other = make_team((1000, 1000))
        other.import_state(state)
        assert other.aggregate == "mean"
        assert other.rating == pytest.approx(team.rating)
        assert other.members[0].rating == pytest.approx(team.members[0].rating)

    def test_json_round_trip(self):
        team = make_team((1200, 1400), aggregate="sum")
        team.beat(make_team((1200, 1400)))
        restored = TeamCompetitor.from_json(team.to_json())
        assert restored.rating == pytest.approx(team.rating)


class TestReset:
    def test_reset_restores_member_ratings(self):
        team = make_team((1200, 1400))
        team.beat(make_team((1200, 1400)))
        assert team.rating > 1300

        team.reset()
        assert team.members[0].rating == 1200
        assert team.members[1].rating == 1400
        assert team.rating == pytest.approx(1300)

    def test_reset_keeps_roster_and_mode(self):
        team = make_team((1200, 1400), aggregate="sum")
        team.reset()
        assert team.aggregate == "sum"
        assert [type(member).__name__ for member in team.members] == ["EloCompetitor", "EloCompetitor"]


class TestArenaBouts:
    def test_team_vs_team_win_in_plain_arena(self):
        team_a = make_team((1200, 1400))
        team_b = make_team((1200, 1400))
        arena = LambdaArena(
            lambda a, b: True,
            initial_state={
                "team_a": team_a.export_state(),
                "team_b": team_b.export_state(),
            },
        )

        arena.matchup("team_a", "team_b", outcome=1.0)

        assert arena.competitors["team_a"].members[0].rating > 1200
        assert arena.competitors["team_b"].members[0].rating < 1200

    def test_team_vs_team_draw_in_plain_arena(self):
        team_a = TeamCompetitor([EloCompetitor(initial_rating=1600), EloCompetitor(initial_rating=1200)])
        team_b = TeamCompetitor([EloCompetitor(initial_rating=1200), EloCompetitor(initial_rating=1600)])
        arena = LambdaArena(
            lambda a, b: None,
            initial_state={
                "team_a": team_a.export_state(),
                "team_b": team_b.export_state(),
            },
        )

        arena.matchup("team_a", "team_b", outcome=0.5)

        # crossed rosters: each draw lifts the lower-rated member of the pair
        assert arena.competitors["team_a"].members[1].rating > 1200
        assert arena.competitors["team_b"].members[0].rating > 1200

    def test_arena_leaderboard_exposes_team_aggregate(self):
        team_a = make_team((1200, 1400), aggregate="sum")
        team_b = make_team((1200, 1400), aggregate="sum")
        arena = LambdaArena(
            lambda a, b: True,
            initial_state={
                "team_a": team_a.export_state(),
                "team_b": team_b.export_state(),
            },
        )

        arena.matchup("team_a", "team_b", outcome=1.0)

        lb = {row["competitor"]: row["rating"] for row in arena.leaderboard()}
        assert lb["team_a"] == pytest.approx(arena.competitors["team_a"].rating)
        assert lb["team_a"] > lb["team_b"]

    def test_arena_export_round_trips_through_team_state(self):
        team_a = make_team((1200, 1400))
        team_b = make_team((1200, 1400))
        arena = LambdaArena(
            lambda a, b: True,
            initial_state={
                "team_a": team_a.export_state(),
                "team_b": team_b.export_state(),
            },
        )
        arena.matchup("team_a", "team_b", outcome=1.0)

        exported = arena.export_state()
        rebuilt = TeamCompetitor.from_state(exported["team_a"])
        assert rebuilt.rating == pytest.approx(arena.competitors["team_a"].rating)
