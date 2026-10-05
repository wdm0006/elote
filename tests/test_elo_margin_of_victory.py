import pytest

from elote import EloCompetitor


def _pair(ra, rb, **kw):
    return EloCompetitor(initial_rating=ra, **kw), EloCompetitor(initial_rating=rb, **kw)


# Expected values computed independently with plain arithmetic:
# new = R_w + K * ln(|m|+1) * 2.2 / (0.001 * (R_w - R_l) + 2.2) * (1 - E_w), K = 32.
@pytest.mark.parametrize(
    "r_w,r_l,margin,exp_w,exp_l",
    [
        (1500, 1500, 1, 1511.090355, 1488.909645),
        (1500, 1500, 20, 1548.712359, 1451.287641),
        (1600, 1400, 10, 1616.898983, 1383.101017),  # favourite wins
        (1400, 1600, 10, 1464.127133, 1535.872867),  # upset
    ],
)
def test_known_values(r_w, r_l, margin, exp_w, exp_l):
    w, lo = _pair(r_w, r_l, margin_of_victory=True)
    w.beat(lo, scores=(margin, 0))
    assert w.rating == pytest.approx(exp_w, abs=1e-5)
    assert lo.rating == pytest.approx(exp_l, abs=1e-5)


def test_lost_to_matches_beat():
    w, lo = _pair(1400, 1600, margin_of_victory=True)
    lo.lost_to(w, scores=(0, 10))
    assert w.rating == pytest.approx(1464.127133, abs=1e-5)
    assert lo.rating == pytest.approx(1535.872867, abs=1e-5)


@pytest.mark.parametrize("method", ["beat", "tied"])
def test_no_scores_equals_option_off(method):
    on = _pair(1500, 1450, margin_of_victory=True)
    off = _pair(1500, 1450)
    getattr(on[0], method)(on[1])
    getattr(off[0], method)(off[1])
    assert on[0].rating == off[0].rating
    assert on[1].rating == off[1].rating


def test_default_ignores_scores():
    a, b = _pair(1500, 1500)
    a.beat(b, scores=(30, 0))
    assert a.rating == pytest.approx(1516.0)


def test_tie_with_scores_unchanged():
    a, b = _pair(1500, 1500, margin_of_victory=True)
    a.tied(b, scores=(2, 2))
    assert a.rating == 1500.0


def test_state_round_trip_preserves_flag():
    a = EloCompetitor(initial_rating=1500, margin_of_victory=True)
    restored = EloCompetitor.from_state(a.export_state())
    assert restored._margin_of_victory is True
    default_state = EloCompetitor().export_state()
    assert "margin_of_victory" not in default_state["parameters"]
    assert EloCompetitor.from_state(default_state)._margin_of_victory is False


def test_pre_change_state_still_imports():
    state = EloCompetitor(initial_rating=1500).export_state()
    assert "margin_of_victory" not in state["parameters"]
    restored = EloCompetitor.from_state(state)
    assert restored._margin_of_victory is False
    assert restored.rating == 1500
