"""Guard the runnable tutorial's selection and final scoring boundaries."""

import importlib.util
from pathlib import Path

from elote import group_by_period, walk_forward


spec = importlib.util.spec_from_file_location(
    "chronological_selection", Path(__file__).parents[1] / "examples" / "chronological_selection.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_tuning_sees_only_development_and_holdout_cannot_change_choice(monkeypatch):
    rows = example.scored_schedule()
    expected = group_by_period(rows)[: example.DEVELOPMENT_PERIODS]
    actual_tune = example.tune
    calls = []

    def checked_tune(system, grid, periods, **kwargs):
        assert periods == expected
        assert kwargs["warmup"] == example.INITIAL_WARMUP
        assert kwargs["score_keys"] == example.SCORE_KEYS
        calls.append(system)
        return actual_tune(system, grid, periods, **kwargs)

    monkeypatch.setattr(example, "tune", checked_tune)
    original = example.select_and_evaluate(rows)
    boundary = expected[-1][-1][3]
    changed = [
        (
            a,
            b,
            1.0 - outcome,
            when,
            {example.SCORE_KEYS[0]: attrs[example.SCORE_KEYS[1]], example.SCORE_KEYS[1]: attrs[example.SCORE_KEYS[0]]},
        )
        if when > boundary
        else (a, b, outcome, when, attrs)
        for a, b, outcome, when, attrs in rows
    ]
    altered = example.select_and_evaluate(changed)
    assert calls == [entry[0] for entry in example.SYSTEMS.values()] * 2
    assert original[1:3] == altered[1:3]
    assert original[0] == altered[0]
    assert all(candidate.report.predictions == 5 for _, candidate in original[0])


def test_final_population_and_direct_replay_agree():
    rows = example.scored_schedule()
    _, name, selected, final = example.select_and_evaluate(rows)
    assert (final.predictions, final.skipped, final.draws) == (4, 1, 1)
    assert [index for index, _, _ in final.by_period] == [4, 5]
    direct = walk_forward(
        example.SYSTEMS[name][0],
        group_by_period(rows),
        competitor_params=selected.params,
        warmup=example.DEVELOPMENT_PERIODS,
        score_keys=example.SCORE_KEYS,
    )
    assert final == direct
