"""Tests for walk-forward evaluation and tuning."""

import datetime
import math
import unittest
from unittest.mock import patch

from elote import (
    EloCompetitor,
    GlickoCompetitor,
    PythagoreanCompetitor,
    WholeHistoryRatingCompetitor,
    group_by_period,
    tune,
    walk_forward,
)
from elote.competitors.base import InvalidParameterException


def _row(a, b, outcome, when, scores=None):
    attributes = None
    if scores is not None:
        attributes = {"home_score": scores[0], "away_score": scores[1]}
    return (a, b, outcome, when, attributes)


def _season(weeks=6, teams=("A", "B", "C", "D")):
    """A deterministic schedule where A > B > C > D, one round of games per week."""
    start = datetime.datetime(2020, 1, 6)
    strength = {name: len(teams) - i for i, name in enumerate(teams)}
    periods = []
    for week in range(weeks):
        when = start + datetime.timedelta(days=7 * week)
        rows = []
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                margin = 7 * (strength[home] - strength[away])
                rows.append(_row(home, away, 1.0, when, (14 + margin, 14)))
        periods.append(rows)
    return periods


class PeriodNativeCompetitor(EloCompetitor):
    period_calls = []
    pairwise_calls = 0

    def beat(self, competitor, *, scores=None):
        type(self).pairwise_calls += 1
        raise AssertionError("period-native training must not replay pairwise results")

    @classmethod
    def apply_rating_period(cls, results, period_end=None):
        cls.period_calls.append((list(results), period_end))

    @classmethod
    def reset_calls(cls):
        cls.period_calls = []
        cls.pairwise_calls = 0


class TestGroupByPeriod(unittest.TestCase):
    def test_groups_by_iso_week_in_order(self):
        rows = [
            _row("A", "B", 1.0, datetime.datetime(2020, 1, 15)),
            _row("C", "D", 1.0, datetime.datetime(2020, 1, 6)),
            _row("E", "F", 1.0, datetime.datetime(2020, 1, 7)),
        ]
        periods = group_by_period(rows)
        self.assertEqual(len(periods), 2)
        self.assertEqual(len(periods[0]), 2)  # the two games in the earlier week
        self.assertEqual(periods[1][0][0], "A")

    def test_custom_key(self):
        rows = [_row(f"A{i}", f"B{i}", 1.0, None) for i in range(4)]
        periods = group_by_period(rows, key=lambda row: row[0][-1])
        self.assertEqual(len(periods), 4)

    def test_rows_without_timestamps_form_one_leading_period(self):
        rows = [
            _row("A", "B", 1.0, None),
            _row("C", "D", 1.0, datetime.datetime(2020, 6, 1)),
        ]
        periods = group_by_period(rows)
        self.assertEqual(len(periods), 2)
        self.assertIsNone(periods[0][0][3])


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        PeriodNativeCompetitor.reset_calls()

    def test_learns_a_separable_schedule(self):
        report = walk_forward(EloCompetitor, _season(), warmup=1)
        self.assertGreater(report.predictions, 0)
        self.assertGreater(report.accuracy, 0.9)
        self.assertLess(report.log_loss, 0.7)

    def test_first_period_is_never_scored_when_warmed_up(self):
        periods = _season()
        with_warmup = walk_forward(EloCompetitor, periods, warmup=1)
        without = walk_forward(EloCompetitor, periods, warmup=0)
        self.assertEqual(without.predictions - with_warmup.predictions, 0)
        # Nothing is predictable in period 0 either way: no competitor exists yet.
        self.assertGreater(without.skipped, 0)

    def test_no_lookahead_within_a_period(self):
        """Every bout in a period is predicted before any of it is learned."""
        periods = [
            [_row("A", "B", 1.0, datetime.datetime(2020, 1, 6))],
            # B beats A twice in one period; the second must not learn from the first.
            [
                _row("B", "A", 1.0, datetime.datetime(2020, 1, 13)),
                _row("B", "A", 1.0, datetime.datetime(2020, 1, 13)),
            ],
        ]
        report = walk_forward(EloCompetitor, periods, warmup=1)
        self.assertEqual(report.predictions, 2)
        # Both bouts saw identical ratings, so both predictions agree.
        self.assertIn(report.accuracy, (0.0, 1.0))

    def test_draws_are_excluded_and_counted(self):
        periods = [
            [_row("A", "B", 1.0, datetime.datetime(2020, 1, 6))],
            [_row("A", "B", 0.5, datetime.datetime(2020, 1, 13))],
        ]
        report = walk_forward(EloCompetitor, periods, warmup=1)
        self.assertEqual(report.draws, 1)
        self.assertEqual(report.predictions, 0)
        self.assertTrue(math.isnan(report.accuracy))

    def test_unseen_competitors_are_skipped_not_invented(self):
        periods = [
            [_row("A", "B", 1.0, datetime.datetime(2020, 1, 6))],
            [_row("Y", "Z", 1.0, datetime.datetime(2020, 1, 13))],
        ]
        report = walk_forward(EloCompetitor, periods, warmup=1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(report.predictions, 0)

    def test_competitor_params_are_restored_afterwards(self):
        before = PythagoreanCompetitor._exponent
        walk_forward(PythagoreanCompetitor, _season(), warmup=1, competitor_params={"exponent": 9.0})
        self.assertEqual(PythagoreanCompetitor._exponent, before)

    def test_valid_class_parameter_changes_metrics(self):
        low = walk_forward(EloCompetitor, _season(), warmup=1, competitor_params={"k_factor": 8})
        high = walk_forward(EloCompetitor, _season(), warmup=1, competitor_params={"k_factor": 64})
        self.assertNotEqual(low.log_loss, high.log_loss)

    def test_rejects_constructor_parameter_with_grid_searchable_default(self):
        with self.assertRaisesRegex(InvalidParameterException, "w2.*default_w2.*base_competitor_kwargs"):
            walk_forward(
                WholeHistoryRatingCompetitor,
                _season(),
                warmup=1,
                competitor_params={"w2": 100.0},
            )

    def test_rejects_constructor_parameter_without_class_default(self):
        with self.assertRaisesRegex(InvalidParameterException, "initial_rating.*base_competitor_kwargs"):
            walk_forward(EloCompetitor, _season(), warmup=1, competitor_params={"initial_rating": 1500})

    def test_rejects_typo_before_training(self):
        periods = [[("A", "B")]]
        with self.assertRaisesRegex(InvalidParameterException, "kfactor"):
            walk_forward(EloCompetitor, periods, competitor_params={"kfactor": 32})

    def test_rejects_a_warmup_that_leaves_nothing_to_score(self):
        periods = _season(weeks=2)
        with self.assertRaises(ValueError):
            walk_forward(EloCompetitor, periods, warmup=2)
        with self.assertRaises(ValueError):
            walk_forward(EloCompetitor, periods, warmup=-1)

    def test_period_native_system_uses_one_batch_call_per_period(self):
        periods = _season(weeks=3)

        walk_forward(PeriodNativeCompetitor, periods, warmup=1)

        self.assertEqual(len(PeriodNativeCompetitor.period_calls), 3)
        self.assertEqual(PeriodNativeCompetitor.pairwise_calls, 0)
        self.assertEqual(
            [period_end for _results, period_end in PeriodNativeCompetitor.period_calls],
            [period[0][3] for period in periods],
        )

    def test_period_native_system_receives_scores_in_caller_order(self):
        period = [
            _row("A", "B", 1.0, datetime.datetime(2020, 1, 6), (21, 7)),
            _row("B", "A", 0.0, datetime.datetime(2020, 1, 7), (3, 10)),
            _row("A", "B", None, datetime.datetime(2020, 1, 8), (0, 0)),
        ]

        walk_forward(
            PeriodNativeCompetitor,
            [period],
            score_keys=("home_score", "away_score"),
        )

        results, period_end = PeriodNativeCompetitor.period_calls[0]
        self.assertEqual([result[3] for result in results], [(21.0, 7.0), (3.0, 10.0)])
        self.assertEqual(period_end, datetime.datetime(2020, 1, 8))

    def test_period_native_system_accepts_a_period_without_timestamps(self):
        walk_forward(PeriodNativeCompetitor, [[_row("A", "B", 1.0, None)]])

        self.assertIsNone(PeriodNativeCompetitor.period_calls[0][1])

    def test_sequential_system_metrics_remain_pinned(self):
        expected = {
            EloCompetitor: (1.0, 0.4189136689010337, 0.12106875000089291),
            GlickoCompetitor: (1.0, 0.129799282281943, 0.02092450055008762),
        }
        for competitor_class, metrics in expected.items():
            with self.subTest(competitor_class=competitor_class.__name__):
                report = walk_forward(competitor_class, _season(), warmup=1)
                self.assertEqual((report.accuracy, report.log_loss, report.brier), metrics)


class TestTune(unittest.TestCase):
    def test_orders_results_by_metric(self):
        results = tune(
            PythagoreanCompetitor,
            {"exponent": [1.0, 2.0, 8.0]},
            _season(),
            metric="log_loss",
            warmup=1,
        )
        self.assertEqual(len(results), 3)
        losses = [r.report.log_loss for r in results]
        self.assertEqual(losses, sorted(losses))

    def test_accuracy_cannot_separate_the_pythagorean_exponent(self):
        """The exponent changes confidence, never which side is favoured.

        `PF**k / (PF**k + PA**k)` is monotone in `PF/PA` for any positive k, and the log5
        combination exceeds 0.5 exactly when one rating exceeds the other, so k cancels out
        of every binary decision. Tuning it on accuracy therefore reports a flat surface,
        which is the reason `tune` defaults to log loss.
        """
        results = tune(
            PythagoreanCompetitor,
            {"exponent": [1.0, 2.0, 8.0]},
            _season(),
            metric="accuracy",
            warmup=1,
        )
        accuracies = {round(r.report.accuracy, 12) for r in results}
        self.assertEqual(len(accuracies), 1)
        log_losses = {round(r.report.log_loss, 12) for r in results}
        self.assertGreater(len(log_losses), 1)

    def test_rejects_unknown_metric_and_empty_grid(self):
        with self.assertRaises(ValueError):
            tune(EloCompetitor, {"k_factor": [16]}, _season(), metric="nonsense")
        with self.assertRaises(ValueError):
            tune(EloCompetitor, {}, _season())

    def test_rejects_bad_grid_before_running_any_point(self):
        with patch("elote.evaluation.walk_forward") as mocked_walk_forward:
            with self.assertRaisesRegex(InvalidParameterException, "kfactor"):
                tune(EloCompetitor, {"kfactor": [8, 16, 32]}, _season())
        mocked_walk_forward.assert_not_called()

    def test_whole_history_class_default_changes_log_loss(self):
        results = tune(
            WholeHistoryRatingCompetitor,
            {"default_w2": [1.0, 5000.0]},
            _season(weeks=8, teams=("A", "B", "C", "D", "E", "F", "G", "H")),
            warmup=1,
        )
        losses = {round(result.report.log_loss, 10) for result in results}
        self.assertEqual(len(losses), 2)


if __name__ == "__main__":
    unittest.main()
