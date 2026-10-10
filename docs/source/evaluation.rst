Walk-Forward Evaluation
=======================

:func:`elote.walk_forward` predicts each period from everything before it, then learns that
period. The returned :class:`elote.WalkForwardReport` carries aggregate accuracy, log loss and
Brier score, and a reliability table showing which probability ranges are overconfident.

Reliability tables
------------------

``report.reliability`` is a tuple of :class:`elote.ReliabilityBin` records, one per
equal-width bin on [0, 1]. Choose the number of bins with ``calibration_bins`` (default 10; a
positive integer, booleans and non-integers raise ``ValueError``).

* Bins are left-inclusive and right-exclusive, except the final bin, which also contains 1.0.
* Each prediction is binned before the log-loss clamp is applied.
* The population is exactly the ``report.predictions`` population: decisive bouts between
  competitors already seen, after the ``warmup`` periods. Draws, missing outcomes, unseen
  competitors and warmup periods are excluded, so bin counts sum to ``report.predictions``.
* ``mean_predicted`` is the mean probability that the first side wins; ``observed_rate`` is
  the fraction of those bouts the first side won. Empty bins have ``count == 0`` and ``None``
  for both.
* Only bin aggregates are stored. This is an empirical expected-score calibration of decisive
  bouts, not a three-outcome draw-probability model.

.. code-block:: python

   from elote import EloCompetitor, group_by_period, walk_forward

   periods = group_by_period(rows)
   report = walk_forward(EloCompetitor, periods, warmup=2, calibration_bins=5)

   for b in report.reliability:
       if b.count:
           print(f"[{b.lower:.1f}, {b.upper:.1f}) n={b.count} "
                 f"predicted={b.mean_predicted:.3f} observed={b.observed_rate:.3f}")

A bin whose ``mean_predicted`` is well above its ``observed_rate`` is overconfident.

Expanding development windows
-----------------------------

Use :func:`elote.expanding_window_evaluate` to inspect a fixed configuration across
seasons or other explicit windows. Supply only development periods; keep a separately
reserved final population outside this helper. Boundaries are exclusive period indexes:
training is ``[0, train_end)`` and scoring is ``[train_end, test_end)``.

.. code-block:: python

   from elote import EloCompetitor, expanding_window_evaluate, group_by_period

   development_periods = group_by_period(development_rows)
   # This example requires at least six development periods.
   folds = expanding_window_evaluate(
       EloCompetitor,
       development_periods,
       [(2, 4), (4, 6)],
       competitor_params={"k_factor": 32},
       calibration_bins=5,
   )
   for fold in folds:
       print(f"train=[0,{fold.train_end}) score=[{fold.train_end},{fold.test_end}) "
             f"training_periods={fold.training_periods} scoring_periods={fold.scoring_periods}")
       print(fold.report)  # Includes the scored prediction count.
       print(fold.report.skipped, fold.report.draws, fold.report.reliability)

Every fold starts with fresh arena state and replays its training prefix. All boundaries
are validated before evaluation: require integer indexes (excluding booleans),
``0 < train_end < test_end <= len(periods)``, and ordered, non-overlapping scoring
windows. Gaps are allowed; empty fold lists are rejected. ``warmup`` is reserved for the
training boundary. Configuration, constructor, comparison, score and calibration options
have the same meanings as in :func:`elote.walk_forward`.

Learning remains adaptive within each scoring window: predict a complete period, then
learn its results before predicting the next. Earlier scoring results become training
history in later folds, making the folds dependent. Inspect individual reports for poor
later windows; this helper performs no search, automatic model choice, pooled aggregate,
final-holdout management or confidence intervals, and makes no uncertainty claims.
