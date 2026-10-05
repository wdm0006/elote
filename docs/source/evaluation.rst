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
