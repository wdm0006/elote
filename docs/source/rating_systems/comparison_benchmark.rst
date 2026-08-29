.. meta::
   :description: A reproducible benchmark comparing every rating system Elote ships, on one interface, one split, and one command you can rerun yourself.

Rating-system comparison benchmark
=====================================

This page benchmarks every rating system Elote ships against every other one,
on one fixed setup: the same seed, the same three synthetic scenarios, the
same train/test split, and the same metrics for all of them. Run
``make compare-systems`` (equivalently,
``uv run python scripts/rating_system_comparison.py``) to reproduce every
number below, or change a constant in that script and rerun it to check a
specific conclusion yourself.

The scope is narrower than "benchmark" can imply: this compares Elote's own
systems to each other, not to another library, and it measures prediction
accuracy on held-out bouts rather than ranking quality — recovery of a known
true ordering, convergence speed, calibration after binning, and behaviour
under adversarial scheduling are all out of scope. See
`What this page does not show`_ for the complete list.

Run it yourself
-------------------

.. code-block:: console

    git clone https://github.com/wdm0006/elote && cd elote
    make compare-systems

    # or, equivalently:
    uv run python scripts/rating_system_comparison.py

The command writes ``rating_system_comparison.csv`` (one row per run,
scenario, and system) and ``rating_system_comparison_env.json`` (the
environment block and scenario definitions) to the current directory. Both
are in ``.gitignore``, so a local run never offers to commit its own output.
Pass ``--repeats 3`` to characterize run-to-run variation, or
``--with-football`` to also try the shipped college football adapter (opt-in:
it needs the ``datasets`` extra and a network round trip, and this page does
not report on it — see `What this page does not show`_).

Fixed inputs
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Input
     - Value
     - Why it is pinned
   * - Seed
     - ``20260809``
     - A module constant (``SEED``) in the runner. Change it in one place and every number below moves together.
   * - Library under test
     - ``elote`` 1.3.2, installed from PyPI
     - The version this page's figures were measured against; check ``rating_system_comparison_env.json`` for the version your own run used.
   * - Python
     - CPython 3.12.11
     - Middle of the supported 3.10-3.12 range.
   * - Machine
     - Apple M4, arm64, macOS 26.2
     - Runtime figures are machine-dependent and mean little without it.
   * - Split
     - Time-ordered, ``test_ratio=0.3``
     - 420 training rows and 180 evaluation rows per scenario.
   * - Starting rating
     - 1500 for every system whose constructor accepts one
     - Otherwise the comparison is partly a comparison of default starting points, which range from 100 to 1500 across the library.
   * - Prediction thresholds
     - Fixed at 0.45 / 0.55
     - Threshold optimization is not used; see `What this page does not show`_.
   * - Per-cell runtime budget
     - 300 seconds
     - A cell over budget is recorded as ``excluded_over_budget`` in the CSV, never dropped silently. No cell reached it in the runs reported here; the slowest was under 13 seconds.

Scenario matrix
-------------------

Three synthetic scenarios from one generator and one seed, moving one axis at
a time.

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 22 16 24

   * - Key
     - Competitors
     - Matchups
     - Generator settings
     - Draws (train / test)
     - What it probes
   * - ``balanced``
     - 30
     - 600
     - library defaults
     - 6 / 2
     - dense schedule, nearly draw-free
   * - ``draw_heavy``
     - 30
     - 600
     - ``draw_probability=0.5``, ``noise_std=200.0``
     - 63 / 26
     - whether draw handling costs anything
   * - ``sparse``
     - 120
     - 600
     - library defaults
     - 3 / 2
     - most pairs have never met

Raising ``draw_probability`` alone barely moves the draw rate, because the
generator only draws when the skill gap is inside one noise standard
deviation; ``noise_std`` has to rise with it, which is why ``draw_heavy``
moves two settings and not one.

Systems compared
--------------------

Every concrete rating system Elote exports at the time of measurement: Elo,
Glicko-1, Glicko-2, TrueSkill, ECF, DWZ, Colley Matrix, Massey, Keener,
Pythagorean, Bradley-Terry, and Whole-History Rating.
``BlendedCompetitor`` is excluded — it is an ensemble that needs an explicit
child configuration, so there is no default form comparable to a single
system out of the box. The runner drives ``LambdaArena`` and the dataset
helpers directly rather than the library's ``benchmark_competitors`` helper,
which keeps every system's own defaults intact.

Metrics
----------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Metric
     - Definition
     - Why it is here
   * - Accuracy
     - The library's own ``History.calculate_metrics`` at fixed 0.45 / 0.55 thresholds; a drawn bout is correct when the prediction falls between them.
     - It is what everyone reports, so it has to be present to be argued with.
   * - Brier score
     - Mean squared error between the predicted probability and the observed expected score, where a win is 1, a draw 0.5, a loss 0. Lower is better.
     - Accuracy is blind to any monotone reshaping of the probability. A proper scoring rule is not, and the library's expected scores are probabilities people use downstream.
   * - Log loss
     - Mean of ``-(y log p + (1-y) log(1-p))``, with ``p`` clipped to ``[1e-12, 1-1e-12]``. Lower is better.
     - It punishes confident errors, which is exactly the failure mode a broken prediction formula produces.
   * - Predictions outside ``[0, 1]``
     - A count, per cell.
     - A rating system returning a negative probability is not a tuning question. This column is what makes the clipping above auditable rather than hidden.
   * - Train seconds / predict seconds
     - ``time.perf_counter`` around the training loop and the evaluation loop separately.
     - Prediction cost and fitting cost differ by orders of magnitude between incremental and global-fit systems, and averaging them hides that.

The observed score for a draw is 0.5 because that is what the library's own
contract means: every ``expected_score`` returns an expected score on a
win-1, draw-half, loss-0 scale, not a win probability.

Results
----------

Every cell below is the median of three runs of
``uv run python scripts/rating_system_comparison.py --repeats 3``
(``make compare-systems`` runs the same script without repeats, for a quick
single pass). Accuracy, Brier, and log loss were identical across all three
runs for every system except Whole-History Rating, whose iterative fit
differs by less than ``1e-5`` in Brier score between runs; see
`Reproducibility`_. Every cell recorded zero predictions outside ``[0, 1]``.
Lower is better for Brier and log loss.

``balanced`` — 30 competitors, 600 matchups, 420 train / 180 test, 6 / 2 draws:

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 17 17

   * - System
     - Accuracy
     - Brier
     - Log loss
     - Train (s)
     - Predict (s)
   * - Elo
     - 0.8778
     - 0.0900
     - 0.3310
     - 0.001
     - 0.000
   * - Glicko-1
     - 0.9167
     - 0.0567
     - 0.2036
     - 0.002
     - 0.000
   * - Glicko-2
     - 0.9167
     - 0.0581
     - 0.2123
     - 0.004
     - 0.000
   * - TrueSkill
     - 0.9111
     - 0.0570
     - 0.1996
     - 0.001
     - 0.000
   * - ECF
     - 0.9056
     - 0.0706
     - 0.2586
     - 0.012
     - 0.001
   * - DWZ
     - 0.9000
     - 0.0806
     - 0.2991
     - 0.001
     - 0.000
   * - Colley
     - 0.8889
     - 0.0845
     - 0.3126
     - 0.047
     - 0.000
   * - Massey
     - 0.9000
     - 0.0707
     - 0.2618
     - 0.045
     - 0.000
   * - Keener
     - 0.7500
     - 0.1484
     - 0.4852
     - 0.135
     - 0.000
   * - Pythagorean
     - 0.9111
     - 0.0594
     - 0.2049
     - 0.000
     - 0.000
   * - **Bradley-Terry**
     - **0.9222**
     - **0.0539**
     - **0.1859**
     - 9.924
     - 0.000
   * - Whole-History Rating
     - 0.9056
     - 0.0627
     - 0.2305
     - 1.144
     - 0.022

``draw_heavy`` — same size, higher noise, 63 / 26 draws:

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 17 17

   * - System
     - Accuracy
     - Brier
     - Log loss
     - Train (s)
     - Predict (s)
   * - Elo
     - 0.6556
     - 0.1219
     - 0.4886
     - 0.001
     - 0.000
   * - Glicko-1
     - **0.7000**
     - 0.1305
     - 0.4976
     - 0.002
     - 0.000
   * - Glicko-2
     - 0.6889
     - 0.1105
     - 0.4436
     - 0.004
     - 0.000
   * - TrueSkill
     - 0.6944
     - **0.1050**
     - **0.4282**
     - 0.001
     - 0.000
   * - ECF
     - **0.7000**
     - 0.1077
     - 0.4454
     - 0.012
     - 0.001
   * - DWZ
     - 0.6667
     - 0.1144
     - 0.4670
     - 0.001
     - 0.000
   * - Colley
     - 0.7111
     - 0.1134
     - 0.4667
     - 0.047
     - 0.000
   * - Massey
     - 0.7222
     - 0.1024
     - 0.4247
     - 0.044
     - 0.000
   * - Keener
     - 0.5389
     - 0.1647
     - 0.5912
     - 0.129
     - 0.000
   * - Pythagorean
     - 0.6944
     - 0.1217
     - 0.4905
     - 0.001
     - 0.000
   * - Bradley-Terry
     - 0.7278
     - 0.1003
     - 0.4192
     - 1.385
     - 0.000
   * - Whole-History Rating
     - 0.6778
     - 0.1165
     - 0.4592
     - 1.174
     - 0.025

``sparse`` — 120 competitors, 600 matchups, 420 train / 180 test, 3 / 2 draws:

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 17 17

   * - System
     - Accuracy
     - Brier
     - Log loss
     - Train (s)
     - Predict (s)
   * - Elo
     - 0.6389
     - 0.1807
     - 0.5553
     - 0.001
     - 0.000
   * - Glicko-1
     - 0.7667
     - 0.1294
     - 0.3991
     - 0.002
     - 0.000
   * - Glicko-2
     - 0.7667
     - 0.1254
     - 0.3931
     - 0.004
     - 0.000
   * - TrueSkill
     - 0.7611
     - **0.1218**
     - **0.3783**
     - 0.001
     - 0.000
   * - ECF
     - 0.7611
     - 0.1278
     - 0.4033
     - 0.006
     - 0.001
   * - DWZ
     - 0.7000
     - 0.1527
     - 0.4882
     - 0.001
     - 0.000
   * - Colley
     - 0.7500
     - 0.1307
     - 0.4255
     - 0.087
     - 0.000
   * - Massey
     - **0.7944**
     - 0.1253
     - 0.3934
     - 0.081
     - 0.000
   * - Keener
     - 0.6944
     - 0.1738
     - 0.5391
     - 1.504
     - 0.000
   * - Pythagorean
     - 0.7722
     - 0.1377
     - 0.4469
     - 0.001
     - 0.000
   * - Bradley-Terry
     - **0.8111**
     - 0.1373
     - 0.4885
     - 11.856
     - 0.000
   * - Whole-History Rating
     - 0.7278
     - 0.1457
     - 0.4593
     - 2.155
     - 0.050

What the tables say
~~~~~~~~~~~~~~~~~~~~~~

**No system wins everywhere.** Bradley-Terry has the best accuracy on
``balanced`` and ``sparse``; TrueSkill has the best Brier score and log loss
on ``draw_heavy`` and ``sparse``. Any copy that names a single winner is
describing one column of one scenario.

**Accuracy and Brier score can disagree about the ordering.** On ``sparse``,
Massey and Bradley-Terry are first and second on accuracy but sit behind
TrueSkill, Glicko-1, and Glicko-2 on Brier score. A reader who uses the
predicted probability for anything beyond picking a winner should not choose
a system on the accuracy column alone.

**Fitting cost and prediction cost are different questions.** Prediction is
sub-millisecond for every system. Training separates them by up to four
orders of magnitude: Bradley-Terry, which refits the whole match graph after
every result, takes roughly 10-12 seconds on ``balanced`` and ``sparse``
where Elo takes about a millisecond, and accounts for the large majority of
each scenario's total sweep time. Whole-History Rating, the other system
that maintains a full history rather than a single running number, is the
next most expensive, at one to two seconds per scenario.

**No prediction fell outside ``[0, 1]`` for any system on this release.**
Earlier development of this library found a TrueSkill defect that produced
impossible probabilities on a fraction of predictions; that defect is fixed
in the release this page measures. See `A version note`_.

Reproducibility
-------------------

Repeating the whole run three times reproduces every figure above exactly,
with one exception: Whole-History Rating's Brier and log-loss figures vary
by less than ``1e-5`` between runs, because its iterative Newton fit's
stopping point is sensitive to floating-point order of operations at the
level of its convergence tolerance. That is roughly four orders of magnitude
below any digit reported here. Every other system, both global-fit and
incremental, is byte-identical across the three runs. Runtime naturally is
not; the runtime columns report the median of the three.

A version note
------------------

An earlier development snapshot of this comparison (see the source specs
this page is adapted from) found that the then-published release returned
probabilities outside ``[0, 1]`` for a large fraction of TrueSkill's
predictions — a since-fixed defect in how the draw correction combined with
the win/loss probabilities. That fix, and the Massey and Keener systems
which were development-branch-only at the time, are now part of the
published release this page measures (``elote`` 1.3.2): every cell above
recorded zero out-of-range predictions, and both systems ran without special
handling. If you are comparing against an older release, do not assume the
numbers here apply — rerun the command above against the version you have.

What this page does not show
--------------------------------

1. **The synthetic generator is a friendly world.** It samples outcomes from
   a latent per-competitor strength, which is the model most of these
   systems assume. Treat these numbers as an upper bound on how well the
   systems agree with each other, not as evidence about messy real data.
2. **One seed.** Three scenarios from one seed are enough to show that the
   ordering is scenario-dependent and not enough to put a confidence
   interval on any single figure. None is claimed anywhere on this page.
3. **Glicko's inactivity-driven RD growth is inert here.** The arena rejects
   a match timestamp earlier than a competitor's creation time, so neither
   Glicko system's rating-deviation decay runs during this back-test.
4. **Default parameters throughout.** No system is tuned. A tuned Elo would
   beat an untuned anything, and a comparison of tuned systems is a
   different and much larger piece of work.
5. **Accuracy uses one fixed threshold pair.** 0.45 / 0.55 is a choice. A
   different pair reorders the accuracy column, which is part of why the
   Brier and log-loss columns are here at all.
6. **Runtime is one machine.** Absolute seconds are Apple M4 numbers. The
   *ratios* between systems are the portable part.
7. **No cross-library comparison, and no real-data run in this page's
   figures.** The runner supports ``--with-football`` for a robustness check
   against the shipped college football adapter, but that scenario is not
   reported here; it is a much larger, uneven, real-world schedule and is
   left for a future revision of this page rather than measured on the same
   timeline as the synthetic scenarios above.

See also
-----------

- :doc:`../choose_a_rating_system` — route from a requirement to a shortlist,
  then measure the shortlist on your own data with the same harness this page
  uses.
- :doc:`elo_vs_glicko_vs_trueskill` — the three most-asked-about systems,
  compared with sourced mathematics.
- :doc:`comparison` — the per-system reference: origins, formulas,
  parameters, strengths and weaknesses.
