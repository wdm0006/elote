Elote vs single-algorithm packages
===================================

Use Elote when the unresolved question is *which* rating system fits your
history. Its narrow advantage is one rating loop across eleven systems, so an
algorithm change does not require an application rewrite.

Use another package when its narrower job is your actual job:

* Use ``openskill`` for genuine multiplayer or team rating; the measurements
  below cover one-on-one updates only.
* Use ``whr`` when Whole-History Rating fitting speed is the binding constraint.
  Its compiled core was 7.03 times faster on the published 500-game fixture.
* Use ``choix`` when you want Bradley-Terry/Luce inference primitives rather
  than a stateful rating loop.
* Use ``trueskill`` or ``glicko2`` when you have already selected that system
  and prefer its focused API.

Measured agreement
------------------

The comparison uses native posterior outputs from each package, not reference
formulas reconstructed from low-level primitives.

.. list-table::
   :header-rows: 1
   :widths: 18 38 20 24

   * - Packages
     - Quantity both compute
     - Matched configuration
     - Result
   * - Elote 1.3.2 / ``trueskill`` 0.4.5
     - Posterior ``mu`` and ``sigma`` for both players after one 1v1 win
     - Equal default priors and draw probability 0.1
     - Maximum absolute delta 0.0003473759; agreement to 3 decimal places
   * - Elote 1.3.2 / ``glicko2`` 2.1.0
     - Posterior rating, RD and volatility after one rating-period result
     - Rating 1500, RD 350, volatility 0.06
     - Maximum absolute delta 0.0000010698; agreement to 5 decimal places
   * - Elote 1.3.2 / ``choix`` 0.4.1
     - Centered native Bradley-Terry log-strengths on one connected graph
     - Regularization 0, tolerance 1e-8, at most 10,000 iterations
     - Maximum absolute delta 0.0000000073; agreement to 8 decimal places
   * - Elote 1.3.2 / ``whr`` 2.2.0
     - Native per-day Elo rating curve
     - No definition-matched setting: Gaussian initial-rating anchor versus
       virtual-game prior
     - No delta quoted; the finding is tracked in `issue 173
       <https://github.com/wdm0006/elote/issues/173>`_

Speed footnote
--------------

On one Apple M4 running CPython 3.12.11, ``whr`` completed construction and 20
fit sweeps over a fixed 500-game fixture in 0.0110 seconds. Elote completed
construction and its lazy fit in 0.0770 seconds: **Elote was 7.03 times
slower** on this machine and protocol.

These fixtures are evidence of agreement, not proof that either implementation
is correct. They do not compare predictive accuracy, rank packages overall, or
establish a general speed ordering.
