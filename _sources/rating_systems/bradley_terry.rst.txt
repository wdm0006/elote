Bradley-Terry Model
===================

Overview
--------

The Bradley-Terry model is a probabilistic model for paired comparisons, introduced by
Ralph Bradley and Milton Terry in 1952. Each competitor is assigned a single latent
*strength*, and the outcome of any head-to-head matchup is modeled as a function of the two
competitors' strengths. It is one of the most widely used models for analyzing paired
comparison data across sports analytics, ranking systems, and preference learning.

Unlike Elo, which nudges ratings incrementally after every game, Bradley-Terry estimates the
strengths by **maximum likelihood** over the full set of observed comparisons. In Elote this
is done the same way as the Colley Matrix method: results are recorded as they happen, and the
strengths of the entire connected group of competitors are re-fit after each result.

How It Works
------------

Each competitor ``i`` has a positive strength ``p_i``. The probability that ``i`` beats ``j`` is

.. math::

   P(i \\text{ beats } j) = \\frac{p_i}{p_i + p_j}

Writing the strengths in the log domain as ``p_i = e^{\\beta_i}`` turns this into the logistic
function

.. math::

   P(i \\text{ beats } j) = \\frac{1}{1 + e^{-(\\beta_i - \\beta_j)}}

which is exactly the functional form of the Elo expected score. Elote reports Bradley-Terry
ratings on an Elo-like scale, ``rating = 1500 + s \\cdot \\beta`` with ``s = 400 / \\ln(10)``,
so that the expected score is numerically identical to Elo's and the ratings are directly
comparable to Elo ratings.

The strengths are fit with the minorization-maximization (MM) update of Hunter (2004). Writing
``W_i`` for the total number of wins by ``i`` and ``n_{ij}`` for the number of games between ``i``
and ``j``, each iteration applies

.. math::

   p_i' = \\frac{W_i}{\\sum_j n_{ij} / (p_i + p_j)}

followed by normalization by the geometric mean of the strengths. This update is guaranteed to
increase the likelihood at every step and converges to the unique regularized maximum. Because the strengths are
only identified up to an overall scale, Elote re-centers them so that the mean log-strength of
a connected group is zero.

A small amount of **regularization** (a virtual win and loss against an average phantom
opponent) is added so that a unique, finite maximum-likelihood estimate exists even when a
competitor is undefeated or winless within its group -- a case where the unregularized estimate
would diverge to infinity.

Ties are counted as half a win for each side.

Advantages
----------

- **Order independent**: the fit depends only on the set of results, not the order they were
  played.
- **Statistically principled**: produces the maximum-likelihood strengths given the data.
- **Elo-compatible scale**: expected scores match Elo, so ratings are easy to interpret.

Limitations
-----------

- **Recomputes globally**: like Colley, every result re-fits the whole connected group, which
  is more expensive than Elo's constant-time update for very large populations.
- **No margin of victory**: only win/loss/tie outcomes are used.
- **Serialization**: the match graph (opponent references) cannot be serialized, so a
  competitor restored from state keeps its rating and aggregate win/loss/tie counts but not the
  live links to its opponents.

Implementation in Elote
-----------------------

.. code-block:: python

   from elote import BradleyTerryCompetitor

   player_a = BradleyTerryCompetitor(initial_rating=1500)
   player_b = BradleyTerryCompetitor(initial_rating=1500)

   # Record results; strengths are re-fit after each one.
   player_a.beat(player_b)
   player_a.beat(player_b)
   player_b.beat(player_a)

   print(player_a.rating, player_b.rating)
   print(player_a.expected_score(player_b))

It also plugs directly into an arena:

.. code-block:: python

   from elote import LambdaArena, BradleyTerryCompetitor

   arena = LambdaArena(comparison_func, base_competitor=BradleyTerryCompetitor)

Customization
-------------

Class-level parameters can be tuned with ``configure_class``:

- ``anchor_rating`` -- the rating assigned to the mean log-strength (default ``1500``).
- ``scale`` -- points per unit of log-strength (default ``400 / ln(10)``).
- ``reg`` -- regularization strength (default ``0.1``).
- ``max_iter`` / ``tol`` -- iteration budget and convergence tolerance for the fit.

Real-World Applications
-----------------------

- Ranking sports teams or players from head-to-head results.
- Aggregating pairwise human preferences (e.g. A/B choices) into a single ranking.
- Any setting where you want a maximum-likelihood ranking rather than an online estimate.

References
----------

- Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The
  Method of Paired Comparisons. *Biometrika*, 39(3/4), 324-345.
- Hunter, D. R. (2004). MM Algorithms for Generalized Bradley-Terry Models. *The Annals of
  Statistics*, 32(1), 384-406.
