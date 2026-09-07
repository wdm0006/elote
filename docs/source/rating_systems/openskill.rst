OpenSkill (Weng-Lin Plackett-Luce)
==================================

See :doc:`Elo vs Glicko vs TrueSkill <elo_vs_glicko_vs_trueskill>` for a measured, sourced head-to-head, or :doc:`How to choose a rating system <../choose_a_rating_system>` for a decision guide across every system.

Overview
--------

OpenSkill is a family of online Bayesian rating algorithms published by Weng and Lin
(2011) as a computationally lighter alternative to TrueSkill. Where TrueSkill relies
on iterative approximate message passing and factor graphs, the Weng-Lin update is a
single closed-form step, and where Elo can only compare two competitors at a time,
the Weng-Lin update is defined over an *ordered participant set of any size*: one
computation ranks a whole bout, ties included.

Each competitor carries a Gaussian belief about its own skill: a mean ``mu`` and a
standard deviation ``sigma``. The displayed rating is the conservative ordinal

.. math::

   \\text{rating} = \\mu - 3 \\sigma

which underestimates true skill with roughly 99.7% confidence, so it only rises as
demonstrated results shrink ``sigma``.

How It Works
------------

A bout is an ordered set of participants with finishing ranks (lower is better, equal
ranks are ties). For participant ``i`` the update combines three quantities computed
over the whole bout: the collective skill-plus-chance scale

.. math::

   c = \\sqrt{\\sum_i (\\sigma_i^2 + \\tau^2) + k \\beta^2}

the likelihood model's win expectation ``exp(mu_i / c) / sum_q`` for the default
Plackett-Luce pairing (the other models swap in their own likelihood, see
`Model variants`_), and the number of teams
sharing each rank. The closed-form update then shifts every ``mu`` by its expected
surplus and shrinks every ``sigma`` in one step. Uncertainty is inflated by an
additive dynamics parameter ``tau`` before every update so beliefs never fully
converge, and a ``kappa`` floor keeps ``sigma`` strictly positive.

Unlike Elo or Glicko, whose updates are defined pairwise, this update is *bout-native*:
genuine 3-way and N-way results enter the same formula as a head-to-head match, and
tied participants share the average rank change. Elote exposes this through the
unified competitor interface: pairwise methods route a single result through the
period hook as a one-bout period, so every caller gets the same formulas, and the
walk-forward evaluation machinery applies results period-by-period as usual.

Model variants
--------------

The Weng-Lin paper derives one update skeleton for four likelihood models, and
``OpenSkillCompetitor`` implements all of them behind the ``model`` constructor
selector. All four share the same Gaussian beliefs, the same ordinal, and the same
bout-native surface; they differ only in the pairwise win model that drives the
update terms.

- ``plackett_luce`` (default) -- Algorithm 4. The full ranking is read as a
  Plackett-Luce distribution over orderings; every participant is compared against
  the whole field above it. The usual choice, and the richest model for wide fields.
- ``bradley_terry_full`` -- Algorithm 1. Every participant is compared against
  every other participant once (winner-vs-loser pairs share one logistic term),
  using the Bradley-Terry logistic likelihood.
- ``bradley_terry_partial`` -- Algorithm 2. Instead of the all-pairs comparison,
  each participant is compared only against nearby ranks within a bounded window,
  with the comparisons averaged over the pair set. Cheaper on very wide fields,
  and less sensitive to rank noise far away in the field.
- ``thurstone`` -- Algorithm 3. Thurstone-Mosteller full pairing: win probabilities
  come from the Gaussian CDF (via a draw margin ``epsilon``) rather than the
  logistic, which some prefer when outcome noise feels more Gaussian than
  Gumbel-like.

All four were verified to 1e-6 against the ``openskill.py`` dev extra over 1v1,
N-way, tie and 12-player bouts; see the notes below. Pick the default unless you
have a reason: ``bradley_terry_full`` and ``thurstone`` mostly differ in tail
behaviour, and ``bradley_terry_partial`` trades a little accuracy for bounded
work on very large fields.

Advantages
----------

- **N-player native**: one closed-form update covers a whole ranked field with ties;
  no inner convergence loop and no pairwise decomposition.
- **Calibrated confidence**: the ``mu - 3*sigma`` ordinal is conservative by
  construction, which suits matchmaking and exploration.
- **Fast**: closed-form update per bout, no matrix refactorization (Colley, Massey,
  Keener) and no iterative inference (TrueSkill).

Limitations
-----------

- **Different rating scale**: beliefs live around ``mu = 25`` with ``sigma = 25/3``,
  not the Elo 1500 scale; ordinals are not comparable to Elo numbers.
- **Rating writes are refused**: the rating is derived from ``mu``/``sigma``; write
  those (or use the serializable state) instead.
- **Score margins are not consumed**: like TrueSkill, the update is rank-based; score
  payloads are validated for consistency with the outcome but do not change the result.

Implementation in Elote
-----------------------

.. code-block:: python

   from elote import OpenSkillCompetitor

   player_a = OpenSkillCompetitor()                    # mu=25, sigma=25/3
   player_b = OpenSkillCompetitor(initial_mu=30.0, initial_sigma=6.0)

   # Record results; beliefs update in one closed-form step.
   player_a.beat(player_b)
   player_b.beat(player_a)
   player_a.tied(player_b)

   print(player_a.rating, player_b.rating)             # mu - 3*sigma ordinals
   print(player_a.expected_score(player_b))            # 1v1 win probability

   # Genuine N-way bouts with ties, updated in one step.
   from elote import OpenSkillCompetitor as OSC

   trio = [OSC(), OSC(), OSC()]
   OSC.apply_bout(trio, ranks=[0, 0, 1])               # first two tied for the win

It also plugs directly into an arena:

.. code-block:: python

   from elote import LambdaArena, OpenSkillCompetitor

   arena = LambdaArena(comparison_func, base_competitor=OpenSkillCompetitor)

Customization
-------------

Constructor parameters:

- ``initial_mu`` -- prior mean skill value (default ``25.0``).
- ``initial_sigma`` -- prior skill standard deviation (default ``25/3``).
- ``model`` -- Weng-Lin variant selector: ``plackett_luce`` (default),
  ``bradley_terry_full``, ``bradley_terry_partial``, or ``thurstone``; see
  `Model variants`_ above. Bouts and rating periods refuse to mix variants.

Class-level constants (``_beta``, ``_tau``, ``_kappa``) can be tuned with
``configure_class``: ``beta`` is the skill-vs-chance deviation baked into every
comparison, ``tau`` the per-update uncertainty inflation, and ``kappa`` the floor on
the posterior variance multiplier.

Notes on the reference implementation
-------------------------------------

Elote's implementation follows the Weng-Lin update as documented for the
``openskill.py`` reference package, for every variant in the ``model`` selector.
Two fidelity notes, both pinned in the reference-value test suite
(``tests/test_OpenSkillCompetitor_known_values.py``, which runs the full battery
for all four variants against the installed ``openskill.py``):

- In ``openskill.py`` 6.2.0 the mu tie adjustment is a no-op (upstream
  `issue #201 <https://github.com/vivekjoshy/openskill.py/issues/201>`_,
  fix in `PR #203 <https://github.com/vivekjoshy/openskill.py/pull/203>`_): tied
  players keep their raw per-player changes instead of the documented average.
  Elote implements the documented rule -- every tied player receives the average mu
  change of its rank group, preserving each player's own prior -- and the tests
  pin both the agreement (sigmas and untied mus to 1e-6) and this documented
  deviation.
- For ``bradley_terry_partial``, elote follows ``openskill.py`` 6.2.0's
  implementation rather than the paper's literal Algorithm 2: a stable rank-sorted
  window of four positions on each side, with the pairwise terms averaged over
  comparison counts. On fields of up to nine participants this coincides with the
  paper's adjacent-rank summation; on wider fields the window bounds the work.
  The reference battery exercises the window on 12-player fields.

Real-World Applications
-----------------------

- Multi-player leaderboards (racing, battle-royale, tournaments) where results are
  full ranked fields rather than pairwise matches.
- Matchmaking where a conservative ordinal keeps uncertain newcomers out of lopsided
  matches.
- Any setting where TrueSkill's quality is wanted without its computational cost.

References
----------

- Weng, R. C., & Lin, C.-J. (2011). A Bayesian Approximation Method for Online
  Ranking. *Journal of Machine Learning Research*, 12, 267-300.
- `openskill.py <https://github.com/vivekjoshy/openskill.py>`_ -- the reference
  Python implementation of the family.
