Whole-History Rating
====================

See :doc:`Elo vs Glicko vs TrueSkill <elo_vs_glicko_vs_trueskill>` for a measured, sourced head-to-head, or :doc:`How to choose a rating system <../choose_a_rating_system>` for a decision guide across every system.

Whole-History Rating (WHR) is a time-aware Bradley-Terry model. It estimates one
rating for each day on which a competitor played and links consecutive ratings with
a Wiener-process prior. Later results can therefore revise earlier ratings while the
curve remains smoother when games are close together.

.. code-block:: python

   from datetime import datetime
   from elote import WholeHistoryRatingCompetitor

   alice = WholeHistoryRatingCompetitor(w2=300.0)
   bob = WholeHistoryRatingCompetitor(w2=300.0)
   alice.beat(bob, match_time=datetime(2025, 1, 10))
   print(alice.rating)
   print(alice.rating_at(datetime(2025, 1, 10)))
   print(alice.rating_history())

Fitting is lazy: results mark the connected component dirty, while ``rating``,
``rating_at``, ``rating_history`` and ``expected_score`` trigger at most
``max_iterations`` Newton sweeps. ``precision`` controls the Elo-point convergence
tolerance. Lower values cost more but give a tighter fit.

``w2`` is the per-day variance in Elo points squared. A larger value permits faster
rating movement between playing days; a smaller value makes the curve smoother.

Reference
---------

Coulom, R. (2008). *Whole-History Rating: A Bayesian Rating System for Players of
Time-Varying Strength*.

