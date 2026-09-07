Arenas
======

Arenas are objects that manage populations of competitors and their matchups. Currently there is only one
type of arena implemented, LambdaArenas

Lambda Arena
------------

Rating periods
~~~~~~~~~~~~~~

A rating period is a batch of results that share one pre-period rating state. Use
:meth:`~elote.LambdaArena.rating_period` when every prediction in the batch should
be made before any result in that batch changes a rating. This differs from calling
:meth:`~elote.LambdaArena.matchup` repeatedly, where each result immediately informs
the next prediction.

Each row is ``(competitor_a, competitor_b, outcome, scores)``. The outcome is
``1.0``, ``0.0``, or ``0.5`` from the first competitor's perspective, and scores
may be ``None``. All currently shipped rating systems use
:meth:`~elote.competitors.base.BaseCompetitor.apply_rating_period`'s sequential
default, so their final ratings match a stream of the same pairwise updates. A
period-native system can override that operation to update its population
simultaneously while keeping the pairwise API intact.

.. autoclass:: elote.arenas.lambda_arena.LambdaArena
    :members:


N-player bouts
--------------

:meth:`~elote.LambdaArena.match_group` runs a single bout between three or more
sides -- or between two sides with rosters, i.e. a team bout. The participants
are given in finishing order (or with explicit ``ranks``), and the whole bout is
updated natively with one bout-level pass rather than as a fan-out of pairwise
results, which is not mathematically equivalent for bout-level models such as
OpenSkill.

.. code-block:: python

    from elote import LambdaArena, OpenSkillCompetitor

    arena = LambdaArena(lambda *args: None, base_competitor=OpenSkillCompetitor)
    # Free-for-all: Ada finishes first, Linus last.
    arena.match_group(["Ada", "Grace", "Linus"])
    # Team bout: the (ada, grace) roster ties the (linus, kurt) roster.
    arena.match_group([
        ("red", ["ada", "grace"]),
        ("blue", ["linus", "kurt"]),
    ], ranks=[0, 0])

``ranks`` gives one finishing rank per participant (lower is better, equal
ranks are ties); ``scores`` is one score per participant and defines the ranks
when ``ranks`` is omitted. Every pre-update prediction is captured before any
participant is updated, and the bout is recorded as a
:class:`~elote.arenas.base.MultiBout` entry.

Only rating systems implementing a bout-level update (``apply_bout``) support
N-way bouts; the first shipped consumer is
:class:`~elote.OpenSkillCompetitor`, including per-member team updates whose
team strength aggregates the members' beliefs.

.. note::

   N-way bouts are excluded from the two-sided analytics --
   ``History.report_results``, ``confusion_matrix``, ``calculate_metrics``,
   ``calculate_metrics_with_draws``, ``optimize_thresholds``, ``random_search``,
   ``accuracy_by_prior_bouts``, ``get_calibration_data`` and the calibration
   plot -- because those are defined over a winner/drawer/loser comparison of
   exactly two competitors. They remain recorded in ``History.bouts`` for
   inspection.


Helpers
-------

.. autoclass:: elote.arenas.base.History
    :members:

.. autoclass:: elote.arenas.base.Bout
    :members:

.. autoclass:: elote.arenas.base.MultiBout
    :members:

