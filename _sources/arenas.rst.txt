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


Helpers
-------

.. autoclass:: elote.arenas.base.History
    :members:

.. autoclass:: elote.arenas.base.Bout
    :members:

