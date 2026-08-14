Pythagorean Expectation
=======================

Overview
--------

Pythagorean expectation is the oldest points-based rating in the sports canon. Bill James
introduced it for baseball in the early 1980s after noticing that a team's winning percentage is
predicted remarkably well by the runs it scored and the runs it allowed, and nothing else. The
name comes from the resemblance of the original :math:`k = 2` form to the Pythagorean theorem.

It is unusual among the systems in Elote in two ways.

- **The rating is already a win expectation.** Every other shipped system produces a strength
  score that has to be mapped through a logistic, a normal CDF or a share before it can be read
  as a probability. A Pythagorean rating is a number in :math:`[0, 1]` that reads directly as
  "the fraction of games this competitor should win".
- **It ignores the opponent graph entirely.** A competitor's rating depends only on its own
  accumulated points for and points against, never on who supplied them. That makes it the
  cheapest system in the library -- a constant-time update with no graph, no matrix and no refit
  -- and it is also its main limitation: there is no strength-of-schedule adjustment at all.

How It Works
------------

Let :math:`PF` be the points a competitor has scored across every game it has played and
:math:`PA` the points it has allowed. Its rating is

.. math::

   w = \frac{PF^{k}}{PF^{k} + PA^{k}}

The exponent :math:`k` is the one free parameter and is fitted per sport: :math:`2` for baseball
(James' original), around :math:`2.37` for American football, and around :math:`14` for
basketball, where scores are much larger and a single point is worth correspondingly less. Elote
defaults to ``2.37``.

Two competitors are compared with the standard **log5** combination of their two win
expectations, also due to Bill James:

.. math::

   E_a = \frac{w_a - w_a w_b}{w_a + w_b - 2 w_a w_b}

This is the probability that :math:`a` beats :math:`b` given the rate at which each of them beats
the field. It is analytically complementary, and the implementation preserves that exactly in
floating point: the two argument orders sum to exactly ``1.0`` and two equal ratings give exactly
``0.5``.

Degenerate inputs
^^^^^^^^^^^^^^^^^

A fresh competitor has :math:`PF = PA = 0`, which makes the rating :math:`0/0`, and an unbeaten
one has :math:`PA = 0`, which makes the rating exactly :math:`1` and log5 in turn a :math:`0/0`.
Elote handles both by adding a small **symmetric prior** :math:`c` (``prior_points``, default
``1.0`` point) to each accumulator rather than by clamping the output:

.. math::

   w = \frac{(PF + c)^{k}}{(PF + c)^{k} + (PA + c)^{k}}

The rating therefore stays a continuous, strictly increasing function of the real totals, a fresh
competitor sits at exactly ``0.5``, and every rating is strictly inside :math:`(0, 1)`, so log5 is
always defined. The prior is comparable in size to the library's unit scores and negligible
against real ones, so it fades as soon as a competitor has played.

Scores
------

Points are what this system is built on, so supplying real ones is the intended use. ``beat`` /
``lost_to`` / ``tied`` accept the common optional ``scores`` payload -- the two competitors'
scores **in the argument order of the call**:

.. code-block:: python

    from elote import PythagoreanCompetitor

    team_a, team_b = PythagoreanCompetitor(), PythagoreanCompetitor()

    team_a.beat(team_b, scores=(28, 14))
    # The same game from the loser's side -- the pair is not reordered:
    # team_b.lost_to(team_a, scores=(14, 28))

Through an arena the pair is always given in ``(a, b)`` order, whichever competitor won, together
with an explicit ``outcome`` so the two can be checked for agreement:

.. code-block:: python

    from elote import LambdaArena, PythagoreanCompetitor

    arena = LambdaArena(lambda a, b: True, base_competitor=PythagoreanCompetitor)
    arena.matchup("Team A", "Team B", outcome=1.0, scores=(28, 14))
    arena.matchup("Team A", "Team C", outcome=0.0, scores=(7, 24))   # C won; order unchanged

.. note::

   When ``scores`` is omitted, the same unit-score fallback the rest of the library uses applies:
   the winner is credited ``1`` and the loser ``0``, and a draw credits each side ``0.5``. On
   unit scores the rating degenerates into a smoothed winning percentage -- still a usable
   baseline, but it throws away the information the method exists to use.

Advantages
----------

- **Directly Interpretable**: the rating *is* a win expectation, with no scale to translate.
- **Cheapest System Here**: constant-time updates, no population fit, no matrix, no eigenvector.
- **Order Independent**: the totals are a sum, so the schedule order cannot matter.
- **Exact Persistence**: there is no opponent graph to lose, so a serialized competitor restores
  exactly, including continued play.
- **A Strong Baseline**: decades of sports analytics have found it hard to beat given only points.

Limitations
-----------

- **No Strength of Schedule**: a 28-14 win over the best competitor in the population and over the
  worst are worth exactly the same.
- **Needs Scores**: on outcome-only data it reduces to a smoothed win percentage.
- **Sport-Specific Exponent**: the default is a football fit; using it on baseball or basketball
  data without changing ``exponent`` will misstate how sharply points map to wins.
- **No Uncertainty**: like Colley, Massey, Keener and Bradley-Terry, the rating is a point
  estimate with no confidence measure attached.
- **Blind to Context**: garbage-time points, blowouts and close games all enter the same sum.

Implementation in Elote
-----------------------

Elote implements Pythagorean expectation through the ``PythagoreanCompetitor`` class:

.. code-block:: python

    from elote import PythagoreanCompetitor

    # Every competitor starts at exactly 0.5
    team_a = PythagoreanCompetitor()
    team_b = PythagoreanCompetitor()

    # Get win probability (log5 of the two expectations)
    print(f"Team A win probability: {team_a.expected_score(team_b):.2%}")

    # Record results with real points
    team_a.beat(team_b, scores=(28, 14))

    print(f"Team A: {team_a.rating:.4f}")   # 0.8267
    print(f"Team B: {team_b.rating:.4f}")   # 0.1733

Parameters
^^^^^^^^^^

- ``exponent`` (constructor): the Pythagorean :math:`k` for this competitor. Default: the class
  exponent. Must be positive.

Note that the constructor deliberately takes **no** ``initial_rating``: the rating is derived from
the points totals rather than stored, so there is no starting rating to hand over, and for the
same reason ``rating`` is read-only.

Class-level parameters, set through ``PythagoreanCompetitor.configure_class(...)``:

- ``exponent``: the default Pythagorean :math:`k` for new competitors. Default ``2.37``, the
  standard American-football fit. Use ``2`` for baseball, or a much larger value for a
  high-scoring sport such as basketball. Must be positive.
- ``prior_points``: the symmetric prior added to both accumulators, in points. Default ``1.0``.
  Larger values pull ratings toward ``0.5`` for longer; must be positive, since a zero prior
  reinstates the degenerate cases it exists to remove.

State and Serialization
^^^^^^^^^^^^^^^^^^^^^^^

``export_state()`` records the two points totals and the number of games played. The rating itself
is not exported because it is a pure function of those totals and the exponent, so restoring the
totals restores the rating exactly.

.. note::

   Unlike Colley, Massey, Keener and Bradley-Terry, this system keeps no per-opponent match graph,
   so nothing is lost on import. A restored competitor continues from exactly where the original
   left off, and further results give ratings identical to those of the competitor it was
   restored from.

References
----------

1. James, B. (1981). *The Bill James Baseball Abstract*. The original :math:`k = 2` formulation.
2. Schatz, A. (2003). "Pythagoras on the Gridiron". *Football Outsiders*. The :math:`k = 2.37` fit
   for American football used as the default here.
3. Miller, S. J. (2007). "A Derivation of the Pythagorean Won-Loss Formula in Baseball".
   *Chance*, 20(1), 40-48. Derives the formula from a Weibull model of scoring.
