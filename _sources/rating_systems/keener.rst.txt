Keener Ratings
==============

Overview
--------

Keener's method, introduced by James Keener in 1993, is the score-based member of the classical
global-fit family that also contains the :doc:`Colley Matrix Method <colley>`, :doc:`Massey
<massey>` and :doc:`Bradley-Terry <bradley_terry>`. Where those three solve a linear system or
maximize a likelihood, Keener builds a square **preference matrix** from the points competitors
have scored on one another and reads the ratings off that matrix's dominant eigenvector.

The eigenvector construction is what gives the method its character: a competitor's rating is
proportional to a weighted average of its opponents' ratings, weighted by how strongly it
outscored them. Beating a strong opponent is therefore worth more than beating a weak one, and
the recursion resolves itself globally rather than through a chain of pairwise updates. Like the
other global-fit systems, the whole connected group is re-solved after every recorded result, so
the ratings depend only on the set of results and not on the order they arrived in.

How It Works
------------

Let :math:`S_{ij}` be the total number of points competitor :math:`i` has scored against
competitor :math:`j` across all their meetings, and :math:`g_i` the number of games :math:`i` has
played.

**1. Smoothed preference.** Raw score shares are unstable for lopsided or barely-played pairs --
a single 3-0 result would read as total dominance -- so Keener applies Laplace smoothing:

.. math::

   a_{ij} = \frac{S_{ij} + 1}{S_{ij} + S_{ji} + 2}

**2. Skew transform.** The smoothed share is pushed away from the middle by Keener's skew
function

.. math::

   h(x) = \frac{1}{2} + \frac{1}{2}\,\mathrm{sgn}\!\left(x - \frac{1}{2}\right)
          \sqrt{\left|2x - 1\right|}

which is monotone, fixes :math:`0`, :math:`\tfrac{1}{2}` and :math:`1`, and satisfies
:math:`h(x) + h(1 - x) = 1`, so the matrix stays antisymmetric about :math:`\tfrac{1}{2}`. The
square root is the point: it damps the influence of very large margins, so running up the score
has sharply diminishing returns.

**3. Games-played normalization.** Row :math:`i` is divided by :math:`g_i`, making each entry an
average preference *per game* so that a competitor cannot accumulate rating merely by playing
more often. Pairs that never met contribute nothing to the row rather than the "no evidence"
value :math:`h(\tfrac{1}{2}) = \tfrac{1}{2}`; on a sparse schedule those entries would otherwise
outnumber the real ones and the fit would degenerate into a ranking by :math:`1/g_i`.

**4. Stabilization.** A small positive constant :math:`\varepsilon` (Keener's :math:`\varepsilon
E` perturbation, default ``1e-4``) is added to every entry:

.. math::

   H_{ij} = \frac{h(a_{ij})\,[\,i \text{ played } j\,]}{g_i} + \varepsilon

This makes :math:`H` strictly positive, so by the Perron-Frobenius theorem its dominant
eigenvalue is real and simple and the corresponding eigenvector is strictly positive and unique
up to scale. Without it, a schedule whose match graph is bipartite or otherwise imprimitive has
no single dominant eigenvector to read, and disconnected sub-groups would have no relation at
all.

The ratings of a connected group are then that dominant eigenvector, scaled so that they average
exactly :math:`1.0`:

.. math::

   H\,r = \lambda_{\max}\,r, \qquad \frac{1}{n}\sum_i r_i = 1

Because they are a positive strength scale rather than a probability scale, the expected score is
the natural share

.. math::

   E_a = \frac{r_a}{r_a + r_b}

computed internally as a logistic of the log rating ratio with a class-configurable scale
(default 1.0, which is exactly the share above). Two equal ratings give exactly 0.5 and the two
argument orders sum to exactly 1.0.

Scores
------

Keener is a score-based method, so supplying real scores is what it is built for. ``beat`` /
``lost_to`` / ``tied`` accept the common optional ``scores`` payload -- the two competitors'
scores **in the argument order of the call**:

.. code-block:: python

    from elote import KeenerCompetitor

    team_a, team_b = KeenerCompetitor(), KeenerCompetitor()

    team_a.beat(team_b, scores=(35, 3))
    # The same game from the loser's side -- the pair is not reordered:
    # team_b.lost_to(team_a, scores=(3, 35))

Through an arena the pair is always given in ``(a, b)`` order, whichever competitor won, together
with an explicit ``outcome`` so the two can be checked for agreement:

.. code-block:: python

    from elote import LambdaArena, KeenerCompetitor

    arena = LambdaArena(lambda a, b: True, base_competitor=KeenerCompetitor)
    arena.matchup("Team A", "Team B", outcome=1.0, scores=(35, 3))
    arena.matchup("Team A", "Team C", outcome=0.0, scores=(7, 24))   # C won; order unchanged

.. note::

   When ``scores`` is omitted, the same unit-score fallback the rest of the library uses applies:
   the winner is credited ``1`` and the loser ``0``, and a draw credits each side ``0.5``. That
   keeps Keener usable on outcome-only data and keeps it consistent with every other system, but
   it discards exactly the information the method is built on -- on win/loss-only data, Colley or
   Massey will generally be the better choice.

Advantages
----------

- **Uses Scores**: the only shipped system that reads margin of victory through a nonlinear,
  bounded transform, so large wins count for more without dominating.
- **Order Independent**: the ratings depend only on the set of results, not the schedule order.
- **Strength of Schedule**: the eigenvector recursion values a win in proportion to the opponent's
  own rating.
- **Well Grounded**: Perron-Frobenius guarantees a unique positive solution on any connected
  schedule.
- **Positive Scale**: ratings are strictly positive and average exactly 1.0, so a rating reads
  directly as "this many times the population average".

Limitations
-----------

- **Recomputes Globally**: like Colley, Massey and Bradley-Terry, each result re-solves the whole
  connected group, which is more expensive than Elo's constant-time update for very large
  populations.
- **Weaker Without Scores**: the unit-score fallback keeps it usable on outcome-only data, but it
  throws away the method's main input.
- **Perturbation Is a Knob**: the ``eps`` that makes the matrix positive also quietly couples
  competitors who never met; very small or very large values change the fit.
- **Connectivity Needed**: groups of competitors that never play each other are rated
  independently of one another.
- **No Uncertainty**: like Colley, Massey and Bradley-Terry, the rating is a point estimate with
  no confidence measure attached.

Comparison with the Other Global-Fit Systems
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - System
     - Uses scores
     - Rating scale
     - Fitted by
   * - **Colley**
     - No
     - Bounded [0, 1], mean 0.5
     - Solving a regularized linear system
   * - **Massey**
     - Yes (margin)
     - Signed, zero mean
     - Least squares on cumulative margins
   * - **Bradley-Terry**
     - No
     - Elo-compatible
     - Maximum likelihood over paired outcomes
   * - **Keener**
     - Yes (score share)
     - Strictly positive, mean 1.0
     - Dominant eigenvector of a preference matrix

Keener and Massey are the two score-based systems, and they use scores very differently. Massey is
**linear** in the margin: a 35-3 win contributes exactly 32, and a 70-6 win contributes exactly
64, so a single blowout can move a rating a long way. Keener is **bounded and concave**: the score
enters only as a share, and the square root in the skew transform means the difference between a
comfortable win and a rout is small. If you want ratings whose differences read as predicted
margins, use Massey; if you want scores to inform the ranking without letting one blowout
dominate it, use Keener.

Against Colley and Bradley-Terry, which see only who beat whom, Keener will usually separate
evenly-matched records that those two must call equal -- but only when real scores are supplied.
On outcome-only data Keener's unit-score fallback is strictly less informative than Colley's
win-percentage model, because the skew transform compresses the single bit of information that is
left.

Implementation in Elote
-----------------------

Elote implements Keener Ratings through the ``KeenerCompetitor`` class:

.. code-block:: python

    from elote import KeenerCompetitor

    # Every competitor starts at 1.0, the mean the fitted ratings are normalized to
    team_a = KeenerCompetitor()
    team_b = KeenerCompetitor()
    team_c = KeenerCompetitor()

    # Get win probability
    print(f"Team A win probability: {team_a.expected_score(team_b):.2%}")

    # Record results; the whole connected group is re-solved after each one
    team_a.beat(team_b, scores=(28, 7))
    team_b.beat(team_c, scores=(14, 7))
    team_a.beat(team_c, scores=(35, 3))

    print(f"Team A: {team_a.rating:.4f}")   # 1.7379
    print(f"Team B: {team_b.rating:.4f}")   # 0.8349
    print(f"Team C: {team_c.rating:.4f}")   # 0.4272

Parameters
^^^^^^^^^^

- ``initial_rating`` (constructor): the rating of a competitor that has not played yet.
  Default ``1.0``, the mean the fitted ratings are normalized to, so an unplayed competitor sits
  exactly at the population average. Must be strictly positive.

Class-level parameters, set through ``KeenerCompetitor.configure_class(...)``:

- ``perturbation``: Keener's :math:`\varepsilon`, added to every matrix entry. Default ``1e-4``.
  Must be positive.
- ``expected_score_scale``: the logistic scale applied to the log rating ratio by
  ``expected_score``. Default ``1.0``, which is the plain Keener share; larger values sharpen
  predictions. Must be positive.
- ``round_decimals``: the number of decimal places fitted ratings are canonicalized to. Default
  ``10``. The eigen-solve's row-order noise is around ``5e-15``, five orders of magnitude below
  this grid, so canonicalizing here is what makes the fit exactly order independent.

State and Serialization
^^^^^^^^^^^^^^^^^^^^^^^

``export_state()`` records the fitted rating, the win/loss/tie counts, and the aggregate score
totals ``points_for`` and ``points_against``.

.. warning::

   As with Colley, Massey and Bradley-Terry, the per-opponent match graph holds live object
   references and cannot be serialized, so it is reset on import. A restored competitor keeps its
   rating and score totals, but continued play re-fits it from the new results alone rather than
   from the full history. To resume a population exactly, keep the original competitor objects
   alive, or re-apply the recorded results to a fresh set.

References
----------

1. Keener, J. P. (1993). "The Perron-Frobenius Theorem and the Ranking of Football Teams".
   *SIAM Review*, 35(1), 80-93.
2. Langville, A. N., & Meyer, C. D. (2012). *Who's #1? The Science of Rating and Ranking*,
   chapter 4. Princeton University Press.
