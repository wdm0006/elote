Glicko-Boost Rating System
==========================

See :doc:`How to choose a rating system <../choose_a_rating_system>` for a decision guide across every system, or :doc:`Glicko <glicko>` for the system Glicko-Boost extends.

Overview
--------

Glicko-Boost is Mark Glickman's extension of Glicko, written for the Deloitte/FIDE Chess Rating Challenge and described in `Glicko-Boost <https://www.glicko.net/glicko/glicko-boost.pdf>`_. It adds four things to Glicko: an advantage for playing white, a second pass that re-rates every player against the opponents' first-pass ratings, an RD "boost" for players whose results were exceptional, and a rating-dependent model for how RD grows between periods.

It is the first system in Elote whose published update is defined over a **rating period** rather than over one game. The two-pass structure cannot be expressed as a stream of pairwise updates -- a player's rating change depends on how their opponents did in the same period -- so ``GlickoBoostCompetitor`` overrides :meth:`~elote.competitors.base.BaseCompetitor.apply_rating_period` and does the whole population at once. The pairwise API is unchanged: ``beat``/``lost_to``/``tied`` run the same algorithm over a one-game period.

How It Works
------------

Every step is built on Glicko updating with a white-advantage term. For a player with rating :math:`r` and rating deviation :math:`RD` who played :math:`J` games with scores :math:`s_j` against opponents :math:`(r_j, RD_j)`:

.. math::

   E(\eta, w_j, r, r_j, RD_j) = \frac{1}{1 + 10^{-g(RD_j)(r + w_j\eta - r_j)/400}}

.. math::

   g(RD) = \frac{1}{\sqrt{1 + 3q^2 RD^2 / \pi^2}}, \qquad q = \frac{\ln 10}{400}

.. math::

   d^2 = \left[q^2 \sum_{j=1}^{J} g(RD_j)^2 E (1 - E)\right]^{-1}

.. math::

   RD' = \sqrt{\left(\frac{1}{RD^2} + \frac{1}{d^2}\right)^{-1}}, \qquad
   r' = r + (RD')^2 q \sum_{j=1}^{J} g(RD_j)(s_j - E)

Here :math:`w_j` is :math:`+1` when the player had white in game :math:`j`, :math:`-1` when black, and :math:`\eta` is the rating advantage of playing white. With :math:`\eta = 0` the term disappears and this is ordinary Glicko updating.

The period update
^^^^^^^^^^^^^^^^^

Given every player's pre-period rating and RD, one period is five steps:

1. Update every player from the pre-period ratings and RDs.
2. Update every player again from the **pre-period** ratings and RDs, but against the opponents' step 1 ratings and RDs. A player who lost to opponents who all had a strong month is now judged against those better ratings.
3. Compute each player's performance z-score and boost their pre-period RD if it is large (below).
4. Repeat step 1 with the boosted RDs.
5. Repeat step 2 on the step 4 results. These are the period's final ratings and RDs.

If no RD was boosted in step 3, steps 4 and 5 reproduce steps 1 and 2 exactly, and Elote skips them.

The RD boost
^^^^^^^^^^^^

The boost catches an improving player whose RD is too small for their rating to move fast enough. The z-score measures how far the period's results ran ahead of what the player's **pre-period** rating predicted, against the step 2 population:

.. math::

   Z = \frac{\sum_j g(RD_j)(s_j - E)}{\sqrt{\sum_j g(RD_j)^2 E (1 - E)}}

.. math::

   RD^{\dagger} = \begin{cases}
     RD & Z \le k \\
     \min\left(RD_{unr},\ (1 + (Z - k)B_1) RD + B_2\right) & Z > k
   \end{cases}

:math:`k = 1.96` is the point beyond which a z-score would occur by chance 2.5% of the time, so on an ordinary month nobody is boosted and steps 4 and 5 never run.

RD increase over time
^^^^^^^^^^^^^^^^^^^^^

Instead of Glicko's constant :math:`c`, the increase depends on both the RD and the rating, so that higher-rated players -- whose strength is more stable -- gain slightly less uncertainty per idle period:

.. math::

   RD_{new} = \min\left(RD_{unr},\ \sqrt{RD^2 + \exp\left(\alpha_0 + \alpha_1 RD + \alpha_2 RD \tfrac{r}{1000} + \alpha_3 \tfrac{r}{1000} + \alpha_4 \left(\tfrac{r}{1000}\right)^2\right)}\right)

Elote applies this when a competitor next takes part, for the rating periods it sat out, the way :class:`~elote.competitors.glicko.GlickoCompetitor` handles inactivity. Results that share a period end therefore never inflate each other's RD.

Prediction
^^^^^^^^^^

``expected_score`` uses the paper's own approximation, which combines the two rating deviations rather than using only the opponent's:

.. math::

   E_A = \frac{1}{1 + 10^{-g\left(\sqrt{RD_A^2 + RD_B^2}\right)(r_A + \eta - r_B)/400}}

With the default :math:`\eta = 0` this is exactly complementary: ``a.expected_score(b) + b.expected_score(a) == 1``.

The colour convention
---------------------

Elote's result API carries no colour argument, and Glicko-Boost does not get one added for it. Instead **argument order is the colour**: in a rating-period row ``(a, b, outcome, scores)``, and in ``a.beat(b)``, ``a`` is white and ``b`` is black.

Callers who have no colour information should leave ``_eta`` at its default of ``0.0``, which is why that -- rather than the paper's 30.0 -- is Elote's default. The update then depends only on rating differences and is invariant to the order in which each row is written.

One exception is worth knowing about: :meth:`~elote.LambdaArena.matchup` dispatches a loss by calling ``beat`` on the *winner*, so a losing row driven through the arena's streaming path has its colours reversed relative to the same row handed to ``apply_rating_period``. With the default ``_eta = 0.0`` nothing changes; if you have set a white advantage, group your results into periods and use :meth:`~elote.LambdaArena.rating_period`, which preserves each row's order.

Key Parameters
--------------

All of these are class-level, so they are settable through ``configure_class`` and reachable from :func:`~elote.walk_forward` and :func:`~elote.tune` through ``competitor_params``.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Elote default
     - Meaning (Glickman's optimized value)
   * - ``eta``
     - 0.0
     - Rating advantage for playing white (30.0)
   * - ``b1``
     - 0.20139
     - RD boost multiplicative factor (0.20139)
   * - ``b2``
     - 17.5
     - RD boost additive factor (17.5)
   * - ``k``
     - 1.96
     - z-score threshold above which the RD is boosted (1.96)
   * - ``alpha0`` .. ``alpha4``
     - 5.83733, -1.75374e-04, -7.080124e-05, 0.001733792, 0.00026706
     - RD-increase-over-time coefficients (same values)
   * - ``rd_unrated``
     - 250.0
     - RD cap, :math:`RD_{unr}` (250.0)
   * - ``rating_period_days``
     - 30.0
     - Days in one rating period, used to turn elapsed time into RD increases

``initial_rating`` (default 1500) and ``initial_rd`` (default 250) are constructor arguments, so they travel through ``base_competitor_kwargs`` rather than ``competitor_params``. Glickman's default rating for an unrated player is 1946.25, which is specific to the FIDE population the system was fitted on; Elote keeps its house default of 1500.

How It Differs From Glicko and Glicko-2
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 24 24 26

   * -
     - Glicko
     - Glicko-2
     - Glicko-Boost
   * - Update granularity
     - One game (Elote replays a period pairwise)
     - One game
     - The whole period, twice over
   * - Colour
     - Not modelled
     - Not modelled
     - :math:`\eta` inside :math:`E()`, from argument order
   * - Reacting to a surprising month
     - Nothing beyond the ordinary update
     - A per-player volatility :math:`\sigma` updated every period
     - A one-off RD boost when :math:`Z > k`, then a re-rate
   * - RD growth while idle
     - :math:`\sqrt{RD^2 + c^2 t}`
     - Driven by volatility
     - :math:`\exp` of a rating-and-RD-dependent polynomial
   * - Expected score
     - :math:`g(RD_B)` only
     - Transformed scale
     - :math:`g\!\left(\sqrt{RD_A^2 + RD_B^2}\right)`, so it is complementary

Glicko-2's volatility is a persistent per-player parameter that is re-estimated every period; Glicko-Boost's boost is a one-shot widening of the prior for the period being rated, discarded afterwards. Glicko-Boost is therefore cheaper and has no iterative solve, at the cost of not tracking volatility over time.

Implementation in Elote
-----------------------

.. code-block:: python

    from datetime import datetime
    from elote import GlickoBoostCompetitor

    # The paper's white advantage; leave it at 0.0 if your data has no colour.
    GlickoBoostCompetitor.configure_class(eta=30.0)

    alice = GlickoBoostCompetitor(initial_rating=2300, initial_rd=140)
    bob = GlickoBoostCompetitor(initial_rating=2295, initial_rd=80)
    cara = GlickoBoostCompetitor(initial_rating=2280, initial_rd=150)

    # One rating period, applied to the whole population at once. The first
    # competitor of each row had white.
    GlickoBoostCompetitor.apply_rating_period(
        [
            (alice, bob, 0.0, None),
            (cara, alice, 1.0, None),
            (bob, cara, 0.5, None),
        ],
        period_end=datetime(2026, 1, 31),
    )

    print(alice.rating, alice.rd)
    print(alice.expected_score(bob))

Through an arena, the same period is one call:

.. code-block:: python

    from elote import LambdaArena, GlickoBoostCompetitor

    arena = LambdaArena(lambda a, b: True, base_competitor=GlickoBoostCompetitor)
    arena.rating_period(
        [("alice", "bob", 0.0, None), ("cara", "alice", 1.0, None)],
        period_end=datetime(2026, 1, 31),
    )
    print(arena.leaderboard())

``beat``, ``lost_to`` and ``tied`` are supported and run the same algorithm over a one-game period, so a single result is never a different formula from a batch. Because of the two-pass structure a lone pairwise call is **not** identical to a Glicko update, and applying a period one row at a time does not give the period's ratings -- group results into periods when you have them.

Advantages
----------

- **Period-native**: rates a whole month the way the published system does, with no within-period ordering effect
- **Opponent-aware**: the second pass judges a player against how their opponents actually did, not just their pre-period ratings
- **Catches improvers**: the RD boost widens the prior for a player whose results were far ahead of their rating, so the re-rate can move them further
- **Colour**: models the advantage of moving first, at no cost to the uniform API
- **Flexible RD growth**: uncertainty grows at a rate that depends on the player's rating and current RD

Limitations
-----------

- **Needs periods**: the value is in the batch; a stream of one-game periods loses the second pass's information
- **Many parameters**: twelve system constants, fitted by Glickman on FIDE data and not necessarily right for another domain
- **No volatility tracking**: unlike Glicko-2, nothing persists between periods about how erratic a player is
- **Colour is positional**: rows have to be written white-first for :math:`\eta` to mean anything
- **Cost**: four Glicko passes over the population per period instead of one

Numeric reference
-----------------

``tests/test_GlickoBoostCompetitor_known_values.py`` checks this implementation against the worked example in Section 4 of Glickman's paper. Two of that table's columns are not reproducible from the paper's own formulas -- the module's docstring records the measured deviations and the sweep that establishes it -- so they are pinned as regression values while the rest of the table is asserted as the published oracle.
