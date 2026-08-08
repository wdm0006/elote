Massey Ratings
==============

Overview
--------

The Massey method is the least-squares counterpart to the :doc:`Colley Matrix Method <colley>`.
Introduced by Kenneth Massey in 1997 and used as one of the computer rankings in the college-football
Bowl Championship Series, it fits every competitor a rating such that the *difference* between two
ratings is a least-squares estimate of the margin by which one would beat the other. Like Colley it
solves a single linear system over the whole match history rather than nudging ratings after each
game, so it is **bias free**: the ratings depend only on the set of results, not on the order they
were played in.

Where Colley produces win-percentage-like ratings bounded to [0, 1], Massey produces zero-mean
ratings on a signed *margin* scale. Roughly half of them are negative, and ``r_a - r_b`` reads
directly as a predicted margin.

How It Works
-----------

Massey's method builds the linear system

.. math::

   M\,r = p

where :math:`M = D - A` is the match graph's Laplacian -- :math:`D_{ii}` is the number of games
played by competitor :math:`i` and :math:`A_{ij}` is the number of games between :math:`i` and
:math:`j` -- and :math:`p_i` is competitor :math:`i`'s cumulative margin.

Because every row of :math:`M` sums to zero, the matrix is singular by construction: adding a
constant to every rating leaves the system unchanged. The standard fix is applied here: the last row
of :math:`M` is replaced with all ones and the last entry of :math:`p` with zero, which adds the
constraint

.. math::

   \sum_i r_i = 0

and makes the solution unique on a connected schedule. As with Colley, the fit is recomputed over the
connected group of competitors after each recorded result, and the expected score between two
competitors is a logistic function of their rating difference:

.. math::

   E_a = \frac{1}{1 + e^{-s\,(r_a - r_b)}}

with a class-configurable scale :math:`s` (default 2.0).

Unit Margins
-----------

.. important::

   Elote implements **unit-margin** Massey. The uniform ``BaseCompetitor`` API exposes only
   ``beat`` / ``lost_to`` / ``tied`` and has no channel for a score differential, so a win
   contributes ``+1`` to the winner's cumulative margin and ``-1`` to the loser's, and a draw
   contributes ``0`` to both while still counting as a game played for each.

Genuine margin-of-victory Massey -- the form used in college-football rankings, where a 35-3 win
counts for more than a 7-6 win -- would require carrying a score through the result methods, and is
not implemented. In practice, unit-margin Massey is a close relative of Colley on a signed scale:
both use only who beat whom, and neither can be gamed by running up the score.

Advantages
---------

- **Order Independent**: the ratings depend only on the set of results, not the schedule order.
- **Interpretable Differences**: a rating difference is a predicted margin, not just an ordering.
- **Well Grounded**: a clean least-squares interpretation with a unique zero-mean solution.
- **Self-Normalizing**: the ratings of a connected group always sum to zero.

Limitations
----------

- **Recomputes Globally**: like Colley and Bradley-Terry, each result re-solves the whole connected
  group, which is more expensive than Elo's constant-time update for very large populations.
- **Unit Margins Only**: see above; the margin channel the classical method uses is not available
  through the uniform API.
- **Negative Ratings**: ratings are zero mean, which is unfamiliar next to a 1500-centered chess
  scale (``_minimum_rating`` is therefore ``-inf`` for this system).
- **Connectivity Needed**: groups of competitors that never play each other cannot be compared, and
  are rated independently of one another.

Implementation in Elote
----------------------

Elote implements Massey Ratings through the ``MasseyCompetitor`` class:

.. code-block:: python

    from elote import MasseyCompetitor

    # Every competitor starts at 0.0
    team_a = MasseyCompetitor()
    team_b = MasseyCompetitor()
    team_c = MasseyCompetitor()

    # Get win probability
    print(f"Team A win probability: {team_a.expected_score(team_b):.2%}")

    # Record results; the whole connected group is re-solved after each one
    team_a.beat(team_b)
    team_a.beat(team_c)
    team_b.beat(team_c)

    print(f"Team A: {team_a.rating:.3f}")   # 0.667
    print(f"Team B: {team_b.rating:.3f}")   # 0.000
    print(f"Team C: {team_c.rating:.3f}")   # -0.667

Customization
------------

Key parameters:

- **initial_rating**: starting rating value (default: 0.0).
- **expected_score_scale**: class-level logistic scale used by ``expected_score`` (default: 2.0),
  settable with ``MasseyCompetitor.configure_class(expected_score_scale=...)``.

Real-World Applications
---------------------

- **College Football**: one of the computer polls used in the BCS era.
- **Sports Rankings**: any competition where a predicted margin is more useful than a bare ordering.
- **Tournament Seeding**: producing an order-independent ranking from recorded results.

References
---------

1. Massey, K. (1997). "Statistical Models Applied to the Rating of Sports Teams".
   Bluefield College undergraduate honors thesis. https://masseyratings.com/theory/massey97.pdf
2. Langville, A. N., & Meyer, C. D. (2012). *Who's #1? The Science of Rating and Ranking*,
   chapter 2. Princeton University Press.
