Colley Matrix Method
===================

Overview
--------

The Colley Matrix Method is a least-squares rating system devised by astrophysicist Wesley Colley.
Rather than nudging ratings after each game the way Elo does, it solves a single system of linear
equations over the entire match history to obtain every competitor's rating at once. It is one of
the computer rankings historically used in the college-football Bowl Championship Series, and is
prized for being **bias free**: the result depends only on who beat whom, not on the order games
were played or on the margin of victory.

How It Works
-----------

Colley's method builds a linear system

.. math::

   C\,r = b

where ``r`` is the vector of ratings, ``C`` is the *Colley matrix* built from each competitor's
game counts and head-to-head schedule, and ``b`` is derived from wins and losses. Every competitor
starts at a rating of 0.5, and solving the system yields final ratings in the range [0, 1]. A useful
property is that the ratings of a group of ``n`` competitors always sum to ``n / 2``.

In Elote the fit is recomputed over the connected group of competitors after each recorded result,
so ratings always reflect the full set of games. The expected score between two competitors is a
logistic function of their rating difference.

Advantages
---------

- **Order Independent**: the ratings depend only on the set of results, not the schedule order.
- **Bias Free**: no margin-of-victory influence, which discourages running up the score.
- **Well Grounded**: a clean least-squares interpretation with a unique solution.
- **Self-Normalizing**: ratings live in [0, 1] and sum to ``n / 2``.

Limitations
----------

- **Recomputes Globally**: like Bradley-Terry, each result re-solves the whole connected group,
  which is more expensive than Elo's constant-time update for very large populations.
- **No Margin of Victory**: only win/loss/tie outcomes are used.
- **Different Scale**: ratings are in [0, 1] rather than on the 1500-centered chess scale.
- **Connectivity Needed**: groups of competitors that never play each other cannot be compared.

Implementation in Elote
----------------------

Elote implements the Colley Matrix Method through the ``ColleyMatrixCompetitor`` class:

.. code-block:: python

    from elote import ColleyMatrixCompetitor

    # Every competitor starts at 0.5
    team_a = ColleyMatrixCompetitor()
    team_b = ColleyMatrixCompetitor()

    # Get win probability
    print(f"Team A win probability: {team_a.expected_score(team_b):.2%}")

    # Record results; the whole connected group is re-solved after each one
    team_a.beat(team_b)
    team_a.beat(team_b)
    team_b.beat(team_a)

    print(f"Team A: {team_a.rating:.3f}")
    print(f"Team B: {team_b.rating:.3f}")

Customization
------------

Key parameter:

- **initial_rating**: starting rating value (default: 0.5).

Real-World Applications
---------------------

- **College Football**: one of the computer polls used in the BCS era.
- **Sports Rankings**: any round-robin or bracketed competition where schedule bias matters.
- **Tournament Seeding**: producing an order-independent ranking from recorded results.

References
---------

1. Colley, W. N. (2002). "Colley's Bias Free College Football Ranking Method: The Colley Matrix
   Explained". https://www.colleyrankings.com/matrate.pdf
