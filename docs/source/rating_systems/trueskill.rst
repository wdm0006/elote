TrueSkill Rating System
======================

Overview
--------

TrueSkill is a Bayesian skill-rating system developed by Microsoft Research. It generalizes Elo and
Glicko to team-based and multiplayer games, and is best known for powering matchmaking on Xbox Live.
Each player's skill is modeled as a Gaussian (normal) distribution with a mean and a standard
deviation, and match results shift both quantities.

How It Works
-----------

Each competitor's skill is described by two quantities:

1. **Mu (μ)**: the estimated skill level (the mean of the belief distribution).
2. **Sigma (σ)**: the uncertainty in that estimate (the standard deviation).

The probability that one player beats another depends on the difference in their means relative to
their combined uncertainty and a skill factor ``beta``. After each match, both players' distributions
are updated via Bayesian inference: the winner's mean rises, the loser's falls, and both players'
sigma typically shrinks as evidence accumulates. A small dynamics factor ``tau`` is added to the
uncertainty over time so that skill can drift.

Because a single Gaussian would over-state confidence, Elote reports a **conservative** rating of
``mu - 3 * sigma`` -- a value you can be roughly 99% sure the player's true skill exceeds.

Advantages
---------

- **Uncertainty Aware**: models both skill and confidence, like Glicko.
- **Team and Multiplayer Support**: naturally extends to games with more than two participants.
- **Fast Convergence**: new players' ratings settle quickly as sigma shrinks.
- **Principled**: grounded in Bayesian inference over factor graphs.

Limitations
----------

- **Complexity**: the underlying factor-graph inference is the most complex of the systems here.
- **No Direct Rating Setter**: the reported rating is derived from mu and sigma, so you set ``mu``
  and ``sigma`` rather than the rating directly.
- **Parameter Sensitivity**: depends on ``beta``, ``tau``, and the assumed draw probability.
- **Different Scale**: works on a mu/sigma scale (default μ = 25) rather than the 1500-centered
  chess scale.

Implementation in Elote
----------------------

Elote implements TrueSkill through the ``TrueSkillCompetitor`` class:

.. code-block:: python

    from elote import TrueSkillCompetitor

    # Create two competitors (defaults: mu=25.0, sigma=8.333)
    player1 = TrueSkillCompetitor()
    player2 = TrueSkillCompetitor(initial_mu=30.0, initial_sigma=5.0)

    # Get win probability
    print(f"Player 2 win probability: {player2.expected_score(player1):.2%}")

    # Record a match result
    player1.beat(player2)

    # The conservative rating is mu - 3 * sigma
    print(f"Player 1: mu={player1.mu}, sigma={player1.sigma}, rating={player1.rating}")

You can also measure how evenly matched two players are:

.. code-block:: python

    quality = TrueSkillCompetitor.match_quality(player1, player2)
    print(f"Match quality: {quality:.2%}")

Customization
------------

Key parameters:

- **initial_mu**: starting mean skill (default: 25.0).
- **initial_sigma**: starting uncertainty (default: 8.333).

Class-level constants can be tuned with ``TrueSkillCompetitor.configure_class(...)``:

- ``beta`` -- skill factor governing how much outcomes depend on skill vs. chance (default 4.166).
- ``tau`` -- dynamics factor adding uncertainty over time (default 0.083).
- ``draw_probability`` -- assumed probability of a draw (default 0.10).

Real-World Applications
---------------------

- **Xbox Live**: the original use case, for matchmaking across many titles.
- **Esports and Multiplayer Games**: team-based skill estimation and matchmaking.
- **Any Multiplayer Ranking**: settings where uncertainty and team composition matter.

References
---------

1. Herbrich, R., Minka, T., & Graepel, T. (2007). "TrueSkill(TM): A Bayesian Skill Rating System".
   Advances in Neural Information Processing Systems 20.
2. Microsoft Research. "TrueSkill Ranking System".
   https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
