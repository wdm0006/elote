Glicko Rating System
==================

Overview
--------

The Glicko rating system was developed by Mark Glickman in 1995 as an improvement over the Elo rating system. The key innovation of Glicko is the introduction of a "rating deviation" (RD) parameter that measures the uncertainty in a player's rating. This addresses one of the main limitations of the Elo system, which doesn't account for rating reliability.

The name "Glicko" is derived from the creator's surname, Glickman. The system has since been further refined into Glicko-2, though Elote currently implements the original Glicko-1 system.

How It Works
-----------

The Glicko system uses three key parameters:

1. **Rating (r)**: Represents the player's skill level, similar to Elo
2. **Rating Deviation (RD)**: Represents the uncertainty in the rating (higher RD = more uncertainty)
3. **Time Factor (c)**: Controls how much the RD increases over time without playing

The expected outcome calculation is similar to Elo but incorporates the rating deviations:

.. math::

   E(A, B) = \frac{1}{1 + 10^{-g(RD_B) \times (r_A - r_B) / 400}}

Where:
- :math:`g(RD) = \frac{1}{\sqrt{1 + 3 \times RD^2 / \pi^2}}`
- :math:`r_A` and :math:`r_B` are the ratings of players A and B
- :math:`RD_A` and :math:`RD_B` are their rating deviations

After a match, both the rating and rating deviation are updated:

.. math::

   r'_A = r_A + \frac{q}{1/RD_A^2 + 1/d^2} \times g(RD_B) \times (S_A - E(A, B))

.. math::

   RD'_A = \sqrt{\frac{1}{1/RD_A^2 + 1/d^2}}

Where:
- :math:`q = \ln(10) / 400`
- :math:`d^2 = 1 / (q^2 \times g(RD_B)^2 \times E(A, B) \times (1 - E(A, B)))`
- :math:`S_A` is the actual score (1 for win, 0.5 for draw, 0 for loss)

When a player doesn't compete for a period, their RD increases:

.. math::

   RD'_A = \min(\sqrt{RD_A^2 + c^2 \times t}, RD_{max})

Where:
- :math:`t` is the time since last competition
- :math:`c` is the volatility constant
- :math:`RD_{max}` is the maximum allowed rating deviation

Advantages
---------

- **Uncertainty Measurement**: Accounts for the reliability of a player's rating
- **Inactivity Handling**: Automatically increases uncertainty for inactive players
- **More Accurate Matchmaking**: Can match players with similar ratings but different uncertainties
- **Faster Convergence**: New players can reach their true skill level faster
- **Better for Sparse Data**: Works well when players don't compete frequently

Limitations
----------

- **Complexity**: More complex to understand and implement than Elo
- **Parameter Sensitivity**: Results depend on proper tuning of multiple parameters
- **Computational Overhead**: Requires more calculations than Elo
- **No Volatility Tracking**: Unlike Glicko-2, doesn't track how volatile a player's performance is
- **Batch Updates**: Originally designed for updating ratings in batches rather than after each game

Implementation in Elote
----------------------

Elote provides an implementation of the Glicko-1 rating system through the ``GlickoCompetitor`` class:

.. code-block:: python

    from elote import GlickoCompetitor

    # Create two competitors with different initial ratings and RDs
    player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
    player2 = GlickoCompetitor(initial_rating=1700, initial_rd=300)

    # Get win probability
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")

    # Record a match result
    player1.beat(player2)  # Player 1 won!

    # Ratings and RDs are automatically updated
    print(f"Player 1 new rating: {player1.rating}, RD: {player1.rd}")
    print(f"Player 2 new rating: {player2.rating}, RD: {player2.rd}")

Customization
------------

The ``GlickoCompetitor`` class allows for customization of several parameters:

.. code-block:: python

    # Create a competitor with custom parameters
    player = GlickoCompetitor(
        initial_rating=1500,
        initial_rd=350,
        volatility=0.06,
        tau=0.5
    )

Key parameters:
- **initial_rating**: Starting rating value (default: 1500)
- **initial_rd**: Starting rating deviation (default: 350)
- **volatility**: How much RD increases over time (default: 0.06)
- **tau**: System constant affecting rating changes (default: 0.5)

Real-World Applications
---------------------

The Glicko rating system is used in various competitive domains:

- **Chess**: Used by the Australian Chess Federation and Free Internet Chess Server
- **Video Games**: Used in modified form by many competitive games
- **Online Platforms**: Used by lichess.org and other competitive platforms
- **Sports Analytics**: Used for player performance analysis in various sports

References
---------

1. Glickman, Mark E. (1995). "A Comprehensive Guide to Chess Ratings". American Chess Journal, 3, 59-102.
2. Glickman, Mark E. (1999). "Parameter estimation in large dynamic paired comparison experiments". Applied Statistics, 48, 377-394.
3. Glickman, Mark E. (2001). "Dynamic paired comparison models with stochastic variances". Journal of Applied Statistics, 28, 673-689.
4. [The Glicko System](http://www.glicko.net/glicko/glicko.pdf) - Original paper by Mark Glickman 