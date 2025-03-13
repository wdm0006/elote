DWZ Rating System
==============

Overview
--------

The Deutsche Wertungszahl (DWZ), or German Evaluation Number, is the official chess rating system of the German Chess Federation (Deutscher Schachbund). Developed in the 1990s as a replacement for the previously used Ingo system, DWZ is similar to the Elo rating system but with some important modifications to better handle tournament play and player development.

The DWZ system is particularly notable for its sophisticated handling of youth players, whose ratings tend to change more rapidly as they improve, and for its detailed approach to calculating expected outcomes based on rating differences.

How It Works
-----------

The DWZ system uses the following key components:

1. **Rating (R)**: Represents the player's skill level
2. **Development Coefficient (E)**: Determines how quickly ratings change, with higher values for younger and less experienced players
3. **Performance Rating (P)**: The rating that would exactly match a player's tournament results

The expected outcome calculation is similar to Elo:

.. math::

   W_e = \frac{1}{1 + 10^{-(R_A - R_B) / 400}}

Where:
- :math:`W_e` is the expected score for player A
- :math:`R_A` is the rating of player A
- :math:`R_B` is the rating of player B

After a tournament, the rating is updated using:

.. math::

   R' = R + E \times (W - W_e)

Where:
- :math:`R'` is the new rating
- :math:`E` is the development coefficient
- :math:`W` is the actual score
- :math:`W_e` is the expected score

The development coefficient is calculated based on:

.. math::

   E = E_0 \times f(A) \times f(n)

Where:
- :math:`E_0` is the base coefficient (typically 30)
- :math:`f(A)` is an age factor (higher for younger players)
- :math:`f(n)` is an experience factor based on number of rated games played

Advantages
---------

- **Age Sensitivity**: Better handles rating changes for youth players
- **Experience Factor**: Accounts for player experience level
- **Tournament Focus**: Designed for batch updates after tournaments
- **National Standardization**: Consistent application across German chess events
- **Detailed Documentation**: Well-documented methodology with regular updates

Limitations
----------

- **Complexity**: More complex to calculate than basic Elo
- **Regional Focus**: Primarily used in Germany and some neighboring countries
- **No Uncertainty Measure**: Unlike Glicko, doesn't explicitly track rating reliability
- **Parameter Sensitivity**: Results depend on proper calibration of multiple factors
- **Less International Recognition**: Not as widely recognized as FIDE Elo ratings

Implementation in Elote
----------------------

Elote provides an implementation of the DWZ rating system through the ``DWZCompetitor`` class:

.. code-block:: python

    from elote import DWZCompetitor

    # Create two competitors with different initial ratings
    player1 = DWZCompetitor(initial_rating=1600)
    player2 = DWZCompetitor(initial_rating=1800)

    # Get win probability
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")

    # Record a match result
    player1.beat(player2)  # Player 1 won!

    # Ratings are automatically updated
    print(f"Player 1 new rating: {player1.rating}")
    print(f"Player 2 new rating: {player2.rating}")

Customization
------------

The ``DWZCompetitor`` class allows for customization of several parameters:

.. code-block:: python

    # Create a competitor with custom parameters
    player = DWZCompetitor(
        initial_rating=1600,
        initial_development_coeff=30,
        base_development_coeff=30
    )

Key parameters:
- **initial_rating**: Starting rating value
- **initial_development_coeff**: Starting development coefficient
- **base_development_coeff**: Base value for development coefficient calculation

DWZ to Elo Conversion
-------------------

While DWZ and Elo use different calculation methods, the numerical values are designed to be roughly comparable. For practical purposes:

.. math::

   \text{DWZ} \approx \text{Elo}

However, due to different update mechanisms, the ratings may diverge over time for the same player.

Real-World Applications
---------------------

The DWZ rating system is used primarily in:

- **German Chess Federation**: Official rating system for all German chess events
- **Youth Development**: Specially calibrated for tracking youth player development
- **Club Championships**: Used for local and regional tournaments in Germany
- **National Rankings**: Determining Germany's top players

References
---------

1. [Deutsche Wertungszahl](https://en.wikipedia.org/wiki/Deutsche_Wertungszahl) - Wikipedia article
2. [Deutscher Schachbund](https://www.schachbund.de/dwz.html) - Official German Chess Federation site
3. Hechenberger, A. (2001). "Die Deutsche Wertungszahl". Schach-Journal.
4. Glickman, Mark E. (1995). "A Comprehensive Guide to Chess Ratings". American Chess Journal, 3, 59-102. 