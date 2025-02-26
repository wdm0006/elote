Elo Rating System
================

Overview
--------

The Elo rating system is one of the most widely used rating systems in the world. Developed by Hungarian-American physics professor Arpad Elo, it was originally designed for chess but has since been adapted for many other competitive domains including video games, basketball, football, and baseball.

The Elo system is named after its creator and was first introduced as the official rating system for the United States Chess Federation in 1960, and later adopted by the World Chess Federation (FIDE) in 1970.

How It Works
-----------

The Elo rating system is based on the following principles:

1. Each player has a rating that represents their skill level
2. The difference between ratings determines the expected outcome of a match
3. After each match, ratings are adjusted based on the actual outcome compared to the expected outcome

The core formula for calculating the expected score (probability of winning) is:

.. math::

   E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}

Where:
- :math:`E_A` is the expected score for player A
- :math:`R_A` is the rating of player A
- :math:`R_B` is the rating of player B

After a match, the ratings are updated using:

.. math::

   R'_A = R_A + K \times (S_A - E_A)

Where:
- :math:`R'_A` is the new rating for player A
- :math:`K` is the K-factor (determines how quickly ratings change)
- :math:`S_A` is the actual score (1 for win, 0.5 for draw, 0 for loss)
- :math:`E_A` is the expected score

Advantages
---------

- **Simplicity**: The Elo system is easy to understand and implement
- **Transparency**: Players can easily see how their rating changes after each match
- **Proven Track Record**: Used successfully for decades in various competitive domains
- **Zero-Sum**: In a two-player game, the rating points one player gains are exactly what the other player loses
- **Self-Correcting**: Ratings naturally adjust over time as more matches are played

Limitations
----------

- **Requires Many Matches**: Needs a significant number of matches to reach an accurate rating
- **No Confidence Intervals**: Unlike Glicko, Elo doesn't account for rating reliability
- **Assumes Stable Performance**: Doesn't account for player improvement or decline over time
- **K-Factor Sensitivity**: Results are highly dependent on the chosen K-factor
- **No Team Dynamics**: In team sports, doesn't account for individual contributions

Implementation in Elote
----------------------

Elote provides a straightforward implementation of the Elo rating system through the ``EloCompetitor`` class:

.. code-block:: python

    from elote import EloCompetitor

    # Create two competitors with different initial ratings
    player1 = EloCompetitor(initial_rating=1500)
    player2 = EloCompetitor(initial_rating=1600)

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

The ``EloCompetitor`` class allows for customization of the K-factor:

.. code-block:: python

    # Create a competitor with a custom K-factor
    player = EloCompetitor(initial_rating=1500, k_factor=32)

A higher K-factor makes ratings change more quickly, while a lower K-factor makes them more stable. Common K-factor values:

- 40: For new players with fewer than 30 games (FIDE standard)
- 20: For players with ratings under 2400 (FIDE standard)
- 10: For elite players with ratings over 2400 (FIDE standard)

Real-World Applications
---------------------

The Elo rating system is used in many domains:

- **Chess**: FIDE and national chess federations
- **Video Games**: League of Legends, DOTA 2, and many other competitive games
- **Sports**: Used for international football rankings
- **Online Matchmaking**: Many platforms use Elo or Elo-derived systems to match players of similar skill

References
---------

1. Elo, Arpad (1978). *The Rating of Chessplayers, Past and Present*. Arco. ISBN 0-668-04721-6.
2. Glickman, Mark E. (1995). "A Comprehensive Guide to Chess Ratings". American Chess Journal, 3, 59-102.
3. Silver, Nate (2015). "How We Calculate NBA Elo Ratings". FiveThirtyEight. 