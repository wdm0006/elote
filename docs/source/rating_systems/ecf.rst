ECF Rating System
===============

Overview
--------

The ECF (English Chess Federation) rating system is the official rating system used for chess players in England. It was developed as an alternative to the Elo system and has been in use since the 1950s, though it has undergone several revisions over the years.

Unlike Elo and Glicko, which use a logistic curve to calculate expected outcomes, the ECF system uses a linear relationship between rating differences and expected game outcomes. This makes it somewhat simpler to calculate by hand, which was an advantage in the pre-computer era.

How It Works
-----------

The ECF rating system is based on the following principles:

1. Each player has a grade (rating) that represents their playing strength
2. The difference between grades determines the expected outcome of a match
3. After each match, grades are adjusted based on the actual outcome compared to the expected outcome

In the ECF system, a difference of 40 grade points is expected to yield approximately a 67% win rate for the stronger player. This is different from Elo, where a 100-point difference corresponds to a 64% win expectancy.

The expected outcome calculation is:

.. math::

   E_A = 0.5 + \frac{R_A - R_B}{F}

Where:
- :math:`E_A` is the expected score for player A
- :math:`R_A` is the grade of player A
- :math:`R_B` is the grade of player B
- :math:`F` is a conversion factor (typically 120)

After a match, the grades are updated using:

.. math::

   R'_A = R_A + K \times (S_A - E_A)

Where:
- :math:`R'_A` is the new grade for player A
- :math:`K` is the K-factor (determines how quickly grades change)
- :math:`S_A` is the actual score (1 for win, 0.5 for draw, 0 for loss)
- :math:`E_A` is the expected score

Advantages
---------

- **Simplicity**: The linear relationship is easier to understand and calculate
- **Local Optimization**: Designed specifically for the English chess community
- **Historical Data**: Long history of use provides extensive comparative data
- **Regular Updates**: The ECF publishes updated ratings multiple times per year
- **Transparency**: Clear calculation methods that players can verify

Limitations
----------

- **Limited Range**: Works best within a certain range of skill differences
- **Less Theoretical Basis**: The linear relationship is less theoretically justified than Elo's logistic curve
- **Regional Focus**: Primarily used in England, limiting international comparability
- **No Uncertainty Measure**: Unlike Glicko, doesn't account for rating reliability
- **Fixed Parameters**: Less flexibility in parameter adjustment compared to other systems

Implementation in Elote
----------------------

Elote provides an implementation of the ECF rating system through the ``ECFCompetitor`` class:

.. code-block:: python

    from elote import ECFCompetitor

    # Create two competitors with different initial grades
    player1 = ECFCompetitor(initial_rating=120)
    player2 = ECFCompetitor(initial_rating=150)

    # Get win probability
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")

    # Record a match result
    player1.beat(player2)  # Player 1 won!

    # Grades are automatically updated
    print(f"Player 1 new grade: {player1.rating}")
    print(f"Player 2 new grade: {player2.rating}")

Customization
------------

The ``ECFCompetitor`` class allows for customization of the K-factor and the conversion factor:

.. code-block:: python

    # Create a competitor with custom parameters
    player = ECFCompetitor(
        initial_rating=120,
        k_factor=20,
        f_factor=120
    )

Key parameters:
- **initial_rating**: Starting grade value
- **k_factor**: Determines how quickly grades change (default: 16)
- **f_factor**: Conversion factor for expected score calculation (default: 120)

ECF to Elo Conversion
--------------------

For those familiar with Elo ratings, ECF grades can be approximately converted to Elo ratings using the formula:

.. math::

   \text{Elo} = 7.5 \times \text{ECF} + 700

This means an ECF grade of 100 is roughly equivalent to an Elo rating of 1450.

Real-World Applications
---------------------

The ECF rating system is primarily used in England for:

- **Chess Tournaments**: Official ECF-rated events throughout England
- **Club Play**: Local chess clubs use ECF grades for team selection and pairing
- **Junior Development**: Tracking progress of young players
- **National Rankings**: Determining England's top players

References
---------

1. [ECF Grading System](http://www.ecfgrading.org.uk/new/help.php#elo) - Official documentation
2. Clarke, P.H. (1982). "The Theory of Grading". British Chess Magazine.
3. Elo, Arpad (1978). *The Rating of Chessplayers, Past and Present*. Arco. ISBN 0-668-04721-6.
4. Sonas, Jeff (2002). "The Sonas Rating Formula - Better than Elo?". ChessBase News. 