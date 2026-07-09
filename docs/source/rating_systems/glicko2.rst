Glicko-2 Rating System
=====================

Overview
--------

Glicko-2 is an extension of the Glicko rating system, both developed by Mark Glickman. It keeps
Glicko's rating and rating deviation (RD) and adds a third quantity, **volatility**, which
measures how erratic a competitor's results have been. A player whose results swing wildly gets a
higher volatility, which in turn lets their rating move more quickly; a player with steady results
gets a lower volatility and a more stable rating.

Internally Glicko-2 works on a transformed scale with mean 0 and standard deviation 1, and converts
back to the familiar mean-1500 display scale (with a scale factor of about 173.7) when reporting a
rating.

How It Works
-----------

Each competitor is described by three quantities:

1. **Rating (r)**: the estimate of the player's skill.
2. **Rating Deviation (RD)**: the uncertainty in that estimate.
3. **Volatility (σ)**: the expected fluctuation in the rating over time.

The expected outcome uses the same logistic form as Glicko, discounting for the opponent's
uncertainty. After a rating period the rating, RD, and volatility are all updated together, with
the volatility found by an iterative procedure controlled by the system constant ``tau``. As with
Glicko, RD grows during periods of inactivity so that a long-absent competitor's rating becomes
less certain.

Advantages
---------

- **Volatility Tracking**: captures how consistent a competitor's results are, not just their skill.
- **Uncertainty Measurement**: like Glicko, models the reliability of each rating.
- **Inactivity Handling**: RD increases for players who have not competed recently.
- **Faster, More Stable Convergence**: erratic players adjust quickly; steady players stay stable.

Limitations
----------

- **Complexity**: the most involved of the Glicko family to understand and implement.
- **Parameter Sensitivity**: results depend on the ``tau`` system constant and initial values.
- **Rating-Period Design**: originally intended to update over rating periods rather than after
  every single game.
- **Computational Overhead**: the iterative volatility update costs more than Elo or Glicko.

Implementation in Elote
----------------------

Elote implements Glicko-2 through the ``Glicko2Competitor`` class:

.. code-block:: python

    from elote import Glicko2Competitor

    # Create two competitors with ratings, RDs, and volatilities
    player1 = Glicko2Competitor(initial_rating=1500, initial_rd=350)
    player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300)

    # Get win probability
    print(f"Player 2 win probability: {player2.expected_score(player1):.2%}")

    # Record a match result (an optional match_time can be supplied)
    player1.beat(player2)

    # Rating, RD, and volatility are all updated
    print(f"Player 1: rating={player1.rating}, rd={player1.rd}, vol={player1.volatility}")

Customization
------------

.. code-block:: python

    player = Glicko2Competitor(
        initial_rating=1500,
        initial_rd=350,
        initial_volatility=0.06,
    )

Key parameters:

- **initial_rating**: starting rating value (default: 1500).
- **initial_rd**: starting rating deviation (default: 350).
- **initial_volatility**: starting volatility (default: 0.06).

The class-level ``tau`` system constant (default 0.5) constrains how much volatility can change
between rating periods and can be adjusted with ``Glicko2Competitor.configure_class(tau=...)``.

Real-World Applications
---------------------

- **Online Chess**: Glicko-2 is the rating system used by lichess.org.
- **Competitive Video Games**: used in modified form by several matchmaking systems.
- **Sports Analytics**: applied where both skill and consistency matter.

References
---------

1. Glickman, Mark E. (2012). "Example of the Glicko-2 system". http://www.glicko.net/glicko/glicko2.pdf
2. Glickman, Mark E. (1999). "Parameter estimation in large dynamic paired comparison experiments".
   Applied Statistics, 48, 377-394.
