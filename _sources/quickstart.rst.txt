Quick Start Guide
===============

This guide will help you get up and running with Elote quickly. We'll cover the basics of creating competitors, recording match results, and using arenas for tournaments.

Basic Usage
----------

Let's start with a simple example using the Elo rating system:

.. code-block:: python

    from elote import EloCompetitor

    # Create two competitors with different initial ratings
    player1 = EloCompetitor(initial_rating=1200)
    player2 = EloCompetitor(initial_rating=1300)

    # Check the probability of player2 beating player1
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")
    # Output: Player 2 win probability: 64.01%

    # Record a match result where player1 beats player2 (an upset!)
    player1.beat(player2)
    # OR equivalently:
    # player2.lost_to(player1)

    # Check the updated ratings
    print(f"Player 1 new rating: {player1.rating}")
    print(f"Player 2 new rating: {player2.rating}")

    # Check the new win probability
    new_probability = player2.expected_score(player1)
    print(f"Player 2 new win probability: {new_probability:.2%}")
    # The probability will have decreased

Recording Draws
--------------

Elote also supports recording draws between competitors:

.. code-block:: python

    from elote import EloCompetitor

    player1 = EloCompetitor(initial_rating=1200)
    player2 = EloCompetitor(initial_rating=1300)

    # Record a draw
    player1.tied(player2)
    # OR equivalently:
    # player2.tied(player1)

    # Check the updated ratings
    print(f"Player 1 new rating: {player1.rating}")
    print(f"Player 2 new rating: {player2.rating}")

Using Different Rating Systems
----------------------------

Elote supports multiple rating systems. Here's how to use Glicko instead of Elo:

.. code-block:: python

    from elote import GlickoCompetitor

    # Create two competitors with Glicko ratings
    player1 = GlickoCompetitor(initial_rating=1500, initial_rd=350)
    player2 = GlickoCompetitor(initial_rating=1600, initial_rd=300)

    # Check win probability
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")

    # Record a match result
    player1.beat(player2)

    # Check the updated ratings and rating deviations
    print(f"Player 1: rating={player1.rating}, rd={player1.rd}")
    print(f"Player 2: rating={player2.rating}, rd={player2.rd}")

Running Tournaments with Arenas
-----------------------------

For managing multiple competitors and matches, use an Arena:

.. code-block:: python

    from elote import LambdaArena
    import random
    import json

    # Define a comparison function (returns True if a beats b)
    def compare_numbers(a, b):
        return a > b

    # Generate 1000 random matchups between numbers 1-10
    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    # Create arena and run tournament with appropriate initial ratings
    arena = LambdaArena(
        compare_numbers, 
        base_competitor_kwargs={"initial_rating": 1200}
    )
    arena.tournament(matchups)

    # Display final rankings
    rankings = arena.leaderboard()
    print(json.dumps(rankings, indent=2))

The output will show the numbers ranked by their ratings, with higher numbers having higher ratings.

Custom Arenas
-----------

You can create custom arenas for specific use cases:

.. code-block:: python

    from elote import Arena
    from elote import EloCompetitor

    class ChessArena(Arena):
        def __init__(self):
            self.competitors = {}
            
        def register_competitor(self, name, rating=1200):
            self.competitors[name] = EloCompetitor(initial_rating=rating)
            
        def record_result(self, name1, name2, winner=None):
            if name1 not in self.competitors:
                self.register_competitor(name1)
            if name2 not in self.competitors:
                self.register_competitor(name2)
                
            if winner == name1:
                self.competitors[name1].beat(self.competitors[name2])
            elif winner == name2:
                self.competitors[name2].beat(self.competitors[name1])
            else:
                self.competitors[name1].tied(self.competitors[name2])
                
        def leaderboard(self):
            return sorted(
                [{"name": name, "rating": competitor.rating} 
                 for name, competitor in self.competitors.items()],
                key=lambda x: x["rating"],
                reverse=True
            )

    # Usage
    chess_arena = ChessArena()
    
    # Register some players with initial ratings
    chess_arena.register_competitor("Magnus", 2800)
    chess_arena.register_competitor("Hikaru", 2750)
    
    # Record match results
    chess_arena.record_result("Magnus", "Hikaru", winner="Magnus")
    chess_arena.record_result("Fabiano", "Magnus", winner="Magnus")  # Auto-registers Fabiano
    
    # Get rankings
    rankings = chess_arena.leaderboard()
    for rank, player in enumerate(rankings, 1):
        print(f"{rank}. {player['name']}: {player['rating']}")

Ensemble Rating System
--------------------

For more robust ratings, you can use the Ensemble rating system that combines multiple rating methods:

.. code-block:: python

    from elote import EnsembleCompetitor
    from elote import EloCompetitor, GlickoCompetitor

    # Create an ensemble with Elo and Glicko components
    player1 = EnsembleCompetitor(
        rating_systems=[
            (EloCompetitor(initial_rating=1200), 0.7),
            (GlickoCompetitor(initial_rating=1500, initial_rd=350), 0.3)
        ]
    )
    
    player2 = EnsembleCompetitor(
        rating_systems=[
            (EloCompetitor(initial_rating=1300), 0.7),
            (GlickoCompetitor(initial_rating=1600, initial_rd=350), 0.3)
        ]
    )

    # Use just like any other competitor
    win_probability = player2.expected_score(player1)
    print(f"Player 2 win probability: {win_probability:.2%}")
    
    player1.beat(player2)
    
    # All underlying ratings are automatically updated
    new_probability = player2.expected_score(player1)
    print(f"Player 2 new win probability: {new_probability:.2%}")

Next Steps
---------

Now that you understand the basics of Elote, you can:

- Explore the different rating systems in more detail
- Learn about advanced arena configurations
- Check out the examples directory for real-world use cases
- Integrate Elote into your own projects

For more detailed information, see the following sections:

- :doc:`rating_systems/elo`
- :doc:`rating_systems/glicko`
- :doc:`rating_systems/ecf`
- :doc:`rating_systems/dwz`
- :doc:`rating_systems/ensemble`
- :doc:`arenas`
- :doc:`examples` 