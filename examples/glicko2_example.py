#!/usr/bin/env python
"""
Example demonstrating the use of the Glicko-2 rating system.

This example shows how to create Glicko-2 competitors, calculate win probabilities,
and update ratings after matches.
"""

from elote import Glicko2Competitor


def main():
    """Run the Glicko-2 example."""
    # Create two competitors with different initial ratings
    player1 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06)
    player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300, initial_volatility=0.06)
    player3 = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.05)

    # Print initial ratings
    print("Initial ratings:")
    print(f"Player 1: Rating={player1.rating}, RD={player1.rd}, Volatility={player1.volatility:.6f}")
    print(f"Player 2: Rating={player2.rating}, RD={player2.rd}, Volatility={player2.volatility:.6f}")
    print(f"Player 3: Rating={player3.rating}, RD={player3.rd}, Volatility={player3.volatility:.6f}")
    print()

    # Calculate win probabilities
    print("Win probabilities:")
    print(f"Player 1 vs Player 2: {player1.expected_score(player2):.4f}")
    print(f"Player 1 vs Player 3: {player1.expected_score(player3):.4f}")
    print(f"Player 2 vs Player 3: {player2.expected_score(player3):.4f}")
    print()

    # Simulate some matches
    print("Simulating matches...")
    print("Match 1: Player 1 beats Player 2 (upset!)")
    player1.beat(player2)

    print("Match 2: Player 3 beats Player 1 (expected)")
    player3.beat(player1)

    print("Match 3: Player 2 and Player 3 tie")
    player2.tied(player3)
    print()

    # Print updated ratings
    print("Updated ratings after matches:")
    print(f"Player 1: Rating={player1.rating:.1f}, RD={player1.rd:.1f}, Volatility={player1.volatility:.6f}")
    print(f"Player 2: Rating={player2.rating:.1f}, RD={player2.rd:.1f}, Volatility={player2.volatility:.6f}")
    print(f"Player 3: Rating={player3.rating:.1f}, RD={player3.rd:.1f}, Volatility={player3.volatility:.6f}")
    print()

    # Calculate new win probabilities
    print("New win probabilities:")
    print(f"Player 1 vs Player 2: {player1.expected_score(player2):.4f}")
    print(f"Player 1 vs Player 3: {player1.expected_score(player3):.4f}")
    print(f"Player 2 vs Player 3: {player2.expected_score(player3):.4f}")
    print()

    # Demonstrate serialization and deserialization
    print("Demonstrating serialization and deserialization...")
    state = player1.export_state()
    player1_copy = Glicko2Competitor.from_state(state)

    print(f"Original player: {player1}")
    print(f"Deserialized player: {player1_copy}")
    print(
        f"Are they equal? {player1.rating == player1_copy.rating and player1.rd == player1_copy.rd and player1.volatility == player1_copy.volatility}"
    )


if __name__ == "__main__":
    main()
