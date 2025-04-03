#!/usr/bin/env python
"""
Example demonstrating the use of the Glicko-2 rating system.

This example shows how to create Glicko-2 competitors, calculate win probabilities,
update ratings after matches, and how ratings change over time due to inactivity.
"""

from elote import Glicko2Competitor
from datetime import datetime, timedelta


def main():
    """Run the Glicko-2 example."""
    # Create initial time and competitors
    initial_time = datetime(2024, 1, 1)
    player1 = Glicko2Competitor(initial_rating=1500, initial_rd=350, initial_volatility=0.06, initial_time=initial_time)
    player2 = Glicko2Competitor(initial_rating=1700, initial_rd=300, initial_volatility=0.06, initial_time=initial_time)
    player3 = Glicko2Competitor(initial_rating=1800, initial_rd=200, initial_volatility=0.05, initial_time=initial_time)

    # Print initial ratings
    print("Initial ratings (January 1st, 2024):")
    print(f"Player 1: Rating={player1.rating}, RD={player1.rd}, Volatility={player1.volatility:.6f}")
    print(f"Player 2: Rating={player2.rating}, RD={player2.rd}, Volatility={player2.volatility:.6f}")
    print(f"Player 3: Rating={player3.rating}, RD={player3.rd}, Volatility={player3.volatility:.6f}")
    print()

    # Calculate initial win probabilities
    print("Initial win probabilities:")
    print(f"Player 1 vs Player 2: {player1.expected_score(player2):.4f}")
    print(f"Player 1 vs Player 3: {player1.expected_score(player3):.4f}")
    print(f"Player 2 vs Player 3: {player2.expected_score(player3):.4f}")
    print()

    # Simulate some matches with time gaps
    print("Simulating matches over time...")

    # First match after 5 days
    match1_time = initial_time + timedelta(days=5)
    print("\nMatch 1 (January 6th): Player 1 beats Player 2 (upset!)")
    print("RDs before match due to 5 days inactivity:")
    print(f"Player 1 RD: {player1.rd:.1f}")
    print(f"Player 2 RD: {player2.rd:.1f}")
    player1.beat(player2, match_time=match1_time)
    print("RDs after match:")
    print(f"Player 1 RD: {player1.rd:.1f}")
    print(f"Player 2 RD: {player2.rd:.1f}")

    # Second match after another 10 days
    match2_time = match1_time + timedelta(days=10)
    print("\nMatch 2 (January 16th): Player 3 beats Player 1")
    print("RDs before match due to 10 days inactivity:")
    print(f"Player 1 RD: {player1.rd:.1f}")
    print(f"Player 3 RD: {player3.rd:.1f}")
    player3.beat(player1, match_time=match2_time)
    print("RDs after match:")
    print(f"Player 1 RD: {player1.rd:.1f}")
    print(f"Player 3 RD: {player3.rd:.1f}")

    # Third match after another 15 days
    match3_time = match2_time + timedelta(days=15)
    print("\nMatch 3 (January 31st): Player 2 and Player 3 tie")
    print("RDs before match due to inactivity:")
    print(f"Player 2 RD: {player2.rd:.1f} (25 days inactive)")
    print(f"Player 3 RD: {player3.rd:.1f} (15 days inactive)")
    player2.tied(player3, match_time=match3_time)
    print("RDs after match:")
    print(f"Player 2 RD: {player2.rd:.1f}")
    print(f"Player 3 RD: {player3.rd:.1f}")
    print()

    # Print final ratings
    print("Final ratings (January 31st, 2024):")
    print(f"Player 1: Rating={player1.rating:.1f}, RD={player1.rd:.1f}, Volatility={player1.volatility:.6f}")
    print(f"Player 2: Rating={player2.rating:.1f}, RD={player2.rd:.1f}, Volatility={player2.volatility:.6f}")
    print(f"Player 3: Rating={player3.rating:.1f}, RD={player3.rd:.1f}, Volatility={player3.volatility:.6f}")
    print()

    # Calculate final win probabilities
    print("Final win probabilities:")
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
