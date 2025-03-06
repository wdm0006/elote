#!/usr/bin/env python
"""
Example demonstrating the use of the TrueSkill rating system.

This example shows how to create TrueSkill competitors, calculate win probabilities,
update ratings after matches, and work with teams.
"""

from elote import TrueSkillCompetitor


def main():
    """Run the TrueSkill example."""
    # Create players with different initial skill levels
    player1 = TrueSkillCompetitor(initial_mu=25.0, initial_sigma=8.333)
    player2 = TrueSkillCompetitor(initial_mu=30.0, initial_sigma=7.0)
    player3 = TrueSkillCompetitor(initial_mu=20.0, initial_sigma=6.0)
    player4 = TrueSkillCompetitor(initial_mu=35.0, initial_sigma=5.0)

    # Print initial ratings
    print("Initial ratings:")
    print(f"Player 1: mu={player1.mu:.2f}, sigma={player1.sigma:.2f}, rating={player1.rating:.2f}")
    print(f"Player 2: mu={player2.mu:.2f}, sigma={player2.sigma:.2f}, rating={player2.rating:.2f}")
    print(f"Player 3: mu={player3.mu:.2f}, sigma={player3.sigma:.2f}, rating={player3.rating:.2f}")
    print(f"Player 4: mu={player4.mu:.2f}, sigma={player4.sigma:.2f}, rating={player4.rating:.2f}")
    print()

    # Calculate win probabilities
    print("Win probabilities:")
    print(f"Player 1 vs Player 2: {player1.expected_score(player2):.4f}")
    print(f"Player 1 vs Player 3: {player1.expected_score(player3):.4f}")
    print(f"Player 2 vs Player 4: {player2.expected_score(player4):.4f}")
    print()

    # Calculate match quality
    print("Match quality:")
    print(f"Player 1 vs Player 2: {TrueSkillCompetitor.match_quality(player1, player2):.4f}")
    print(f"Player 1 vs Player 3: {TrueSkillCompetitor.match_quality(player1, player3):.4f}")
    print(f"Player 2 vs Player 4: {TrueSkillCompetitor.match_quality(player2, player4):.4f}")
    print()

    # Simulate some matches
    print("Simulating matches...")
    print("Match 1: Player 1 beats Player 2 (upset!)")
    player1.beat(player2)

    print("Match 2: Player 3 beats Player 1 (another upset!)")
    player3.beat(player1)

    print("Match 3: Player 2 and Player 4 tie")
    player2.tied(player4)
    print()

    # Print updated ratings
    print("Updated ratings after matches:")
    print(f"Player 1: mu={player1.mu:.2f}, sigma={player1.sigma:.2f}, rating={player1.rating:.2f}")
    print(f"Player 2: mu={player2.mu:.2f}, sigma={player2.sigma:.2f}, rating={player2.rating:.2f}")
    print(f"Player 3: mu={player3.mu:.2f}, sigma={player3.sigma:.2f}, rating={player3.rating:.2f}")
    print(f"Player 4: mu={player4.mu:.2f}, sigma={player4.sigma:.2f}, rating={player4.rating:.2f}")
    print()

    # Calculate new win probabilities
    print("New win probabilities:")
    print(f"Player 1 vs Player 2: {player1.expected_score(player2):.4f}")
    print(f"Player 1 vs Player 3: {player1.expected_score(player3):.4f}")
    print(f"Player 2 vs Player 4: {player2.expected_score(player4):.4f}")
    print()

    # Demonstrate team creation
    print("Team creation:")
    team1_mu, team1_sigma = TrueSkillCompetitor.create_team([player1, player3])
    team2_mu, team2_sigma = TrueSkillCompetitor.create_team([player2, player4])
    print(f"Team 1 (Players 1 & 3): mu={team1_mu:.2f}, sigma={team1_sigma:.2f}")
    print(f"Team 2 (Players 2 & 4): mu={team2_mu:.2f}, sigma={team2_sigma:.2f}")
    print()

    # Demonstrate serialization and deserialization
    print("Demonstrating serialization and deserialization...")
    state = player1.export_state()
    player1_copy = TrueSkillCompetitor.from_state(state)

    print(f"Original player: {player1}")
    print(f"Deserialized player: {player1_copy}")
    print(f"Are they equal? {player1.mu == player1_copy.mu and player1.sigma == player1_copy.sigma}")


if __name__ == "__main__":
    main()
