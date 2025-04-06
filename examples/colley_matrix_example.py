"""
Colley Matrix Method example using Elote.

The Colley Matrix Method is a least-squares rating system that solves a system of linear
equations to obtain rankings. It's widely used in sports rankings, particularly college
football.

This example demonstrates:
1. Creating ColleyMatrixCompetitor instances
2. Recording match results
3. Examining how ratings change with match outcomes
4. Visualizing the rating changes over time
"""

import os
import matplotlib.pyplot as plt
from elote import ColleyMatrixCompetitor


def main():
    # Create competitors with default initial rating of 0.5
    team_a = ColleyMatrixCompetitor()
    team_b = ColleyMatrixCompetitor()
    team_c = ColleyMatrixCompetitor()
    team_d = ColleyMatrixCompetitor()

    # Initial ratings and expectations
    print("Initial ratings:")
    print(f"Team A: {team_a.rating:.3f}")
    print(f"Team B: {team_b.rating:.3f}")
    print(f"Team C: {team_c.rating:.3f}")
    print(f"Team D: {team_d.rating:.3f}")

    print("\nInitial win probabilities:")
    print(f"Team A vs Team B: {team_a.expected_score(team_b):.2%}")
    print(f"Team A vs Team C: {team_a.expected_score(team_c):.2%}")

    # Record match results in a tournament
    print("\nSimulating a small tournament...")

    # Track rating history
    a_ratings = [team_a.rating]
    b_ratings = [team_b.rating]
    c_ratings = [team_c.rating]
    d_ratings = [team_d.rating]

    # Round 1
    team_a.beat(team_b)  # A beats B
    team_c.beat(team_d)  # C beats D

    a_ratings.append(team_a.rating)
    b_ratings.append(team_b.rating)
    c_ratings.append(team_c.rating)
    d_ratings.append(team_d.rating)

    # Round 2 - simplified to avoid network issues
    team_b.beat(team_d)  # B beats D

    a_ratings.append(team_a.rating)
    b_ratings.append(team_b.rating)
    c_ratings.append(team_c.rating)
    d_ratings.append(team_d.rating)

    # Round 3 - simplified to avoid network issues
    team_c.beat(team_b)  # C beats B

    a_ratings.append(team_a.rating)
    b_ratings.append(team_b.rating)
    c_ratings.append(team_c.rating)
    d_ratings.append(team_d.rating)

    # Final ratings
    print("\nFinal ratings:")
    print(f"Team A: {team_a.rating:.3f} (won 1, lost 0)")
    print(f"Team B: {team_b.rating:.3f} (won 1, lost 2)")
    print(f"Team C: {team_c.rating:.3f} (won 2, lost 0)")
    print(f"Team D: {team_d.rating:.3f} (won 0, lost 2)")

    # Final win probabilities
    print("\nFinal win probabilities:")
    print(f"Team A vs Team B: {team_a.expected_score(team_b):.2%}")
    print(f"Team A vs Team C: {team_a.expected_score(team_c):.2%}")
    print(f"Team B vs Team C: {team_b.expected_score(team_c):.2%}")
    print(f"Team B vs Team D: {team_b.expected_score(team_d):.2%}")

    # Verify a key property of Colley Matrix ratings: sum of ratings equals n/2
    total_rating = team_a.rating + team_b.rating + team_c.rating + team_d.rating
    print(f"\nSum of all ratings: {total_rating:.3f}")
    print(f"Expected sum (n/2): {4 / 2}")

    # Demonstrate a tie
    print("\nSimulating a tie between Team B and Team D...")
    team_b.tied(team_d)
    print(f"Team B rating after tie: {team_b.rating:.3f}")
    print(f"Team D rating after tie: {team_d.rating:.3f}")

    # Plot rating changes over time
    plt.figure(figsize=(10, 6))
    rounds = range(4)  # Initial + 3 rounds

    plt.plot(rounds, a_ratings, "o-", label="Team A")
    plt.plot(rounds, b_ratings, "s-", label="Team B")
    plt.plot(rounds, c_ratings, "^-", label="Team C")
    plt.plot(rounds, d_ratings, "x-", label="Team D")

    plt.xlabel("Round")
    plt.ylabel("Rating")
    plt.title("Colley Matrix Ratings Over Tournament Rounds")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks(rounds)

    # Save the plot
    plt.savefig(os.path.join("images", "colley_matrix_ratings.png"))
    print("\nRating history plot saved as 'colley_matrix_ratings.png'")

    # Show the plot if running interactively
    # plt.show()


if __name__ == "__main__":
    main()
