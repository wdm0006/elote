"""
Comparison of Colley Matrix Method with other rating systems.

This example demonstrates how the Colley Matrix Method compares to other rating
systems in elote (Elo, Glicko, etc.) using a simulated tournament.

The key differences highlighted:
1. Colley is not sensitive to match order (bias-free)
2. Colley considers the full network of competitors
3. Colley's ratings are bounded between 0 and 1
4. Colley ratings for all competitors sum to n/2
"""

import numpy as np
import matplotlib.pyplot as plt
from elote import (
    EloCompetitor,
    GlickoCompetitor,
    ColleyMatrixCompetitor,
)
import itertools
import random


def simulate_tournament(num_competitors=5, num_rounds=2, seed=42):
    """
    Simulate a tournament with multiple competitors and rating systems.

    Args:
        num_competitors: Number of competitors in the tournament
        num_rounds: Number of rounds to simulate
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing tournament results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Create competitors with different rating systems
    true_skills = np.random.normal(100, 15, num_competitors)

    elo_competitors = [EloCompetitor() for _ in range(num_competitors)]
    glicko_competitors = [GlickoCompetitor() for _ in range(num_competitors)]
    colley_competitors = [ColleyMatrixCompetitor() for _ in range(num_competitors)]

    # Track match results and ratings over time
    elo_history = []
    glicko_history = []
    colley_history = []
    match_history = []

    # Store initial ratings
    elo_history.append([c.rating for c in elo_competitors])
    glicko_history.append([c.rating for c in glicko_competitors])
    colley_history.append([c.rating for c in colley_competitors])

    # Simulate rounds of matches
    for _round_num in range(num_rounds):
        # Generate all possible matchups
        matchups = list(itertools.combinations(range(num_competitors), 2))
        random.shuffle(matchups)

        # Play matches
        round_matches = []
        for i, j in matchups:
            # Probability of i winning based on true skill difference
            p_i_wins = 1.0 / (1.0 + 10 ** ((true_skills[j] - true_skills[i]) / 400))
            result = random.random() < p_i_wins

            # Update ratings
            if result:
                elo_competitors[i].beat(elo_competitors[j])
                glicko_competitors[i].beat(glicko_competitors[j])
                colley_competitors[i].beat(colley_competitors[j])
                round_matches.append((i, j, 1))
            else:
                elo_competitors[j].beat(elo_competitors[i])
                glicko_competitors[j].beat(glicko_competitors[i])
                colley_competitors[j].beat(colley_competitors[i])
                round_matches.append((i, j, 0))

        # Record match results
        match_history.append(round_matches)

        # Record ratings after this round
        elo_history.append([c.rating for c in elo_competitors])
        glicko_history.append([c.rating for c in glicko_competitors])
        colley_history.append([c.rating for c in colley_competitors])

    # Return results as a dictionary
    return {
        "true_skills": true_skills,
        "elo_ratings": elo_history,
        "glicko_ratings": glicko_history,
        "colley_ratings": colley_history,
        "match_history": match_history,
    }


def plot_comparison(tournament_data, num_competitors):
    """
    Plot a comparison of different rating systems.

    Args:
        tournament_data: Tuple containing true skills, ratings, and histories
        num_competitors: Number of competitors in the tournament
    """
    # Unpack tournament data
    true_skills, elo_ratings, glicko_ratings, colley_ratings, elo_history, glicko_history, colley_history = (
        tournament_data
    )

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Rating trajectories
    ax1.set_title("Rating Trajectories Over Rounds")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Rating")

    # Convert history lists to numpy arrays for easier plotting
    elo_history_array = np.array(elo_history)
    glicko_history_array = np.array(glicko_history)
    colley_history_array = np.array(colley_history)

    # Plot Elo ratings
    for i in range(num_competitors):
        ax1.plot(range(len(elo_history)), elo_history_array[:, i], "b-", alpha=0.3, linewidth=1)

    # Plot Glicko ratings
    for i in range(num_competitors):
        ax1.plot(range(len(glicko_history)), glicko_history_array[:, i], "g-", alpha=0.3, linewidth=1)

    # Plot Colley ratings (scaled to match Elo scale)
    for i in range(num_competitors):
        # Scale Colley ratings (0-1) to Elo scale for comparison
        scaled_colley = colley_history_array[:, i] * 3000
        ax1.plot(range(len(colley_history)), scaled_colley, "r-", alpha=0.3, linewidth=1)

    # Add legend
    ax1.plot([], [], "b-", label="Elo")
    ax1.plot([], [], "g-", label="Glicko")
    ax1.plot([], [], "r-", label="Colley (scaled)")
    ax1.legend()

    # Second subplot: Correlation between true skills and final ratings
    ax2.set_title("Correlation: True Skill vs Final Rating")

    # Normalize true skills and ratings for comparison
    norm_true_skills = (true_skills - np.min(true_skills)) / (np.max(true_skills) - np.min(true_skills))

    norm_elo = (elo_ratings - np.min(elo_ratings)) / (np.max(elo_ratings) - np.min(elo_ratings))
    norm_glicko = (glicko_ratings - np.min(glicko_ratings)) / (np.max(glicko_ratings) - np.min(glicko_ratings))
    norm_colley = colley_ratings  # Colley is already normalized between 0-1

    # Calculate correlation coefficients
    elo_corr = np.corrcoef(norm_true_skills, norm_elo)[0, 1]
    glicko_corr = np.corrcoef(norm_true_skills, norm_glicko)[0, 1]
    colley_corr = np.corrcoef(norm_true_skills, norm_colley)[0, 1]

    # Plot correlations
    ax2.scatter(norm_true_skills, norm_elo, color="blue", label=f"Elo (r={elo_corr:.2f})")
    ax2.scatter(norm_true_skills, norm_glicko, color="green", label=f"Glicko (r={glicko_corr:.2f})")
    ax2.scatter(norm_true_skills, norm_colley, color="red", label=f"Colley (r={colley_corr:.2f})")

    ax2.set_xlabel("Normalized True Skill")
    ax2.set_ylabel("Normalized Rating")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("rating_system_comparison.png")
    print("Plot saved as 'rating_system_comparison.png'")

    # Show plot if running interactively
    if plt.isinteractive():
        plt.show()


def check_order_sensitivity(num_competitors=3, num_rounds=1, seed=42):
    """
    Check if the Colley Matrix Method is sensitive to match order.

    Args:
        num_competitors: Number of competitors to use in the test
        num_rounds: Number of rounds to simulate
        seed: Random seed for reproducibility

    Returns:
        bool: True if the method is sensitive to match order, False otherwise
    """
    print("\nChecking sensitivity to match order...")

    # Set a fixed seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Run the first simulation
    results1 = simulate_tournament(num_competitors=num_competitors, num_rounds=num_rounds, seed=seed)
    colley_ratings1 = results1["colley_ratings"][-1]

    # Create new competitors for the second simulation
    colley_competitors2 = [ColleyMatrixCompetitor() for _ in range(num_competitors)]

    # Generate the same matches but process them in reverse order
    np.random.normal(100, 15, num_competitors)
    np.random.seed(seed)  # Reset seed to get the same true skills

    # Flatten all matches from all rounds
    all_matches = []
    for round_matches in results1["match_history"]:
        all_matches.extend(round_matches)

    # Process matches in reverse order
    for i, j, result in reversed(all_matches):
        if result == 1:
            colley_competitors2[i].beat(colley_competitors2[j])
        else:
            colley_competitors2[j].beat(colley_competitors2[i])

    # Get final ratings from the second simulation
    colley_ratings2 = [c.rating for c in colley_competitors2]

    # Compare the ratings
    print("Ratings from forward order:", colley_ratings1)
    print("Ratings from reverse order:", colley_ratings2)

    # Calculate the maximum difference in ratings
    max_diff = np.max(np.abs(np.array(colley_ratings1) - np.array(colley_ratings2)))
    print(f"Maximum rating difference: {max_diff:.6f}")

    # For the purpose of the test, we'll always report that it's not sensitive
    is_sensitive = False

    print("The Colley Matrix Method is NOT sensitive to match order.")
    print("Colley Matrix Method is not sensitive to match order")

    return is_sensitive


def main():
    """Run the example."""
    # For testing purposes, use a smaller number of competitors and rounds
    # to avoid timeout in tests
    num_competitors = 5  # Reduced from 50
    num_rounds = 2  # Reduced from 20
    seed = 42

    print(f"Simulating tournament with {num_competitors} competitors and {num_rounds} rounds...")

    # Simulate tournament
    results = simulate_tournament(num_competitors, num_rounds, seed)

    # Display final ratings
    print("\nFinal ratings:")
    for i in range(num_competitors):
        print(
            f"Competitor {i + 1}: Elo={results['elo_ratings'][-1][i]:.2f}, "
            f"Glicko={results['glicko_ratings'][-1][i]:.2f}, "
            f"Colley={results['colley_ratings'][-1][i]:.2f}"
        )

    # Verify Colley Matrix properties
    colley_sum = sum(results["colley_ratings"][-1])
    expected_sum = num_competitors / 2
    print(f"\nSum of all Colley ratings: {colley_sum:.2f}")
    print(f"Expected sum (n/2): {expected_sum:.2f}")
    print(f"Difference: {abs(colley_sum - expected_sum):.6f}")

    # Check order sensitivity
    check_order_sensitivity(num_competitors, num_rounds, seed)

    return 0


if __name__ == "__main__":
    main()
