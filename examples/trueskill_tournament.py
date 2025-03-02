#!/usr/bin/env python
"""
Example demonstrating the use of TrueSkill in a tournament setting.

This example shows how to use the TrueSkill rating system with the LambdaArena
to run a tournament and rank competitors.
"""

import random
import json
from elote import TrueSkillCompetitor, LambdaArena


def main():
    """Run the TrueSkill tournament example."""

    # Create a comparison function that compares two numbers
    # Returns True if a beats b (i.e., a > b)
    def comparison_func(a, b):
        return a > b

    # Create a LambdaArena with TrueSkill competitors
    arena = LambdaArena(
        comparison_func,
        base_competitor=TrueSkillCompetitor,
        base_competitor_kwargs={"initial_mu": 25.0, "initial_sigma": 8.333},
    )

    # Generate 1000 random matchups between numbers 1-10
    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    # Run the tournament
    print("Running tournament with 1000 matchups...")
    arena.tournament(matchups)

    # Display the leaderboard
    print("\nFinal rankings:")
    leaderboard = arena.leaderboard()
    print(json.dumps(leaderboard, indent=4))

    # Display detailed competitor information
    print("\nDetailed competitor information:")
    for entry in leaderboard:
        competitor_id = entry["competitor"]
        rating = entry["rating"]
        competitor = arena.competitors[competitor_id]
        print(f"Competitor {competitor_id}: rating={rating:.2f}, mu={competitor.mu:.2f}, sigma={competitor.sigma:.2f}")

    # Calculate match quality between top competitors
    if len(leaderboard) >= 2:
        top1_id = leaderboard[0]["competitor"]
        top2_id = leaderboard[1]["competitor"]
        top1 = arena.competitors[top1_id]
        top2 = arena.competitors[top2_id]
        match_quality = TrueSkillCompetitor.match_quality(top1, top2)
        print(f"\nMatch quality between top two competitors ({top1_id} vs {top2_id}): {match_quality:.4f}")


if __name__ == "__main__":
    main()
