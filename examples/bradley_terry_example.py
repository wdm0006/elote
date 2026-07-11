"""
Bradley-Terry model example using Elote.

The Bradley-Terry model is a maximum-likelihood paired-comparison system. Each competitor
is assigned a single latent strength, and the probability that one competitor beats another
is a logistic function of the difference in their strengths -- exactly the functional form
of the Elo expected score. Unlike Elo, which nudges ratings incrementally, Bradley-Terry
re-fits the strengths of the whole connected group of competitors by maximum likelihood
after every result, so the final ratings depend only on the set of outcomes, not their order.

This example demonstrates:
1. Creating BradleyTerryCompetitor instances (on an Elo-compatible rating scale)
2. Recording match results and watching the ratings re-fit
3. The order-independence of the maximum-likelihood fit
4. That the expected score matches Elo's for the same rating gap
5. Using BradleyTerryCompetitor inside a LambdaArena tournament
"""

import os
import json
import matplotlib.pyplot as plt
from elote import BradleyTerryCompetitor, EloCompetitor, LambdaArena


def basic_bouts():
    # Ratings use the same 1500-centered scale as Elo.
    team_a = BradleyTerryCompetitor(initial_rating=1500)
    team_b = BradleyTerryCompetitor(initial_rating=1500)
    team_c = BradleyTerryCompetitor(initial_rating=1500)
    team_d = BradleyTerryCompetitor(initial_rating=1500)

    print("Initial ratings:")
    for name, team in [("A", team_a), ("B", team_b), ("C", team_c), ("D", team_d)]:
        print(f"Team {name}: {team.rating:.1f}")

    print("\nInitial win probability A vs B:", f"{team_a.expected_score(team_b):.2%}")

    print("\nSimulating a small tournament...")

    a_ratings = [team_a.rating]
    b_ratings = [team_b.rating]
    c_ratings = [team_c.rating]
    d_ratings = [team_d.rating]

    def snapshot():
        a_ratings.append(team_a.rating)
        b_ratings.append(team_b.rating)
        c_ratings.append(team_c.rating)
        d_ratings.append(team_d.rating)

    # Round 1
    team_a.beat(team_b)  # A beats B
    team_c.beat(team_d)  # C beats D
    snapshot()

    # Round 2
    team_a.beat(team_c)  # A beats C
    team_b.beat(team_d)  # B beats D
    snapshot()

    # Round 3
    team_a.beat(team_d)  # A beats D
    team_b.beat(team_c)  # B beats C
    snapshot()

    print("\nFinal ratings (higher = stronger):")
    print(f"Team A: {team_a.rating:.1f} (won 3, lost 0)")
    print(f"Team B: {team_b.rating:.1f} (won 1, lost 1)")
    print(f"Team C: {team_c.rating:.1f} (won 1, lost 1)")
    print(f"Team D: {team_d.rating:.1f} (won 0, lost 2)")

    print("\nFinal win probabilities:")
    print(f"Team A vs Team B: {team_a.expected_score(team_b):.2%}")
    print(f"Team A vs Team D: {team_a.expected_score(team_d):.2%}")

    # Plot rating changes over time
    plt.figure(figsize=(10, 6))
    rounds = range(len(a_ratings))
    plt.plot(rounds, a_ratings, "o-", label="Team A")
    plt.plot(rounds, b_ratings, "s-", label="Team B")
    plt.plot(rounds, c_ratings, "^-", label="Team C")
    plt.plot(rounds, d_ratings, "x-", label="Team D")
    plt.xlabel("Round")
    plt.ylabel("Rating")
    plt.title("Bradley-Terry Ratings Over Tournament Rounds")
    plt.legend()
    plt.grid(True)
    plt.xticks(list(rounds))

    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", "bradley_terry_ratings.png"))
    print("\nRating history plot saved as 'images/bradley_terry_ratings.png'")


def order_independence():
    """The maximum-likelihood fit depends only on the set of results, not their order."""
    print("\n--- Order independence ---")

    # Sequence 1: alternating
    x1, y1 = BradleyTerryCompetitor(), BradleyTerryCompetitor()
    x1.beat(y1)
    y1.beat(x1)
    x1.beat(y1)

    # Sequence 2: same results, different order
    x2, y2 = BradleyTerryCompetitor(), BradleyTerryCompetitor()
    y2.beat(x2)
    x2.beat(y2)
    x2.beat(y2)

    print(f"Order 1 -> X: {x1.rating:.2f}, Y: {y1.rating:.2f}")
    print(f"Order 2 -> X: {x2.rating:.2f}, Y: {y2.rating:.2f}")
    print("Ratings match regardless of order:", round(x1.rating, 6) == round(x2.rating, 6))


def matches_elo():
    """For a given rating gap, Bradley-Terry's expected score matches Elo's."""
    print("\n--- Expected score matches Elo ---")
    bt_strong = BradleyTerryCompetitor(initial_rating=1600)
    bt_weak = BradleyTerryCompetitor(initial_rating=1400)
    elo_strong = EloCompetitor(initial_rating=1600)
    elo_weak = EloCompetitor(initial_rating=1400)
    print(f"Bradley-Terry: {bt_strong.expected_score(bt_weak):.6f}")
    print(f"Elo:           {elo_strong.expected_score(elo_weak):.6f}")


def arena_tournament():
    """BradleyTerryCompetitor plugs straight into a LambdaArena."""
    print("\n--- LambdaArena tournament ---")

    # Latent strengths 0..5; higher wins. The arena should recover this order.
    matchups = [(a, b) for a in range(6) for b in range(6) if a != b]

    arena = LambdaArena(lambda a, b: a > b, base_competitor=BradleyTerryCompetitor)
    arena.tournament(matchups)

    print("Leaderboard (best first):")
    print(json.dumps(arena.leaderboard(), indent=2, default=str))


def main():
    basic_bouts()
    order_independence()
    matches_elo()
    arena_tournament()


if __name__ == "__main__":
    main()
