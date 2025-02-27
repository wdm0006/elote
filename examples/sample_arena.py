from elote import LambdaArena, EloCompetitor
import json
import random


# sample bout function which just compares the two inputs
def func(a, b):
    if a == b:
        return None
    else:
        return a > b


# Configure the EloCompetitor class with a moderate k_factor
# Note: Using a more moderate k_factor (20) to prevent ratings from changing too drastically
EloCompetitor.configure_class(k_factor=20)

# Create arena with a higher initial rating for all competitors
# Using 1200 as initial rating (standard chess starting rating) to prevent negative ratings
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]
arena = LambdaArena(func, base_competitor=EloCompetitor, base_competitor_kwargs={"initial_rating": 1200})
arena.tournament(matchups)

print("Arena results:")
print(json.dumps(arena.leaderboard(), indent=4))
