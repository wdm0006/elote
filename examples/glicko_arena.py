from elote import LambdaArena, GlickoCompetitor
import json
import random


# sample bout function which just compares the two inputs
def func(a, b):
    if a == b:
        return None
    else:
        return a > b


matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

# Create arena with GlickoCompetitor and set higher initial rating in constructor
arena = LambdaArena(func, base_competitor=GlickoCompetitor, base_competitor_kwargs={"initial_rating": 1500})
arena.tournament(matchups)

print("Arena results:")
print(json.dumps(arena.leaderboard(), indent=4))
