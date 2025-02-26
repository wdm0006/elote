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

arena = LambdaArena(func, base_competitor=GlickoCompetitor)
arena.tournament(matchups)

print("Arena results:")
print(json.dumps(arena.leaderboard(), indent=4))
