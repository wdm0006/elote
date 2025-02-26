from elote import LambdaArena, GlickoCompetitor
import json
import random


# sample bout function which just compares the two inputs
def func(a, b):
    if a == b:
        return None
    else:
        return a > b


# start scoring, stop and save state
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)]
arena = LambdaArena(func, base_competitor=GlickoCompetitor)
arena.tournament(matchups)
print(json.dumps(arena.leaderboard(), indent=4))

saved_state = arena.export_state()

# restart with some new class level args
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(100)]
arena = LambdaArena(func, base_competitor=GlickoCompetitor, initial_state=saved_state)
arena.set_competitor_class_var("_c", 5)
arena.tournament(matchups)
print(json.dumps(arena.leaderboard(), indent=4))
