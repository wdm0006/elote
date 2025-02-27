from elote import LambdaArena, GlickoCompetitor
import json
import random
import copy


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
print("Arena results:")
print(json.dumps(arena.leaderboard(), indent=4))

# Export state and create a deep copy to avoid modifying the original
saved_state = copy.deepcopy(arena.export_state())

# Create a new arena with the saved state
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(100)]
new_arena = LambdaArena(func, base_competitor=GlickoCompetitor)

# Use from_state to recreate competitors
for k, v in saved_state.items():
    new_arena.competitors[k] = GlickoCompetitor.from_state(v)

# Run more matches
new_arena.tournament(matchups)
print("Arena results:")
print(json.dumps(new_arena.leaderboard(), indent=4))
