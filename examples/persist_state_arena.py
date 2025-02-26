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
# We'll manually recreate the competitors to avoid the class_vars issue
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(100)]
new_arena = LambdaArena(func, base_competitor=GlickoCompetitor)

# Manually add competitors from saved state
for k, v in saved_state.items():
    # Filter out any class_vars if present
    competitor_args = {key: value for key, value in v.items() 
                      if key != 'class_vars'}
    new_arena.competitors[k] = GlickoCompetitor(**competitor_args)

# Set class variable after initialization
new_arena.set_competitor_class_var("_c", 5)
new_arena.tournament(matchups)
print("Arena results:")
print(json.dumps(new_arena.leaderboard(), indent=4))
