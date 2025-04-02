from elote import LambdaArena, GlickoCompetitor
import json
import random
from datetime import datetime, timedelta


# sample bout function which just compares the two inputs
def func(a, b):
    if a == b:
        return None
    else:
        return a > b


# Create initial time and a list of matchups with timestamps spread over a month
initial_time = datetime(2024, 1, 1)
matchups_with_time = []
for _i in range(1000):
    # Random matchup
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    # Random time within the month (0-30 days from initial time)
    match_time = initial_time + timedelta(
        days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )
    matchups_with_time.append((a, b, match_time))

# Sort matchups by time to ensure chronological order
matchups_with_time.sort(key=lambda x: x[2])

# Create arena with GlickoCompetitor and set initial time
arena = LambdaArena(
    func,
    base_competitor=GlickoCompetitor,
    base_competitor_kwargs={"initial_rating": 1500, "initial_rd": 350, "initial_time": initial_time},
)

# Process matches in chronological order
for a, b, match_time in matchups_with_time:
    # Use matchup() instead of tournament() to handle match times
    if func(a, b):  # If a wins
        arena.matchup(a, b, match_time=match_time)
    else:  # If b wins
        arena.matchup(b, a, match_time=match_time)

print("\nArena results after one month of matches:")
print("(Notice how less active players have higher RD values)")
leaderboard = arena.leaderboard()

# Convert leaderboard list to a dictionary and add RD values and last activity times
leaderboard_dict = {}
for entry in leaderboard:
    player_id = entry["competitor"]
    leaderboard_dict[player_id] = entry
    competitor = arena.competitors.get(player_id)
    if competitor:
        leaderboard_dict[player_id]["rd"] = round(competitor.rd, 2)
        leaderboard_dict[player_id]["last_activity"] = competitor._last_activity.strftime("%Y-%m-%d %H:%M")

print(json.dumps(leaderboard_dict, indent=4))
