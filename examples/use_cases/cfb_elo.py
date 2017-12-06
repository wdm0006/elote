from elote import LambdaArena
import json


# we already know the winner, so the lambda here is trivial
def func(a, b):
    return True


# the matchups are filtered down to only those between teams deemed 'reasonable', by me.
filt = {x for _, x in json.load(open('../data/cfb_teams_filtered.json', 'r')).items()}
matchups = [(x.get('winner'), x.get('loser')) for x in json.load(open('../data/cfb.json', 'r')) if x.get('winner') in filt and x.get('loser') in filt]

# we use the default EloCompetitor, but adjust the k_factor to 50 before running the tournament
arena = LambdaArena(func)
arena.set_competitor_class_var('_k_factor', 50)
arena.tournament(matchups)

# then we print out the top 25
rankings = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get('rating'))[:25]
for idx, item in enumerate(rankings):
    print('%d) %s' % (idx + 1, item.get('competitor')))


# Top 25 Teams on Dec 6, 2017 according to this:
# 1) Clemson
# 2) Georgia
# 3) Oklahoma
# 4) Ohio_St
# 5) Auburn
# 6) USC
# 7) Stanford
# 8) Alabama
# 9) TCU
# 10) Miami_FL
# 11) Wisconsin
# 12) Oklahoma_St
# 13) LSU
# 14) Washington
# 15) Boise_St
# 16) Penn_St
# 17) Washington_St
# 18) Memphis
# 19) Notre_Dame
# 20) UCLA
# 21) UCF
# 22) Northwestern
# 23) South_Carolina
# 24) Michigan_St
# 25) Arizona_St



