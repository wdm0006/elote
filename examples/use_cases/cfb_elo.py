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
# 2) Oklahoma
# 3) Ohio_St
# 4) Georgia
# 5) Auburn
# 6) Alabama
# 7) UCF
# 8) USC
# 9) Miami_FL
# 10) LSU
# 11) Memphis
# 12) Penn_St
# 13) Wisconsin
# 14) TCU
# 15) Oklahoma_St
# 16) Northwestern
# 17) Washington
# 18) Stanford
# 19) Washington_St
# 20) Michigan_St
# 21) Houston
# 22) Notre_Dame
# 23) Boise_St
# 24) South_Carolina
# 25) Missouri

# Final AP Poll that year
# 1. Alabama (57) (13-1) 1521
# 2. Georgia (13-2) 1454
# 3. Oklahoma (12-1) 1374
# 4. Clemson (12-2) 1292
# 5. Ohio State (12-2) 1286
# 6. UCF (4) (13-0) 1248
# 7. Wisconsin (13-1) 1194
# 8. Penn State (11-2) 1120
# 9. TCU (11-3) 974
# 10. Auburn (10-4)
# 11. Notre Dame (10-3) 857
# 12. USC (11-3) 839
# 13. Miami (10-3) 769
# 14. Oklahoma State (10-3) 758
# 15. Michigan State (10-3) 705
# 16. Washington (10-3) 668
# 17. Northwestern (10-3) 528
# 18. LSU (9-4) 368
# 19. Mississippi State (9-4) 359
# 20. Stanford (9-5) 336
# 21. USF (10-2) 267
# 22. Boise State (11-3) 251
# 23. NC State (9-4) 232
# 24. Virginia Tech (9-4) 125
# 25. Memphis (10-3) 119



