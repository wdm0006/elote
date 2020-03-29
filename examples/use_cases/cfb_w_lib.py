import CFBScrapy as cfb
from elote import LambdaArena


# pull API data
train_df = cfb.get_game_info(year=2000)
for year in range(1, 18):
    train_df.append(cfb.get_game_info(year=2000 + year))
test_df = cfb.get_game_info(year=2018).append(cfb.get_game_info(year=2019))

# sort the dates and drop unneeded cols
train_df = train_df.reindex(columns=['start_date', 'home_team', 'away_team', 'home_points', 'away_points'])
test_df = test_df.reindex(columns=['start_date', 'home_team', 'away_team', 'home_points', 'away_points'])
train_df = train_df.sort_values(by='start_date')
test_df = test_df.sort_values(by='start_date')


# then form matchup objects (winner first). First sort the data so the matchups happen in true date order
train_matchups = list()
for idx, row in train_df.iterrows():
    train_matchups.append((
        row.home_team,
        row.away_team,
        {"home_points": row.home_points, "away_points": row.away_points}
    ))

test_matchups = list()
for idx, row in test_df.iterrows():
    test_matchups.append((
        row.home_team,
        row.away_team,
        {"home_points": row.home_points, "away_points": row.away_points}
    ))

# we already know the winner, so the lambda here is trivial
def func(a, b, attributes=None):
    if attributes.get('home_points', 0.0) > attributes.get('away_points', 0.0):
        return True
    else:
        return False

# we use the default EloCompetitor, but adjust the k_factor to 50 before running the tournament
arena = LambdaArena(func)
arena.set_competitor_class_var('_k_factor', 800)
arena.tournament(train_matchups)

# do a threshold search and clear the history for validation
_, thresholds = arena.history.random_search(trials=10_000)
tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(*thresholds)
print('\t\tTrain Set: thresholds=%s' % (str(thresholds), ))
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))
print('win pct: %s%%' % (100 * ((tp + tn)/(tp + tn + fp + fn + do_nothing))))
arena.clear_history()

# then we print out the top 25 as of the end of our training dataset
print('\n\nTop 25 as of start of validation:')
rankings = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get('rating'))[:25]
for idx, item in enumerate(rankings):
    print('\t%d) %s' % (idx + 1, item.get('competitor')))

# now validation
print('\n\nStarting Validation Step...')
arena.tournament(test_matchups)
report = arena.history.report_results()

tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(0.4, 0.6)
print('\n\nTest Set: using 0.4/0.6 thresholds')
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))
print('win pct: %s%%' % (100 * ((tp + tn)/(tp + tn + fp + fn + do_nothing))))

tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(*thresholds)
print('\n\nTest Set: using learned thresholds: %s' % (str(thresholds), ))
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))
print('win pct: %s%%' % (100 * ((tp + tn)/(tp + tn + fp + fn + do_nothing))))