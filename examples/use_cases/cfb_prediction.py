from elote import LambdaArena
import datetime
import json

# split into train/test
with open('../data/cfb.json', 'r') as f:
    data = json.load(f)
data = [x for x in data if x.get('score') != '0-0']

# sample data
# {
#     "date": "20170824",
#     "home_team": "Campbellsville",
#     "loser": "Pikeville",
#     "neutral_site": false,
#     "score": "28-14",
#     "winner": "Campbellsville"
# }

# first we split by sorted dates, so all test set is after the train set
dates = sorted(list({datetime.datetime.strptime(x.get('date'), '%Y%m%d') for x in data}))
train_dates = [x.strftime('%Y%m%d') for x in dates[:55]]
test_dates = [x.strftime('%Y%m%d') for x in dates[55:]]

# then form matchup objects (winner first). First sort the data so the matchups happen in true date order
data = sorted(data, key=lambda x: datetime.datetime.strptime(x.get('date'), '%Y%m%d'))
train_matchups = [(x.get('winner'), x.get('loser'), x) for x in data if x.get('date') in train_dates]
test_matchups = [(x.get('winner'), x.get('loser'), x) for x in data if x.get('date') in test_dates]


# we already know the winner, so the lambda here is trivial
def func(a, b):
    return True


# we use the default EloCompetitor, but adjust the k_factor to 50 before running the tournament
arena = LambdaArena(func)
arena.set_competitor_class_var('_k_factor', 400)
arena.tournament(train_matchups)

# do a threshold search and clear the history for validation
_, thresholds = arena.history.random_search(trials=10_000)
arena.clear_history()

# then we print out the top 25 as of the end of our training dataset
rankings = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get('rating'))[:25]
for idx, item in enumerate(rankings):
    print('%d) %s' % (idx + 1, item.get('competitor')))

# now validation
print('Validation Step')
arena.tournament(test_matchups)
arena.history.report_results()
tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(0.4, 0.6)
print('All data')
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))


tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(0.4, 0.6, attribute_filter={'neutral_site': True})
print('only neutral sites')
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))


tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(*thresholds, attribute_filter={'neutral_site': True})
print('using learned thresholds: %s' % (str(thresholds), ))
print('wins: %s' % (tp + tn, ))
print('losses: %s' % (fp + fn, ))
print('do_nothing: %s' % (do_nothing, ))