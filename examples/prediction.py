from elote import EloCompetitor

good = EloCompetitor(initial_rating=400)
better = EloCompetitor(initial_rating=500)

print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))
print('probability of good beating better: %5.2f%%' % (good.expected_score(better) * 100, ))

good.beat(better)

print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))
print('probability of good beating better: %5.2f%%' % (good.expected_score(better) * 100, ))