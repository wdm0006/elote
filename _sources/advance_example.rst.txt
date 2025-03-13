Advanced Examples
=================


College Football Ranking
------------------------

In this example we are going to use a ``LambdaArena`` and the ``CFPScrapy`` library to build a rating system for college
football and see how it performs.

To start with we need historical data on games to seed our ratings with. Luckily there is a nice library/API for that:

.. code-block:: python

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

Next we need to make a lamba to execute the matchups with. Since we have the scores available in the attributes of our
matchup dataset, we can simply check the score to see if the first competitor won or lost:

.. code-block:: python

    # we already know the winner, so the lambda here is trivial
    def func(a, b, attributes=None):
        if attributes.get('home_points', 0.0) > attributes.get('away_points', 0.0):
            return True
        else:
            return False

 To start with we will use an Elo competitor with a ``_k_factor`` of 400. We will train the ratings with a tournament
 on the first couple of decades of data:

.. code-block:: python

    # we use the default EloCompetitor, but adjust the k_factor to 400 before running the tournament
    arena = LambdaArena(func)
    arena.set_competitor_class_var('_k_factor', 400)
    arena.tournament(train_matchups)

Once we've developed some ratings, let's take a look at the training set and how the ratings performed, and use that
to select some potential thresholds:

.. code-block:: python

    # do a threshold search and clear the history for validation
    _, thresholds = arena.history.random_search(trials=10_000)
    tp, fp, tn, fn, do_nothing = arena.history.confusion_matrix(*thresholds)
    print('\n\nTrain Set: thresholds=%s' % (str(thresholds), ))
    print('wins: %s' % (tp + tn, ))
    print('losses: %s' % (fp + fn, ))
    print('do_nothing: %s' % (do_nothing, ))
    print('win pct: %s%%' % (100 * ((tp + tn)/(tp + tn + fp + fn + do_nothing))))
    arena.clear_history()

This will return:

.. code-block::

    Train Set: thresholds=[0.6350196774347375, 0.9364243175248251]
    wins: 267
    losses: 236
    do_nothing: 171
    win pct: 39.61424332344214%

And while we are here let's also print out what the rankings would have been to start the 2018 season:

.. code-block:: python

    # then we print out the top 25 as of the end of our training dataset
    print('\n\nTop 25 as of start of validation:')
    rankings = sorted(arena.leaderboard(), reverse=True, key=lambda x: x.get('rating'))[:25]
    for idx, item in enumerate(rankings):
        print('\t%d) %s' % (idx + 1, item.get('competitor')))

 Which will print:

 .. code-block::

    Top 25 as of start of validation:
    1) Miami
    2) Oklahoma
    3) Florida State
    4) Oregon State
    5) Texas
    6) Georgia Tech
    7) Washington
    8) Virginia Tech
    9) Kansas State
    10) Notre Dame
    11) Cincinnati
    12) TCU
    13) Michigan
    14) Arkansas
    15) Toledo
    16) Air Force
    17) Tennessee
    18) Auburn
    19) Florida
    20) Boise State
    21) Louisville
    22) Middle Tennessee
    23) North Carolina
    24) Pittsburgh
    25) Oregon

Now let's take a look at some hold out validation by using these ratings to take a look at the 2018 and 2019 seasons. The
ratings will of course still update as the games are evaluated:

.. code-block:: python

    # now validation
    print('\n\nStarting Validation Step...')
    arena.tournament(test_matchups)
    report = arena.history.report_results()

We can then look at the results from just this set (notice we ran ``clear_history()`` up above to wipe out the train set
results from our history tracker:

.. code-block:: python

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

Which will print out:

.. code-block::

    Test Set: using 0.4/0.6 thresholds
    wins: 1045
    losses: 456
    do_nothing: 193
    win pct: 61.68831168831169%

    Test Set: using learned thresholds: [0.6350196774347375, 0.9364243175248251]
    wins: 804
    losses: 483
    do_nothing: 407
    win pct: 47.4616292798111%

Not awesome. This is probably related to ``k_factor`` which tunes how quickly ratings will respond to new matchups. Let's
try doubling it to 800 and rerunning. Now you will see the final output:

.. code-block::

    Test Set: using 0.4/0.6 thresholds
    wins: 1095
    losses: 503
    do_nothing: 96
    win pct: 64.63990554899645%


    Test Set: using learned thresholds: [0.5277889558418678, 0.6981558136040092]
    wins: 1093
    losses: 526
    do_nothing: 75
    win pct: 64.52184179456907%

Before we get too excited about this, let's take a look at the post-game win probabilities provided by the same API we
are getting data from:

.. code-block::

    Test Set: using probabilities from dataset as baseline
    wins: 1481
    losses: 117
    do_nothing: 96
    win pct: 87.42621015348288%

So we're not exactly going to Vegas.