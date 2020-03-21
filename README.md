Elote
=====

[![Coverage Status](https://coveralls.io/repos/wdm0006/elote/badge.svg?branch=master&service=github)](https://coveralls.io/github/wdm0006/elote?branch=master)  ![travis status](https://travis-ci.org/wdm0006/elote.svg?branch=master) 

A python package for rating competitors based on bouts. The classical example of this would be rating chess players based
on repeated head to head matches between different players. The first rating system implemented in elote, the Elo rating
system, was made for just that [3]. Another well known use case would be for college football rankings.

There are a whole bunch of other use-cases for this sort of system other than sports though, including collaborative
ranking from a group (for voting, prioritizing, or other similar activities).

Currently implemented rating systems are:

 * Elo [3]
 * Glicko-1 [1]
 * ECF [4]
 * DWZ [5]

Usage
=====

There are a bunch of examples in the examples directory if you want to dive a little deeper, but 
Elote is pretty simple to use. The two objects we care about are Competitors and Arenas. Competitors 
are the things you are rating, and an Arena is a mechanism to schedule Bouts between them. First, Competitors:

Competitors
-----------

    from elote import EloCompetitor
    
    good = EloCompetitor(initial_rating=400)
    better = EloCompetitor(initial_rating=500)
    
    print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))
    print('probability of good beating better: %5.2f%%' % (good.expected_score(better) * 100, ))

This creates two competitors, intilized with different starting ratings. Right away we can see how a match
between them is likely to go:

    probability of better beating good: 64.01%
    probability of good beating better: 35.99%

If we actually held the match, and there was an upset, updating their rankings is as easy as:

    good.beat(better)
    
or

    better.lost_to(good)
    
We can then rerun the predictions and see updated probabilities:

    probability of better beating good: 61.25%
    probability of good beating better: 38.75%
    
Not a huge change using default settings.

Arenas
------

Arenas are a useful abstraction for scheduling large numbers of bouts or matches. The LambdaArena object 
takes in a lambda function with two arguments which returns a boolean for if the first argument won (None for ties). Without 
ever manually setting up any competitors, as long as the arguments are hash-able, the Arena object will create
all of the competitors, run the matches and rank them all. 

Here's a toy example which uses a lambda function that just compares two integers. With this, we've implemented
the worst performing, most over complicated sorting algorithm conceivable, but it works:

    from elote import LambdaArena
    import json
    import random
    
    
    # sample bout function which just compares the two inputs
    def func(a, b):
        return a > b
    
    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]
    
    arena = LambdaArena(func)
    arena.tournament(matchups)
    
    print(json.dumps(arena.leaderboard(), indent=4))

The final leaderboard looks like:

    [
        {
            "rating": 560.0,
            "competitor": 1
        },
        {
            "rating": 803.3256886926524,
            "competitor": 2
        },
        {
            "rating": 994.1660057704563,
            "competitor": 3
        },
        {
            "rating": 1096.0912814220258,
            "competitor": 4
        },
        {
            "rating": 1221.000354671287,
            "competitor": 5
        },
        {
            "rating": 1351.4243548137367,
            "competitor": 6
        },
        {
            "rating": 1401.770230395329,
            "competitor": 7
        },
        {
            "rating": 1558.934907485894,
            "competitor": 8
        },
        {
            "rating": 1607.6971796462033,
            "competitor": 9
        },
        {
            "rating": 1708.3786662956998,
            "competitor": 10
        }
    ]


Examples
========

In the examples directory there are a bunch of basic examples using generated data to show different features of elote,
as well as some use cases using real data, so far all from MasseyRatings.com, but transformed into JSON [2].

Installation
============

Soon to be on PyPI, but for now you can fork the repo or install from git.

Supporting only python 3.4+

Contributing
============

This is very much still a work in progress, so if you're interested in contributing, there is lots to do. Open up an
issue or a PR and we can coordinate efforts.

References
==========

 - [1] http://www.glicko.net/glicko/glicko.pdf
 - [2] MasseyRatings.com
 - [3] Elo, Arpad (1978). The Rating of Chessplayers, Past and Present. Arco. ISBN 0-668-04721-6.
 - [4] http://www.ecfgrading.org.uk/new/help.php#elo
 - [5] https://en.wikipedia.org/wiki/Deutsche_Wertungszahl