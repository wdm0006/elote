Examples
========

Sample Bout
-----------

.. code-block:: python

    from elote import EloCompetitor

    good = EloCompetitor()
    better = EloCompetitor()
    best = EloCompetitor()

    print('Starting ratings:')
    print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

    print('\nAfter matches')

    for _ in range(10):
        better.beat(good)
        best.beat(better)
        print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

.. code-block::

    Starting ratings:
     400.00,  400.00,  400.00

    After matches
     384.00,  399.26,  416.74
     368.70,  398.66,  432.64
     354.08,  398.18,  447.75
     340.10,  397.79,  462.11
     326.73,  397.49,  475.78
     313.95,  397.25,  488.80
     301.71,  397.08,  501.21
     289.99,  396.95,  513.05
     278.77,  396.87,  524.36
     268.01,  396.81,  535.18

Bout with Initialization
------------------------

.. code-block:: python

    from elote import EloCompetitor

    good = EloCompetitor(initial_rating=500)
    better = EloCompetitor(initial_rating=450)
    best = EloCompetitor(initial_rating=400)

    print('Starting ratings:')
    print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

    print('\nAfter matches')

    for _ in range(20):
        better.beat(good)
        best.beat(better)
        print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

.. code-block::

    Starting ratings:
     500.00,  450.00,  400.00

    After matches
     481.71,  449.18,  419.10
     464.22,  448.50,  437.28
     447.50,  447.94,  454.57
     431.52,  447.49,  471.00
     416.25,  447.13,  486.62
     401.67,  446.86,  501.47
     387.74,  446.65,  515.61
     374.43,  446.51,  529.07
     361.70,  446.41,  541.89
     349.52,  446.35,  554.13
     337.87,  446.32,  565.81
     326.71,  446.31,  576.98
     316.01,  446.33,  587.66
     305.74,  446.36,  597.90
     295.89,  446.40,  607.71
     286.42,  446.45,  617.13
     277.31,  446.51,  626.19
     268.53,  446.57,  634.89
     260.08,  446.64,  643.28
     251.93,  446.70,  651.36

Bout with Ties
--------------

.. code-block:: python

    from elote import EloCompetitor

    good = EloCompetitor(initial_rating=500)
    better = EloCompetitor(initial_rating=450)
    best = EloCompetitor(initial_rating=400)
    also_best = EloCompetitor(initial_rating=400)

    print('Starting ratings:')
    print('%7.2f, %7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, also_best.rating, ))

    print('\nAfter matches')

    for _ in range(20):
        better.beat(good)
        better.lost_to(best)
        best.tied(also_best)
        print('%7.2f, %7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, also_best.rating, ))

.. code-block::

    Starting ratings:
     500.00,  450.00,  400.00,  400.00

    After matches
     481.71,  449.18,  418.23,  400.88
     464.22,  448.46,  434.81,  402.51
     447.49,  447.79,  449.93,  404.78
     431.51,  447.14,  463.75,  407.60
     416.23,  446.48,  476.42,  410.87
     401.62,  445.80,  488.06,  414.53
     387.64,  445.07,  498.78,  418.51
     374.26,  444.30,  508.69,  422.75
     361.44,  443.48,  517.86,  427.22
     349.15,  442.60,  526.39,  431.86
     337.36,  441.66,  534.33,  436.65
     326.02,  440.68,  541.75,  441.55
     315.12,  439.64,  548.70,  446.54
     304.62,  438.56,  555.22,  451.60
     294.50,  437.44,  561.36,  456.71
     284.73,  436.28,  567.15,  461.84
     275.30,  435.09,  572.62,  466.99
     266.18,  433.87,  577.81,  472.14
     257.35,  432.62,  582.74,  477.28
     248.80,  431.35,  587.44,  482.40

Prediction
----------

.. code-block:: python

    from elote import EloCompetitor

    good = EloCompetitor(initial_rating=400)
    better = EloCompetitor(initial_rating=500)

    print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))
    print('probability of good beating better: %5.2f%%' % (good.expected_score(better) * 100, ))

    good.beat(better)

    print('probability of better beating good: %5.2f%%' % (better.expected_score(good) * 100, ))
    print('probability of good beating better: %5.2f%%' % (good.expected_score(better) * 100, ))

.. code-block::

    probability of better beating good: 64.01%
    probability of good beating better: 35.99%
    probability of better beating good: 58.42%
    probability of good beating better: 41.58%

Sample Arena
------------

.. code-block:: python

    from elote import LambdaArena
    import json
    import random


    # sample bout function which just compares the two inputs
    def func(a, b):
        if a == b:
            return None
        else:
            return a > b

    # Configure the EloCompetitor class with a moderate k_factor
    # Note: Using a more moderate k_factor (20) to prevent ratings from changing too drastically
    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    arena = LambdaArena(func)
    arena.set_competitor_class_var('_k_factor', 20)
    # Using 1200 as initial rating (standard chess starting rating) to prevent negative ratings
    arena.tournament(matchups, base_competitor_kwargs={"initial_rating": 1200})

    print("Arena results:")
    print(json.dumps(arena.leaderboard(), indent=4))

.. code-block::

    Arena results:
    [
        {
            "competitor": 1,
            "rating": 1013.2827659706424
        },
        {
            "competitor": 2,
            "rating": 1058.4698600862191
        },
        ...
        {
            "competitor": 9,
            "rating": 1328.7158907591229
        },
        {
            "competitor": 10,
            "rating": 1435.6511383857194
        }
    ]

DWZ Arena
---------

.. code-block:: python

    from elote import LambdaArena, DWZCompetitor
    import json
    import random


    # sample bout function which just compares the two inputs
    def func(a, b):
        if a == b:
            return None
        else:
            return a > b


    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    # Create arena with DWZCompetitor and set higher initial rating
    arena = LambdaArena(
        func, 
        base_competitor=DWZCompetitor,
        base_competitor_kwargs={"initial_rating": 1200}
    )
    arena.tournament(matchups)

    print("Arena results:")
    print(json.dumps(arena.leaderboard(), indent=4))

.. code-block::

    Arena results:
    [
        {
            "competitor": 1,
            "rating": 1012.72183429478434
        },
        {
            "competitor": 2,
            "rating": 1044.28657694745118
        },
        ...
        {
            "competitor": 9,
            "rating": 1358.7172315502228
        },
        {
            "competitor": 10,
            "rating": 1411.6297448123669
        }
    ]

ECF Arena
---------

.. code-block:: python

    from elote import LambdaArena, ECFCompetitor
    import json
    import random


    # sample bout function which just compares the two inputs
    def func(a, b):
        if a == b:
            return None
        else:
            return a > b


    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    # Create arena with ECFCompetitor
    # Note: ECFCompetitor now has a default initial rating of 100 (minimum rating)
    arena = LambdaArena(func, base_competitor=ECFCompetitor)
    arena.tournament(matchups)

    print("Arena results:")
    print(json.dumps(arena.leaderboard(), indent=4))

.. code-block::

    Arena results:
    [
        {
            "competitor": 1,
            "rating": 100.0
        },
        {
            "competitor": 2,
            "rating": 106.99453852992116
        },
        ...
        {
            "competitor": 9,
            "rating": 134.9562091605315
        },
        {
            "competitor": 10,
            "rating": 145.9090955086219
        }
    ]

Glicko Arena
------------

.. code-block:: python

    from elote import LambdaArena, GlickoCompetitor
    import json
    import random


    # sample bout function which just compares the two inputs
    def func(a, b):
        if a == b:
            return None
        else:
            return a > b

    matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

    # Create arena with GlickoCompetitor and set higher initial rating in constructor
    arena = LambdaArena(
        func, 
        base_competitor=GlickoCompetitor, 
        base_competitor_kwargs={"initial_rating": 1500}
    )
    arena.tournament(matchups)

    print("Arena results:")
    print(json.dumps(arena.leaderboard(), indent=4))

.. code-block::

    Arena results:
    [
        {
            "competitor": 1,
            "rating": 1126.83348029853865
        },
        {
            "competitor": 2,
            "rating": 1395.7055271953318
        },
        ...
        {
            "competitor": 9,
            "rating": 2421.232857002417
        },
        {
            "competitor": 10,
            "rating": 2984.275564273142
        }
    ]

Persisting State from an Arena
------------------------------

.. code-block:: python

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
    # Create arena with GlickoCompetitor and set higher initial rating
    arena = LambdaArena(
        func, 
        base_competitor=GlickoCompetitor,
        base_competitor_kwargs={"initial_rating": 1500}
    )
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

.. code-block::

    Arena results:
    [
        {
            "competitor": 2,
            "rating": 1455.4305146831251
        },
        {
            "competitor": 6,
            "rating": 1515.7289656415517
        },
        ...
        {
            "competitor": 4,
            "rating": 1673.3452669422734
        },
        {
            "competitor": 7,
            "rating": 1967.7914038679244
        }
    ]
    Arena results:
    [
        {
            "competitor": 2,
            "rating": 1304.09061767108824
        },
        {
            "competitor": 1,
            "rating": 1339.4265505050406
        },
        ...
        {
            "competitor": 9,
            "rating": 2253.7482792356805
        },
        {
            "competitor": 10,
            "rating": 2403.5118562872744
        }
    ]