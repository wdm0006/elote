Getting Started
===============

To install latest release:

.. code-block::

    pip install elote

To install bleeding edge, clone the repository and run:

.. code-block::

    pip install -e .


Basic Usage
-----------

The most basic object in ``elote`` is a competitor. To start with, let's take a look at ``EloCompetitor``. Let's make 3
objects, one for each of 3 players in a game:

.. code-block:: python

    from elote import EloCompetitor

    good_player = EloCompetitor(initial_rating=1200)
    better_player = EloCompetitor(initial_rating=1200)
    best_player = EloCompetitor(initial_rating=1200)

    print('Starting ratings:')
    print('%7.2f, %7.2f, %7.2f' % (good_player.rating, better_player.rating, best_player.rating, ))

All we do is initialize them, and print out their starting ratings. Rating is our measure of how good we think a
competitor is with the information at hand. Here we don't really have any information, so they are all rated the same:

.. code-block::

    Starting ratings:
    1200.00,  1200.00,  1200.00

To make things a little more interesting, let's do 20 ``matches``. A ``match`` is an instance where two players compete,
and one of them wins. This gives us some new information to update our ratings with. For each of the matches we simulate
we will have ``better_player`` beat ``good_player`` or ``best_player`` beat ``better_player``. At each iteration, we will
print out the ratings to get an idea of how they change over time.


.. code-block:: python

    print('\nAfter matches')
    for _ in range(10):
        better_player.beat(good_player)
        best_player.beat(better_player)
        print('%7.2f, %7.2f, %7.2f' % (good_player.rating, better_player.rating, best_player.rating, ))

.. code-block::

    After matches
    good,    better,  best
    1184.00,  1199.26,  1216.74
    1168.70,  1198.66,  1232.64
    1154.08,  1198.18,  1247.75
    1140.10,  1197.79,  1262.11
    1126.73,  1197.49,  1275.78
    1113.95,  1197.25,  1288.80
    1101.71,  1197.08,  1301.21
    1089.99,  1196.95,  1313.05
    1078.77,  1196.87,  1324.36
    1068.01,  1196.81,  1335.18

So as you can see, over time, the scores gradually update to reflect our hierarchy.

For more infromation on types of competitors we have, or different configuraiton options, please see the detailed API
docs on the competitors page.