Getting Started
===============

Installation: TODO


Examples
--------

The most basic object in ``elote`` is a competitor. To start with, let's take a look at ``EloCompetitor``. Let's make 3
objects, one for each of 3 players in a game:

.. code-block:: python

    from elote import EloCompetitor

    good_player = EloCompetitor()
    better_player = EloCompetitor()
    best_player = EloCompetitor()

    print('Starting ratings:')
    print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

All we do is initialize them, and print out their starting ratings. Rating is our measure of how good we think a
competitor is with the information at hand. Here we don't really have any information, so they are all rated the same:

.. code-block::

    Starting ratings:
    400.00,  400.00,  400.00

To make things a little more interesting, let's do 20 ``matches``. A ``match`` is an instance where two players compete,
and one of them wins. This gives us some new information to update our ratings with. For each of the matches we simulate
we will have ``better_player`` beat ``good_player`` or ``best_player`` beat ``better_player``. At each iteration, we will
print out the ratings to get an idea of how they change over time.


.. code-block:: python

    print('\nAfter matches')
    for _ in range(10):
        better.beat(good)
        best.beat(better)
        print('%7.2f, %7.2f, %7.2f' % (good.rating, better.rating, best.rating, ))

.. code-block::

    After matches
    good,    better,  best
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

So as you can see, over time, the scores gradually update to reflect our hierarchy.

For more infromation on types of competitors we have, or different configuraiton options, please see the detailed API
docs on the competitors page.