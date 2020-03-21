.. elote documentation master file, created by
   sphinx-quickstart on Sat Mar 21 13:38:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Elote
=====

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   competitors
   arenas



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
