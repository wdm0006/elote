.. elote documentation master file, created by
   sphinx-quickstart on Sat Mar 21 13:38:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Elote: Elegant Rating Systems in Python
=======================================

**Elote** is a powerful Python library for implementing and comparing rating systems. Whether you're ranking chess players, sports teams, or prioritizing features in your product backlog, Elote provides a simple, elegant API for all your competitive ranking needs.

Rating systems allow you to rank competitors based on their performance in head-to-head matchups. The most famous example is the Elo rating system used in chess, but these systems have applications far beyond sports:

- Ranking products based on A/B comparisons
- Prioritizing features through pairwise voting
- Creating recommendation systems
- Matchmaking in games and competitions
- Collaborative filtering and ranking

Elote makes implementing these systems simple and intuitive, with a clean API that handles all the mathematical complexity for you.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started
   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   competitors
   arenas
   serialization
   
.. toctree::
   :maxdepth: 1
   :caption: Rating Systems

   rating_systems/elo
   rating_systems/glicko
   rating_systems/ecf
   rating_systems/dwz
   rating_systems/ensemble
   rating_systems/comparison

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
   advance_example
   use_cases/product_ranking
   use_cases/matchmaking
   use_cases/feature_prioritization

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/competitors
   api/arenas

.. toctree::
   :maxdepth: 1
   :caption: Resources

   blog_posts

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
