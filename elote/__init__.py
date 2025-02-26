"""Elote: Elegant Rating Systems in Python.

Elote is a Python library for implementing and comparing rating systems. Whether you're
ranking chess players, sports teams, or prioritizing features in your product backlog,
Elote provides a simple, elegant API for all your competitive ranking needs.

Rating systems allow you to rank competitors based on their performance in head-to-head
matchups. The most famous example is the Elo rating system used in chess, but these
systems have applications far beyond sports:

- Ranking products based on A/B comparisons
- Prioritizing features through pairwise voting
- Creating recommendation systems
- Matchmaking in games and competitions
- Collaborative filtering and ranking

Elote makes implementing these systems simple and intuitive, with a clean API that
handles all the mathematical complexity for you.

Available rating systems:
- Elo: The classic chess rating system
- Glicko: An improvement on Elo that accounts for rating reliability
- ECF: The English Chess Federation rating system
- DWZ: The Deutsche Wertungszahl (German evaluation number) system
- Ensemble: A meta-rating system that combines multiple rating systems
"""

from elote.competitors.elo import EloCompetitor
from elote.competitors.glicko import GlickoCompetitor
from elote.competitors.ecf import ECFCompetitor
from elote.competitors.dwz import DWZCompetitor
from elote.competitors.ensemble import BlendedCompetitor

from elote.arenas.lambda_arena import LambdaArena


__all__ = [
    "EloCompetitor",
    "ECFCompetitor",
    "DWZCompetitor",
    "GlickoCompetitor",
    "LambdaArena",
    "BlendedCompetitor",
]
