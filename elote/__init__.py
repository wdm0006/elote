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
- Glicko-2: A further improvement on Glicko that adds volatility tracking
- TrueSkill: Microsoft's Bayesian skill rating system for multiplayer games
- ECF: The English Chess Federation rating system
- DWZ: The Deutsche Wertungszahl (German evaluation number) system
- Colley Matrix: A least-squares rating system that solves a system of linear equations
- Ensemble: A meta-rating system that combines multiple rating systems

Datasets:
- Chess: Chess games from the Lichess database
- College Football: College football games
- Synthetic: Synthetic data generator for testing

Analysis and Benchmarking:
- Benchmarking: Tools for comparing different rating systems
- Visualization: Tools for visualizing rating system performance
"""

from elote.competitors.elo import EloCompetitor
from elote.competitors.glicko import GlickoCompetitor
from elote.competitors.glicko2 import Glicko2Competitor
from elote.competitors.trueskill import TrueSkillCompetitor
from elote.competitors.ecf import ECFCompetitor
from elote.competitors.dwz import DWZCompetitor
from elote.competitors.colley import ColleyMatrixCompetitor
from elote.competitors.ensemble import BlendedCompetitor

from elote.arenas.lambda_arena import LambdaArena

from elote.datasets.base import BaseDataset, DataSplit
from elote.datasets.chess import ChessDataset
from elote.datasets.football import CollegeFootballDataset
from elote.datasets.synthetic import SyntheticDataset
from elote.datasets.utils import train_arena_with_dataset, evaluate_arena_with_dataset, train_and_evaluate_arena

# Import new benchmark and visualization modules
from elote.benchmark import evaluate_competitor, benchmark_competitors
from elote.visualization import (
    plot_rating_system_comparison,
    plot_optimized_accuracy_comparison,
    plot_accuracy_by_prior_bouts,
)

__all__ = [
    # Competitors
    "EloCompetitor",
    "GlickoCompetitor",
    "Glicko2Competitor",
    "TrueSkillCompetitor",
    "ECFCompetitor",
    "DWZCompetitor",
    "ColleyMatrixCompetitor",
    "BlendedCompetitor",
    # Arenas
    "LambdaArena",
    # Datasets
    "BaseDataset",
    "DataSplit",
    "ChessDataset",
    "CollegeFootballDataset",
    "SyntheticDataset",
    # Dataset utilities
    "train_arena_with_dataset",
    "evaluate_arena_with_dataset",
    "train_and_evaluate_arena",
    # Benchmarking
    "evaluate_competitor",
    "benchmark_competitors",
    # Visualization
    "plot_rating_system_comparison",
    "plot_optimized_accuracy_comparison",
    "plot_accuracy_by_prior_bouts",
]
