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

Datasets (require optional dependencies):
- Chess: Chess games from the Lichess database (requires: python-chess, pyzstd)
- College Football: College football games (requires: sportsdataverse)
- Synthetic: Synthetic data generator for testing (always available)

Analysis and Benchmarking:
- Benchmarking: Tools for comparing different rating systems
- Visualization: Tools for visualizing rating system performance
"""

# Core competitors - always available
from elote.competitors.elo import EloCompetitor
from elote.competitors.glicko import GlickoCompetitor
from elote.competitors.glicko2 import Glicko2Competitor
from elote.competitors.trueskill import TrueSkillCompetitor
from elote.competitors.ecf import ECFCompetitor
from elote.competitors.dwz import DWZCompetitor
from elote.competitors.colley import ColleyMatrixCompetitor
from elote.competitors.ensemble import BlendedCompetitor

# Core arenas - always available
from elote.arenas.lambda_arena import LambdaArena

# Core datasets - always available
from elote.datasets.base import BaseDataset, DataSplit
from elote.datasets.synthetic import SyntheticDataset
from elote.datasets.utils import train_arena_with_dataset, evaluate_arena_with_dataset, train_and_evaluate_arena

# Core analysis modules - always available
from elote.benchmark import evaluate_competitor, benchmark_competitors
from elote.visualization import (
    plot_rating_system_comparison,
    plot_optimized_accuracy_comparison,
    plot_accuracy_by_prior_bouts,
)

# Make logging easily accessible
from .logging import logger, set_level, add_handler, basic_config

# Type imports
from typing import Any, List, Dict

# Base __all__ list with always-available exports
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
    # Core Datasets
    "BaseDataset",
    "DataSplit",
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
    # Logging
    "logger",
    "set_level",
    "add_handler",
    "basic_config",
]

# Optional datasets - only import if dependencies are available
_optional_imports = {}

# Try to import ChessDataset
try:
    from elote.datasets.chess import ChessDataset
    _optional_imports["ChessDataset"] = ChessDataset
    __all__.append("ChessDataset")
except ImportError as e:
    # Store the import error for later reference
    _optional_imports["ChessDataset"] = e

# Try to import CollegeFootballDataset
try:
    from elote.datasets.football import CollegeFootballDataset
    _optional_imports["CollegeFootballDataset"] = CollegeFootballDataset
    __all__.append("CollegeFootballDataset")
except ImportError as e:
    # Store the import error for later reference
    _optional_imports["CollegeFootballDataset"] = e


def _get_missing_dependency_error(dataset_name: str, import_error: ImportError) -> str:
    """Generate a helpful error message for missing optional dependencies."""
    if dataset_name == "ChessDataset":
        return (
            f"ChessDataset requires optional dependencies that are not installed.\n"
            f"Install them with: pip install 'elote[datasets]' or pip install python-chess pyzstd\n"
            f"Original error: {import_error}"
        )
    elif dataset_name == "CollegeFootballDataset":
        return (
            f"CollegeFootballDataset requires optional dependencies that are not installed.\n"
            f"Install them with: pip install 'elote[datasets]' or pip install 'sportsdataverse[all]'\n"
            f"Original error: {import_error}"
        )
    else:
        return f"Optional component {dataset_name} is not available: {import_error}"


def __getattr__(name: str) -> Any:
    """Handle access to optional imports with helpful error messages."""
    if name in _optional_imports:
        obj = _optional_imports[name]
        if isinstance(obj, ImportError):
            raise ImportError(_get_missing_dependency_error(name, obj))
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Add a convenience function to check what's available
def list_available_datasets() -> List[str]:
    """List all available dataset classes."""
    available = ["SyntheticDataset"]  # Always available
    
    for dataset_name in ["ChessDataset", "CollegeFootballDataset"]:
        if dataset_name in _optional_imports and not isinstance(_optional_imports[dataset_name], ImportError):
            available.append(dataset_name)
    
    return available


def list_missing_datasets() -> List[Dict[str, str]]:
    """List dataset classes that are not available due to missing dependencies."""
    missing = []
    
    for dataset_name in ["ChessDataset", "CollegeFootballDataset"]:
        if dataset_name in _optional_imports and isinstance(_optional_imports[dataset_name], ImportError):
            missing.append({
                "name": dataset_name,
                "error": _get_missing_dependency_error(dataset_name, _optional_imports[dataset_name])
            })
    
    return missing


# Add these utility functions to __all__
__all__.extend(["list_available_datasets", "list_missing_datasets"])
