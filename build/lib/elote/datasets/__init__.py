"""
Datasets module for elote.

This module provides a common interface for getting datasets and splitting them into train and test sets
for evaluating different rating algorithms.
"""

# Core datasets - always available
from elote.datasets.base import BaseDataset, DataSplit
from elote.datasets.synthetic import SyntheticDataset
from typing import Any

# Base __all__ list with always-available exports
__all__ = [
    "BaseDataset",
    "DataSplit",
    "SyntheticDataset",
]

# Optional datasets - only import if dependencies are available
_optional_imports = {}

# Try to import ChessDataset
try:
    from elote.datasets.chess import ChessDataset
    _optional_imports["ChessDataset"] = ChessDataset
    __all__.append("ChessDataset")
except ImportError as e:
    _optional_imports["ChessDataset"] = e

# Try to import CollegeFootballDataset  
try:
    from elote.datasets.football import CollegeFootballDataset
    _optional_imports["CollegeFootballDataset"] = CollegeFootballDataset
    __all__.append("CollegeFootballDataset")
except ImportError as e:
    _optional_imports["CollegeFootballDataset"] = e


def __getattr__(name: str) -> Any:
    """Handle access to optional imports with helpful error messages."""
    if name in _optional_imports:
        obj = _optional_imports[name]
        if isinstance(obj, ImportError):
            if name == "ChessDataset":
                raise ImportError(
                    f"ChessDataset requires optional dependencies that are not installed.\n"
                    f"Install them with: pip install 'elote[datasets]' or pip install python-chess pyzstd\n"
                    f"Original error: {obj}"
                )
            elif name == "CollegeFootballDataset":
                raise ImportError(
                    f"CollegeFootballDataset requires optional dependencies that are not installed.\n"
                    f"Install them with: pip install 'elote[datasets]' or pip install 'sportsdataverse[all]'\n"
                    f"Original error: {obj}"
                )
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
