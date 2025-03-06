"""
Datasets module for elote.

This module provides a common interface for getting datasets and splitting them into train and test sets
for evaluating different rating algorithms.
"""

from elote.datasets.base import BaseDataset, DataSplit
from elote.datasets.chess import ChessDataset
from elote.datasets.football import CollegeFootballDataset
from elote.datasets.synthetic import SyntheticDataset

__all__ = [
    "BaseDataset",
    "DataSplit",
    "ChessDataset",
    "CollegeFootballDataset",
    "SyntheticDataset",
]
