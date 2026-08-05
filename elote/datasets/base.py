"""
Base dataset interface for elote.

This module provides a common interface for getting datasets and splitting them into train and test sets
for evaluating different rating algorithms.
"""

import abc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Sequence, Set, Iterator
import datetime
import pandas as pd
import numpy as np
import gc


@dataclass
class DataSplit:
    """
    A container for train and test data splits.

    Attributes:
        train: List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        test: List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
    """

    train: Sequence[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]
    test: Sequence[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]

    def __len__(self) -> int:
        """Return the total number of matchups in both train and test sets."""
        return len(self.train) + len(self.test)

    def __str__(self) -> str:
        """Return a string representation of the data split."""
        return f"DataSplit(train={len(self.train)}, test={len(self.test)})"

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the train and test sets to pandas DataFrames.

        Returns:
            Tuple of (train_df, test_df)
        """
        columns = ["competitor_a", "competitor_b", "outcome", "timestamp", "attributes"]
        train_df = pd.DataFrame(self.train, columns=columns)
        test_df = pd.DataFrame(self.test, columns=columns)
        return train_df, test_df


class BaseDataset(abc.ABC):
    """
    Base abstract class for all dataset implementations.

    Datasets provide a common interface for getting data and splitting it into train and test sets
    for evaluating different rating algorithms.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_memory_mb: int = 1024):
        """
        Initialize a dataset.

        Args:
            cache_dir: Directory to cache downloaded data. If None, a temporary directory will be used.
            max_memory_mb: Maximum memory usage in MB for dataset operations.
        """
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self._data: Optional[List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]] = None
        self._memory_efficient = max_memory_mb < 2048  # Use memory-efficient mode for < 2GB

    @abc.abstractmethod
    def download(self) -> None:
        """
        Download the dataset from its source.

        This method should handle downloading the data from its source and storing it in the cache directory.
        It should be idempotent, i.e., if the data is already downloaded, it should not download it again.
        """
        pass

    @abc.abstractmethod
    def load(self) -> List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]:
        """
        Load the dataset into memory.

        Returns:
            List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
            where outcome is 1.0 if competitor_a won, 0.0 if competitor_b won, and 0.5 for a draw.
        """
        pass

    def get_data(self) -> List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]:
        """
        Get the dataset, downloading it if necessary.

        Returns:
            List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        """
        if self._data is None:
            self.download()
            self._data = self.load()
            
            # Force garbage collection after loading large datasets
            if self._memory_efficient:
                gc.collect()
                
        return self._data

    @staticmethod
    def _validate_test_ratio(test_ratio: float) -> None:
        """Reject a test ratio outside the documented 0.0 to 1.0 range."""
        if not 0.0 <= test_ratio <= 1.0:
            raise ValueError(f"test_ratio must be between 0.0 and 1.0, got {test_ratio!r}")

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        """Reject a non-positive batch size."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def get_data_iterator(self, batch_size: int = 1000) -> Iterator[List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]]:
        """
        Get an iterator over the dataset in batches for memory-efficient processing.

        Args:
            batch_size: Number of matchups per batch.

        Yields:
            Batches of matchup tuples.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        self._validate_batch_size(batch_size)
        return self._iter_batches(batch_size)

    def _iter_batches(self, batch_size: int) -> Iterator[List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]]]:
        data = self.get_data()
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def clear_cache(self) -> None:
        """Clear the in-memory cache to free up memory."""
        self._data = None
        gc.collect()

    def get_memory_usage_mb(self) -> float:
        """
        Estimate the memory usage of the loaded dataset in MB.
        
        Returns:
            Estimated memory usage in MB.
        """
        if self._data is None:
            return 0.0
        
        # Rough estimate: each tuple takes ~200 bytes on average
        return len(self._data) * 200 / (1024 * 1024)

    def time_split(self, test_ratio: float = 0.2) -> DataSplit:
        """
        Split the dataset into train and test sets based on time.

        Args:
            test_ratio: Ratio of data to use for testing (0.0 to 1.0)

        Returns:
            DataSplit object containing train and test sets

        Raises:
            ValueError: If test_ratio is outside the 0.0 to 1.0 range.
        """
        self._validate_test_ratio(test_ratio)

        data = self.get_data()

        # Sort by timestamp if available
        data_with_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in data if ts is not None]
        data_without_time = [(a, b, outcome, ts, attrs) for a, b, outcome, ts, attrs in data if ts is None]

        if not data_with_time:
            # If no timestamps, fall back to random split
            return self.random_split(test_ratio)

        # Sort by timestamp
        data_with_time.sort(key=lambda x: x[3])

        # Split based on time
        split_idx = int(len(data_with_time) * (1 - test_ratio))
        train_data = data_with_time[:split_idx] + data_without_time
        test_data = data_with_time[split_idx:]

        return DataSplit(train=train_data, test=test_data)

    def random_split(self, test_ratio: float = 0.2, seed: Optional[int] = None) -> DataSplit:
        """
        Split the dataset into train and test sets randomly.

        Args:
            test_ratio: Ratio of data to use for testing (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            DataSplit object containing train and test sets

        Raises:
            ValueError: If test_ratio is outside the 0.0 to 1.0 range.
        """
        self._validate_test_ratio(test_ratio)

        data = self.get_data()

        rng = np.random.RandomState(seed)

        # Shuffle the data
        indices = rng.permutation(len(data))
        split_idx = int(len(data) * (1 - test_ratio))

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]

        return DataSplit(train=train_data, test=test_data)

    def competitor_split(self, test_ratio: float = 0.2, seed: Optional[int] = None) -> DataSplit:
        """
        Split the dataset into train and test sets based on competitors.

        This ensures that some competitors are only in the test set, which is useful for
        evaluating how well the rating system generalizes to new competitors.

        Args:
            test_ratio: Ratio of competitors to reserve for testing (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            DataSplit object containing train and test sets

        Raises:
            ValueError: If test_ratio is outside the 0.0 to 1.0 range.
        """
        self._validate_test_ratio(test_ratio)

        data = self.get_data()

        rng = np.random.RandomState(seed)

        # Get all unique competitors, in first-appearance order so the shuffle below
        # depends only on the seed and not on set iteration order.
        seen: Dict[Any, None] = {}
        for a, b, _, _, _ in data:
            seen[a] = None
            seen[b] = None

        all_competitors_list: List[Any] = list(seen)
        rng.shuffle(all_competitors_list)

        # Split competitors
        split_idx = int(len(all_competitors_list) * (1 - test_ratio))
        train_competitors: Set[Any] = set(all_competitors_list[:split_idx])
        test_competitors: Set[Any] = set(all_competitors_list[split_idx:])

        # Split data based on competitors
        train_data: List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]] = []
        test_data: List[Tuple[Any, Any, float, Optional[datetime.datetime], Optional[Dict[str, Any]]]] = []

        for matchup in data:
            a, b = matchup[0], matchup[1]
            if a in train_competitors and b in train_competitors:
                train_data.append(matchup)
            elif a in test_competitors or b in test_competitors:
                test_data.append(matchup)

        return DataSplit(train=train_data, test=test_data)
