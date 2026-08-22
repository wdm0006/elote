"""
Tests for the datasets module.
"""

import unittest
import datetime
import os
import random
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import shutil
import pytest

from elote import (
    DataSplit,
    SyntheticDataset,
    LambdaArena,
    EloCompetitor,
    train_arena_with_dataset,
    evaluate_arena_with_dataset,
    train_and_evaluate_arena,
    list_available_datasets,
    WholeHistoryRatingCompetitor,
)
from elote.arenas.base import Bout, History

# Conditionally import optional datasets
try:
    from elote import ChessDataset
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False
    ChessDataset = None

try:
    from elote import CollegeFootballDataset
    # Test if sportsdataverse.cfb is actually available and working
    # Note: sportsdataverse may fail to import due to xgboost model compatibility issues
    import sportsdataverse.cfb  # noqa: F401
    HAS_FOOTBALL = True
except Exception:
    # Catch any exception (including xgboost.core.XGBoostError for deprecated model formats)
    # Test if sportsdataverse is actually available
    HAS_FOOTBALL = False
    CollegeFootballDataset = None

# The adapter class itself imports fine without sportsdataverse (that extra is only needed to
# download), so the identifier/cache tests below run unconditionally against a local fixture.
from elote.datasets.football import CACHE_SCHEMA_VERSION, CollegeFootballDataset as FootballDataset


class TestDataSplit(unittest.TestCase):
    """Tests for the DataSplit class."""

    def test_data_split_initialization(self):
        """Test that the DataSplit class initializes correctly."""
        train_data = [
            ("A", "B", 1.0, datetime.datetime.now(), {"attr": "value"}),
            ("C", "D", 0.0, datetime.datetime.now(), {"attr": "value"}),
        ]
        test_data = [
            ("E", "F", 1.0, datetime.datetime.now(), {"attr": "value"}),
            ("G", "H", 0.5, datetime.datetime.now(), {"attr": "value"}),
        ]

        data_split = DataSplit(train=train_data, test=test_data)

        self.assertEqual(len(data_split.train), 2)
        self.assertEqual(len(data_split.test), 2)
        self.assertEqual(len(data_split), 4)

        # Test string representation
        self.assertEqual(str(data_split), "DataSplit(train=2, test=2)")

        # Test to_dataframe method
        train_df, test_df = data_split.to_dataframe()
        self.assertEqual(train_df.shape, (2, 5))
        self.assertEqual(test_df.shape, (2, 5))
        self.assertEqual(list(train_df.columns), ["competitor_a", "competitor_b", "outcome", "timestamp", "attributes"])


class TestSyntheticDataset(unittest.TestCase):
    """Tests for the SyntheticDataset class."""

    def test_synthetic_dataset_initialization(self):
        """Test that the SyntheticDataset class initializes correctly."""
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=20,
            skill_distribution="normal",
            skill_mean=1500,
            skill_std=300,
            noise_std=100,
            draw_probability=0.1,
            time_span_days=365,
            seed=42,
        )

        self.assertEqual(dataset.num_competitors, 10)
        self.assertEqual(dataset.num_matchups, 20)
        self.assertEqual(dataset.skill_distribution, "normal")
        self.assertEqual(dataset.skill_mean, 1500)
        self.assertEqual(dataset.skill_std, 300)
        self.assertEqual(dataset.noise_std, 100)
        self.assertEqual(dataset.draw_probability, 0.1)
        self.assertEqual(dataset.time_span_days, 365)
        self.assertEqual(dataset.seed, 42)

    def test_synthetic_dataset_generation(self):
        """Test that the SyntheticDataset generates data correctly."""
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=20,
            seed=42,
        )

        data = dataset.get_data()

        self.assertEqual(len(data), 20)

        # Check that all matchups have the expected format
        for a, b, outcome, timestamp, attributes in data:
            self.assertTrue(a.startswith("competitor_"))
            self.assertTrue(b.startswith("competitor_"))
            self.assertIn(outcome, [0.0, 0.5, 1.0])
            self.assertIsInstance(timestamp, datetime.datetime)
            self.assertIn("true_skill_a", attributes)
            self.assertIn("true_skill_b", attributes)
            self.assertIn("skill_diff", attributes)

    def test_synthetic_dataset_splits(self):
        """Test that the SyntheticDataset splits data correctly."""
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=100,
            seed=42,
        )

        # Test time split
        time_split = dataset.time_split(test_ratio=0.2)
        self.assertEqual(len(time_split.train), 80)
        self.assertEqual(len(time_split.test), 20)

        # Test random split
        random_split = dataset.random_split(test_ratio=0.3, seed=42)
        self.assertEqual(len(random_split.train), 70)
        self.assertEqual(len(random_split.test), 30)

        # Test competitor split
        competitor_split = dataset.competitor_split(test_ratio=0.4, seed=42)
        self.assertGreater(len(competitor_split.train), 0)
        self.assertGreater(len(competitor_split.test), 0)

    def test_memory_management_features(self):
        """Test memory management features in datasets."""
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=100,
            seed=42,
            max_memory_mb=512  # Set low memory limit
        )

        # Test memory usage estimation
        initial_memory = dataset.get_memory_usage_mb()
        self.assertEqual(initial_memory, 0.0)  # No data loaded yet

        # Load data and check memory usage
        dataset.get_data()
        loaded_memory = dataset.get_memory_usage_mb()
        self.assertGreater(loaded_memory, 0.0)

        # Test data iterator
        batch_count = 0
        total_items = 0
        for batch in dataset.get_data_iterator(batch_size=25):
            batch_count += 1
            total_items += len(batch)
            self.assertLessEqual(len(batch), 25)

        self.assertEqual(total_items, 100)
        self.assertEqual(batch_count, 4)

        # Test cache clearing
        dataset.clear_cache()
        cleared_memory = dataset.get_memory_usage_mb()
        self.assertEqual(cleared_memory, 0.0)

    def test_memory_efficient_mode(self):
        """Test memory-efficient mode activation."""
        # Test with low memory limit (should activate memory-efficient mode)
        dataset_low = SyntheticDataset(num_competitors=5, num_matchups=10, max_memory_mb=1024)
        self.assertTrue(dataset_low._memory_efficient)

        # Test with high memory limit (should not activate memory-efficient mode)
        dataset_high = SyntheticDataset(num_competitors=5, num_matchups=10, max_memory_mb=4096)
        self.assertFalse(dataset_high._memory_efficient)


class TestDatasetArgumentValidation(unittest.TestCase):
    """Tests for argument validation on the BaseDataset split and iterator helpers."""

    def _dataset(self):
        return SyntheticDataset(num_competitors=12, num_matchups=100, seed=42)

    def test_splits_reject_ratios_outside_the_unit_interval(self):
        dataset = self._dataset()

        for split_name in ("time_split", "random_split", "competitor_split"):
            for test_ratio in (-1.0, -0.2, 1.2, 2.0):
                with self.subTest(split=split_name, test_ratio=test_ratio):
                    with self.assertRaisesRegex(ValueError, "test_ratio must be between 0.0 and 1.0"):
                        getattr(dataset, split_name)(test_ratio=test_ratio)

    def test_splits_accept_the_endpoint_ratios(self):
        dataset = self._dataset()
        total = len(dataset.get_data())
        self.assertEqual(total, 100)

        for split_name in ("time_split", "random_split", "competitor_split"):
            with self.subTest(split=split_name):
                no_test = getattr(dataset, split_name)(test_ratio=0.0)
                self.assertEqual(len(no_test.train), total)
                self.assertEqual(len(no_test.test), 0)

                no_train = getattr(dataset, split_name)(test_ratio=1.0)
                self.assertEqual(len(no_train.train), 0)
                self.assertEqual(len(no_train.test), total)

    def test_get_data_iterator_rejects_non_positive_batch_sizes(self):
        for batch_size in (0, -1):
            with self.subTest(batch_size=batch_size):
                dataset = self._dataset()
                with self.assertRaisesRegex(ValueError, "batch_size must be a positive integer"):
                    dataset.get_data_iterator(batch_size=batch_size)

                # The error is raised eagerly, before the dataset is loaded.
                self.assertIsNone(dataset._data)

    def test_get_data_iterator_still_batches_positive_sizes(self):
        dataset = self._dataset()

        batches = list(dataset.get_data_iterator(batch_size=30))

        self.assertEqual([len(batch) for batch in batches], [30, 30, 30, 10])


def _fingerprint(rows):
    """Reduce generated rows to the part a seed is supposed to determine.

    Absolute timestamps are anchored to ``datetime.now()`` at generation time, so only
    the spacing between rows is reproducible; everything else must match exactly.
    """
    base = rows[0][3]
    return [(a, b, outcome, (ts - base).total_seconds(), attrs) for a, b, outcome, ts, attrs in rows]


class TestDatasetReproducibility(unittest.TestCase):
    """A seed must fully determine the data, and must not reach outside the instance."""

    def _dataset(self, seed):
        return SyntheticDataset(num_competitors=10, num_matchups=50, seed=seed)

    def test_seeded_generation_ignores_other_datasets(self):
        """Constructing and loading another dataset in between must not change the data."""
        first = self._dataset(42)
        other = self._dataset(7)
        other.get_data()
        first_rows = first.get_data()

        independent = self._dataset(42)

        self.assertEqual(_fingerprint(first_rows), _fingerprint(independent.get_data()))

    def test_seeded_generation_ignores_the_global_random_stream(self):
        """Unrelated draws between construction and load must not change the data."""
        dataset = self._dataset(42)
        random.random()
        np.random.random()
        rows = dataset.get_data()

        self.assertEqual(_fingerprint(rows), _fingerprint(self._dataset(42).get_data()))

    def test_clear_cache_regenerates_identical_rows(self):
        """One seeded object must reproduce itself across a cache clear."""
        dataset = self._dataset(42)
        first = _fingerprint(dataset.get_data())

        dataset.clear_cache()
        second = _fingerprint(dataset.get_data())

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_data(self):
        """Guards the reproducibility tests above against a constant generator."""
        self.assertNotEqual(
            _fingerprint(self._dataset(42).get_data()),
            _fingerprint(self._dataset(7).get_data()),
        )

    def test_generation_preserves_the_callers_global_streams(self):
        random.seed(1234)
        np.random.seed(1234)
        expected_python = [random.random() for _ in range(5)]
        expected_numpy = np.random.random(5).tolist()

        random.seed(1234)
        np.random.seed(1234)
        self._dataset(42).get_data()

        self.assertEqual([random.random() for _ in range(5)], expected_python)
        self.assertEqual(np.random.random(5).tolist(), expected_numpy)

    def test_splits_preserve_the_callers_global_streams(self):
        random.seed(1234)
        np.random.seed(1234)
        expected_python = [random.random() for _ in range(5)]
        expected_numpy = np.random.random(5).tolist()

        random.seed(1234)
        np.random.seed(1234)
        dataset = self._dataset(42)
        dataset.random_split(test_ratio=0.3, seed=1)
        dataset.competitor_split(test_ratio=0.3, seed=1)

        self.assertEqual([random.random() for _ in range(5)], expected_python)
        self.assertEqual(np.random.random(5).tolist(), expected_numpy)

    def test_random_split_is_reproducible(self):
        dataset = self._dataset(42)

        first = dataset.random_split(test_ratio=0.3, seed=1)
        second = dataset.random_split(test_ratio=0.3, seed=1)

        self.assertEqual(list(first.train), list(second.train))
        self.assertEqual(list(first.test), list(second.test))
        self.assertNotEqual(list(first.train), list(dataset.random_split(test_ratio=0.3, seed=2).train))

    def test_competitor_split_is_reproducible(self):
        dataset = self._dataset(42)

        first = dataset.competitor_split(test_ratio=0.3, seed=1)
        second = dataset.competitor_split(test_ratio=0.3, seed=1)

        self.assertEqual(list(first.train), list(second.train))
        self.assertEqual(list(first.test), list(second.test))

    def test_competitor_split_ordering_does_not_depend_on_set_iteration(self):
        """The shuffled competitor list must come from the data order, not a set."""
        data = [
            (3, 1, 1.0, None, None),
            (2, 0, 0.0, None, None),
            (1, 2, 0.5, None, None),
        ]
        dataset = self._dataset(42)
        dataset._data = data

        split = dataset.competitor_split(test_ratio=0.5, seed=1)

        # Small ints hash to themselves, so ``list({3, 1, 2, 0})`` is ``[0, 1, 2, 3]``
        # under every PYTHONHASHSEED -- reliably different from first-appearance order.
        rng = np.random.RandomState(1)
        order = [3, 1, 2, 0]
        rng.shuffle(order)
        train_competitors = set(order[:2])
        expected_train = [row for row in data if row[0] in train_competitors and row[1] in train_competitors]

        self.assertEqual(list(split.train), expected_train)


class TestAvailableDatasets(unittest.TestCase):
    """Tests for dataset availability functions."""

    def test_list_available_datasets(self):
        """Test that list_available_datasets works correctly."""
        available = list_available_datasets()
        self.assertIn("SyntheticDataset", available)
        self.assertIsInstance(available, list)
        for dataset_name in available:
            self.assertIsInstance(dataset_name, str)


@pytest.mark.skipif(not HAS_CHESS, reason="ChessDataset requires python-chess and pyzstd")
class TestChessDataset(unittest.TestCase):
    """Tests for the ChessDataset class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a sample PGN file for testing
        self.sample_pgn = os.path.join(self.temp_dir, "lichess_2013-01.pgn")
        with open(self.sample_pgn, "w") as f:
            f.write("""[Event "Rated Blitz game"]
[Site "https://lichess.org/abcdefgh"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[UTCDate "2013.01.01"]
[UTCTime "12:00:00"]
[WhiteElo "1500"]
[BlackElo "1400"]
[TimeControl "300+0"]
[ECO "C00"]
[Opening "French Defense"]
[Termination "Normal"]

1. e4 e6 2. d4 d5 3. exd5 exd5 4. Nf3 Nf6 5. Bd3 Bd6 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/12345678"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[UTCDate "2013.01.02"]
[UTCTime "13:00:00"]
[WhiteElo "1600"]
[BlackElo "1700"]
[TimeControl "300+0"]
[ECO "B01"]
[Opening "Scandinavian Defense"]
[Termination "Normal"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5 4. d4 Nf6 5. Nf3 Bg4 0-1

[Event "Rated Blitz game"]
[Site "https://lichess.org/87654321"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]
[UTCDate "2013.01.03"]
[UTCTime "14:00:00"]
[WhiteElo "1800"]
[BlackElo "1800"]
[TimeControl "300+0"]
[ECO "A00"]
[Opening "Uncommon Opening"]
[Termination "Normal"]

1. a3 a6 2. b3 b6 3. c3 c6 4. d3 d6 5. e3 e6 1/2-1/2
""")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.sample_pgn):
            os.remove(self.sample_pgn)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch("elote.datasets.chess.requests.get")
    @patch("elote.datasets.chess.pyzstd.decompress")
    def test_chess_dataset_initialization(self, mock_decompress, mock_get):
        """Test that the ChessDataset class initializes correctly."""
        # Mock the response and pyzstd
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"test data"]
        mock_get.return_value = mock_response

        mock_decompress.return_value = b"decompressed data"

        # Create a dataset with a specific cache directory
        dataset = ChessDataset(cache_dir=self.temp_dir, max_games=10, year=2013, month=1)

        self.assertEqual(dataset.max_games, 10)
        self.assertEqual(dataset.year, 2013)
        self.assertEqual(dataset.month, 1)
        self.assertEqual(dataset.cache_dir, self.temp_dir)
        self.assertEqual(
            dataset.data_url, "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
        )
        self.assertEqual(dataset.compressed_file, os.path.join(self.temp_dir, "lichess_2013-01.pgn.zst"))
        self.assertEqual(dataset.decompressed_file, os.path.join(self.temp_dir, "lichess_2013-01.pgn"))

    def test_parse_pgn_game(self):
        """Test that _parse_pgn_game correctly parses a PGN game."""
        import chess.pgn

        # Create a dataset
        dataset = ChessDataset(cache_dir=self.temp_dir)

        # Open the sample PGN file
        with open(self.sample_pgn, "r") as f:
            # Read the first game
            game = chess.pgn.read_game(f)

            # Parse the game
            matchup = dataset._parse_pgn_game(game)

            # Check the parsed data
            self.assertEqual(matchup[0], "Player1")  # White player
            self.assertEqual(matchup[1], "Player2")  # Black player
            self.assertEqual(matchup[2], 1.0)  # White wins
            self.assertEqual(matchup[3].year, 2013)
            self.assertEqual(matchup[3].month, 1)
            self.assertEqual(matchup[3].day, 1)
            self.assertEqual(matchup[4]["white_rating"], 1500)
            self.assertEqual(matchup[4]["black_rating"], 1400)
            self.assertEqual(matchup[4]["eco"], "C00")
            self.assertEqual(matchup[4]["opening"], "French Defense")

            # Read the second game
            game = chess.pgn.read_game(f)

            # Parse the game
            matchup = dataset._parse_pgn_game(game)

            # Check the parsed data
            self.assertEqual(matchup[0], "Player3")  # White player
            self.assertEqual(matchup[1], "Player4")  # Black player
            self.assertEqual(matchup[2], 0.0)  # Black wins

            # Read the third game
            game = chess.pgn.read_game(f)

            # Parse the game
            matchup = dataset._parse_pgn_game(game)

            # Check the parsed data
            self.assertEqual(matchup[0], "Player5")  # White player
            self.assertEqual(matchup[1], "Player6")  # Black player
            self.assertEqual(matchup[2], 0.5)  # Draw

    @patch("elote.datasets.chess.ChessDataset.download")
    def test_load_from_pgn(self, mock_download):
        """Test that load correctly loads games from a PGN file."""

        # Mock the download method to set the decompressed file to our sample PGN
        def mock_download_impl():
            # No self parameter needed here
            return None  # Just return None, we're setting decompressed_file directly below

        mock_download.side_effect = mock_download_impl

        # Create a dataset
        dataset = ChessDataset(cache_dir=self.temp_dir)
        dataset.decompressed_file = self.sample_pgn

        # Load the data
        matchups = dataset.load()

        # Check that we loaded the expected number of games
        self.assertEqual(len(matchups), 3)

        # Check the first matchup
        self.assertEqual(matchups[0][0], "Player1")  # White player
        self.assertEqual(matchups[0][1], "Player2")  # Black player
        self.assertEqual(matchups[0][2], 1.0)  # White wins

        # Check the second matchup
        self.assertEqual(matchups[1][0], "Player3")  # White player
        self.assertEqual(matchups[1][1], "Player4")  # Black player
        self.assertEqual(matchups[1][2], 0.0)  # Black wins

        # Check the third matchup
        self.assertEqual(matchups[2][0], "Player5")  # White player
        self.assertEqual(matchups[2][1], "Player6")  # Black player
        self.assertEqual(matchups[2][2], 0.5)  # Draw


@pytest.mark.skipif(not HAS_FOOTBALL, reason="CollegeFootballDataset requires sportsdataverse")
class TestCollegeFootballDataset(unittest.TestCase):
    """Tests for the CollegeFootballDataset class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    @patch("sportsdataverse.cfb.espn_cfb_schedule")
    def test_college_football_dataset_initialization(self, mock_espn_cfb_schedule):
        # Setup mock
        mock_espn_cfb_schedule.return_value = pd.DataFrame()

        # Initialize dataset
        dataset = CollegeFootballDataset(cache_dir=self.temp_dir, start_year=2020, end_year=2021)

        # Check attributes
        self.assertEqual(dataset.start_year, 2020)
        self.assertEqual(dataset.end_year, 2021)
        self.assertEqual(dataset.cache_dir, self.temp_dir)

    @patch("sportsdataverse.cfb.espn_cfb_schedule")
    def test_college_football_dataset_download(self, mock_espn_cfb_schedule):
        # Setup mock data with the correct structure
        mock_data = pd.DataFrame(
            {
                "season": [2020, 2020, 2021],
                "home_name": ["Team A", "Team C", "Team A"],
                "away_name": ["Team B", "Team D", "Team C"],
                "home_score": [28, 35, 21],
                "away_score": [21, 28, 14],
                "date": ["2020-09-05", "2020-09-12", "2021-09-04"],
                "status_type_completed": [True, True, True],
                "start_date": ["2020-09-05", "2020-09-12", "2021-09-04"],
            }
        )

        # Configure mock to return different data for each year
        def side_effect(dates, season_type, groups, limit, return_as_pandas):
            # Return data for the specific year
            if dates == 2020:
                data = mock_data[mock_data["season"] == 2020].copy()
                return data
            elif dates == 2021:
                data = mock_data[mock_data["season"] == 2021].copy()
                return data
            return pd.DataFrame()  # Empty DataFrame for other years

        mock_espn_cfb_schedule.side_effect = side_effect

        # Initialize dataset
        dataset = CollegeFootballDataset(cache_dir=self.temp_dir, start_year=2020, end_year=2021)

        # Download data
        dataset.download()

        # Check that the data file was created
        self.assertTrue(os.path.exists(dataset.data_file))

        # Check that the data was saved correctly
        saved_data = pd.read_csv(dataset.data_file)
        # We should have data in the saved file
        self.assertGreater(len(saved_data), 0)

    @patch("sportsdataverse.cfb.espn_cfb_schedule")
    def test_college_football_dataset_load(self, mock_espn_cfb_schedule):
        # Setup mock data with the correct structure
        mock_data = pd.DataFrame(
            {
                "season": [2020, 2020, 2021],
                "home_name": ["Team A", "Team C", "Team A"],
                "away_name": ["Team B", "Team D", "Team C"],
                "home_score": [28, 35, 21],
                "away_score": [21, 28, 14],
                "date": ["2020-09-05", "2020-09-12", "2021-09-04"],
                "status_type_completed": [True, True, True],
                "start_date": ["2020-09-05", "2020-09-12", "2021-09-04"],
            }
        )

        # Configure mock to return different data for each year
        def side_effect(dates, season_type, groups, limit, return_as_pandas):
            # Return data for the specific year
            if dates == 2020:
                return mock_data[mock_data["season"] == 2020].copy()
            elif dates == 2021:
                return mock_data[mock_data["season"] == 2021].copy()
            return pd.DataFrame()  # Empty DataFrame for other years

        mock_espn_cfb_schedule.side_effect = side_effect

        # Initialize dataset
        dataset = CollegeFootballDataset(cache_dir=self.temp_dir, start_year=2020, end_year=2021)

        # Download and load data
        matchups = dataset.load()

        # Check that matchups were loaded
        self.assertGreater(len(matchups), 0)

        # Check the format of the matchups
        for matchup in matchups:
            self.assertEqual(len(matchup), 5)  # (team_a, team_b, outcome, timestamp, attributes)
            self.assertIsInstance(matchup[0], str)  # team_a
            self.assertIsInstance(matchup[1], str)  # team_b
            self.assertIsInstance(matchup[2], float)  # outcome
            self.assertIsInstance(matchup[3], datetime.datetime)  # timestamp
            self.assertIsInstance(matchup[4], dict)  # attributes


class TestCollegeFootballDatasetIdentifiers(unittest.TestCase):
    """Tests that the college football dataset names programs rather than mascots.

    These build the cached CSV from a local fixture frame, so they need neither
    sportsdataverse nor network access.
    """

    # Mirrors the ESPN feed's shape: home_name/away_name are the mascot, while
    # home_display_name/away_display_name name the program. The second row is two
    # different programs that share the mascot "Tigers".
    RAW_GAMES = pd.DataFrame(
        {
            "date": ["2021-09-04", "2021-09-11", "2021-09-18"],
            "start_date": ["2021-09-04", "2021-09-11", "2021-09-18"],
            "home_name": ["Tigers", "Tigers", "Wildcats"],
            "away_name": ["Wildcats", "Tigers", "Tigers"],
            "home_display_name": ["Auburn Tigers", "Clemson Tigers", "Kentucky Wildcats"],
            "away_display_name": ["Arizona Wildcats", "LSU Tigers", "Missouri Tigers"],
            "home_score": [30, 10, 21],
            "away_score": [10, 27, 21],
            "status_type_completed": [True, True, True],
        }
    )

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def _dataset_from(self, raw_games):
        """Build a dataset whose cache holds the prepared form of ``raw_games``."""
        dataset = FootballDataset(cache_dir=self.temp_dir, start_year=2021, end_year=2021)
        dataset._prepare_games(raw_games).to_csv(dataset.data_file, index=False)
        return dataset

    def test_programs_sharing_a_mascot_stay_distinct(self):
        """Two programs sharing a mascot must yield two distinct competitor identifiers."""
        matchups = self._dataset_from(self.RAW_GAMES).load()

        identifiers = {m[0] for m in matchups} | {m[1] for m in matchups}
        self.assertEqual(
            identifiers,
            {
                "Auburn Tigers",
                "Arizona Wildcats",
                "Clemson Tigers",
                "LSU Tigers",
                "Kentucky Wildcats",
                "Missouri Tigers",
            },
        )

    def test_emitted_rows_match_the_fixture(self):
        """Each row names its two programs and carries the outcome implied by the scores."""
        matchups = self._dataset_from(self.RAW_GAMES).load()

        self.assertEqual(
            [(a, b, outcome) for a, b, outcome, _timestamp, _attributes in matchups],
            [
                ("Auburn Tigers", "Arizona Wildcats", 1.0),
                ("Clemson Tigers", "LSU Tigers", 0.0),
                ("Kentucky Wildcats", "Missouri Tigers", 0.5),
            ],
        )

    def test_no_row_pairs_a_competitor_with_itself(self):
        """A competitor must never appear on both sides of the same row."""
        matchups = self._dataset_from(self.RAW_GAMES).load()

        for competitor_a, competitor_b, _outcome, _timestamp, _attributes in matchups:
            self.assertNotEqual(competitor_a, competitor_b)

    def test_falls_back_to_name_columns_when_display_names_missing(self):
        """A feed without the display-name columns still yields identifiers."""
        raw_games = self.RAW_GAMES.drop(columns=["home_display_name", "away_display_name"])
        matchups = self._dataset_from(raw_games).load()

        identifiers = {m[0] for m in matchups} | {m[1] for m in matchups}
        self.assertEqual(identifiers, {"Tigers", "Wildcats"})

    def test_cache_path_carries_the_schema_version(self):
        """The cached CSV path is versioned, so a stale cache cannot suppress the mapping."""
        dataset = FootballDataset(cache_dir=self.temp_dir, start_year=2020, end_year=2021)

        self.assertEqual(
            os.path.basename(dataset.data_file),
            f"college_football_games_v{CACHE_SCHEMA_VERSION}_2020_2021.csv",
        )

        # A cache written by the pre-fix code lives at the unversioned path and is ignored.
        legacy_path = os.path.join(self.temp_dir, "college_football_games_2020_2021.csv")
        pd.DataFrame(
            {
                "start_date": ["2020-09-05"],
                "home_team": ["Tigers"],
                "away_team": ["Tigers"],
                "home_points": [28],
                "away_points": [21],
            }
        ).to_csv(legacy_path, index=False)

        self.assertNotEqual(dataset.data_file, legacy_path)
        self.assertFalse(os.path.exists(dataset.data_file))


class TestDatasetUtils(unittest.TestCase):
    """Tests for the dataset utility functions."""

    def test_train_arena_with_dataset(self):
        """Test that train_arena_with_dataset works correctly."""
        # Create a synthetic dataset
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=50,
            seed=42,
        )

        # Create an arena
        arena = LambdaArena(
            lambda a, b, attributes=None: True,  # Updated to accept attributes parameter
            base_competitor=EloCompetitor,
        )

        # Train the arena
        trained_arena = train_arena_with_dataset(arena, dataset.get_data())

        # Check that competitors were created
        self.assertGreater(len(trained_arena.competitors), 0)

        # Check that history was recorded
        self.assertGreater(len(trained_arena.history.bouts), 0)

    def test_train_arena_with_empty_dataset(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        progress_callback = MagicMock()

        trained_arena = train_arena_with_dataset(arena, [], progress_callback=progress_callback)

        self.assertIs(trained_arena, arena)
        progress_callback.assert_not_called()

    def test_evaluate_arena_with_dataset(self):
        """Test that evaluate_arena_with_dataset works correctly."""
        # Create a synthetic dataset
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=50,
            seed=42,
        )

        # Create and train an arena
        arena = LambdaArena(
            lambda a, b, attributes=None: True,  # Updated to accept attributes parameter
            base_competitor=EloCompetitor,
        )
        trained_arena = train_arena_with_dataset(arena, dataset.get_data())

        # Evaluate the arena
        history = evaluate_arena_with_dataset(trained_arena, dataset.get_data())

        # Check that history was recorded
        self.assertGreater(len(history.bouts), 0)

    def test_evaluate_arena_with_empty_dataset(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        progress_callback = MagicMock()

        with self.assertNoLogs("elote", level="WARNING"):
            history = evaluate_arena_with_dataset(arena, [], progress_callback=progress_callback)

        self.assertEqual(history.bouts, [])
        progress_callback.assert_not_called()

    def test_evaluate_arena_warns_once_for_skipped_unseen_competitors(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        train_arena_with_dataset(arena, [("A", "B", 1.0, None, None)])
        test_data = [
            ("A", "B", 1.0, None, None),
            ("A", "C", 1.0, None, None),
            ("D", "B", 0.0, None, None),
            ("C", "D", 0.5, None, None),
        ]

        with self.assertLogs("elote", level="WARNING") as captured:
            history = evaluate_arena_with_dataset(arena, test_data)

        self.assertEqual(len(history.bouts), 1)
        self.assertEqual(
            captured.output,
            ["WARNING:elote:Skipped 3/4 evaluation bouts: competitor not found in training history."],
        )

    def test_evaluate_arena_warns_when_nonempty_test_set_yields_no_bouts(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        train_arena_with_dataset(arena, [("A", "B", 1.0, None, None)])

        with self.assertLogs("elote", level="WARNING") as captured:
            history = evaluate_arena_with_dataset(arena, [("A", "C", 1.0, None, None)])

        self.assertEqual(history.bouts, [])
        self.assertEqual(
            captured.output,
            [
                "WARNING:elote:Skipped 1/1 evaluation bouts: competitor not found in training history.",
                "WARNING:elote:Evaluation history is empty after evaluating 1 test bouts; metrics are not meaningful.",
            ],
        )

    def test_evaluate_arena_does_not_warn_when_every_row_is_evaluable(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        rows = [("A", "B", 1.0, None, None)]
        train_arena_with_dataset(arena, rows)

        with self.assertNoLogs("elote", level="WARNING"):
            history = evaluate_arena_with_dataset(arena, rows)

        self.assertEqual(len(history.bouts), 1)

    def test_dataset_helpers_reject_non_positive_batch_sizes(self):
        arena = LambdaArena(lambda a, b, attributes=None: True)
        data = [("A", "B", 1.0, None, None)]

        for helper in (train_arena_with_dataset, evaluate_arena_with_dataset):
            for batch_size in (0, -1):
                for dataset in ([], data):
                    with self.subTest(helper=helper.__name__, batch_size=batch_size, empty=not dataset):
                        with self.assertRaisesRegex(ValueError, "batch_size must be a positive integer"):
                            helper(arena, dataset, batch_size=batch_size)

    def test_dataset_helpers_report_batched_progress(self):
        data = [
            ("A", "B", 1.0, None, None),
            ("A", "C", 1.0, None, None),
            ("B", "C", 1.0, None, None),
        ]
        arena = LambdaArena(lambda a, b, attributes=None: True)
        train_progress = MagicMock()
        eval_progress = MagicMock()

        train_arena_with_dataset(arena, data, batch_size=2, progress_callback=train_progress)
        history = evaluate_arena_with_dataset(arena, data, batch_size=2, progress_callback=eval_progress)

        self.assertEqual(train_progress.call_args_list, [unittest.mock.call(2, 3), unittest.mock.call(3, 3)])
        self.assertEqual(eval_progress.call_args_list, [unittest.mock.call(2, 3), unittest.mock.call(3, 3)])
        self.assertEqual(len(history.bouts), 3)

    def test_train_arena_records_a_bout_for_every_row_including_draws(self):
        """Every training row, drawn or decisive, must appear in the arena history."""
        dataset = SyntheticDataset(
            num_competitors=12,
            num_matchups=200,
            seed=7,
            draw_probability=0.9,
            noise_std=200.0,
        )
        rows = dataset.get_data()
        draws = [row for row in rows if row[2] == 0.5]
        self.assertEqual(len(draws), 62)

        arena = LambdaArena(lambda a, b, attributes=None: True, base_competitor=EloCompetitor)
        trained_arena = train_arena_with_dataset(arena, rows)

        self.assertEqual(len(trained_arena.history.bouts), len(rows))

    def test_train_arena_records_draw_bouts_with_outcome_and_attributes(self):
        """Drawn rows must record a draw outcome and carry their attributes through."""
        draw_attributes = {"event": "round-3", "board": 7}
        win_attributes = {"event": "round-4", "board": 2}
        data = [
            ("A", "B", 0.5, None, draw_attributes),
            ("A", "B", 1.0, None, win_attributes),
        ]

        arena = LambdaArena(lambda a, b, attributes=None: True, base_competitor=EloCompetitor)
        trained_arena = train_arena_with_dataset(arena, data)

        self.assertEqual(len(trained_arena.history.bouts), 2)
        draw_bout, win_bout = trained_arena.history.bouts
        self.assertEqual((draw_bout.a, draw_bout.b), ("A", "B"))
        self.assertEqual(draw_bout._normalized_outcome(), "draw")
        self.assertEqual(draw_bout.attributes, draw_attributes)
        self.assertEqual(win_bout._normalized_outcome(), "a")
        self.assertEqual(win_bout.attributes, win_attributes)

    def test_accuracy_by_prior_bouts_counts_drawn_training_bouts(self):
        """Prior-bout bins must be seeded from the complete training history."""
        dataset = SyntheticDataset(
            num_competitors=12,
            num_matchups=200,
            seed=7,
            draw_probability=0.9,
            noise_std=200.0,
        )
        split = dataset.time_split(test_ratio=0.3)
        self.assertEqual(len(split.train), 140)
        self.assertEqual(sum(1 for row in split.train if row[2] == 0.5), 43)

        arena = LambdaArena(lambda a, b, attributes=None: True, base_competitor=EloCompetitor)
        train_arena_with_dataset(arena, split.train)

        # competitor_0 and competitor_2 appear in 23 and 24 training rows respectively,
        # of which 20 and 17 are decisive -- so the pre-fix history binned this bout at 17.
        history = History()
        history.add_bout(Bout("competitor_0", "competitor_2", 0.5, 0.5))
        binned = history.accuracy_by_prior_bouts(arena, bin_size=1)["binned"]

        self.assertEqual(list(binned.keys()), [23])
        self.assertEqual(binned[23]["total"], 1)

    def test_train_arena_ratings_unchanged_by_draw_bout_recording(self):
        """Recording draw bouts must not change the ratings the draws produce.

        The expected values were captured before the draw branch was routed through
        ``LambdaArena.matchup``, when draws called ``competitor.tied()`` directly.
        """
        dataset = SyntheticDataset(
            num_competitors=12,
            num_matchups=200,
            seed=7,
            draw_probability=0.9,
            noise_std=200.0,
        )

        arena = LambdaArena(
            lambda a, b, attributes=None: True,
            base_competitor=EloCompetitor,
            base_competitor_kwargs={"initial_rating": 1500},
        )
        trained_arena = train_arena_with_dataset(arena, dataset.get_data())

        expected = [
            ("competitor_8", 1713.983015442147),
            ("competitor_0", 1693.4325854310873),
            ("competitor_3", 1587.1903055628625),
            ("competitor_9", 1533.9165724655888),
            ("competitor_5", 1527.8487497240067),
            ("competitor_2", 1512.8963021933225),
            ("competitor_6", 1505.589946609411),
            ("competitor_11", 1484.3664770734704),
            ("competitor_1", 1436.529021706998),
            ("competitor_10", 1385.3428838446823),
            ("competitor_4", 1376.9278018383588),
            ("competitor_7", 1241.976338108064),
        ]
        leaderboard = trained_arena.leaderboard()
        self.assertEqual(len(leaderboard), len(expected))
        for entry, (name, rating) in zip(leaderboard, expected, strict=True):
            with self.subTest(competitor=name):
                self.assertEqual(entry["competitor"], name)
                self.assertAlmostEqual(entry["rating"], rating, places=10)

    def test_train_and_evaluate_arena(self):
        """Test that train_and_evaluate_arena works correctly."""
        # Create a synthetic dataset
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=50,
            seed=42,
        )

        # Split the data
        data_split = dataset.time_split(test_ratio=0.3)

        # Create an arena
        arena = LambdaArena(
            lambda a, b, attributes=None: True,  # Updated to accept attributes parameter
            base_competitor=EloCompetitor,
        )

        # Train and evaluate the arena
        trained_arena, evaluation_history = train_and_evaluate_arena(arena, data_split)

        # Check that competitors were created
        self.assertGreater(len(trained_arena.competitors), 0)

        # Check that evaluation history was recorded
        self.assertGreater(len(evaluation_history.bouts), 0)

    def test_competitor_split_warns_for_every_unevaluable_test_row(self):
        dataset = SyntheticDataset(num_competitors=30, num_matchups=600, seed=7)
        data_split = dataset.competitor_split(test_ratio=0.2, seed=7)
        arena = LambdaArena(lambda a, b, attributes=None: True, base_competitor=EloCompetitor)

        with self.assertLogs("elote", level="WARNING") as captured:
            _, history = train_and_evaluate_arena(arena, data_split)

        skipped_message = (
            f"WARNING:elote:Skipped {len(data_split.test)}/{len(data_split.test)} evaluation bouts: "
            "competitor not found in training history."
        )
        self.assertEqual(history.bouts, [])
        self.assertIn(skipped_message, captured.output)



class TestCollegeFootballPartialDownload(unittest.TestCase):
    """A season that fails to fetch must not be silently cached.

    The cache file is keyed only by the requested year range, so writing a short result
    after a transient failure makes every later run read a dataset with a season missing
    and nothing to signal it.
    """

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.cache_dir, True)

    @staticmethod
    def _season_frame(year, rows=3):
        return pd.DataFrame(
            {
                "start_date": [f"{year}-09-0{i + 1}T17:00Z" for i in range(rows)],
                "home_display_name": [f"Home {year} {i}" for i in range(rows)],
                "away_display_name": [f"Away {year} {i}" for i in range(rows)],
                "home_score": [20 + i for i in range(rows)],
                "away_score": [10 + i for i in range(rows)],
                "neutral_site": [False] * rows,
                "status_type_completed": [True] * rows,
            }
        )

    def _run_download(self, schedule_side_effect):
        from elote.datasets.football import CollegeFootballDataset

        dataset = CollegeFootballDataset(cache_dir=self.cache_dir, start_year=2019, end_year=2021)
        fake_cfb = MagicMock()
        fake_cfb.espn_cfb_schedule.side_effect = schedule_side_effect
        fake_pkg = MagicMock()
        fake_pkg.cfb = fake_cfb
        # `import sportsdataverse.cfb as cfb` resolves the submodule from sys.modules.
        with patch.dict(
            "sys.modules",
            {"sportsdataverse": fake_pkg, "sportsdataverse.cfb": fake_cfb},
        ):
            dataset.download()
        return dataset

    def test_a_failed_season_raises_and_caches_nothing(self):
        def side_effect(dates, **_kwargs):
            if dates == 2020:
                raise ConnectionError("upstream timed out")
            return self._season_frame(dates)

        with self.assertRaises(RuntimeError) as ctx:
            self._run_download(side_effect)

        message = str(ctx.exception)
        self.assertIn("2020", message)
        self.assertIn("ConnectionError", message)
        self.assertEqual(os.listdir(self.cache_dir), [], "a partial download was cached")

    def test_a_complete_download_is_cached(self):
        dataset = self._run_download(lambda dates, **_kwargs: self._season_frame(dates))
        self.assertTrue(os.path.exists(dataset.data_file))
        cached = pd.read_csv(dataset.data_file)
        self.assertEqual(len(cached), 9)



class TestTrainingForwardsMatchTime(unittest.TestCase):
    """A dataset row's timestamp must reach the competitors, not just the sort.

    Whole-History Rating keeps one latent rating per playing day, Glicko and Glicko-2 inflate
    rating deviation for time since last match, and DWZ's development coefficient depends on
    age at the time of the match. None of them has any other source for the date, so dropping
    it makes every row in a dataset look simultaneous.
    """

    class _RecordingArena:
        def __init__(self):
            self.match_times = []

        def matchup(self, a, b, attributes=None, match_time=None, outcome=None, scores=None):
            self.match_times.append(match_time)

    def _rows(self):
        return [
            ("A", "B", 1.0, datetime.datetime(2020, 1, 6), None),
            ("C", "D", 0.0, datetime.datetime(2020, 2, 6), None),
            ("E", "F", 0.5, datetime.datetime(2020, 3, 6), None),
        ]

    def test_match_time_is_forwarded_for_every_outcome(self):
        arena = self._RecordingArena()
        train_arena_with_dataset(arena, self._rows())

        self.assertEqual(
            arena.match_times,
            [
                datetime.datetime(2020, 1, 6),
                datetime.datetime(2020, 2, 6),
                datetime.datetime(2020, 3, 6),
            ],
        )

    def test_rows_without_a_timestamp_still_train(self):
        arena = self._RecordingArena()
        train_arena_with_dataset(arena, [("A", "B", 1.0, None, None)])
        self.assertEqual(arena.match_times, [None])

    def test_whr_builds_one_rating_per_playing_day(self):
        """End to end: distinct dates must produce distinct days in the fitted curve."""
        arena = LambdaArena(lambda a, b: True, base_competitor=WholeHistoryRatingCompetitor)
        train_arena_with_dataset(
            arena,
            [
                ("A", "B", 1.0, datetime.datetime(2020, 1, 6), None),
                ("A", "B", 1.0, datetime.datetime(2020, 6, 6), None),
                ("A", "B", 1.0, datetime.datetime(2020, 11, 6), None),
            ],
        )
        history = arena.competitors["A"].rating_history()
        self.assertEqual(len(history), 3, f"expected three playing days, got {history}")
        self.assertEqual(
            [day for day, _rating in history],
            [datetime.date(2020, 1, 6), datetime.date(2020, 6, 6), datetime.date(2020, 11, 6)],
        )


if __name__ == "__main__":
    unittest.main()
