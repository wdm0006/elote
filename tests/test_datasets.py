"""
Tests for the datasets module.
"""

import unittest
import datetime
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
import shutil

from elote import (
    DataSplit,
    SyntheticDataset,
    ChessDataset,
    CollegeFootballDataset,
    LambdaArena,
    EloCompetitor,
    train_arena_with_dataset,
    evaluate_arena_with_dataset,
    train_and_evaluate_arena,
)


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

    def test_train_and_evaluate_arena(self):
        """Test that train_and_evaluate_arena works correctly."""
        # Create a synthetic dataset
        dataset = SyntheticDataset(
            num_competitors=10,
            num_matchups=50,
            seed=42,
        )

        # Split the dataset
        data_split = dataset.time_split(test_ratio=0.2)

        # Create an arena
        arena = LambdaArena(
            lambda a, b, attributes=None: True,  # Updated to accept attributes parameter
            base_competitor=EloCompetitor,
        )

        # Train and evaluate the arena
        trained_arena, history = train_and_evaluate_arena(arena, data_split)

        # Check that competitors were created
        self.assertGreater(len(trained_arena.competitors), 0)

        # Check that bouts were recorded
        self.assertGreater(len(history.bouts), 0)


if __name__ == "__main__":
    unittest.main()
