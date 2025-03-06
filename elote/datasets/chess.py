"""
Chess dataset for elote.

This module provides a dataset of chess games for testing and evaluating different rating algorithms.
"""

import os
import datetime
import tempfile
import requests
import chess.pgn
import pyzstd
from typing import List, Tuple, Dict, Any, Optional

from elote.datasets.base import BaseDataset


class ChessDataset(BaseDataset):
    """
    Chess dataset for testing and evaluating different rating algorithms.

    This dataset contains chess games from the Lichess database.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_games: int = 10000, year: int = 2013, month: int = 1):
        """
        Initialize a chess dataset.

        Args:
            cache_dir: Directory to cache downloaded data. If None, a temporary directory will be used.
            max_games: Maximum number of games to load (to limit memory usage)
            year: Year of the Lichess database to use (2013-present)
            month: Month of the Lichess database to use (1-12)
        """
        super().__init__(cache_dir=cache_dir)
        self.max_games = max_games
        self.year = year
        self.month = month

        if self.cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "elote_datasets", "chess")

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # URL for the Lichess database
        self.data_url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"

        # File paths
        self.compressed_file = os.path.join(self.cache_dir, f"lichess_{year}-{month:02d}.pgn.zst")
        self.decompressed_file = os.path.join(self.cache_dir, f"lichess_{year}-{month:02d}.pgn")

    def download(self) -> None:
        """
        Download the chess dataset from Lichess.
        """
        # Check if the decompressed file already exists
        if os.path.exists(self.decompressed_file):
            print(f"Using existing decompressed file: {self.decompressed_file}")
            return

        # Check if the compressed file already exists
        if not os.path.exists(self.compressed_file):
            print(f"Downloading chess dataset from {self.data_url}...")

            # Download the data
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()

            with open(self.compressed_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Download complete: {self.compressed_file}")
        else:
            print(f"Using existing compressed file: {self.compressed_file}")

        # Decompress the file using pyzstd
        print(f"Decompressing {self.compressed_file}...")

        try:
            with open(self.compressed_file, "rb") as compressed:
                with open(self.decompressed_file, "wb") as decompressed:
                    decompressed.write(pyzstd.decompress(compressed.read()))
            print(f"Decompression complete: {self.decompressed_file}")
        except Exception as e:
            raise RuntimeError(f"Error decompressing the zstd file: {e}") from e

    def _parse_pgn_game(
        self, game: chess.pgn.Game
    ) -> Optional[Tuple[str, str, float, datetime.datetime, Dict[str, Any]]]:
        """
        Parse a PGN game into a matchup tuple.

        Args:
            game: A chess.pgn.Game object

        Returns:
            A matchup tuple (white_player, black_player, outcome, timestamp, attributes) or None if parsing fails
        """
        try:
            # Extract headers
            headers = game.headers

            # Get player IDs
            white_player = headers.get("White", "Unknown")
            black_player = headers.get("Black", "Unknown")

            # Get ratings
            white_rating = int(headers.get("WhiteElo", "1500"))
            black_rating = int(headers.get("BlackElo", "1500"))

            # Get result
            result = headers.get("Result", "*")
            if result == "1-0":
                outcome = 1.0  # White wins
            elif result == "0-1":
                outcome = 0.0  # Black wins
            elif result == "1/2-1/2":
                outcome = 0.5  # Draw
            else:
                return None  # Unknown result

            # Get timestamp
            date_str = headers.get("UTCDate", "")
            time_str = headers.get("UTCTime", "")

            if date_str and time_str:
                try:
                    timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
                except ValueError:
                    # Try alternative format
                    try:
                        timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M")
                    except ValueError:
                        timestamp = None
            else:
                timestamp = None

            # Get additional attributes
            attributes = {
                "white_rating": white_rating,
                "black_rating": black_rating,
                "time_control": headers.get("TimeControl", ""),
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "termination": headers.get("Termination", ""),
                "game_id": headers.get("Site", "").split("/")[-1] if "Site" in headers else None,
            }

            return (white_player, black_player, outcome, timestamp, attributes)

        except Exception as e:
            print(f"Error parsing game: {e}")
            return None

    def load(self) -> List[Tuple[str, str, float, datetime.datetime, Dict[str, Any]]]:
        """
        Load the chess dataset into memory.

        Returns:
            List of matchup tuples (white_player, black_player, outcome, timestamp, attributes)
            where outcome is 1.0 if white won, 0.0 if black won, and 0.5 for a draw.
        """
        # Ensure the data is downloaded and decompressed
        self.download()

        print(f"Loading chess games from {self.decompressed_file}...")

        # Parse the PGN file
        matchups = []
        games_loaded = 0

        with open(self.decompressed_file, "r", encoding="utf-8", errors="ignore") as pgn_file:
            while games_loaded < self.max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file

                matchup = self._parse_pgn_game(game)
                if matchup is not None:
                    matchups.append(matchup)
                    games_loaded += 1

                if games_loaded % 1000 == 0:
                    print(f"Loaded {games_loaded} games...")

        print(f"Loaded {len(matchups)} chess games.")
        return matchups
