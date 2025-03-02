"""
College football dataset for elote.

This module provides a dataset of college football games for testing and evaluating different rating algorithms.
"""

import os
import datetime
import tempfile
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from elote.datasets.base import BaseDataset


class CollegeFootballDataset(BaseDataset):
    """
    College football dataset for testing and evaluating different rating algorithms.

    This dataset contains college football games from the College Football Data API using sportsdataverse.
    """

    def __init__(self, cache_dir: Optional[str] = None, start_year: int = 2015, end_year: int = 2022):
        """
        Initialize a college football dataset.

        Args:
            cache_dir: Directory to cache downloaded data. If None, a temporary directory will be used.
            start_year: First year to include in the dataset
            end_year: Last year to include in the dataset
        """
        super().__init__(cache_dir=cache_dir)
        self.start_year = start_year
        self.end_year = end_year

        if self.cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "elote_datasets", "college_football")

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # File to cache the data
        self.data_file = os.path.join(self.cache_dir, f"college_football_games_{start_year}_{end_year}.csv")

    def download(self) -> None:
        """
        Download the college football dataset using sportsdataverse.
        """
        if os.path.exists(self.data_file):
            # Data already downloaded
            print(f"Using cached college football data from {self.data_file}")
            return

        try:
            # Try to import sportsdataverse directly
            import sportsdataverse.cfb as cfb
        except ImportError as e:
            if "pkg_resources" in str(e):
                raise ImportError(
                    "The 'pkg_resources' module is missing. This is part of setuptools. "
                    "Please install it with: pip install setuptools"
                ) from e
            elif "xgboost" in str(e) or "libxgboost" in str(e):
                raise ImportError(
                    "XGBoost dependency issue detected. On macOS, you need to install libomp: brew install libomp"
                ) from e
            else:
                raise ImportError(
                    "sportsdataverse is required to download college football data. "
                    "Please install it with: pip install sportsdataverse"
                ) from e

        print(f"Downloading college football data for years {self.start_year}-{self.end_year}...")

        # Initialize an empty DataFrame
        all_games = pd.DataFrame()

        # Download data for each year
        for year in range(self.start_year, self.end_year + 1):
            print(f"Fetching data for {year}...")
            try:
                # Get schedule data for the year using espn_cfb_schedule
                # season_type=2 is for regular season
                year_data = cfb.espn_cfb_schedule(
                    dates=year,
                    season_type=2,
                    groups=80,  # FBS games
                    limit=1000,
                    return_as_pandas=True,
                )

                if year_data is not None and not year_data.empty:
                    # Keep only completed games
                    year_data = year_data[year_data["status_type_completed"] == True]  # noqa: E712
                    all_games = pd.concat([all_games, year_data], ignore_index=True)
            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                print("Continuing with next year...")
                continue

        if all_games.empty:
            raise ValueError("No games data was retrieved. Check your internet connection or try again later.")

        # Select and rename relevant columns based on the actual column names in the data
        # First, print the columns to debug
        print(f"Available columns: {all_games.columns.tolist()}")

        # Map the columns from sportsdataverse to our expected format
        column_mapping = {
            "date": "game_date",  # Use a different name to avoid conflicts
            "home_name": "home_team",
            "away_name": "away_team",
            "home_score": "home_points",
            "away_score": "away_points",
        }

        # Check which columns actually exist and create a valid mapping
        valid_mapping = {}
        for src, dst in column_mapping.items():
            if src in all_games.columns:
                valid_mapping[src] = dst
            else:
                print(f"Warning: Column '{src}' not found in data")

        # Rename columns that exist
        if valid_mapping:
            all_games = all_games.rename(columns=valid_mapping)

        # Define the columns we want to keep
        columns_to_keep = []

        # Add required columns if they exist
        required_columns = ["start_date", "home_team", "away_team", "home_points", "away_points"]
        for col in required_columns:
            if col in all_games.columns:
                columns_to_keep.append(col)
            else:
                # Try to find alternative column names
                if col == "start_date" and "game_date" in all_games.columns:
                    all_games["start_date"] = all_games["game_date"]
                    columns_to_keep.append("start_date")
                elif col == "home_points" and "home_score" in all_games.columns:
                    all_games["home_points"] = all_games["home_score"]
                    columns_to_keep.append("home_points")
                elif col == "away_points" and "away_score" in all_games.columns:
                    all_games["away_points"] = all_games["away_score"]
                    columns_to_keep.append("away_points")
                else:
                    print(f"Warning: Required column '{col}' not found in data")

        # Add optional columns if they exist
        optional_columns = ["home_conference", "away_conference", "neutral_site", "conference_game", "venue"]

        for col in optional_columns:
            if col in all_games.columns:
                columns_to_keep.append(col)

        # Check if we have all required columns
        missing_required = set(required_columns) - set(columns_to_keep)
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        # Keep only the columns we need
        all_games = all_games[columns_to_keep]

        # Sort by date
        if "start_date" in all_games.columns:
            all_games = all_games.sort_values(by="start_date")

        # Save to CSV
        all_games.to_csv(self.data_file, index=False)

        print(f"Downloaded {len(all_games)} games and saved to {self.data_file}")

    def load(self) -> List[Tuple[str, str, float, datetime.datetime, Dict[str, Any]]]:
        """
        Load the college football dataset into memory.

        Returns:
            List of matchup tuples (home_team, away_team, outcome, timestamp, attributes)
            where outcome is 1.0 if home team won, 0.0 if away team won, and 0.5 for a tie.
        """
        # Ensure data is downloaded
        self.download()

        # Load the CSV file
        df = pd.read_csv(self.data_file)

        # Convert to matchup tuples
        matchups = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # Convert scores to outcome (1.0 for home win, 0.0 for away win, 0.5 for tie)
            home_score = row["home_points"]
            away_score = row["away_points"]

            if home_score > away_score:
                outcome = 1.0
            elif away_score > home_score:
                outcome = 0.0
            else:
                outcome = 0.5

            # Parse timestamp
            try:
                # Format depends on output, typically ISO format
                date_str = row["start_date"]
                timestamp = pd.to_datetime(date_str)
            except (ValueError, TypeError, KeyError):
                # If timestamp parsing fails, use None
                timestamp = None

            # Add attributes
            attributes = {
                "home_score": home_score,
                "away_score": away_score,
            }

            # Add optional attributes if they exist in the data
            optional_attrs = [
                "venue",
                "neutral_site",
                "conference_game",
                "home_conference",
                "away_conference",
            ]

            for attr in optional_attrs:
                if attr in row and not pd.isna(row[attr]):
                    attributes[attr] = row[attr]

            matchups.append((home_team, away_team, outcome, timestamp, attributes))

        return matchups
