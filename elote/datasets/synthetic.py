"""
Synthetic data generator for elote.

This module provides a synthetic data generator for testing and evaluating different rating algorithms.
"""

import datetime
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from elote.datasets.base import BaseDataset


class SyntheticDataset(BaseDataset):
    """
    Synthetic data generator for testing and evaluating different rating algorithms.

    This dataset generates random matchups between competitors with configurable parameters.
    The outcome of each matchup is determined by the true skill of each competitor plus some noise.
    """

    def __init__(
        self,
        num_competitors: int = 100,
        num_matchups: int = 1000,
        skill_distribution: str = "normal",
        skill_mean: float = 1500,
        skill_std: float = 300,
        noise_std: float = 100,
        draw_probability: float = 0.1,
        time_span_days: int = 365,
        seed: Optional[int] = None,
    ):
        """
        Initialize a synthetic dataset generator.

        Args:
            num_competitors: Number of competitors to generate
            num_matchups: Number of matchups to generate
            skill_distribution: Distribution of true skills ("normal", "uniform", or "pareto")
            skill_mean: Mean of the skill distribution (for normal distribution)
            skill_std: Standard deviation of the skill distribution (for normal distribution)
            noise_std: Standard deviation of the noise added to skills during matchups
            draw_probability: Probability of a draw when competitors are closely matched
            time_span_days: Number of days to spread the matchups over
            seed: Random seed for reproducibility
        """
        super().__init__(cache_dir=None)
        self.num_competitors = num_competitors
        self.num_matchups = num_matchups
        self.skill_distribution = skill_distribution
        self.skill_mean = skill_mean
        self.skill_std = skill_std
        self.noise_std = noise_std
        self.draw_probability = draw_probability
        self.time_span_days = time_span_days
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def download(self) -> None:
        """
        No download needed for synthetic data.
        """
        pass

    def _generate_skills(self) -> Dict[str, float]:
        """
        Generate true skills for all competitors.

        Returns:
            Dictionary mapping competitor IDs to their true skills
        """
        skills = {}

        for i in range(self.num_competitors):
            competitor_id = f"competitor_{i}"

            if self.skill_distribution == "normal":
                skill = np.random.normal(self.skill_mean, self.skill_std)
            elif self.skill_distribution == "uniform":
                skill = np.random.uniform(
                    self.skill_mean - self.skill_std * 1.73,  # Matching variance of normal
                    self.skill_mean + self.skill_std * 1.73,
                )
            elif self.skill_distribution == "pareto":
                # Pareto distribution for more realistic skill distribution with few very skilled competitors
                skill = np.random.pareto(3) * self.skill_std + self.skill_mean - self.skill_std
            else:
                raise ValueError(f"Unknown skill distribution: {self.skill_distribution}")

            skills[competitor_id] = skill

        return skills

    def _generate_matchups(
        self, skills: Dict[str, float]
    ) -> List[Tuple[str, str, float, datetime.datetime, Dict[str, Any]]]:
        """
        Generate random matchups between competitors.

        Args:
            skills: Dictionary mapping competitor IDs to their true skills

        Returns:
            List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        """
        matchups = []
        competitors = list(skills.keys())

        # Generate timestamps spanning the time_span_days
        start_date = datetime.datetime.now() - datetime.timedelta(days=self.time_span_days)

        for _i in range(self.num_matchups):
            # Select two random competitors
            a, b = random.sample(competitors, 2)

            # Generate a timestamp
            days_offset = random.uniform(0, self.time_span_days)
            timestamp = start_date + datetime.timedelta(days=days_offset)

            # Determine the outcome based on true skills plus noise
            skill_a = skills[a] + np.random.normal(0, self.noise_std)
            skill_b = skills[b] + np.random.normal(0, self.noise_std)

            # Calculate skill difference and normalize
            skill_diff = skill_a - skill_b

            # Determine if it's a draw
            if abs(skill_diff) < self.noise_std and random.random() < self.draw_probability:
                outcome = 0.5  # Draw
            else:
                outcome = 1.0 if skill_diff > 0 else 0.0

            # Add attributes with true skills for evaluation
            attributes = {
                "true_skill_a": skills[a],
                "true_skill_b": skills[b],
                "skill_diff": skill_diff,
            }

            matchups.append((a, b, outcome, timestamp, attributes))

        # Sort by timestamp
        matchups.sort(key=lambda x: x[3])

        return matchups

    def load(self) -> List[Tuple[str, str, float, datetime.datetime, Dict[str, Any]]]:
        """
        Generate and load the synthetic dataset.

        Returns:
            List of matchup tuples (competitor_a, competitor_b, outcome, timestamp, attributes)
        """
        # Generate true skills for all competitors
        skills = self._generate_skills()

        # Generate random matchups
        matchups = self._generate_matchups(skills)

        return matchups
