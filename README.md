# Elote 🏆

[![PyPI version](https://badge.fury.io/py/elote.svg)](https://badge.fury.io/py/elote)
[![Python Versions](https://img.shields.io/pypi/pyversions/elote.svg)](https://pypi.org/project/elote/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Elote** is a powerful Python library for implementing and comparing rating systems. Whether you're ranking chess players, sports teams, or prioritizing features in your product backlog, Elote provides a simple, elegant API for all your competitive ranking needs.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Competitors](#competitors)
  - [Arenas](#arenas)
- [Development](#development)
- [Contributing](#contributing)
- [Blog Posts](#blog-posts)
- [References](#references)

## Overview

Rating systems allow you to rank competitors based on their performance in head-to-head matchups. The most famous example is the Elo rating system used in chess, but these systems have applications far beyond sports:

- Ranking products based on A/B comparisons
- Prioritizing features through pairwise voting
- Creating recommendation systems
- Matchmaking in games and competitions
- Collaborative filtering and ranking

Elote makes implementing these systems simple and intuitive, with a clean API that handles all the mathematical complexity for you.

## Features

Currently implemented rating systems:

- **Elo** - The classic chess rating system
- **Glicko-1** - An improvement on Elo that accounts for rating reliability
- **Glicko-2** - A further improvement on Glicko that adds volatility tracking
- **TrueSkill** - Microsoft's Bayesian skill rating system for multiplayer games
- **ECF** - The English Chess Federation rating system
- **DWZ** - The Deutsche Wertungszahl (German evaluation number) system

## Installation

### For Users

```bash
pip install elote
```

### For Developers

We use a modern Python packaging approach with `pyproject.toml`. Most things you need are in the `Makefile`:

```bash
# Using Make (recommended)
make install-dev

# Or using pip
pip install -e ".[dev]"

# Or using uv
uv pip install -e ".[dev]"
```

### Requirements

- Python 3.8 or higher

## Quick Start

```python
from elote import EloCompetitor

# Create two competitors with different initial ratings
player1 = EloCompetitor(initial_rating=1500)
player2 = EloCompetitor(initial_rating=1600)

# Get win probability
print(f"Player 2 win probability: {player2.expected_score(player1):.2%}")

# Record a match result
player1.beat(player2)  # Player 1 won!

# Ratings are automatically updated
print(f"Player 1 new rating: {player1.rating}")
print(f"Player 2 new rating: {player2.rating}")
```

## Usage Examples

Elote is built around two main concepts: **Competitors** and **Arenas**.

### Competitors

Competitors represent the entities you're rating. Here's how to use them:

```python
from elote import EloCompetitor

good = EloCompetitor(initial_rating=400)
better = EloCompetitor(initial_rating=500)

# Check win probabilities
print(f"Probability of better beating good: {better.expected_score(good):.2%}")
print(f"Probability of good beating better: {good.expected_score(better):.2%}")
```

Output:
```
Probability of better beating good: 64.01%
Probability of good beating better: 35.99%
```

If a match occurs, updating ratings is simple:

```python
# If good wins (an upset!)
good.beat(better)
# OR
better.lost_to(good)

# Check updated probabilities
print(f"Probability of better beating good: {better.expected_score(good):.2%}")
print(f"Probability of good beating better: {good.expected_score(better):.2%}")
```

Output:
```
Probability of better beating good: 61.25%
Probability of good beating better: 38.75%
```

### Arenas

Arenas handle large numbers of matchups automatically. The `LambdaArena` takes a comparison function and manages all competitors for you:

```python
from elote import LambdaArena
import json
import random

# Define a comparison function (returns True if a beats b)
def comparison(a, b):
    return a > b

# Generate 1000 random matchups between numbers 1-10
matchups = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(1000)]

# Create arena and run tournament
arena = LambdaArena(comparison)
arena.tournament(matchups)

# Display final rankings
print("Arena results:")
print(json.dumps(arena.leaderboard(), indent=4))
```

This example effectively implements a sorting algorithm using a rating system - not efficient, but demonstrates how Elote works with any comparable objects!

## Development

The project includes a Makefile that simplifies common development tasks:

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint code
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format

# Build package
make build

# Build documentation
make docs
```

## Contributing

Contributions are welcome! If you'd like to help improve Elote:

1. Check the [issues](https://github.com/yourusername/elote/issues) for open tasks
2. Fork the repository
3. Create a feature branch
4. Add your changes
5. Submit a pull request

For major changes, please open an issue first to discuss what you'd like to change.

## Blog Posts

Here are some blog posts about Elote:

- [Elote: A Python Package for Rating Systems](https://mcginniscommawill.com/posts/2017-12-06-elote-python-package-rating-systems/) - Introduction to the library
- [Using Cursor for Library Maintenance](https://mcginniscommawill.com/posts/2025-03-09-cursor-for-library-maintenance/#how-cursor-helps-with-maintenance) - How Cursor helps maintain Elote
- [Year's End: Looking Back at 2017](https://mcginniscommawill.com/posts/2017-12-28-years-end-looking-back-2017/) - Reflections including Elote development

## References

1. [Glicko Rating System](http://www.glicko.net/glicko/glicko.pdf)
2. [Glicko-2 Rating System](http://www.glicko.net/glicko/glicko2.pdf)
3. [Massey Ratings](https://masseyratings.com)
4. Elo, Arpad (1978). The Rating of Chessplayers, Past and Present. Arco. ISBN 0-668-04721-6.
5. [ECF Grading System](http://www.ecfgrading.org.uk/new/help.php#elo)
6. [Deutsche Wertungszahl](https://en.wikipedia.org/wiki/Deutsche_Wertungszahl)
7. [TrueSkill: A Bayesian Skill Rating System](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/)