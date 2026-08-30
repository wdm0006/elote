# Elote

[![PyPI version](https://badge.fury.io/py/elote.svg)](https://pypi.org/project/elote/)
[![Python versions](https://img.shields.io/pypi/pyversions/elote.svg)](https://pypi.org/project/elote/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**The scikit-learn of rating algorithms: compare rating models behind one
largely uniform Python interface, then keep the one that fits your data.**

Use Elote to turn pairwise wins, losses, and draws into expected scores and
rankings. Start with Elo, switch to Glicko, TrueSkill, Bradley-Terry, or another
included model without rewriting the surrounding workflow, and evaluate the
alternatives on the same match history.

```bash
pip install elote
```

## Compare and rank in 60 seconds

The same matchups can drive different rating systems:

```python
from elote import EloCompetitor, GlickoCompetitor, LambdaArena

winner_loser_pairs = [
    ("Ada", "Grace"),
    ("Grace", "Linus"),
    ("Ada", "Linus"),
    ("Grace", "Linus"),
]

for model in (EloCompetitor, GlickoCompetitor):
    arena = LambdaArena(lambda winner, loser: True, base_competitor=model)
    for winner, loser in winner_loser_pairs:
        arena.matchup(winner, loser)

    ranking = [row["competitor"] for row in arena.leaderboard()]
    print(f"{model.__name__}: {' > '.join(ranking)}")
```

```text
EloCompetitor: Ada > Grace > Linus
GlickoCompetitor: Ada > Grace > Linus
```

`LambdaArena` owns the population, creates competitors as identifiers appear,
records each bout, and returns the leaderboard best-first. Change
`base_competitor` to compare another model with the same application code.

## Choose your task

| I want to... | Start with |
|---|---|
| Rate two competitors directly | Create two matching competitor objects; call `expected_score()`, then `beat()`, `lost_to()`, or `tied()`. See [Getting started](https://elote.mcginniscommawill.com/getting_started.html). |
| Manage a changing population | Use `LambdaArena` with your identifiers and results, then read `leaderboard()` and `history`. See [Arenas](https://elote.mcginniscommawill.com/arenas.html). |
| Compare models on held-out data | Build a `DataSplit`, then use `evaluate_competitor()` or `benchmark_competitors()` and the plotting helpers. See [`elote/benchmark.py`](elote/benchmark.py). |
| Save and resume ratings | Call `export_state()` or `to_json()` and store the result yourself; restore with `from_state()`, `from_json()`, or `LambdaArena(initial_state=...)`. See [Serialization](https://elote.mcginniscommawill.com/serialization.html). |

Install the dataset adapters when you want the Lichess chess or college-football
loaders:

```bash
pip install "elote[datasets]"
```

`SyntheticDataset` is included in the base installation.

## Rating models

Every row below is exported from `elote` and supports the shared competitor
workflow. Constructor parameters and algorithm-specific state still differ.

| Model | Class | Model shape |
|---|---|---|
| Elo | `EloCompetitor` | Incremental rating |
| Glicko-1 | `GlickoCompetitor` | Incremental rating and rating deviation |
| Glicko-2 | `Glicko2Competitor` | Incremental rating, deviation, and volatility |
| TrueSkill | `TrueSkillCompetitor` | Bayesian mean and uncertainty |
| ECF | `ECFCompetitor` | English Chess Federation rating |
| DWZ | `DWZCompetitor` | German chess rating |
| Colley Matrix | `ColleyMatrixCompetitor` | Global fit over the matchup graph |
| Massey | `MasseyCompetitor` | Global least-squares fit using score margins |
| Keener | `KeenerCompetitor` | Global eigenvector fit using scores |
| Pythagorean | `PythagoreanCompetitor` | Points-based win expectation from points scored and allowed |
| Bradley-Terry | `BradleyTerryCompetitor` | Global paired-comparison fit |
| Whole-History Rating | `WholeHistoryRatingCompetitor` | Time-aware Bradley-Terry rating curve |
| Blended ensemble | `BlendedCompetitor` | Composition of multiple competitor models |

The common surface covers expected scores, results, ratings, reset, configuration,
and state serialization. It does not erase meaningful differences in parameters,
uncertainty, rating scale, or whether a model updates incrementally or refits a
connected population.

## Evaluation tools

Elote keeps pre-result predictions in `History` objects so you can inspect
accuracy, precision, recall, F1, draw-aware metrics, confusion counts, calibration
data, optimized decision thresholds, and accuracy by prior bouts. Dataset helpers
train on one split and evaluate on another without updating ratings on the held-out
rows. Benchmark results retain the trained arena and history for further analysis.

The package also includes plotting helpers for rating-system comparisons,
optimized accuracy, calibration, and accuracy by prior bouts.

Curious how they stack up against each other? Run `make compare-systems` (or
[`scripts/rating_system_comparison.py`](scripts/rating_system_comparison.py)
directly) to train every system on the same split and compare accuracy, Brier
score, and log loss — see
[`docs/source/rating_systems/comparison_benchmark.rst`](docs/source/rating_systems/comparison_benchmark.rst)
for the methodology and measured results.

## Library boundaries

Elote is an in-process Python library. It does not provide a CLI, hosted API,
server, user interface, accounts, authentication, or a database. Your application
supplies identifiers and results, decides how a model is selected, and stores
exported dictionaries or JSON in the persistence system it owns.

## Trust and project status

- Python 3.10 or later
- Typed-package marker (`py.typed`)
- MIT licensed
- Behavioral and known-value tests across the model catalog, plus tests for
  serialization, datasets, benchmarking, histories, plots, and the shared interface
- Published with an Alpha development classifier

See the [full documentation](https://elote.mcginniscommawill.com/) and the
[`examples/`](examples/) directory for more workflows.

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

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md), then open an
[issue](https://github.com/wdm0006/elote/issues) or pull request.

```bash
git clone https://github.com/wdm0006/elote.git
cd elote
make install-dev
make test       # or: make test-cov
make lint       # or: make lint-fix, make format
make docs       # or: make build
```

Elote is available under the [MIT License](LICENSE.md).
