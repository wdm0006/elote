# elote — agent guide

**Repo:** `wdm0006/elote` — "The scikit-learn of rating algorithms."
**What it is:** an in-process Python library for rating bouts. You feed pairwise wins,
losses, and draws; it produces expected scores and rankings. Rating models ship behind
one uniform competitor interface (Elo, Glicko-1, Glicko-2, Glicko-Boost, TrueSkill,
ECF, DWZ, Colley Matrix, Massey, Keener, Pythagorean, Bradley-Terry, Whole-History
Rating, plus a Blended ensemble), with arenas, evaluation/benchmarking, plotting, and
dataset helpers around them.

## Stack

| Layer | Choice |
|---|---|
| Language | Python `>=3.10` (CI matrix: 3.10 / 3.11 / 3.12; local dev venv: 3.11) |
| Package manager | uv — Makefile, CI, and tox all drive `uv`; `uv.lock` is committed |
| Build | setuptools (`python -m build`) |
| Tests | pytest (+ `pytest-cov`, `pytest-benchmark`); tox for the 3-version matrix |
| Lint / format | ruff (line-length 120, rules E/F/B, E501 ignored) |
| Types | mypy (strict-ish, `elote/` only; tests/examples/scripts/docs excluded) |
| Docs | Sphinx (`docs/source`) |
| Runtime shape | In-process library — **no CLI, no server, no ports, no database, no required env vars** |

## Commands

First-time setup (uv is not preinstalled on a fresh sandbox):

```bash
pip install uv
uv venv --python=3.11                  # uv downloads CPython 3.11 if absent
uv pip install -e ".[dev,datasets]"    # datasets extra needed for the full suite (mirrors CI)
```

`make setup` wraps the first two lines but also runs `brew install libomp`
(macOS-only — skip that step on Linux).

| Task | Command |
|---|---|
| Install (dev only) | `make install-dev` |
| Run tests | `make test` |
| One file / class / method | `make test PYTEST_ARGS="tests/test_unified_interface.py::TestUnifiedInterface::test_base_methods_elo"` |
| Coverage | `make test-cov` |
| Benchmarks | `make benchmark` |
| All Python versions | `make test-all` (tox py310/py311/py312) |
| Lint | `make lint` (also `make lint-fix`, `make format`) |
| Typecheck | `make typecheck` (or `make typecheck FILE=elote/competitors/elo.py`) |
| Run an example | `make run-example EXAMPLE=sample_bout.py` |
| Compare all rating systems | `make compare-systems` |
| Docs | `make docs` |
| Build dist | `make build` |

Makefile targets shell out to `uv run`, so run them from the repo root with `.venv`
present.

## Codebase map

See [`codebase-map.md`](codebase-map.md). Hot spots: `elote/competitors/base.py`
(the shared `BaseCompetitor` contract every model implements),
`elote/arenas/lambda_arena.py` (the arena most examples use), `elote/benchmark.py`
(evaluation/split tooling).

## Local Verification Summary

Verified 2026-09-06 in the onboarding sandbox (Python 3.11.16, uv 0.12.10).
Dev stack status: **healthy**.

| Check | Command | Result |
|---|---|---|
| Unit + behavioral tests | `make test` | **612 passed**, 2 warnings, ~55s |
| Lint | `make lint` | All checks passed |
| Types | `make typecheck` | Success: no issues in 30 source files |
| Primary user flow | README quickstart script (below) | Output matches README exactly |
| Serialization | `export_state`/`initial_state`, `to_json`/`from_json` | Round-trips OK |
| Example run | `make run-example EXAMPLE=sample_bout.py` | Exit 0 |

Primary user flow exercised (the library equivalent of an app smoke test — the README
"Compare and rank in 60 seconds" workflow):

```python
from elote import EloCompetitor, GlickoCompetitor, LambdaArena

pairs = [("Ada", "Grace"), ("Grace", "Linus"), ("Ada", "Linus"), ("Grace", "Linus")]
for model in (EloCompetitor, GlickoCompetitor):
    arena = LambdaArena(lambda w, l: True, base_competitor=model)
    for winner, loser in pairs:
        arena.matchup(winner, loser)
    print(f"{model.__name__}: {' > '.join(r['competitor'] for r in arena.leaderboard())}")
# -> EloCompetitor: Ada > Grace > Linus
# -> GlickoCompetitor: Ada > Grace > Linus
```

## Sandbox snapshot

- Snapshot (e2b template): `5lptge4r97x1hkeuhq88:default`
- Built at: `2026-09-06T20:19:51.962Z`
- Contents: repo at `master` + `.venv` (Python 3.11.16) with `.[dev,datasets]` installed via uv.

## Gotchas

- `make setup` includes `brew install libomp` — macOS-only; on Linux skip it, everything works without it.
- The full test suite needs the `datasets` extra; `make install-dev` alone leaves dataset-adapter tests unrunnable.
- Running the suite regenerates `images/*_ratings.png`; `uv run` may rewrite `uv.lock`. Both show as dirty in `git status` — restore with `git checkout -- images/ uv.lock` unless your change intends them.
- `tests/test_examples.py` executes the example scripts, so a broken example fails the suite.
- CI runs tox py310–py312 + ruff on every PR to `master` (`.github/workflows/test-suite.yml`); docs build is validated separately in `test-docs-build.yml`.
- Conventions live in `.cursor/rules/*.mdc` (testing via Makefile, implementing new competitors, pytest/python/sphinx standards) — read them before adding a rating model.
