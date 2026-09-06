---
name: local-dev
description: Stand up a working local dev environment for wdm0006/elote — uv + Python 3.11 venv, full test suite, lint, typecheck, and a primary-flow smoke test
---

# local-dev — elote onboarding record

Durable record of the LOCAL-DEV onboarding run. Last verified 2026-09-06.
Everything below was executed and observed in the repo sandbox, not inferred.

## Environment facts

- Sandbox is Linux; system Python is 3.13 — do not use it for this project. Create the project venv with `uv venv --python=3.11` (uv downloads CPython 3.11.16 automatically; matches `make setup`).
- **uv is not preinstalled** — `pip install uv` first (uv 0.12.10 verified).
- No Docker, no databases, no services, no env vars, no ports. The "app" is the importable `elote` package.

## Setup steps (in order)

1. `pip install uv`
2. `uv venv --python=3.11`
3. `uv pip install -e ".[dev,datasets]"`

Notes:
- Do NOT run `make setup` verbatim on Linux — its `brew install libomp` line is macOS-only and fails there. The two uv lines above are the portable equivalent.
- The `datasets` extra is required for the full suite (CI installs `.[dev,datasets]`). `make install-dev` installs dev only.

## Verification ladder (all must pass)

| Step | Command | Observed 2026-09-06 |
|---|---|---|
| Tests | `make test` | 612 passed, 2 warnings, ~55s |
| Lint | `make lint` | All checks passed |
| Types | `make typecheck` | Success: no issues in 30 source files |
| Primary flow | README quickstart (below) | `EloCompetitor: Ada > Grace > Linus`, `GlickoCompetitor: Ada > Grace > Linus` — matches README |
| Serialization | `export_state`/`initial_state`, `to_json`/`from_json` | Round-trips OK |
| Example | `make run-example EXAMPLE=sample_bout.py` | Exit 0 |

## Primary-flow smoke script

```bash
uv run python - <<'PY'
from elote import EloCompetitor, GlickoCompetitor, LambdaArena

pairs = [("Ada", "Grace"), ("Grace", "Linus"), ("Ada", "Linus"), ("Grace", "Linus")]
for model in (EloCompetitor, GlickoCompetitor):
    arena = LambdaArena(lambda w, l: True, base_competitor=model)
    for winner, loser in pairs:
        arena.matchup(winner, loser)
    print(f"{model.__name__}: {' > '.join(r['competitor'] for r in arena.leaderboard())}")

arena = LambdaArena(lambda w, l: True, base_competitor=EloCompetitor)
arena.matchup("a", "b")
restored = LambdaArena(lambda w, l: True, initial_state=arena.export_state())
assert restored.leaderboard()[0]["competitor"] == "a"
print("state round-trip OK")
PY
```

Serialization gotcha: `to_json()`/`from_json()` are **competitor** methods; arenas use
`export_state()` + `LambdaArena(initial_state=...)`. `LambdaArena` has no `to_json`.

## Known side effects (restore before committing)

- `make test` regenerates `images/bradley_terry_ratings.png` and `images/colley_matrix_ratings.png`.
- `uv run` may rewrite `uv.lock`.
- Clean up with: `git checkout -- images/ uv.lock` (`.venv/` is gitignored).

## CI parity

PRs to `master` run `.github/workflows/test-suite.yml` (tox py310/py311/py312 with
`.[datasets]`, plus ruff lint) and `test-docs-build.yml` (Sphinx build). A green local
`make test && make lint && make typecheck` on 3.11 is a good pre-push proxy.
