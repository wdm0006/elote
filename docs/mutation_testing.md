# Mutation testing campaign report — mutmut 3.7.0 full rerun

**Project:** `wdm0006/elote` · **Campaign date:** 2026-09-07 · **Tool:** mutmut 3.7.0
**Tree measured:** local integration of PRs #185–#189 (see §2)
**Baseline:** 55.0% non-skipped score on 10,496 mutants (research findings §3)
**Target:** ≥70% non-skipped score on the canonical configuration

## 1. Headline result

| Metric | Baseline (§3) | This campaign | Delta |
|---|---|---|---|
| Total mutants | 10,496 | 11,681 | +1,185 (+11.3%) |
| Killed | 5,256 | 6,578 | +1,322 |
| Survived | 4,275 | 4,599 | +324 |
| Timeout | 22 | 37 | +15 |
| No tests (uncovered) | 943 | 467 | −476 |
| Suspicious / skipped | 0 / 0 | 0 / 0 | — |
| **Non-skipped score** | **55.0%** | **58.7%** (6,578 / 11,214) | **+3.7 pts** |

**Verdict: the ≥70% target was not met.** The campaign would need **1,272 more kills**
(7,850 of 11,214 non-skipped). The score improved +3.7 points despite an 11.3% larger
mutant set — the hardening waves' new tests kill mutants strictly faster than the new
code (#187) generates them — but the remaining gap is concentrated in behavioral
clusters no wave addressed (§7). Of the 4,599 survivors, **1,988 fall inside the
locked waiver register** (§6) and **2,611 are behavioral**. Score net of waivers:
**71.3%** (6,578 / 9,226) — above target, reported for decision-making only; the locked
gate is the raw non-skipped score.

## 2. Tree measured

All five hardening PRs (#185–#189) were still open when the campaign started, so per the
task brief the campaign ran on a local integration of the five PR branches rather than on
merged `origin/master`:

- Branch: `scratch/campaign-integration` (merged cleanly from `origin/master@fd9fd6f`
  plus the five PR heads; no conflicts).
- Test tree at measurement time: **1,098 tests + 678 subtests**, including the canonical
  accuracy anchors (#186), per-system rating-math kill tests (#188), and the
  serialization/state + ensemble coverage (#189).
- One integration-only adjustment was required (see §3).

If any of the five PRs change before merging, only §1's and §5's numbers need
re-measuring; the waiver register and shortfall analysis (§6–§8) are structural.

## 3. Tree verification (pre-campaign)

Run on the exact measured tree, immediately before launching the campaign:

| Check | Result |
|---|---|
| `pytest tests/ -q` | **1,098 passed**, 678 subtests, 0 failed |
| `ruff check .` | All checks passed |
| `mypy elote/` (30 files) | No issues |

**Integration finding (cross-PR hazard, documented here for the record):** #187's lazy
solve caching defers Massey's `_rating_scale` update to the first public `rating` read,
while #186's white-box test
(`tests/test_rating_math_massey.py::test_rating_scale_floored_for_degenerate_groups`)
asserted the private scale immediately after `tied()`. On the integrated tree this fails
(`1.0 != 1e-9`); each PR passes in isolation. The campaign tree applies the minimal fix —
read the public `a.rating` before asserting the deferred private values — preserving the
documented lazy semantics without touching production code. **When #186 and #187 merge
together, master's suite will need the same one-line adjustment; it is not included in
this docs-only PR.**

## 4. Methodology

Canonical configuration (from #185, `pyproject.toml`):

```toml
[tool.mutmut]
paths_to_mutate = ["elote/"]
tests_dir = ["tests/"]
pytest_add_cli_args = ["--benchmark-disable",
                       "--ignore=tests/test_examples.py"]
```

- `tests/test_examples.py` remains excluded: it spawns subprocesses against the installed
  (unmutated) package and cannot kill mutants.
- Unlike the baseline run, `tests/test_unified_interface.py` is **in scope** (#185
  restored class-state hygiene, removing the `_base_rating` leak that had forced its
  exclusion in the research campaign).
- Runner: mutmut 3.7.0, parallel workers, `PYTHONHASHSEED=0`; wall time **41 minutes**
  (08:30:43–09:11:45 UTC) for 11,681 mutants ≈ 4.7 mutants/s.
- Score definition (identical to baseline §3): `killed / (killed + survived + timeout)`;
  `no tests` mutants are uncovered code and are excluded from the denominator.
- Per-mutant test selection is mutmut's stats-driven mapping (tests are attributed to the
  mutated function by execution during the stats phase; mutants in functions with no
  mapped tests run the full selection). The campaign used this canonical selection
  unmodified — no forced full-selection runs are reported here.
- The baseline excluded `test_unified_interface.py` and had 943 no-tests mutants; this
  run re-included that file and covers more of `elote/__init__.py` (467 no-tests).

Mutant count grew +11.3% over baseline because #187 added new code paths (lazy solve
caching and fallback helpers in colley/massey/keener) that themselves generate mutants.

## 5. Per-module results (baseline → campaign)

**A note on per-module baseline comparison:** the baseline report's per-module survivor
counts were partially captured (its own `base.py` row flags "13, but 234 in the deeper
function breakdown", and its module rows sum to ~291 of 4,275 total survivors), so a
row-by-row baseline→campaign survivor table would compare exact numbers against partial
ones. This report therefore records the complete final per-module data below; baseline
per-module figures should be read from the research findings §3 with that caveat in
mind. Exact comparison is valid at the totals level (§1).

| Module | Mutants | Killed | Survived | No-tests | Timeout | Score | Behavioral (post-waiver) |
|---|---|---|---|---|---|---|---|
| `elote/competitors/glicko_boost.py` | 506 | 396 | 102 | 0 | 8 | 78.3% | 84 |
| `elote/competitors/whr.py` | 563 | 367 | 91 | 103 | 2 | 79.8% | 86 |
| `elote/competitors/massey.py` | 447 | 313 | 133 | 0 | 1 | 70.0% | 53 |
| `elote/competitors/keener.py` | 552 | 344 | 167 | 32 | 9 | 66.2% | 72 |
| `elote/competitors/pythagorean.py` | 232 | 156 | 76 | 0 | 0 | 67.2% | 37 |
| `elote/competitors/bradley_terry.py` | 424 | 256 | 167 | 0 | 1 | 60.4% | 85 |
| `elote/arenas/lambda_arena.py` | 483 | 289 | 186 | 8 | 0 | 60.8% | 90 |
| `elote/arenas/base.py` | 1,225 | 617 | 405 | 196 | 7 | 60.0% | 228 |
| `elote/competitors/base.py` | 519 | 309 | 210 | 0 | 0 | 59.5% | 30 |
| `elote/competitors/elo.py` | 282 | 163 | 119 | 0 | 0 | 57.8% | 39 |
| `elote/competitors/colley.py` | 620 | 355 | 264 | 0 | 1 | 57.3% | 113 |
| `elote/competitors/trueskill.py` | 708 | 397 | 311 | 0 | 0 | 56.1% | 146 |
| `elote/competitors/ensemble.py` | 448 | 244 | 204 | 0 | 0 | 54.5% | 37 |
| `elote/competitors/dwz.py` | 438 | 222 | 216 | 0 | 0 | 50.7% | 77 |
| `elote/competitors/glicko.py` | 537 | 268 | 265 | 0 | 4 | 49.9% | 112 |
| `elote/datasets/chess.py` | 296 | 119 | 120 | 57 | 0 | 49.8% | 120 |
| `elote/competitors/glicko2.py` | 866 | 417 | 445 | 0 | 4 | 48.2% | 258 |
| `elote/competitors/ecf.py` | 290 | 115 | 175 | 0 | 0 | 39.7% | 75 |
| `elote/datasets/football.py` | 340 | 140 | 200 | 0 | 0 | 41.2% | 160 |
| `elote/visualization.py` | 796 | 316 | 480 | 0 | 0 | 39.7% | 479 |
| `elote/evaluation.py` | 305 | 233 | 72 | 0 | 0 | 76.4% | 62 |
| `elote/datasets/utils.py` | 246 | 197 | 49 | 0 | 0 | 80.1% | 49 |
| `elote/datasets/synthetic.py` | 126 | 83 | 43 | 0 | 0 | 65.9% | 43 |
| `elote/benchmark.py` | 225 | 157 | 68 | 0 | 0 | 69.8% | 54 |
| `elote/datasets/base.py` | 122 | 100 | 22 | 0 | 0 | 82.0% | 22 |
| `elote/__init__.py` | 48 | 5 | 9 | 34 | 0 | 35.7% | 0 |
| `elote/logging.py` | 24 | 0 | 0 | 24 | 0 | — | 0 |
| `elote/datasets/__init__.py` | 13 | 0 | 0 | 13 | 0 | — | 0 |
| **Total** | **11,681** | **6,578** | **4,599** | **467** | **37** | **58.7%** | **2,611** |

Baseline clusters the hardening waves closed (raw survivor counts; the residue that
remains is overwhelmingly the waived log-string class inside validation branches):

| Baseline cluster (§3) | Baseline raw | Final raw | Final behavioral |
|---|---|---|---|
| `base._validate_state_dict` | 75 | 40 | **2** |
| `base.import_state` | 33 | 32 | **2** |
| `colley._recalculate_ratings` matrix assembly | 65 | 64 | **27** (reduced, not closed) |
| Elo escalation/clamp branches | 16 (module) | 119 (module) | 39 (module; serde defaults dominate) |

The serialization/state waves (#189) genuinely closed the *assertable* residue of the
base-class validation and import paths; what survives there is log-string mutants inside
the same branches. Colley's matrix-assembly path improved but retains 27 behavioral
survivors (degenerate-matrix fallbacks and connectivity pruning), and Elo's module
survivors are now dominated by `from_state`/`_import_parameters` defaults, not the
escalation/clamp branches #188 covered.

## 6. Waiver register (locked classes)

Survivors are classified by rendering each mutant's diff and matching against the locked
waiver classes from the hardening spec. Classification is mechanical and exhaustive
(all 4,599 survivors classified; 0 classification errors):

| Waiver class (locked) | Survivors | Rationale |
|---|---|---|
| Inert log/format-string mutants | 1,922 | Mutations inside `logger.*(...)` arguments or format strings; no test asserts log content for these call sites, and normal log levels make them behaviorally inert. Concentrated in validation branches (each guard logs before raising — the guard's raise is asserted, the log call is not). |
| `cast()` mutants | 33 | Type-annotation casts only; no runtime effect. |
| Exception-message strings | 20 | Message text changes; the exception type and raise site are unchanged and asserted. |
| `__hash__` identity mutants | 4 | Identity-based hash bodies; behavior contract is object identity. |
| `elote/__init__.py` dataset/dependency IO helpers | 9 | Optional-dependency probing helpers; the module's 34 no-tests mutants are already excluded from the score, and the 9 inert survivors round out the class. |
| **Total waived** | **1,988** | — |

**Score net of waivers: 71.3%** (6,578 / (11,214 − 1,988)). The ≥70% gate is evaluated on
the raw non-skipped score per the locked spec; the net figure is reported for
decision-making only.

**Per-mutant waivers beyond the locked classes: none.** Every remaining survivor has a
non-inert diff and is treated as behavioral (§7). The 37 timeouts are loop-breaking
mutants in iterative/graph code (`optimize_thresholds` ×7, `glicko_boost._advance_to` ×8,
`keener._recalculate_ratings` ×6, `_get_connected_competitors` ×4 across
colley/massey/keener/bradley-terry, `glicko2.update_ratings` ×4, glicko export ×4, WHR
node/component ×2, `keener.tied` ×1) — real behavior changes, counted in the denominator
like the baseline did.

## 7. Shortfall itemization (2,611 behavioral survivors)

Each cluster names representative surviving mutants (reproducible with
`mutmut show <key>`) and what killing them would require.

### 7.1 Visualization plotting paths — 479

Representative: `elote.visualization.x_plot_calibration_curve__mutmut_60` — removes the
`prob_true` argument from `ax.plot(...)`, silently plotting the calibration curve against
the wrong data. Distribution: `plot_calibration_comparison` 136,
`plot_rating_system_comparison` 129, `plot_calibration_curve` 80,
`plot_accuracy_by_prior_bouts` 73, `plot_optimized_accuracy_comparison` 41,
`compute_calibration_data` 20.

Why they survive: no test inspects matplotlib artists. Killing these requires
artist-level assertions (`ax.get_lines()[0].get_ydata()`, label/legend checks) in an
Agg-backend test module. No hardening wave included a visualization test module; this is
the largest single cluster of the shortfall.

### 7.2 Competitor `from_state` / `_import_*` defaults and validation — 631

Representative:
`elote.competitors.glicko2.xǁGlicko2Competitorǁfrom_state__mutmut_67` — changes the
omitted-key default `state.get("initial_rating", 1500)` to `1501`; a state dict missing
`initial_rating` silently reconstructs with the wrong rating. The same
omitted-key-default / key-rename shape accounts for the bulk of this cluster across all
14 competitor modules (glicko2 258, trueskill 146, colley 113, glicko 112, …).

Why they survive: #189 covered base-class and ensemble serialization; each subclass's own
`from_state`/`_import_parameters` omitted-key defaults were not individually asserted.
Killing these needs one omitted-key round-trip assertion per system (~14 small tests).

### 7.3 Dataset ETL and parsing — 394

Representative:
`elote.datasets.football.xǁCollegeFootballDatasetǁ_prepare_games__mutmut_80` — renames a
constructed column (`home_points` → `XXhome_pointsXX`), dropping it from downstream
frames. Distribution: football `_prepare_games` 76 + `download` 50 + `load` 18,
chess `_parse_pgn_game` 84 + `load` 20, `datasets/utils` 49, `synthetic` 43,
`datasets/base` 22.

Why they survive: loaders are exercised through optional-dependency integration tests
(skipped when `sportsdataverse`/`python-chess` are absent) and network paths excluded
from the suite. Killing these needs small fixture-CSV/PGN unit tests — no network, no
optional deps.

### 7.4 Arena analysis utilities — 318

Representative: `elote.arenas.base.xǁHistoryǁoptimize_thresholds__mutmut_143` —
`low < midpoint < high` → `low < midpoint <= high`, an off-by-boundary change that only
manifests with duplicate values. Distribution: `optimize_thresholds` 47,
`accuracy_by_prior_bouts` 44, `_normalized_outcome` 31, `random_search` 30,
`confusion_matrix` 28, plus `lambda_arena` 90 (`process_history`, `evaluate_performance`,
`validate`, `rating_period`).

Why they survive: calibration/search helpers run inside larger flows whose assertions are
coarse; boundary-value and duplicate-value unit tests would kill them.

### 7.5 Walk-forward evaluation — 62

Representative: `elote.evaluation.x_walk_forward__mutmut_104` — `period_count += 1` →
`+= 2`, double-counting samples in per-period metrics. Distribution: `walk_forward` 43,
`tune` 14, `_period_key` 5.

Why they survive: evaluation metrics are asserted at aggregate level; per-period metric
assertions (counts, log-loss/brier per period) would kill them.

### 7.6 Long tail — 727

Remaining behavioral survivors: benchmark harness (54), plus residual rating-math gaps —
glicko2 `update_ratings` 48 and `update_rd_for_inactivity` 20, colley
`_recalculate_ratings` 27, trueskill `update_team` 13, scattered
`configure_class`/validation boundaries and window edges. Each is a small targeted-test
candidate; none is waivable under the locked classes.

## 8. Path to the 70% target

Killing **1,272** more mutants reaches 70% (non-skipped). Ranked by kill-efficiency from
§7 — the two largest clusters alone cover the gap:

1. Competitor import-default assertions (~631 kills, ~14 small tests).
2. Visualization artist assertions (~479 kills, one new test module).
3. Dataset fixture-CSV/PGN unit tests (~394 kills, no network needed).
4. Arena/evaluation boundary and per-period assertions (~380 kills).

Clusters 1+2 total ~1,110 kills; adding any part of 3–4 clears 70%. If the ≥70% gate is
re-evaluated net of the locked waiver register, the campaign already stands at 71.3%.

## 9. Reproducibility

```bash
# On the integration tree (scratch/campaign-integration) with the canonical config:
uv pip install -e '.[dev]' mutmut==3.7.0
PYTHONHASHSEED=0 .venv/bin/mutmut run          # full campaign, parallel workers, ~41 min
.venv/bin/mutmut results                        # per-mutant statuses
.venv/bin/mutmut show <mutant-key>              # render any survivor cited above
.venv/bin/mutmut tests_for_mutant <mutant-key>  # inspect per-mutant test selection
```

Raw campaign data behind this report (per-module/per-function survivor counts, waiver
classification keys, timeout list) was produced from `mutants/*.meta` by the session
collector scripts; representative survivors are verifiable directly with `mutmut show`.
