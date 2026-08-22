v1.3.0
======

Four new rating systems, margin-of-victory support end to end, and an evaluation path that is
deterministic and reproducible from the dataset split through to the chosen threshold. Twelve rating
systems now ship, against eight in v1.2.0.

New rating systems

 * Added `MasseyCompetitor`, the least-squares classical rating system, which fits every result at
   once and can use point margins rather than outcomes alone (#120).
 * Added `PythagoreanCompetitor`, the Pythagorean expectation system, rating competitors from points
   scored and allowed (#142).
 * Added `WholeHistoryRatingCompetitor`, a lazy, bounded time-aware Bradley-Terry MAP system with a
   queryable per-playing-day rating curve (#150).
 * Added `KeenerCompetitor`, the score-based member of the classical global-fit family. Keener builds
   a smoothed, skew-transformed preference matrix from the points competitors have scored on one
   another and reads ratings off its dominant eigenvector (#124).

Margin of victory

 * Added an optional score channel to the result API, so a bout can carry the score as well as the
   outcome. Systems that cannot use a margin ignore it (#124).
 * Dataset point scores are threaded into the margin-aware rating systems, so Massey and Pythagorean
   see real margins in benchmarks and arena runs rather than synthesized ones (#139).

Evaluation and benchmarking

 * Threshold optimization is now an exact, deterministic sweep rather than a stochastic search. The
   same history produces the same thresholds every time (#116).
 * `SyntheticDataset` and split seeding are instance-local and reproducible, so two datasets built in
   one process no longer share generator state (#115).
 * `accuracy_by_prior_bouts` reports small bins instead of dropping them, and normalizes outcomes the
   same way the rest of the metrics do, which makes cold-start calibration readable (#141, #148).
 * Dataset evaluation warns when it skips bouts rather than silently narrowing the sample (#151).
 * Arena evaluation skips competitors it has never seen instead of inventing them at the default
   rating (#140).
 * Global-fit rating floors are preserved in benchmarks, and the benchmark's initial-rating
   normalization is applied for real without mutating competitor class state (#95, #127).
 * College football competitors are identified by program rather than mascot, so a program that
   changes its mascot keeps one rating (#153).
 * Calibration outcome normalization was corrected, so calibration curves and accuracy agree on what
   a result was (#121).

Improvements

 * The Bradley-Terry MM refit is vectorized with NumPy, which makes repeated global refits practical
   on larger pools (#104).
 * A `LambdaArena` can be restored from its own `export_state()` output, and a restored global fit
   keeps its comparison graph consistent (#96, #138).
 * Global-fit competitor identity no longer depends on the order state was loaded in (#152).
 * `LambdaArena.expected_score` is a pure read and no longer mutates arena state (#137).
 * Glicko competitors can start from a historical match time, so an imported history does not begin
   with an inactivity penalty (#131).
 * Colley's tie perturbation was removed; ties are handled by the linear system itself (#103).
 * Elo ratings saturate at the configured minimum instead of raising (#97).
 * Validation on the way in: blended competitor compositions must match (#102), bulk arena APIs
   validate their outcomes (#149), dataset split ratios and iterator batch sizes are checked (#112),
   and the dataset helpers handle empty inputs (#108).

Bug fixes

 * Fixed an inverted TrueSkill draw margin and unified its three call sites (#119).
 * Fixed the TrueSkill `expected_score` draw adjustment (#111).
 * Fixed `MasseyCompetitor.expected_score` to normalize by the fitted rating scale, so its
   probabilities are comparable with the other systems (#154).
 * A drawn training matchup now records a `Bout`, so draws reach the metrics (#110).

Compatibility notes

 * Predicted probabilities change for TrueSkill on drawn bouts and for Massey. Recalibrate any
   thresholds tuned against v1.2.1.
 * The score channel is optional everywhere; existing result-reporting calls are unaffected.

v1.2.1
======

A correctness release. Every shipped rating system except Elo, DWZ, Colley and Bradley-Terry had at
least one numeric defect, and the prediction-quality metrics silently discarded drawn bouts. Ratings
and predicted probabilities produced by this version differ from v1.2.0; state exported by v1.2.0
still loads without changes.

Bug fixes

 * Fixed `GlickoCompetitor.expected_score` and the Glicko rating update squaring the rating deviation
   twice and reading the wrong player's deviation. Expected scores had collapsed to roughly 0.5 for
   any rating gap; a 2500-vs-1500 matchup now returns 0.979 instead of 0.504.
 * Fixed `Glicko2Competitor.beat`/`tied` applying the inactivity rating-deviation inflation three
   times per match, and updating the two competitors sequentially so the second update read the
   first competitor's already-updated state. Both competitors are now updated from the same
   pre-match snapshot, and `update_rd_for_inactivity` is idempotent for a repeated timestamp.
 * Fixed `TrueSkillCompetitor.expected_score` using `sqrt(beta^2 + sigma_i^2 + sigma_j^2)` where
   two-player TrueSkill uses `sqrt(2*beta^2 + ...)`. Predictions were too sharp.
 * Fixed the TrueSkill win, loss and draw updates: skill variance is now inflated by `tau^2` before
   the match, mean corrections are scaled by `sigma^2/c` and variance corrections by `sigma^2/c^2`,
   and draws apply both the mean correction and the positive draw correction. Regression tested
   against the reference `trueskill` package.
 * Fixed `TrueSkillCompetitor.rating` clamping the conservative estimate at the inherited minimum
   rating. It now returns `mu - 3*sigma` unclamped, so comparisons and `LambdaArena.leaderboard()`
   order competitors on TrueSkill's own scale.
 * Fixed ECF rating updates to apply the rating-difference cap independently for each competitor.
   Draws between widely-separated competitors were argument-order dependent, and a competitor could
   gain rating from losing an expected result.
 * Fixed `History.report_results()` comparing a predicted competitor identifier against a bout slot
   label when computing `"correct"`, which made the field wrong for essentially every bout.
 * Fixed `History.confusion_matrix`, `calculate_metrics`, `calculate_metrics_with_draws`,
   `optimize_thresholds` and `random_search` discarding every drawn bout. Draw handling in those
   methods had been unreachable; draws now count as true positives when predicted, false positives
   when wrongly predicted, and false negatives when missed.

Improvements

 * `History.report_results()` now reports the winner under the correctly spelled key
   `predicted_winner`. The historical misspelling `predicted_winnder` is retained as a deprecated
   alias with the same value and will be removed in a future release.
 * CI now runs the test suite against every supported Python version (3.10, 3.11, 3.12) instead of
   3.10 only, lints in a separate job, and uses current GitHub Actions.

Compatibility notes

 * Metrics computed from a `History` will change for any dataset containing draws, and predicted
   probabilities change for Glicko, Glicko-2 and TrueSkill. Recalibrate any thresholds tuned against
   v1.2.0.
 * `TrueSkillCompetitor.rating` may now be negative for a competitor that has played few matches.
 * Recording a Glicko-2 match timestamped before a competitor's last recorded activity now raises
   `InvalidParameterException` where it previously could be accepted.

v1.2.0
======

New rating systems

 * Added the Bradley-Terry model (`BradleyTerryCompetitor`), a maximum-likelihood paired-comparison
   system reported on an Elo-compatible rating scale.

Bug fixes

 * Fixed a `TypeError` crash in `LambdaArena.process_history`, `evaluate_performance`, and `validate`.
 * Fixed `LambdaArena.leaderboard()` to return competitors best-first, matching its docstring.
 * Fixed `EloCompetitor.transformed_rating` to honor a configurable `_base_rating` divisor.
 * Fixed DWZ to correctly calculate the development coefficient based on the competitor's age at
   the time of the match.

Improvements

 * Added type hints across the competitor and arena modules.
 * Added a configurable logging module (`elote.logging`).
 * Expanded the documentation to cover every competitor (Glicko-2, TrueSkill, Colley Matrix, and
   Bradley-Terry), added a Bradley-Terry example, and refreshed the rating-system comparison guide.
 * Updated the Sphinx theme (Google Analytics and dark mode) and general linting/dependency cleanup.
 * Routine dependency bumps (uv, pygments, requests, black, pillow, filelock, virtualenv).
 
v1.1.0
======

 * Glicko and Glicko-2 now properly handle time since last match 
 * Bugfix in evaluation of draws in benchmarking

v1.0.0
======

 * [] Added end to end examples using the chess and cfb datasets
 * [] Added Glicko-2 Competitor 
 * [] Added TrueSkill Competitor
 * [] Added datasets module to read sample data for development
 * [] Added a visualization module to plot rating systems performance
 * [] Added a benchmark module to compare rating systems
 * [] Added scipy optimization to find optimal thresholds for rating systems
 * [CORE-3] Standardized the `Competitor` serialization formats
 * [CORE-1] Fixed minimum rating enforcement across all competitor classes
 * [CORE-1] Updated documentation examples to use higher initial ratings
 * [CORE-1] Made `reset` method abstract in `BaseCompetitor` class
 * [CORE-1] Updated ECFCompetitor default initial rating from 40 to 100
 * [CORE-1] Fixed benchmark tests to prevent negative ratings

v0.1.0
======

 * Many bugfixes
 * Improved testing and documentation
 * Added notion of history object and bout objects for arenas to track progress
 
v0.0.3,4 and 5
==============

 * No change, debugging CI
 
v0.0.2
======

 * bugfixes in glicko expected score
 * bugfixes in elo score that wouldn't allow ratings to drop properly
 * added some testing and CI
 
v0.0.1
======

 * initial release
 * lambda arena added
 * elo competitor added
 * glicko competitor added
