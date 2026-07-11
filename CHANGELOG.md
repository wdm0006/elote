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