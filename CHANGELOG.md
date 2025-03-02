v1.0.0 (Unreleased)
======

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