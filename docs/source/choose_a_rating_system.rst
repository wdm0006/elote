.. meta::
   :description: Route from problem characteristics — uncertainty, draws, score margins, online updates, global fits, persistence — to a shortlist of rating systems, then measure them on your own data.

How to choose a rating system
==============================

.. note::

   The measured tables and code output below come from a snapshot verified on
   2026-08-09 against ten of Elote's rating systems. The library has since
   added two more concrete systems, :doc:`Pythagorean <rating_systems/pythagorean>`
   and :doc:`Whole-History Rating <rating_systems/whr>`, which are not covered
   by this comparison. The runnable examples were re-checked against a current
   install and produce the same qualitative results; a handful of measured
   figures (noted inline) differ from this snapshot in later decimal places
   because the library has changed since. Treat every number here as a
   demonstration of the method, and use "Run it on your own data" below to get
   figures for your own history and current install.

**Route by what your problem looks like, not by which algorithm is famous.** If
you need a per-competitor confidence value, three of the ten systems here give
you one. If your results carry real scores, two of them read the margin. If your
ranking must not depend on the order games were played in, four of them fit the
whole result set at once. Everything else is a preference until you measure it —
and because every system in Elote answers the same four calls, measuring it on
your own history is a loop over a dictionary, not a rewrite.

No system on this page is the best one. The measured ordering changes with the
data: on three generated scenarios run through the same split, no system won all
three, and on one of them the system that ranked first on accuracy ranked
seventh of ten on Brier score.

Start here
----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 25

   * - If your problem looks like this
     - Start with
     - Then read
   * - Two competitors, a win or a loss, results arriving continuously
     - Elo
     - :doc:`Elo <rating_systems/elo>`
   * - The same, but you need to know how sure the rating is
     - Glicko-1 or Glicko-2
     - :doc:`Elo vs Glicko vs TrueSkill <rating_systems/elo_vs_glicko_vs_trueskill>`
   * - Competitors appear, play a handful of games, and go quiet
     - Glicko-2
     - :doc:`Glicko-2 <rating_systems/glicko2>`
   * - Teams, or more than two sides per result
     - TrueSkill
     - :doc:`TrueSkill <rating_systems/trueskill>`
   * - Draws are common and carry information
     - Glicko-1, Glicko-2, or TrueSkill
     - `Draws`_
   * - Results carry real scores and the margin matters
     - Massey or Keener
     - `Score margins`_
   * - A fixed season of results, ranked once, order must not matter
     - Colley, Massey, Keener, or Bradley-Terry
     - `Incremental updates or a global fit`_
   * - You want predicted point spreads, not just probabilities
     - Massey
     - :doc:`Massey <rating_systems/massey>`
   * - A national chess federation's published formula
     - ECF or DWZ
     - :doc:`ECF <rating_systems/ecf>`, :doc:`DWZ <rating_systems/dwz>`
   * - You genuinely do not know
     - Any two rows above, then measure
     - `Run it on your own data`_

The ten systems, by decision axis
----------------------------------

Every concrete system Elote exported at the time this page was measured.
``Uncertainty`` means the object publishes a per-competitor spread you can
read. ``Reads score margins`` means supplying a score changes the rating; the
systems marked no accept the score channel and ignore it. ``Keeps your prior``
means ``initial_rating`` still affects predictions after results exist.

.. list-table::
   :header-rows: 1
   :widths: 16 14 14 16 14 16 12

   * - System
     - Update model
     - Uncertainty
     - Reads score margins
     - Keeps your prior
     - Rating scale
     - Default start
   * - Elo
     - Incremental
     - no
     - no
     - yes
     - Chess-like points
     - 400
   * - Glicko-1
     - Incremental
     - ``rd``
     - no
     - yes
     - Chess-like points
     - 1500
   * - Glicko-2
     - Incremental
     - ``rd``, ``volatility``
     - no
     - yes
     - Chess-like points
     - 1500
   * - TrueSkill
     - Incremental
     - ``sigma``
     - no
     - yes
     - ``mu`` / ``sigma``
     - ``mu`` 25, ``sigma`` 8.333
   * - ECF
     - Incremental
     - no
     - no
     - yes
     - ECF grading points
     - 100
   * - DWZ
     - Incremental
     - no
     - no
     - yes
     - Chess-like points
     - 400
   * - Colley Matrix
     - Global fit
     - no
     - no
     - no
     - ``[0, 1]``, sums to n/2
     - 0.5
   * - Massey
     - Global fit
     - no
     - **yes**
     - no
     - Zero-mean margin
     - 0.0
   * - Keener
     - Global fit
     - no
     - **yes**
     - no
     - Positive, mean 1.0
     - 1.0
   * - Bradley-Terry
     - Global fit
     - no
     - no
     - no
     - Elo-compatible points
     - 1500

``BlendedCompetitor`` is a weighted ensemble over any of the above rather than
a rating system in its own right, so it has no row: its behaviour on every
axis is whatever its components' behaviour is. See
:doc:`Ensemble <rating_systems/ensemble>`. ``PythagoreanCompetitor`` and
``WholeHistoryRatingCompetitor`` are newer additions not covered by this
comparison; see their own pages for their axes.

Two consequences of that table are worth stating before you read further,
because both surprise people:

- The default starting rating is **not** the same across systems — it ranges
  from 100 to 1500. If you compare systems without passing ``initial_rating``,
  part of what you measure is the difference between those defaults. Set it
  explicitly.
- For the four global-fit systems, ``initial_rating`` is a placeholder, not a
  prior. Started at 1600 against 1400, Bradley-Terry predicts 0.7597 for the
  favourite; after a single drawn game it predicts exactly 0.5000, with both
  ratings back at 1500. The fit sees results, not your beliefs about them.

One workflow, ten systems
---------------------------

The reason a comparison is cheap here is that the workflow does not change.
Feed results to an arena; read a leaderboard and an expected score out of it.
Swapping the rating system is one constructor argument:

.. code-block:: python

    from elote import EloCompetitor, GlickoCompetitor, LambdaArena

    # Your own results, always from a's point of view:
    # 1.0 means a won, 0.0 means b won, 0.5 means a draw.
    results = [
        ("ann", "bob", 1.0),
        ("bob", "cid", 1.0),
        ("ann", "cid", 0.5),
        ("dee", "ann", 1.0),
        ("dee", "bob", 1.0),
    ]


    def rate(system, **kwargs):
        arena = LambdaArena(lambda a, b: None, base_competitor=system, base_competitor_kwargs=kwargs)
        for a, b, outcome in results:
            arena.matchup(a, b, outcome=outcome)
        return arena


    elo = rate(EloCompetitor, initial_rating=1500)
    for entry in elo.leaderboard():
        print(f"{entry['competitor']:>4} {entry['rating']:8.1f}")
    print(f"Elo    P(ann beats bob) = {elo.expected_score('ann', 'bob'):.3f}")

    # Changing rating system is one argument. Nothing else about the loop changes.
    glicko = rate(GlickoCompetitor, initial_rating=1500)
    print(f"Glicko P(ann beats bob) = {glicko.expected_score('ann', 'bob'):.3f}")

.. code-block:: text

     dee   1531.9
     ann   1497.8
     bob   1485.5
     cid   1484.8
    Elo    P(ann beats bob) = 0.518
    Glicko P(ann beats bob) = 0.508

The ``lambda`` is the arena's fallback for deciding a result it was not given.
It is never called here, because every call passes an explicit ``outcome``.
See :doc:`Arenas <arenas>` for the form where the arena decides outcomes
itself.

Uncertainty
-----------

Three systems publish a per-competitor spread: Glicko-1 as ``rd``, Glicko-2 as
``rd`` plus ``volatility``, and TrueSkill as ``sigma``. The other seven give
you a number with no attached notion of how much evidence is behind it.

.. code-block:: python

    from elote import EloCompetitor, GlickoCompetitor, TrueSkillCompetitor


    def after(system, games, **kwargs):
        a, b = system(**kwargs), system(**kwargs)
        for _ in range(games):
            a.beat(b)
        return a


    print(f"{'games':>6}{'elo rating':>12}{'glicko rd':>12}{'trueskill sigma':>18}")
    for games in (0, 1, 5, 20):
        elo = after(EloCompetitor, games, initial_rating=1500)
        glicko = after(GlickoCompetitor, games, initial_rating=1500)
        trueskill = after(TrueSkillCompetitor, games)
        print(f"{games:>6}{elo.rating:>12.1f}{glicko.rd:>12.2f}{trueskill.sigma:>18.3f}")

.. code-block:: text

     games  elo rating   glicko rd   trueskill sigma
         0      1500.0      350.00             8.333
         1      1516.0      290.23             7.171
         5      1566.8      223.26             5.585
        20      1665.4      177.98             4.413

Choose one of these three when a downstream decision depends on confidence:
matchmaking that should not pair a settled competitor against an unknown one,
a leaderboard that should hold back a new entrant, or any place you would
otherwise hand-roll a "minimum games played" rule.

One caveat specific to this library rather than to the algorithms. Glicko-1
and Glicko-2 also inflate ``rd`` for competitors who have not played
recently, but the arena rejects a match timestamp earlier than the
competitor's creation time, so that decay cannot be exercised through an
arena on historical data. On a back-test it is inert.

Draws
-----

All ten systems accept a drawn result. They disagree sharply about what one
means. Starting a favourite at 1600 against 1400 and recording one draw, the
favourite's predicted score moves:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - System
     - Before
     - After one draw
   * - Elo
     - 0.7597
     - 0.7418
   * - DWZ
     - 0.7597
     - 0.6797
   * - Glicko-1
     - 0.6836
     - 0.5786
   * - Glicko-2
     - 0.6836
     - 0.5785
   * - ECF
     - 0.7034
     - 0.5000
   * - Bradley-Terry
     - 0.7597
     - 0.5000

Elo barely moves, because one result is one K-factor's worth of evidence. ECF
collapses to level, because an ECF rating is an average of recent
performance grades and there is only one of them. Bradley-Terry collapses
because it re-fits from the results alone and, with one drawn game on
record, the maximum-likelihood answer is that the two are equal.

None of those is wrong. They are different amounts of memory. If draws are a
large share of your results, that difference stops being a detail. On a
generated scenario carrying 26 drawn bouts in a 180-bout evaluation set,
against 2 in the otherwise identically sized nearly draw-free scenario, each
of Elo, Glicko-1, Glicko-2 and TrueSkill lost more than twenty points of
accuracy. That scenario also raises the generator's noise, so the drop is not
attributable to draws alone — which is itself the point: draw-heavy data is
harder in more than one way at once.

Draws are also the one place where a scoring rule matters more than accuracy
— see `Accuracy is not enough`_.

Score margins
-------------

Two systems read the optional score channel and change their ratings because
of it: Massey and Keener. The other eight accept ``scores=`` and ignore it.

.. code-block:: python

    from elote import KeenerCompetitor, LambdaArena, MasseyCompetitor

    games = [("ann", "bob", 1.0, (35, 3)), ("bob", "cid", 1.0, (17, 14)), ("ann", "cid", 1.0, (21, 20))]

    for system in (MasseyCompetitor, KeenerCompetitor):
        arena = LambdaArena(lambda a, b: None, base_competitor=system)
        for a, b, outcome, scores in games:
            arena.matchup(a, b, outcome=outcome, scores=scores)
        board = {e["competitor"]: e["rating"] for e in arena.leaderboard()}
        print(f"{system.__name__:<20} " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(board.items())))

.. code-block:: text

    MasseyCompetitor     ann=11.000  bob=-9.667  cid=-1.333
    KeenerCompetitor     ann=1.350  bob=0.735  cid=0.915

Massey's rating differences are predicted margins on the same scale as the
scores, so ``ann`` is rated about 20.7 points better than ``bob``. Keener
passes scores through a bounded, concave transform, so a blowout counts for
more than a narrow win but cannot dominate the table.

Prefer Massey when you want to predict a spread, Keener when you want scores
to inform a ranking without letting one lopsided result decide it, and
neither when your results are outcome-only — with no scores supplied, both
fall back to unit margins and Keener in particular loses most of its input.

The score channel shown here (``scores=`` on ``beat``/``lost_to``/``tied``,
and through an arena) is present in the current published release; see
:doc:`Quickstart <quickstart>` for the full contract, including validation
rules and how dataset helpers read scores from row attributes.

Incremental updates or a global fit
--------------------------------------

This is the largest structural division in the table, and it decides more
than the choice between two chess-derived formulas does.

**Incremental** systems (Elo, Glicko-1, Glicko-2, TrueSkill, ECF, DWZ) adjust
two competitors per result and forget the result afterwards. Cost per result
is constant, ratings are a running estimate, and the answer depends on the
order results arrived.

**Global-fit** systems (Colley, Massey, Keener, Bradley-Terry) re-solve the
whole connected group of competitors after every result. The answer does not
depend on order, and the cost per result grows with the size of the group.

You can watch the difference rather than take it on faith:

.. code-block:: python

    import random

    from elote import BradleyTerryCompetitor, EloCompetitor, LambdaArena

    results = [("ann", "bob", 1.0), ("bob", "cid", 1.0), ("cid", "ann", 1.0), ("ann", "dee", 1.0), ("dee", "bob", 1.0)]


    def final_ratings(system, rows, **kwargs):
        arena = LambdaArena(lambda a, b: None, base_competitor=system, base_competitor_kwargs=kwargs)
        for a, b, outcome in rows:
            arena.matchup(a, b, outcome=outcome)
        return {e["competitor"]: round(e["rating"], 6) for e in arena.leaderboard()}


    shuffled = list(results)
    random.Random(20260809).shuffle(shuffled)
    for system in (EloCompetitor, BradleyTerryCompetitor):
        same = final_ratings(system, results, initial_rating=1500) == final_ratings(system, shuffled, initial_rating=1500)
        print(f"{system.__name__:<24} same ratings after reordering the same results: {same}")

.. code-block:: text

    EloCompetitor            same ratings after reordering the same results: False
    BradleyTerryCompetitor   same ratings after reordering the same results: True

Order independence is the right property for a completed season ranked once.
It is the wrong property for a ladder, where the whole point is that a
result from last year should count for less than one from this morning.

The cost is real and it is not small. On 420 training rows of one generated
scenario, Bradley-Terry took about 9.3 seconds against Elo's roughly a
millisecond — a factor of several thousand — and accounted for most of a
ten-system sweep on its own. Refit-per-result is a fine default for hundreds
of competitors and a poor one for hundreds of thousands.

How much data you have
-----------------------

Global-fit systems need a **connected** schedule: competitors who have never
met, directly or through a chain of shared opponents, are fitted in separate
groups and cannot be meaningfully compared. Elote will still return a number
for such a pair. With two never-meeting pairs on record, Bradley-Terry
returns 0.6152 across the gap and Massey returns exactly 0.5000 — neither is
evidence about that matchup, because there is none.

Incremental systems degrade more gracefully here, but not for free: they
simply carry each competitor's prior further. On a generated sparse
scenario, where 120 competitors share 600 matchups, Elo scored around 0.64
accuracy against around 0.76 for both Glicko systems and TrueSkill. The
uncertainty-tracking systems are worth their extra complexity exactly when
data is thin, which is the case Glicko was designed for.

If your schedule is thin *and* fragmented, prefer an incremental system with
uncertainty and treat cross-group comparisons as unsupported rather than as
predictions.

Persistence
-----------

All ten systems round-trip through ``export_state()`` and ``from_state()``,
and all ten restore the current rating exactly. See
:doc:`Serialization <serialization>`.

There is one caveat the serialization page does not cover, and it follows
from the section above. The four global-fit systems persist aggregate
counts, not the match graph, so a restored competitor no longer knows *who*
it played. The restored object agrees with the live one about the present
and disagrees about the next result.

So: for an incremental system, saving state is a complete answer. For a
global-fit system, keep the results themselves as the system of record and
treat the fitted ratings as a derived artifact you can always recompute.

Run it on your own data
--------------------------

This is the part worth actually doing. Pick two or three candidates from the
table above, train them on the same split, and score them on the same
held-out results.

.. code-block:: python

    from elote import (
        BradleyTerryCompetitor,
        EloCompetitor,
        Glicko2Competitor,
        GlickoCompetitor,
        LambdaArena,
        SyntheticDataset,
        TrueSkillCompetitor,
        train_and_evaluate_arena,
    )

    SHORTLIST = {
        "Elo": (EloCompetitor, {"initial_rating": 1500}),
        "Glicko-1": (GlickoCompetitor, {"initial_rating": 1500}),
        "Glicko-2": (Glicko2Competitor, {"initial_rating": 1500}),
        "TrueSkill": (TrueSkillCompetitor, {}),
        "Bradley-Terry": (BradleyTerryCompetitor, {"initial_rating": 1500}),
    }


    def compare(split):
        """Train every shortlisted system on the same split and score it on the same held-out bouts."""
        rows = []
        for name, (system, kwargs) in SHORTLIST.items():
            arena = LambdaArena(
                lambda a, b, attributes=None: True,  # unused: outcomes come from the data
                base_competitor=system,
                base_competitor_kwargs=kwargs,
            )
            _, history = train_and_evaluate_arena(arena, split)
            scored = [b for b in history.bouts if b.predicted_outcome is not None]
            brier = sum((b.predicted_outcome - b.outcome) ** 2 for b in scored) / len(scored)
            out_of_range = sum(1 for b in scored if not 0.0 <= b.predicted_outcome <= 1.0)
            accuracy = history.calculate_metrics(0.45, 0.55)["accuracy"]
            rows.append((name, accuracy, brier, out_of_range, len(scored)))

        print(f"{'system':<14}{'accuracy':>10}{'brier':>9}{'outside [0,1]':>15}{'bouts':>7}")
        for name, accuracy, brier, out_of_range, n in sorted(rows, key=lambda r: r[2]):
            print(f"{name:<14}{accuracy:>10.4f}{brier:>9.4f}{out_of_range:>15d}{n:>7d}")


    if __name__ == "__main__":
        # Replace this with your own rows: (a, b, outcome, timestamp, attributes),
        # where outcome is 1.0 if a won, 0.0 if b won, and 0.5 for a draw.
        dataset = SyntheticDataset(num_competitors=30, num_matchups=600, seed=20260809)
        dataset.load()
        compare(dataset.time_split(test_ratio=0.3))

.. code-block:: text

    system          accuracy    brier  outside [0,1]  bouts
    Bradley-Terry     0.9222   0.0539              0    180
    TrueSkill         0.9111   0.0570              0    180
    Glicko-2          0.9167   0.0581              0    180
    Glicko-1          0.9167   0.0567              0    180
    Elo               0.8778   0.0900              0    180

Three things about that harness are deliberate, and you should keep them
when you swap in your own results:

- **The split is time-ordered, not random.** A random split lets a system
  learn from a competitor's future when predicting its past.
- **Every system sees the identical split** and starts from the same rating
  where it accepts one, so what varies between rows is the algorithm.
- **Predictions outside** ``[0, 1]`` **are counted, not clipped away.** A
  rating system returning an impossible probability is a defect, and a
  metric that quietly clips it hides that.

To use your own history, replace the last three lines. ``train_and_evaluate_arena``
takes a ``DataSplit``, and a ``DataSplit`` is two ordered lists of
``(a, b, outcome, timestamp, attributes)`` rows — no dataset class required:

.. code-block:: python

    import datetime as dt

    from elote import DataSplit, EloCompetitor, LambdaArena, train_and_evaluate_arena


    def row(a, b, outcome, day):
        return (a, b, outcome, dt.datetime(2026, 1, day), None)


    split = DataSplit(
        train=[row("ann", "bob", 1.0, 1), row("bob", "cid", 1.0, 2), row("ann", "cid", 1.0, 3), row("dee", "ann", 1.0, 4)],
        test=[row("ann", "bob", 1.0, 5), row("cid", "dee", 0.0, 6)],
    )
    arena = LambdaArena(
        lambda a, b, attributes=None: True,
        base_competitor=EloCompetitor,
        base_competitor_kwargs={"initial_rating": 1500},
    )
    _, history = train_and_evaluate_arena(arena, split)
    print(len(history.bouts), [round(b.predicted_outcome, 4) for b in history.bouts])

.. code-block:: text

    2 [0.5178, 0.4305]

Sort your rows by time before splitting them, and put the split point at a
date rather than at a row count if your results arrive in bursts.

Accuracy is not enough
------------------------

Accuracy answers "did the favourite win". It is blind to any reshaping of a
probability that keeps the ordering, so a system that says 0.99 when it
should say 0.60 scores the same. A proper scoring rule — Brier score or log
loss — is not blind to it, and Elote's expected scores are numbers people
use downstream.

The two genuinely disagree. On a generated sparse scenario, one system
ranked first of ten on accuracy and seventh of ten on Brier score. And in one
measured case a version of a system with a known prediction defect scored
*higher* on accuracy than the fixed version while being worse on Brier
score, worse on log loss, and returning dozens of impossible probabilities.

Report at least accuracy, one proper scoring rule, and a count of
out-of-range predictions. If two candidates are close on all three, they are
close, and you should pick on the other axes in this guide.

What this page does not tell you
-----------------------------------

- The figures here come from a **generated** dataset that samples outcomes
  from a latent per-competitor strength — the model most of these systems
  assume. Their agreement with it is an upper bound, not evidence about
  messy data.
- Nothing here compares Elote to another library. Every runtime figure
  compares Elote's own systems to each other.
- The comparison measures prediction on held-out results only: not recovery
  of a known true ordering, not convergence speed, not calibration after
  binning, and not behaviour under adversarial scheduling.
- The library's own ``benchmark_competitors`` helper has had known issues
  running Massey in the past because of how it enforces a minimum rating.
  Drive the arena and dataset helpers directly, as above, if you hit that.

Where to go next
--------------------

- :doc:`Elo vs Glicko vs TrueSkill <rating_systems/elo_vs_glicko_vs_trueskill>`
  — the three systems people name, compared with sourced mathematics and
  measured results.
- :doc:`Comparing rating systems <rating_systems/comparison>` — the
  per-system reference: origins, formulas, parameters, strengths and
  weaknesses.
- :doc:`Quickstart <quickstart>` — the shortest path from a list of results
  to a leaderboard.
- :doc:`Arenas <arenas>` and :doc:`Serialization <serialization>` — running
  matchups and persisting state.
- The system pages: :doc:`Elo <rating_systems/elo>`,
  :doc:`Glicko <rating_systems/glicko>`,
  :doc:`Glicko-2 <rating_systems/glicko2>`,
  :doc:`TrueSkill <rating_systems/trueskill>`,
  :doc:`ECF <rating_systems/ecf>`, :doc:`DWZ <rating_systems/dwz>`,
  :doc:`Colley <rating_systems/colley>`, :doc:`Massey <rating_systems/massey>`,
  :doc:`Keener <rating_systems/keener>`,
  :doc:`Bradley-Terry <rating_systems/bradley_terry>`,
  :doc:`Pythagorean <rating_systems/pythagorean>`,
  :doc:`Whole-History Rating <rating_systems/whr>`.
