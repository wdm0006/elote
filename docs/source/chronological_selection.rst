Chronological tuning and final evaluation
=========================================

Reserve later periods before searching for parameters. Here we use six local,
deterministic weekly periods with point scores, comparing small Elo and
Pythagorean grids. The first four periods are development data; the last two
remain untouched by selection. This illustrative schedule establishes no
universally best rating system.

Run the example from a checkout with the base package installed::

    uv run --no-sync python examples/chronological_selection.py

No dataset downloads, plotting packages or optional extras are needed.

The executable source
---------------------

.. literalinclude:: ../../examples/chronological_selection.py
   :language: python

Reading the protocol
--------------------

``group_by_period`` orders the timestamped rows into ISO weeks. Tuning uses only
periods 0-3 and an explicit one-period initial warmup. Each grid candidate scores
five decisive development bouts. ``tune`` ranks configurations by development
log loss; the example then chooses across both systems using that same metric.
These are **selection scores**, reused to make the choice, not independent final
evidence. Class parameters belong in the grids; constructor options, if needed,
would be supplied separately through ``base_competitor_kwargs``.

Once selected, the system and parameters are fixed. The final ``walk_forward``
starts from fresh ratings, replays the complete schedule, and uses all four
development periods as warmup. Only periods 4-5 contribute to the final metrics.
Using the initial tuning warmup here would incorrectly include development
predictions in the final report.

Every prediction within a period uses ratings from earlier periods: no result
in that period informs another prediction in the same period. After scoring,
the whole period is learned. This also happens on the final holdout: parameters
stay fixed while ratings evolve, so this is adaptive predict-then-learn
evaluation. It differs from predicting an entire holdout with frozen ratings.
The holdout is untouched by search, not withheld forever from rating updates.

``score_keys`` forwards the custom point-score attributes through tuning and
final replay. Pythagorean uses those points; the default Elo configuration here
learns win/loss/draw outcomes without margin scaling.

The final population is exactly four predictions, one skipped bout and one draw.
Team E first appears in period 4, so its decisive bout is skipped because it has
no prior rating; it is learned and can be scored in period 5. Draws are also
learned but excluded from accuracy, log loss and Brier. Warmup bouts contribute
to fitting, not to these reported counts. The output prints period dates,
scoring boundaries, warmup, protocol and population beside every metric.

Keep the final report separate from selection scores, and do not revise the
choice after inspecting it. If you change the search based on final results,
reserve new later data for another independent evaluation.
