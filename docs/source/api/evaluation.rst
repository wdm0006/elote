Evaluation API Reference
========================

Walk-forward evaluation
-----------------------

``benchmark_competitors`` compares systems on one frozen train/test split. Walk-forward
evaluation predicts each period from everything before it and then learns that period, and it
reports log loss and Brier score alongside accuracy. ``compare_walk_forward`` runs that
protocol for several systems over the same periods and warmup, so the numbers are comparable.

.. code-block:: python

    from elote import (
        EloCompetitor, GlickoBoostCompetitor, SyntheticDataset,
        compare_walk_forward, group_by_period,
    )

    periods = group_by_period(SyntheticDataset(num_competitors=20, num_matchups=1000, seed=1).load())
    result = compare_walk_forward(
        {
            "elo": EloCompetitor,
            "elo_k32": (EloCompetitor, {"competitor_params": {"k_factor": 32}}),
            "glicko_boost": GlickoBoostCompetitor,
        },
        periods,
        warmup=4,
    )
    print(result)
    result.ranking()[0]["system"]  # best log loss

Each system is a class, or a ``(class, options)`` pair where ``options`` may hold
``competitor_params`` and ``base_competitor_kwargs``. Invalid systems raise before anything is
evaluated. Walk-forward numbers are not interchangeable with frozen-split numbers: they score
a different population of bouts, and ``warmup`` decides how much early history is excluded.
``same_population`` is ``False`` when systems scored different bout counts.

.. automodule:: elote.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
