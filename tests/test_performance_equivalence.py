"""Equivalence guards and benchmarks for the hot-path performance refactors.

Two semantics-preserving changes are pinned here:

1. :class:`~elote.EloCompetitor` ``beat``/``tied`` compute the opponent's expected score
   as the logistic complement ``1 - win_es`` instead of a second ``10 **`` evaluation.
   The complement must agree with the direct two-pow form to <= 1e-12 across a grid of
   fixture rating pairs.

2. :class:`~elote.ColleyMatrixCompetitor`, :class:`~elote.MasseyCompetitor` and
   :class:`~elote.KeenerCompetitor` defer their matrix/eigenvector fit to the first
   rating read (dirty-flag caching) instead of re-fitting after every recorded game.
   Ratings from the deferred fit must match ratings from the eager per-game fit -- the
   historical behavior, reproduced here by reading after every game -- to <= 1e-12, and
   repeated reads with no new games must be exactly stable.
"""

import random

import pytest

from elote import ColleyMatrixCompetitor, EloCompetitor, KeenerCompetitor, MasseyCompetitor

MATRIX_SYSTEMS = [ColleyMatrixCompetitor, MasseyCompetitor, KeenerCompetitor]

# Rating pairs spanning the Elo scale: defaults, known-value fixtures, extremes, and a
# pair whose loser is clamped at the minimum rating.
ELO_RATING_PAIRS = [
    (400, 400),
    (1200, 1000),
    (1000, 1200),
    (1500, 1500),
    (1500, 1200),
    (2000, 1000),
    (2700, 1200),
    (100, 900),
    (100, 200),
    (2500, 2400),
]

# A deterministic 8-competitor schedule: everyone connected, mixed beat/lost_to/tied,
# played twice (second leg reverses the results so no competitor goes undefeated).
SCHEDULE = [
    (0, 1, "beat"),
    (2, 3, "beat"),
    (4, 5, "beat"),
    (6, 7, "beat"),
    (1, 2, "lost"),
    (3, 4, "lost"),
    (5, 6, "lost"),
    (7, 0, "lost"),
    (0, 3, "beat"),
    (1, 4, "beat"),
    (2, 5, "beat"),
    (6, 3, "beat"),
    (5, 1, "lost"),
    (7, 2, "lost"),
    (4, 0, "lost"),
    (3, 6, "lost"),
    (0, 2, "tied"),
    (4, 6, "tied"),
    (1, 3, "tied"),
]


def _play(system, incremental, use_scores):
    """Build an 8-competitor population and play the fixed schedule on it.

    With ``incremental`` every game is followed by a forced rating read on every
    competitor, which reproduces the historical eager re-fit behavior (one matrix solve
    per recorded game). Without it, all games are recorded first and the fit is forced
    exactly once at the end -- the deferred/cached behavior.
    """
    comps = [system() for _ in range(8)]
    scores = None
    draw_scores = None
    if use_scores:
        scores = (3.0, 1.0)
        draw_scores = (2.0, 2.0)

    for winner, loser, kind in SCHEDULE:
        if kind == "beat":
            comps[winner].beat(comps[loser], scores=scores)
        elif kind == "lost":
            comps[loser].lost_to(comps[winner], scores=None if scores is None else (scores[1], scores[0]))
        else:
            comps[winner].tied(comps[loser], scores=None if draw_scores is None else draw_scores)
        if incremental:
            _ = [c.rating for c in comps]

    _ = [c.rating for c in comps]
    return comps


# ---------------------------------------------------------------------------
# Elo: single-pow beat/tied via the logistic complement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("rating_a", "rating_b"), ELO_RATING_PAIRS)
def test_elo_expected_scores_are_complementary(rating_a, rating_b):
    """lose_es must equal 1 - win_es to <= 1e-12: the identity the single-pow beat relies on."""
    a = EloCompetitor(initial_rating=rating_a)
    b = EloCompetitor(initial_rating=rating_b)

    win_es = a.expected_score(b)
    lose_es = b.expected_score(a)

    assert abs(win_es + lose_es - 1.0) <= 1e-12


@pytest.mark.parametrize(("rating_a", "rating_b"), ELO_RATING_PAIRS)
def test_elo_beat_matches_double_pow_reference(rating_a, rating_b):
    """beat() must reproduce the historical two-pow update to <= 1e-12."""
    actual_a = EloCompetitor(initial_rating=rating_a)
    actual_b = EloCompetitor(initial_rating=rating_b)
    ref_a = EloCompetitor(initial_rating=rating_a)
    ref_b = EloCompetitor(initial_rating=rating_b)

    # Reference: the historical form, spelled out -- two independent expected-score
    # evaluations (two 10 ** evaluations) feeding the same K-factor update.
    win_es = ref_a.expected_score(ref_b)
    lose_es = ref_b.expected_score(ref_a)
    ref_a_rating = ref_a._new_rating(1.0, win_es)
    ref_b_rating = ref_b._new_rating(0.0, lose_es)

    actual_a.beat(actual_b)

    assert abs(actual_a.rating - ref_a_rating) <= 1e-12
    assert abs(actual_b.rating - ref_b_rating) <= 1e-12


@pytest.mark.parametrize(("rating_a", "rating_b"), ELO_RATING_PAIRS)
def test_elo_tied_matches_double_pow_reference(rating_a, rating_b):
    """tied() must reproduce the historical two-pow update to <= 1e-12."""
    actual_a = EloCompetitor(initial_rating=rating_a)
    actual_b = EloCompetitor(initial_rating=rating_b)
    ref_a = EloCompetitor(initial_rating=rating_a)
    ref_b = EloCompetitor(initial_rating=rating_b)

    win_es = ref_a.expected_score(ref_b)
    lose_es = ref_b.expected_score(ref_a)
    ref_a_rating = ref_a._new_rating(0.5, win_es)
    ref_b_rating = ref_b._new_rating(0.5, lose_es)

    actual_a.tied(actual_b)

    assert abs(actual_a.rating - ref_a_rating) <= 1e-12
    assert abs(actual_b.rating - ref_b_rating) <= 1e-12


@pytest.mark.parametrize(("rating_a", "rating_b"), ELO_RATING_PAIRS)
@pytest.mark.parametrize("k_factor", [8, 17])
def test_elo_single_pow_holds_for_custom_k_factors(rating_a, rating_b, k_factor):
    """The complement identity is independent of the K-factor."""
    actual_a = EloCompetitor(initial_rating=rating_a, k_factor=k_factor)
    actual_b = EloCompetitor(initial_rating=rating_b, k_factor=k_factor)
    ref_a = EloCompetitor(initial_rating=rating_a, k_factor=k_factor)
    ref_b = EloCompetitor(initial_rating=rating_b, k_factor=k_factor)

    win_es = ref_a.expected_score(ref_b)
    lose_es = ref_b.expected_score(ref_a)
    ref_a_rating = ref_a._new_rating(1.0, win_es)
    ref_b_rating = ref_b._new_rating(0.0, lose_es)

    actual_a.beat(actual_b)

    assert abs(actual_a.rating - ref_a_rating) <= 1e-12
    assert abs(actual_b.rating - ref_b_rating) <= 1e-12


# ---------------------------------------------------------------------------
# Matrix systems: dirty-flag caching of the deferred fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", MATRIX_SYSTEMS)
@pytest.mark.parametrize("use_scores", [False, True], ids=["unit_scores", "real_scores"])
def test_matrix_cached_ratings_match_eager_per_game_fits(system, use_scores):
    """Deferred (cached) fits must equal eager per-game fits to <= 1e-12, ratings and expected scores."""
    eager = _play(system, incremental=True, use_scores=use_scores)
    cached = _play(system, incremental=False, use_scores=use_scores)

    for eager_comp, cached_comp in zip(eager, cached, strict=True):
        assert abs(eager_comp.rating - cached_comp.rating) <= 1e-12
        # Reads must not mutate the recorded match record.
        assert eager_comp.num_games == cached_comp.num_games

    for i in range(len(eager)):
        for j in range(len(eager)):
            if i != j:
                eager_es = eager[i].expected_score(eager[j])
                cached_es = cached[i].expected_score(cached[j])
                assert abs(eager_es - cached_es) <= 1e-12


@pytest.mark.parametrize("system", MATRIX_SYSTEMS)
def test_matrix_repeated_reads_are_exactly_stable(system):
    """Repeated rating reads with no new games return bit-identical values (O(1) amortized reads)."""
    comps = _play(system, incremental=False, use_scores=False)
    first = [c.rating for c in comps]
    games_before = [c.num_games for c in comps]

    for _ in range(10):
        again = [c.rating for c in comps]
        assert again == first

    assert [c.num_games for c in comps] == games_before


@pytest.mark.parametrize("system", MATRIX_SYSTEMS)
def test_matrix_new_game_after_read_invalidates_cache(system):
    """A game recorded after a read must be reflected in the next read."""
    comps = [system() for _ in range(4)]
    comps[0].beat(comps[1])
    _ = comps[0].rating  # forces the fit for the current state

    stale = [c.rating for c in comps]
    comps[2].beat(comps[0])  # new game after the read
    fresh = [c.rating for c in comps]

    assert any(a != b for a, b in zip(stale, fresh, strict=True))
    # And a third read without new games is stable again.
    assert [c.rating for c in comps] == fresh


# ---------------------------------------------------------------------------
# Benchmarks (pytest-benchmark): run with --benchmark-compare against the
# t5_before run taken on the pre-refactor code for the before/after tables.
# ---------------------------------------------------------------------------


def test_benchmark_elo_beat(benchmark):
    """Elo beat: one 10 ** evaluation per update (complement) instead of two."""

    a = EloCompetitor(initial_rating=1500)
    b = EloCompetitor(initial_rating=1200)

    def beat():
        a.beat(b)
        return a.rating

    result = benchmark.pedantic(beat, rounds=1000)
    assert result > 0


def test_benchmark_elo_tied(benchmark):
    """Elo tied: one 10 ** evaluation per update (complement) instead of two."""

    a = EloCompetitor(initial_rating=1500)
    b = EloCompetitor(initial_rating=1200)

    def tied():
        a.tied(b)
        return a.rating

    result = benchmark.pedantic(tied, rounds=1000)
    assert result > 0


@pytest.mark.parametrize("system", MATRIX_SYSTEMS)
def test_benchmark_matrix_record_games(benchmark, system):
    """Per-game recording cost: a dirty-flag mark instead of a full matrix fit per game."""
    rng = random.Random(20240907)
    games = [(rng.randrange(30), rng.randrange(30)) for _ in range(100)]
    comps = [system() for _ in range(30)]

    def play():
        for winner, loser in games:
            comps[winner].beat(comps[loser])
        return comps[0].rating

    result = benchmark.pedantic(play, rounds=5)
    assert result is not None


@pytest.mark.parametrize("system", MATRIX_SYSTEMS)
def test_benchmark_matrix_repeated_rating_reads(benchmark, system):
    """Repeated rating reads with no new games: one deferred fit, then O(1) per read."""
    comps = [system() for _ in range(30)]
    rng = random.Random(20240907)
    for _ in range(200):
        winner, loser = rng.randrange(30), rng.randrange(30)
        comps[winner].beat(comps[loser])

    def read_all():
        return sum(c.rating for c in comps)

    result = benchmark.pedantic(read_all, rounds=200)
    # Sanity only: each fit pins its own scale (Colley sums to n/2, Massey to a zero
    # mean, Keener to a mean of 1), so the expected sum of 30 ratings is per-system.
    expected_sum = 30 * {ColleyMatrixCompetitor: 0.5, MasseyCompetitor: 0.0, KeenerCompetitor: 1.0}[system]
    assert result == pytest.approx(expected_sum, abs=1e-3)
