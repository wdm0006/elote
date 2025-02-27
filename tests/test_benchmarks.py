import random
import pytest
from elote import EloCompetitor, GlickoCompetitor, ECFCompetitor, DWZCompetitor, BlendedCompetitor, LambdaArena


# Utility functions for benchmarks
def generate_random_scores(n=100):
    """Generate random scores for benchmarking."""
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]


def generate_random_competitors(competitor_class, n=100):
    """Generate random competitors of a given class."""
    return [competitor_class(initial_rating=random.randint(1000, 2000)) for _ in range(n)]


# Benchmark individual competitor operations
@pytest.mark.parametrize(
    "competitor_class",
    [
        EloCompetitor,
        GlickoCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    ],
)
def test_competitor_expected_score(benchmark, competitor_class):
    """Benchmark the expected_score method for different competitor types."""

    def setup():
        comp1 = competitor_class(initial_rating=1500)
        comp2 = competitor_class(initial_rating=1200)
        return comp1, comp2

    def expected_score(comp1, comp2):
        return comp1.expected_score(comp2)

    result = benchmark.pedantic(expected_score, args=setup(), rounds=1000)
    assert result > 0


@pytest.mark.parametrize(
    "competitor_class",
    [
        EloCompetitor,
        GlickoCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    ],
)
def test_competitor_beat(benchmark, competitor_class):
    """Benchmark the beat method for different competitor types."""

    def setup():
        # Use higher initial ratings to prevent negative ratings
        comp1 = competitor_class(initial_rating=1500)
        comp2 = competitor_class(initial_rating=1200)
        return comp1, comp2

    def beat(comp1, comp2):
        comp1.beat(comp2)
        return comp1.rating

    result = benchmark.pedantic(beat, args=setup(), rounds=1000)
    assert result > 0


@pytest.mark.parametrize(
    "competitor_class",
    [
        EloCompetitor,
        GlickoCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    ],
)
def test_competitor_tied(benchmark, competitor_class):
    """Benchmark the tied method for different competitor types."""

    def setup():
        comp1 = competitor_class(initial_rating=1500)
        comp2 = competitor_class(initial_rating=1200)
        return comp1, comp2

    def tied(comp1, comp2):
        comp1.tied(comp2)
        return comp1.rating

    result = benchmark.pedantic(tied, args=setup(), rounds=1000)
    assert result > 0


# Benchmark blended competitor operations
def test_blended_competitor_expected_score(benchmark):
    """Benchmark the expected_score method for BlendedCompetitor."""

    def setup():
        comp1 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
            ]
        )
        comp2 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1200}},
            ]
        )
        return comp1, comp2

    def expected_score(comp1, comp2):
        return comp1.expected_score(comp2)

    result = benchmark.pedantic(expected_score, args=setup(), rounds=1000)
    assert result > 0


def test_blended_competitor_beat(benchmark):
    """Benchmark the beat method for BlendedCompetitor."""

    def setup():
        # Use higher initial ratings to prevent negative ratings
        comp1 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
            ]
        )
        comp2 = BlendedCompetitor(
            competitors=[
                {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1200}},
                {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1200}},
            ]
        )
        return comp1, comp2

    def beat(comp1, comp2):
        comp1.beat(comp2)
        return comp1.rating

    result = benchmark.pedantic(beat, args=setup(), rounds=1000)
    assert result > 0


# Benchmark arena operations
@pytest.mark.parametrize(
    "competitor_class",
    [
        EloCompetitor,
        GlickoCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    ],
)
def test_arena_matchup(benchmark, competitor_class):
    """Benchmark the matchup method for different competitor types in an arena."""

    def setup():
        # Use appropriate initial ratings for each competitor type
        base_competitor_kwargs = {}
        if competitor_class == ECFCompetitor:
            base_competitor_kwargs = {"initial_rating": 100}  # Minimum rating for ECF

        arena = LambdaArena(
            lambda a, b: a > b, base_competitor=competitor_class, base_competitor_kwargs=base_competitor_kwargs
        )
        return arena

    def matchup(arena):
        for _ in range(100):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            arena.matchup(a, b)
        return len(arena.competitors)

    result = benchmark.pedantic(matchup, args=(setup(),), rounds=10)
    assert result > 0


@pytest.mark.parametrize(
    "competitor_class",
    [
        EloCompetitor,
        GlickoCompetitor,
        ECFCompetitor,
        DWZCompetitor,
    ],
)
def test_arena_tournament(benchmark, competitor_class):
    """Benchmark the tournament method for different competitor types in an arena."""

    def setup():
        # Use a base_competitor_kwargs to set higher initial ratings
        arena = LambdaArena(
            lambda a, b: a > b, base_competitor=competitor_class, base_competitor_kwargs={"initial_rating": 1500}
        )
        competitors = list(range(1, 21))  # 20 competitors
        return arena, competitors

    def tournament(arena, competitors):
        arena.tournament([(a, b) for a in competitors for b in competitors if a != b])
        return len(arena.history.bouts)

    result = benchmark.pedantic(tournament, args=setup(), rounds=10)
    assert result > 0


# Benchmark large-scale operations
def test_large_tournament(benchmark):
    """Benchmark a large tournament with many competitors."""

    def setup():
        arena = LambdaArena(lambda a, b: a > b)
        competitors = list(range(1, 101))  # 100 competitors
        return arena, competitors

    def large_tournament(arena, competitors):
        # Create matchups for the tournament
        matchups = [(a, b) for a in competitors[:10] for b in competitors[10:20] if a != b]
        arena.tournament(matchups)
        return len(arena.history.bouts)

    result = benchmark.pedantic(large_tournament, args=setup(), rounds=3)
    assert result > 0


def test_many_sequential_matchups(benchmark):
    """Benchmark many sequential matchups."""

    def setup():
        arena = LambdaArena(lambda a, b: a > b)
        matchups = [(random.randint(1, 1000), random.randint(1, 1000)) for _ in range(500)]
        return arena, matchups

    def many_matchups(arena, matchups):
        for a, b in matchups:
            arena.matchup(a, b)
        return len(arena.history.bouts)

    result = benchmark.pedantic(many_matchups, args=setup(), rounds=3)
    assert result > 0
