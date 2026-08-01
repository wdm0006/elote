import itertools
import unittest

from elote import EloCompetitor, GlickoCompetitor, LambdaArena
from elote.competitors.base import InvalidRatingValueException, MissMatchedCompetitorTypesException


class TestElo(unittest.TestCase):
    def test_Improvement(self):
        initial_rating = 100
        player1 = EloCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = EloCompetitor(initial_rating=800)
            player1.beat(player2)
            self.assertGreater(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Decay(self):
        initial_rating = 800
        player1 = EloCompetitor(initial_rating=initial_rating)

        # if player1 beats someone with a high rating, their rating should go up.
        for _ in range(10):
            player2 = EloCompetitor(initial_rating=100)
            player2.beat(player1)
            self.assertLess(player1.rating, initial_rating)
            initial_rating = player1.rating

    def test_Expectation(self):
        player1 = EloCompetitor(initial_rating=1000)
        player2 = EloCompetitor(initial_rating=100)
        self.assertGreater(player1.expected_score(player2), player2.expected_score(player1))

    def test_Exceptions(self):
        player1 = EloCompetitor(initial_rating=1000)
        player2 = GlickoCompetitor(initial_rating=100)

        with self.assertRaises(MissMatchedCompetitorTypesException):
            player1.verify_competitor_types(player2)


class TestEloMinimumRating(unittest.TestCase):
    """Behaviour of EloCompetitor at the _minimum_rating floor."""

    def test_weakest_competitor_saturates_at_the_floor(self):
        """A long arena run with a clearly weakest member saturates instead of raising."""
        pairs = list(itertools.permutations([1, 2, 3, 4], 2))
        matchups = [pairs[i % len(pairs)] for i in range(200)]

        arena = LambdaArena(lambda a, b: a > b)
        arena.tournament(matchups)

        ratings = {entry["competitor"]: entry["rating"] for entry in arena.leaderboard()}
        self.assertEqual(ratings[1], EloCompetitor._minimum_rating)
        self.assertGreater(ratings[2], EloCompetitor._minimum_rating)
        self.assertGreater(ratings[3], ratings[2])
        self.assertGreater(ratings[4], ratings[3])

    def test_both_participants_saturate_at_the_floor(self):
        """Neither side of a match may be driven below the floor, whichever is the caller."""
        winner = EloCompetitor(initial_rating=EloCompetitor._minimum_rating)
        loser = EloCompetitor(initial_rating=EloCompetitor._minimum_rating)

        # The loser is already at the floor and would otherwise go below it.
        winner.beat(loser)
        self.assertEqual(loser.rating, EloCompetitor._minimum_rating)
        self.assertGreater(winner.rating, EloCompetitor._minimum_rating)

        # The caller's own side is only reachable through a draw it was favoured to win,
        # and needs a large K-factor to cross the floor in one step.
        favourite = EloCompetitor(initial_rating=EloCompetitor._minimum_rating + 10, k_factor=1000)
        underdog = EloCompetitor(initial_rating=EloCompetitor._minimum_rating, k_factor=1000)
        favourite.tied(underdog)
        self.assertEqual(favourite.rating, EloCompetitor._minimum_rating)

    def test_explicit_assignment_below_the_floor_still_raises(self):
        """The rating setter keeps rejecting explicit out-of-range assignments."""
        player = EloCompetitor(initial_rating=400)

        with self.assertRaises(InvalidRatingValueException):
            player.rating = EloCompetitor._minimum_rating - 1

        self.assertEqual(player.rating, 400)

    def test_ratings_away_from_the_floor_are_unchanged(self):
        """Ordinary Elo arithmetic is untouched: equal 400s exchange exactly K/2."""
        winner = EloCompetitor(initial_rating=400, k_factor=32)
        loser = EloCompetitor(initial_rating=400, k_factor=32)

        winner.beat(loser)

        self.assertAlmostEqual(winner.rating, 416.0)
        self.assertAlmostEqual(loser.rating, 384.0)
