import unittest

from elote import BradleyTerryCompetitor, EloCompetitor
from elote.competitors.base import (
    BaseCompetitor,
    MissMatchedCompetitorTypesException,
    InvalidParameterException,
)


class TestBradleyTerryCompetitor(unittest.TestCase):
    """Behavioral / interface tests for BradleyTerryCompetitor."""

    def test_initial_rating(self):
        player = BradleyTerryCompetitor()
        self.assertAlmostEqual(player.rating, 1500.0)

        player = BradleyTerryCompetitor(initial_rating=1600)
        self.assertAlmostEqual(player.rating, 1600.0)

    def test_rating_setter(self):
        player = BradleyTerryCompetitor()
        player.rating = 1700
        self.assertAlmostEqual(player.rating, 1700.0)

    def test_expected_score_symmetry(self):
        a = BradleyTerryCompetitor(initial_rating=1600)
        b = BradleyTerryCompetitor(initial_rating=1400)
        self.assertAlmostEqual(a.expected_score(b) + b.expected_score(a), 1.0)

        equal_a = BradleyTerryCompetitor()
        equal_b = BradleyTerryCompetitor()
        self.assertAlmostEqual(equal_a.expected_score(equal_b), 0.5)

    def test_expected_score_matches_elo(self):
        # Bradley-Terry with scale 400/ln(10) has the same expected-score form as Elo.
        bt_a = BradleyTerryCompetitor(initial_rating=1650)
        bt_b = BradleyTerryCompetitor(initial_rating=1480)
        elo_a = EloCompetitor(initial_rating=1650)
        elo_b = EloCompetitor(initial_rating=1480)
        self.assertAlmostEqual(bt_a.expected_score(bt_b), elo_a.expected_score(elo_b))

    def test_beat_updates_ratings(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        old_a, old_b = a.rating, b.rating
        a.beat(b)
        # A global re-fit changes both ratings; the winner ends up ahead of the loser.
        self.assertNotEqual(a.rating, old_a)
        self.assertNotEqual(b.rating, old_b)
        self.assertGreater(a.rating, b.rating)

    def test_lost_to(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.lost_to(b)
        self.assertGreater(b.rating, a.rating)

    def test_tied(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.tied(b)
        # Symmetric tie leaves the two competitors on equal footing.
        self.assertAlmostEqual(a.rating, b.rating, places=5)
        self.assertEqual(a._ties, 1)
        self.assertEqual(b._ties, 1)

    def test_win_loss_tie_counts(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        a.beat(b)
        b.beat(a)
        a.tied(b)
        self.assertEqual(a._wins, 1)
        self.assertEqual(a._losses, 1)
        self.assertEqual(a._ties, 1)
        self.assertEqual(a.num_games, 3)

    def test_undefeated_stays_finite(self):
        a = BradleyTerryCompetitor()
        b = BradleyTerryCompetitor()
        c = BradleyTerryCompetitor()
        a.beat(b)
        a.beat(c)
        a.beat(b)
        self.assertTrue(abs(a.rating) < float("inf"))
        self.assertGreater(a.rating, b.rating)
        self.assertGreater(a.rating, c.rating)

    def test_reset(self):
        a = BradleyTerryCompetitor(initial_rating=1550)
        b = BradleyTerryCompetitor()
        a.beat(b)
        a.reset()
        self.assertAlmostEqual(a.rating, 1550.0)
        self.assertEqual(a._wins, 0)
        self.assertEqual(a._losses, 0)
        self.assertEqual(a._opponents, {})

    def test_comparison_operators(self):
        a = BradleyTerryCompetitor(initial_rating=1400)
        b = BradleyTerryCompetitor(initial_rating=1600)
        self.assertTrue(a < b)
        self.assertTrue(b > a)
        self.assertFalse(a == b)

    def test_serialization_roundtrip(self):
        a = BradleyTerryCompetitor(initial_rating=1550)
        b = BradleyTerryCompetitor()
        a.beat(b)
        a.beat(b)
        b.beat(a)

        state = a.export_state()
        restored = BaseCompetitor.from_state(state)
        self.assertIsInstance(restored, BradleyTerryCompetitor)
        self.assertAlmostEqual(restored.rating, a.rating)
        self.assertEqual(restored._wins, 2)
        self.assertEqual(restored._losses, 1)

    def test_json_roundtrip(self):
        a = BradleyTerryCompetitor(initial_rating=1520)
        json_str = a.to_json()
        restored = BradleyTerryCompetitor.from_json(json_str)
        self.assertAlmostEqual(restored.rating, a.rating)

    def test_type_mismatch_raises(self):
        a = BradleyTerryCompetitor()
        with self.assertRaises(MissMatchedCompetitorTypesException):
            a.expected_score(EloCompetitor())
        with self.assertRaises(MissMatchedCompetitorTypesException):
            a.beat(EloCompetitor())

    def test_configure_class_validation(self):
        with self.assertRaises(InvalidParameterException):
            BradleyTerryCompetitor.configure_class(reg=-1)
        with self.assertRaises(InvalidParameterException):
            BradleyTerryCompetitor.configure_class(max_iter=0)
        with self.assertRaises(InvalidParameterException):
            BradleyTerryCompetitor.configure_class(scale=0)


if __name__ == "__main__":
    unittest.main()
