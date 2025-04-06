import unittest
from elote import DWZCompetitor
import math


class TestDWZKnownValues(unittest.TestCase):
    """Tests for DWZCompetitor with known values to verify correctness after optimization."""

    def test_initial_rating(self):
        """Test that initial rating is set correctly."""
        player = DWZCompetitor(initial_rating=400)
        self.assertEqual(player.rating, 400)

        player = DWZCompetitor(initial_rating=1000)
        self.assertEqual(player.rating, 1000)

    def test_expected_score(self):
        """Test expected_score with known values."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Calculate expected value manually
        expected = 1 / (1 + 10 ** ((1200 - 1000) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected)

        # Test with different ratings
        player1 = DWZCompetitor(initial_rating=1500)
        player2 = DWZCompetitor(initial_rating=1300)

        expected = 1 / (1 + 10 ** ((1300 - 1500) / 400))

        self.assertAlmostEqual(player1.expected_score(player2), expected)

    def test_calculate_E_method(self):
        """Test the internal _calculate_E method with various scenarios."""
        # Scenario 1: Teenager (<=20), winning (W_a > W_e), rating > 1300
        player_teen_win = DWZCompetitor(initial_rating=1500)
        opponent_weak = DWZCompetitor(initial_rating=1000)
        w_a = 1.0
        w_e = player_teen_win.expected_score(opponent_weak)  # Should be > 0.5
        player_teen_win._count = 5  # Set a game count

        # Manual E calculation for teenager, winning
        age = 18
        J = 5
        Z_A = 1500.0
        E0 = (Z_A / 1000.0) ** 4 + J
        a = Z_A / 2000.0  # a = 0.75
        a = max(0.5, min(1.0, a))
        B = 0.0  # Rating >= 1300
        E_raw = a * E0 + B
        i = player_teen_win._count + 1  # i = 6
        E_upper_bound = min(30.0, 5.0 * i)  # min(30, 30) = 30
        expected_E_teen_win = max(5.0, min(E_raw, E_upper_bound))
        calculated_E_teen_win = player_teen_win._calculate_E(age=age, W_a=w_a, W_e=w_e)
        self.assertAlmostEqual(calculated_E_teen_win, expected_E_teen_win)

        # Scenario 2: Adult (>25), losing (W_a < W_e), rating < 1300
        player_adult_loss_low_rating = DWZCompetitor(initial_rating=1000)
        opponent_strong = DWZCompetitor(initial_rating=1500)
        w_a = 0.0
        w_e = player_adult_loss_low_rating.expected_score(opponent_strong)  # Should be < 0.5
        player_adult_loss_low_rating._count = 2  # Set a game count

        # Manual E calculation for adult, losing, low rating
        age = 30
        J = 15
        Z_A = 1000.0
        E0 = (Z_A / 1000.0) ** 4 + J  # E0 = 1^4 + 15 = 16
        a = 1.0  # Not teen or not winning
        B = math.exp((1300.0 - Z_A) / 150.0) - 1.0  # B = exp(300/150) - 1 = exp(2) - 1
        E_raw = a * E0 + B  # E_raw = 1 * 16 + exp(2) - 1 = 15 + exp(2)
        i = player_adult_loss_low_rating._count + 1  # i = 3
        E_upper_bound = 150.0  # B > 0
        expected_E_adult_loss = max(5.0, min(E_raw, E_upper_bound))
        calculated_E_adult_loss = player_adult_loss_low_rating._calculate_E(age=age, W_a=w_a, W_e=w_e)
        self.assertAlmostEqual(calculated_E_adult_loss, expected_E_adult_loss)

        # Scenario 3: Default age (None -> 26), draw (W_a == W_e implicitly tested if We = 0.5)
        player_default_draw = DWZCompetitor(initial_rating=1200)
        opponent_equal = DWZCompetitor(initial_rating=1200)
        w_a = 0.5
        w_e = player_default_draw.expected_score(opponent_equal)  # Should be 0.5
        player_default_draw._count = 10

        # Manual E calculation for default age (adult), draw, rating < 1300
        age = None  # Defaults to 26 internally
        J = 15  # Adult
        Z_A = 1200.0
        E0 = (Z_A / 1000.0) ** 4 + J  # E0 = 1.2^4 + 15
        a = 1.0  # Not teen or not winning
        # B depends on W_a <= W_e. Here W_a = W_e = 0.5. So B is calculated.
        B = math.exp((1300.0 - Z_A) / 150.0) - 1.0  # B = exp(100/150) - 1 = exp(2/3) - 1
        E_raw = a * E0 + B
        i = player_default_draw._count + 1  # i = 11
        E_upper_bound = 150.0  # B > 0
        expected_E_default_draw = max(5.0, min(E_raw, E_upper_bound))
        calculated_E_default_draw = player_default_draw._calculate_E(age=age, W_a=w_a, W_e=w_e)
        self.assertAlmostEqual(calculated_E_default_draw, expected_E_default_draw)

    def test_beat_with_known_values(self):
        """Test beat method with known values using default age."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Store original ratings and counts
        p1_original_rating = player1.rating
        p2_original_rating = player2.rating
        p1_original_count = player1._count
        p2_original_count = player2._count

        # Calculate expected new ratings manually using _new_rating logic (which calls _calculate_E)
        # Player 1 wins (W_a = 1.0), age = None (default 26 -> adult)
        p1_w_a = 1.0
        p1_w_e = player1.expected_score(player2)
        p1_E = player1._calculate_E(age=None, W_a=p1_w_a, W_e=p1_w_e)
        p1_new_rating = p1_original_rating + (800.0 / (p1_E + 1)) * (p1_w_a - p1_w_e)
        p1_new_rating = max(player1._minimum_rating, p1_new_rating)

        # Player 2 loses (W_a = 0.0), age = None (default 26 -> adult)
        p2_w_a = 0.0
        p2_w_e = player2.expected_score(player1)
        p2_E = player2._calculate_E(age=None, W_a=p2_w_a, W_e=p2_w_e)
        p2_new_rating = p2_original_rating + (800.0 / (p2_E + 1)) * (p2_w_a - p2_w_e)
        p2_new_rating = max(player2._minimum_rating, p2_new_rating)

        # Player1 beats player2 (default age)
        player1.beat(player2)

        # Check new ratings and counts
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)
        self.assertEqual(player1._count, p1_original_count + 1)
        self.assertEqual(player2._count, p2_original_count + 1)

        # Simple check: Does passing an age change the outcome?
        # Use higher ratings and count to avoid clamping E at 5
        player1_teen = DWZCompetitor(initial_rating=1600)
        player2_opponent = DWZCompetitor(initial_rating=1400)
        player1_default = DWZCompetitor(initial_rating=1600)  # Need a separate instance for default age comparison
        player2_opponent_default = DWZCompetitor(initial_rating=1400)

        # Set count to avoid initial E clamping
        player1_teen._count = 10
        player1_default._count = 10

        player1_teen.beat(player2_opponent, age=15)
        player1_default.beat(player2_opponent_default)  # Default age

        self.assertNotEqual(player1_teen.rating, player1_default.rating)  # Rating should differ due to age

    def test_tied_with_known_values(self):
        """Test tied method with known values using default age."""
        player1 = DWZCompetitor(initial_rating=1000)
        player2 = DWZCompetitor(initial_rating=1200)

        # Store original ratings and counts
        p1_original_rating = player1.rating
        p2_original_rating = player2.rating
        p1_original_count = player1._count
        p2_original_count = player2._count

        # Calculate expected new ratings manually (W_a = 0.5)
        # Player 1 ties (W_a = 0.5), age = None (default 26 -> adult)
        p1_w_a = 0.5
        p1_w_e = player1.expected_score(player2)
        p1_E = player1._calculate_E(age=None, W_a=p1_w_a, W_e=p1_w_e)
        p1_new_rating = p1_original_rating + (800.0 / (p1_E + 1)) * (p1_w_a - p1_w_e)
        p1_new_rating = max(player1._minimum_rating, p1_new_rating)

        # Player 2 ties (W_a = 0.5), age = None (default 26 -> adult)
        p2_w_a = 0.5
        p2_w_e = player2.expected_score(player1)
        p2_E = player2._calculate_E(age=None, W_a=p2_w_a, W_e=p2_w_e)
        p2_new_rating = p2_original_rating + (800.0 / (p2_E + 1)) * (p2_w_a - p2_w_e)
        p2_new_rating = max(player2._minimum_rating, p2_new_rating)

        # Players tie (default age)
        player1.tied(player2)

        # Check new ratings and counts
        self.assertAlmostEqual(player1.rating, p1_new_rating)
        self.assertAlmostEqual(player2.rating, p2_new_rating)
        self.assertEqual(player1._count, p1_original_count + 1)
        self.assertEqual(player2._count, p2_original_count + 1)

        # Simple check: Does passing an age change the outcome?
        # Use higher ratings and count to avoid clamping E at 5
        player1_young_adult = DWZCompetitor(initial_rating=1600)
        player2_opponent = DWZCompetitor(initial_rating=1400)
        player1_default = DWZCompetitor(initial_rating=1600)  # Need a separate instance for default age comparison
        player2_opponent_default = DWZCompetitor(initial_rating=1400)

        # Set count to avoid initial E clamping
        player1_young_adult._count = 10
        player1_default._count = 10

        player1_young_adult.tied(player2_opponent, age=22)
        player1_default.tied(player2_opponent_default)  # Default age

        self.assertNotEqual(player1_young_adult.rating, player1_default.rating)  # Rating should differ due to age


if __name__ == "__main__":
    unittest.main()
