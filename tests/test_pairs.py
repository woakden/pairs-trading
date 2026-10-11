"""Tests on simulated prices, so they run without downloading anything.

Run from the project folder:  python3 -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pairs  # noqa: E402
from backtest import list_trades, performance, positions_from_z, run_backtest  # noqa: E402
from run import analyse, parse_args  # noqa: E402


def simulated_pair(n=1500, beta=0.8, seed=1, linked=True):
    """Two simulated stocks. If linked, log(Y) = beta * log(X) + a gap that keeps pulling back to 0."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    log_x = np.log(50) + np.cumsum(rng.normal(0, 0.015, n))
    if linked:
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.9 * gap[t - 1] + rng.normal(0, 0.02)
        log_y = 1.0 + beta * log_x + gap
    else:
        log_y = np.log(80) + np.cumsum(rng.normal(0, 0.015, n))
    return pd.DataFrame({"Y": np.exp(log_y), "X": np.exp(log_x)}, index=dates)


class StatisticsTests(unittest.TestCase):
    def test_hedge_ratio_recovers_true_beta(self):
        p = simulated_pair(beta=0.8)
        beta = pairs.rolling_hedge_ratio(np.log(p["Y"]), np.log(p["X"]), 252).dropna()
        self.assertAlmostEqual(beta.median(), 0.8, delta=0.1)

    def test_linked_pair_is_cointegrated(self):
        p = simulated_pair()
        _, p_value = pairs.engle_granger(np.log(p["Y"]), np.log(p["X"]))
        self.assertLess(p_value, 0.01)

    def test_unrelated_pair_is_not_cointegrated(self):
        p = simulated_pair(linked=False, seed=3)
        _, p_value = pairs.engle_granger(np.log(p["Y"]), np.log(p["X"]))
        self.assertGreater(p_value, 0.05)

    def test_half_life_matches_simulation(self):
        # The gap shrinks by 10% a day, so it should halve in about ln(2) / 0.1 = 6.6 days.
        rng = np.random.default_rng(0)
        gap = np.zeros(5000)
        for t in range(1, 5000):
            gap[t] = 0.9 * gap[t - 1] + rng.normal()
        self.assertAlmostEqual(pairs.half_life(gap), np.log(2) / 0.1, delta=1.0)

    def test_random_walk_has_no_half_life(self):
        walk = np.cumsum(np.random.default_rng(5).normal(size=3000)) + np.arange(3000) * 0.05
        self.assertEqual(pairs.half_life(walk), float("inf"))


class SignalTests(unittest.TestCase):
    def test_entry_exit_and_stop(self):
        z = pd.Series([0, 2.5, 1.0, 0.3, 0, -2.2, -4.5, -3.0, -1.0, -2.5, 0.0])
        expected = [0, -1, -1, 0, 0, 1, 0, 0, 0, 1, 0]
        # 2.5: short; 0.3: back inside exit band, close; -2.2: long; -4.5: stop out;
        # -3.0: still outside entry after the stop, so wait; -1.0: cooled off; -2.5: long again.
        self.assertEqual(positions_from_z(z, 2.0, 0.5, 4.0).tolist(), expected)

    def test_missing_z_means_flat(self):
        z = pd.Series([np.nan, 3.0, np.nan, 3.0])
        self.assertEqual(positions_from_z(z, 2.0, 0.5, 4.0).tolist(), [0, -1, 0, -1])


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.args = parse_args(["--y", "Y", "--x", "X", "--cost-bps", "0"])

    def test_makes_money_on_a_mean_reverting_pair(self):
        out = analyse(simulated_pair(), self.args)
        self.assertGreater(out["performance"]["total_return"], 0.1)
        self.assertGreater(len(out["trades"]), 10)

    def test_costs_reduce_returns(self):
        p = simulated_pair()
        free = analyse(p, self.args)["performance"]["total_return"]
        costly = analyse(p, parse_args(["--y", "Y", "--x", "X", "--cost-bps", "20"]))["performance"]["total_return"]
        self.assertLess(costly, free)

    def test_no_look_ahead(self):
        # Changing prices after a date must not change any position held up to that date.
        p = simulated_pair()
        cut = p.index[1000]
        shocked = p.copy()
        shocked.loc[shocked.index > cut, "Y"] *= 1.5
        a = analyse(p, self.args)["result"]["position"]
        b = analyse(shocked, self.args)["result"]["position"]
        pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])

    def test_trade_returns_add_up(self):
        out = analyse(simulated_pair(), self.args)
        trades, r = out["trades"], out["result"]["strategy_return"]
        # With no costs, compounding every trade gives the same total as compounding every day.
        self.assertAlmostEqual(float(np.prod(1 + trades["return"])), float(np.prod(1 + r)), places=10)

    def test_performance_of_known_returns(self):
        r = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range("2020-01-01", periods=3))
        perf = performance(r)
        self.assertAlmostEqual(perf["total_return"], 1.01 * 0.98 * 1.03 - 1)
        self.assertAlmostEqual(perf["max_drawdown"], -0.02)


if __name__ == "__main__":
    unittest.main()
