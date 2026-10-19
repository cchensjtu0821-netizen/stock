import json
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:
    raise unittest.SkipTest("波段回测测试需安装 requirements-research.txt") from error

from stock_analysis.relationships import regression
from stock_analysis.swing import backtest, fees, make_features, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def execution_fixture():
    index = pd.bdate_range("2024-01-01", periods=6)
    frame = pd.DataFrame(index=index)
    for col in ("open", "close", "analysis_open", "analysis_close"):
        frame[col] = 10.0
    frame["volume"] = 100000
    frame["cash_dividend"] = 0.0
    frame["atr14"] = 1.0
    frame["ma20"] = 9.0
    frame["pullback_signal"] = [True, False, False, False, False, False]
    frame["breakout_signal"] = False
    for col in ("gold_filter", "sector_filter", "market_filter"):
        frame[col] = True
    return frame


class SwingExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/zijin_swing.json").read_text())
        self.frame = execution_fixture()
        self.config.update(test_start=str(self.frame.index[0].date()), test_end=str(self.frame.index[-1].date()))

    def run_replay(self, frame=None):
        return backtest(self.frame if frame is None else frame, self.config, "pullback_all")

    def test_signal_at_close_enters_next_open_and_exit_respects_t_plus_one(self):
        self.frame.loc[self.frame.index[1], "ma20"] = 11
        result = self.run_replay()
        self.assertEqual(len(result["trades"]), 1)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["signal_date"], "2024-01-01")
        self.assertEqual(trade["entry_date"], "2024-01-02")
        self.assertEqual(trade["exit_date"], "2024-01-03")
        self.assertEqual(trade["holding_sessions"], 1)

    def test_limit_down_defers_exit_without_changing_original_signal(self):
        self.frame.loc[self.frame.index[1], "ma20"] = 11
        self.frame.loc[self.frame.index[2], ["open", "close"]] = 9
        self.frame.loc[self.frame.index[3], ["open", "close"]] = 9.2
        result = self.run_replay()
        self.assertEqual(result["stats"]["blocked_sells"], 1)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["exit_date"], "2024-01-04")
        self.assertEqual(trade["exit_signal_date"], "2024-01-02")

    def test_blocked_buy_does_not_execute_stale_signal_next_day(self):
        self.frame.loc[self.frame.index[1], ["open", "close"]] = 11
        result = self.run_replay()
        self.assertEqual(result["stats"]["blocked_buys"], 1)
        self.assertTrue((result["equity"]["shares"] == 0).all())

    def test_dividend_belongs_to_prior_holder_even_when_sold_on_ex_date(self):
        self.frame.loc[self.frame.index[1], "ma20"] = 11
        self.frame.loc[self.frame.index[2], "cash_dividend"] = .2
        self.frame.loc[self.frame.index[2], ["open", "close"]] = 9.8
        result = self.run_replay()
        trade = result["trades"].iloc[0]
        self.assertAlmostEqual(trade["gross_dividends"], trade["shares"] * .2)
        self.assertAlmostEqual(trade["dividend_tax"], trade["gross_dividends"] * .2)
        self.assertAlmostEqual(trade["net_pnl"], -trade["entry_cost"] - trade["exit_cost"] - trade["dividend_tax"])

    def test_ex_date_buyer_does_not_receive_dividend(self):
        self.frame.loc[self.frame.index[1], "cash_dividend"] = .2
        result = self.run_replay()
        self.assertTrue((result["equity"]["dividend_receivable"] == 0).all())

    def test_minimum_commission_and_sell_only_stamp(self):
        buy = fees(100, "buy", self.config)
        sell = fees(100, "sell", self.config)
        self.assertEqual(buy["commission"], 5)
        self.assertEqual(buy["stamp"], 0)
        self.assertAlmostEqual(sell["stamp"], .05)

    def test_round_lot_cap_and_accounting_reconcile(self):
        self.frame.loc[self.frame.index[1], "ma20"] = 11
        result = self.run_replay()
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["shares"] % 100, 0)
        self.assertLessEqual(trade["entry_total"], self.config["initial_cash"] * self.config["max_position_fraction"])
        self.assertLessEqual(trade["shares"] * 2, self.config["initial_cash"] * self.config["risk_per_trade"])
        self.assertAlmostEqual(result["stats"]["ending_equity"] - self.config["initial_cash"], trade["net_pnl"])

    def test_open_trade_is_excluded_from_success_rate(self):
        result = self.run_replay()
        self.assertIsNotNone(result["stats"]["open_position"])
        self.assertEqual(result["stats"]["closed_trades"], 0)
        self.assertIsNone(result["stats"]["win_rate"])
        self.assertLess(result["stats"]["ending_equity"], self.config["initial_cash"])

    def test_new_period_does_not_enter_from_pre_period_signal(self):
        result = backtest(self.frame, self.config, "pullback_all", start="2024-01-02")
        self.assertTrue((result["equity"]["shares"] == 0).all())

    def test_zero_trade_interval_and_small_sample_uncertainty(self):
        self.assertEqual(wilson_interval(0, 0), [None, None])
        interval = wilson_interval(1, 1)
        self.assertLess(interval[0], .21)
        self.assertAlmostEqual(interval[1], 1)


class InformationTimingTests(unittest.TestCase):
    def test_appending_future_prices_and_dividend_cannot_change_past_features(self):
        index = pd.bdate_range("2024-01-01", periods=110)
        values = 10 + np.sin(np.arange(110) / 5) + np.arange(110) * .02
        prices = pd.DataFrame({"open": values, "close": values, "high": values + .1,
                               "low": values - .1, "volume": 100000}, index=index)
        dividends = pd.DataFrame({"ex_date": [index[-5]], "cash_per10": [4.0]})
        empty = pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "cash_per10": pd.Series(dtype=float)})
        past = prices.iloc[:100]
        before = make_features(past, past, past, past, empty)
        after = make_features(prices, prices, prices, prices, dividends).iloc[:100]
        pd.testing.assert_frame_equal(before, after)

    def test_missing_factor_observation_does_not_become_a_zero_return(self):
        index = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame({"open": 10, "close": np.arange(100) * .1 + 10,
                               "high": np.arange(100) * .1 + 10, "low": 10, "volume": 100000}, index=index)
        external = prices.drop(index[80])
        empty = pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "cash_per10": pd.Series(dtype=float)})
        frame = make_features(prices, prices, external, prices, empty)
        self.assertTrue(pd.isna(frame.loc[index[80], "sector_close"]))
        self.assertFalse(frame.loc[index[80], "sector_filter"])

    def test_multifactor_regression_recovers_independent_coefficients(self):
        rng = np.random.default_rng(42)
        factors = rng.normal(0, .01, (500, 3))
        returns = pd.DataFrame(factors, columns=["gold", "sector", "sse"])
        returns["zijin"] = factors @ np.array([.2, .5, 1.0]) + rng.normal(0, .0001, 500)
        result = regression(returns)
        for name, expected in (("gold", .2), ("sector", .5), ("sse", 1.0)):
            self.assertAlmostEqual(result["coefficients"][name]["beta"], expected, delta=.003)
        self.assertGreater(result["r_squared"], .99)


if __name__ == "__main__":
    unittest.main()
