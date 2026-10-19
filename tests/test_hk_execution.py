import json
from pathlib import Path
import unittest

import pandas as pd

from stock_analysis.swing import backtest, can_fill, dividend_tax_rate, fees
from test_swing import execution_fixture


class HongKongExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config/zijin_swing.json").read_text())
        self.config.update(market="HK", lot_size=500, commission_rate=.0003, minimum_commission=3,
                           platform_fee=15, dividend_tax_rate=0)
        self.frame = execution_fixture()
        self.config.update(test_start=str(self.frame.index[0].date()), test_end=str(self.frame.index[-1].date()))

    def test_stamp_duty_date_change_both_sides_and_round_up(self):
        old = fees(10001, "buy", self.config, "2023-11-16")
        new = fees(10001, "buy", self.config, "2023-11-17")
        self.assertEqual(old["stamp"], 14)
        self.assertEqual(new["stamp"], 11)
        self.assertEqual(new["stamp"], fees(10001, "sell", self.config, "2023-11-17")["stamp"])

    def test_settlement_fee_removes_minimum_on_effective_date(self):
        self.assertEqual(fees(10000, "buy", self.config, "2025-06-27")["settlement"], 2)
        self.assertEqual(fees(10000, "buy", self.config, "2025-06-30")["settlement"], .42)

    def test_hong_kong_large_gap_is_not_a_share_limit(self):
        row, previous = self.frame.iloc[1].copy(), self.frame.iloc[0]
        row["open"] = 15
        self.assertFalse(can_fill(row, previous, "buy"))
        self.assertTrue(can_fill(row, previous, "buy", "HK"))
        row["open"] = 8
        self.assertFalse(can_fill(row, previous, "sell"))
        self.assertTrue(can_fill(row, previous, "sell", "HK"))
        row["volume"] = 0
        self.assertFalse(can_fill(row, previous, "sell", "HK"))

    def test_board_lot_and_accounting_for_hk_round_trip(self):
        self.frame.loc[self.frame.index[1], "ma20"] = 11
        result = backtest(self.frame, self.config, "pullback_all")
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["shares"] % 500, 0)
        self.assertLessEqual(trade["entry_total"], 30000)
        self.assertAlmostEqual(result["stats"]["ending_equity"] - 100000, trade["net_pnl"])

    def test_hk_tax_is_explicit_and_not_a_share_holding_period_rule(self):
        entry, exit = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")
        self.assertEqual(dividend_tax_rate(entry, exit, self.config), 0)
        self.config["dividend_tax_rate"] = .2
        self.assertEqual(dividend_tax_rate(entry, exit, self.config), .2)


if __name__ == "__main__": unittest.main()
