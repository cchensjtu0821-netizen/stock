import json
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:
    raise unittest.SkipTest("指标研究测试需安装 requirements-research.txt") from error

from stock_analysis.indicator_lab import add_indicators, candidates, chinese_sma, cross_down, cross_up
from stock_analysis.swing import backtest, make_features


def feature_fixture(count=140):
    index = pd.bdate_range("2024-01-01", periods=count)
    close = 10 + np.arange(count) * .03 + np.sin(np.arange(count) / 7)
    raw = pd.DataFrame({"open": close, "close": close, "high": close + .2,
                        "low": close - .2, "volume": 100000 + np.arange(count) * 100}, index=index)
    dividends = pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "cash_per10": pd.Series(dtype=float)})
    return make_features(raw, raw, raw, raw, dividends)


class IndicatorFormulaTests(unittest.TestCase):
    def test_kdj_recursive_smoothing_has_known_seed(self):
        result = chinese_sma(pd.Series([np.nan, 0, 0, 0]))
        self.assertTrue(pd.isna(result.iloc[0]))
        np.testing.assert_allclose(result.iloc[1:], [100 / 3, 200 / 9, 400 / 27])

    def test_cross_requires_a_previous_observation_and_only_fires_once(self):
        values = pd.Series([-1, 0, 1, 2, 1, 0, -1])
        self.assertEqual(cross_up(values, 0).tolist(), [False, False, True, False, False, False, False])
        self.assertEqual(cross_down(values, 0).tolist(), [False, False, False, False, False, False, True])

    def test_cci_mean_deviation_and_directional_indicators(self):
        frame = feature_fixture(100)
        for col in ("analysis_open", "analysis_close", "analysis_high", "analysis_low"):
            frame[col] = np.arange(1, 101, dtype=float)
        frame["analysis_high"] += .25
        frame["analysis_low"] -= .25
        result = add_indicators(frame)
        self.assertAlmostEqual(result["cci20"].iloc[-1], 126.66666666666667)
        self.assertAlmostEqual(result["mfi14"].iloc[-1], 100)
        self.assertAlmostEqual(result["adx14"].iloc[-1], 100)
        self.assertEqual(result["di_minus14"].iloc[-1], 0)
        self.assertGreater(result["di_plus14"].iloc[-1], 0)

    def test_flat_market_avoids_infinite_indicators(self):
        frame = feature_fixture(100)
        for col in ("analysis_open", "analysis_close", "analysis_high", "analysis_low"):
            frame[col] = 10.0
        result = add_indicators(frame)
        for col in ("kdj_k", "kdj_d", "mfi14"):
            self.assertAlmostEqual(result[col].iloc[-1], 50)
        self.assertAlmostEqual(result["williams_r14"].iloc[-1], -50)
        self.assertFalse(np.isinf(result.select_dtypes(include=["number"])).any().any())

    def test_future_bars_do_not_change_any_past_indicator_or_candidate(self):
        frame = feature_fixture(180)
        prefix = add_indicators(frame.iloc[:140])
        complete = add_indicators(frame)
        pd.testing.assert_frame_equal(prefix, complete.iloc[:140])
        for before, after in zip(candidates(prefix), candidates(complete)):
            pd.testing.assert_series_equal(before.entry, after.entry.iloc[:140])
            pd.testing.assert_series_equal(before.exit, after.exit.iloc[:140])

    def test_candidate_catalog_is_complete_and_unique(self):
        specs = candidates(add_indicators(feature_fixture()))
        self.assertEqual(len(specs), 28)
        self.assertEqual(len({spec.key for spec in specs}), 28)
        self.assertEqual(sum(spec.category == "单指标" for spec in specs), 16)
        for spec in specs:
            self.assertEqual(spec.entry.dtype, bool)
            self.assertEqual(spec.exit.dtype, bool)


class ExitComparisonTests(unittest.TestCase):
    def setUp(self):
        self.frame = feature_fixture(100).iloc[-6:].copy()
        for col in ("open", "close", "analysis_open", "analysis_close"):
            self.frame[col] = 10.0
        self.frame["atr14"] = .5
        self.frame["ma20"] = 11.0
        self.frame["cash_dividend"] = 0.0
        self.signals = pd.Series([True, False, False, False, False, False], index=self.frame.index)
        self.exits = pd.Series([False, False, False, True, False, False], index=self.frame.index)
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config/zijin_swing.json").read_text())
        self.config.update(test_start=str(self.frame.index[0].date()), test_end=str(self.frame.index[-1].date()))

    def test_native_exit_is_evaluated_at_close_and_filled_next_day(self):
        result = backtest(self.frame, self.config, "custom", signals=self.signals, indicator_exit=self.exits, use_ma20_exit=False)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["entry_date"], str(self.frame.index[1].date()))
        self.assertEqual(trade["exit_date"], str(self.frame.index[4].date()))
        self.assertEqual(trade["exit_reason"], "indicator_exit")

    def test_common_risk_mode_does_not_apply_universal_ma20_or_native_exit(self):
        result = backtest(self.frame, self.config, "custom", signals=self.signals, use_ma20_exit=False)
        self.assertEqual(result["stats"]["closed_trades"], 0)
        self.assertIsNotNone(result["stats"]["open_position"])

    def test_classic_indicator_mode_really_disables_atr_exit_and_risk_sizing(self):
        self.frame.loc[self.frame.index[2], ["close", "analysis_close"]] = 9
        self.frame.loc[self.frame.index[3], ["open", "analysis_open"]] = 9.1
        controlled = backtest(self.frame, self.config, "custom", signals=self.signals, use_ma20_exit=False)
        classic = backtest(self.frame, self.config, "custom", signals=self.signals, use_ma20_exit=False, use_atr_risk_controls=False)
        self.assertEqual(controlled["trades"].iloc[0]["exit_reason"], "closing_atr_stop")
        self.assertEqual(classic["stats"]["closed_trades"], 0)
        self.assertGreater(classic["stats"]["open_position"]["shares"], controlled["trades"].iloc[0]["shares"])


if __name__ == "__main__":
    unittest.main()
