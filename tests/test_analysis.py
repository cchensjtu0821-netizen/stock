import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from stock_analysis.engine import analyze
from stock_analysis.io import load_fundamentals, load_prices


def price_rows(values, symbol="TEST"):
    rows = []
    day = date(2024, 1, 1)
    for value in values:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        rows.append({"symbol": symbol, "date": day.isoformat(), "close": value,
                     "adj_close": value, "volume": 1000, "currency": "USD", "source": "test"})
        day += timedelta(days=1)
    return rows


def financial_snapshot(**overrides):
    row = {"symbol": "TEST", "period_end": "2024-09-30", "published_date": "2024-11-01",
           "currency": "USD", "source": "test", "revenue_ttm": 2000, "revenue_prior_ttm": 1600,
           "eps_ttm": 4, "eps_prior_ttm": 2, "net_income_ttm": 400, "average_equity": 2000,
           "operating_income_ttm": 600, "operating_cash_flow_ttm": 500, "capex_ttm": 100,
           "shares_outstanding": 100, "debt": 1000, "cash": 200, "ebitda_ttm": 500}
    row.update(overrides)
    return row


class CalculationTests(unittest.TestCase):
    def setUp(self):
        self.prices = price_rows(range(100, 353))
        self.as_of = self.prices[-1]["date"]

    def metrics(self, snapshots=None, prices=None, benchmark=None):
        return analyze(self.prices if prices is None else prices,
                       [financial_snapshot()] if snapshots is None else snapshots,
                       "TEST", self.as_of, benchmark)["metrics"]

    def test_financial_ratios_have_known_values(self):
        metrics = self.metrics()
        expected = {"revenue_growth": .25, "eps_growth": 1, "roe": .2,
                    "operating_margin": .3, "cash_conversion": 1.25, "fcf_margin": .2,
                    "pe_ttm": 88, "fcf_yield": 400 / 35200, "net_debt_ebitda": 1.6}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(metrics[key]["value"], value)

    def test_price_window_endpoints_and_means(self):
        metrics = self.metrics()
        self.assertAlmostEqual(metrics["momentum_12_1"]["value"], 331 / 100 - 1)
        self.assertAlmostEqual(metrics["ma50_gap"]["value"], 352 / 327.5 - 1)
        self.assertAlmostEqual(metrics["ma200_gap"]["value"], 352 / 252.5 - 1)
        self.assertEqual(metrics["max_drawdown_252d"]["value"], 0)

    def test_drawdown_tracks_peak_before_trough(self):
        prices = price_rows([100] * 248 + [200, 80, 160, 180, 200])
        self.assertAlmostEqual(self.metrics(prices=prices)["max_drawdown_252d"]["value"], -.6)

    def test_volatility_uses_sample_standard_deviation(self):
        values = [100.0]
        for daily_return in [0, .1] * 30:
            values.append(values[-1] * (1 + daily_return))
        value = self.metrics(prices=price_rows(values))["volatility_60d"]["value"]
        self.assertAlmostEqual(value, .05 * math.sqrt(60 / 59) * math.sqrt(252))

    def test_raw_close_is_used_for_valuation_and_turnover(self):
        prices = deepcopy(self.prices)
        for row in prices:
            row["adj_close"] /= 2
        metrics = self.metrics(prices=prices)
        self.assertEqual(metrics["pe_ttm"]["value"], 88)
        self.assertEqual(metrics["turnover_20d_proxy"]["value"], 342500)
        self.assertAlmostEqual(metrics["ma50_gap"]["value"], 352 / 327.5 - 1)

    def test_losses_and_negative_equity_do_not_become_cheap_pe(self):
        metrics = self.metrics([financial_snapshot(
            eps_ttm=-4, eps_prior_ttm=-2, net_income_ttm=-400,
            average_equity=-2000, ebitda_ttm=0, revenue_prior_ttm=0)])
        for key in ("pe_ttm", "eps_growth", "cash_conversion", "roe", "net_debt_ebitda", "revenue_growth"):
            self.assertIsNone(metrics[key]["value"], key)

    def test_net_cash_and_negative_fcf_preserve_sign(self):
        metrics = self.metrics([financial_snapshot(debt=100, cash=200, capex_ttm=600)])
        self.assertAlmostEqual(metrics["net_debt_ebitda"]["value"], -.2)
        self.assertAlmostEqual(metrics["fcf_margin"]["value"], -.05)
        self.assertLess(metrics["fcf_yield"]["value"], 0)

    def test_insufficient_history_is_unavailable(self):
        metrics = self.metrics(prices=self.prices[:19], snapshots=[])
        self.assertTrue(all(metric["value"] is None for metric in metrics.values()))

    def test_incomplete_volume_does_not_shrink_window(self):
        prices = deepcopy(self.prices)
        prices[-1]["volume"] = None
        self.assertIsNone(self.metrics(prices=prices)["turnover_20d_proxy"]["value"])

    def test_future_prices_and_future_disclosures_are_excluded(self):
        future = dict(self.prices[-1], date="2099-01-01", close=999999, adj_close=999999)
        snapshots = [financial_snapshot(), financial_snapshot(published_date="2099-01-02", eps_ttm=100)]
        result = analyze(self.prices + [future], snapshots, "TEST", self.as_of)
        self.assertEqual(result["close"], 352)
        self.assertEqual(result["price_rows"], 253)
        self.assertEqual(result["financial_published_date"], "2024-11-01")
        self.assertEqual(result["metrics"]["pe_ttm"]["value"], 88)

    def test_latest_known_revision_is_selected_without_newer_period_leak(self):
        snapshots = [financial_snapshot(), financial_snapshot(published_date="2024-11-15", eps_ttm=8),
                     financial_snapshot(period_end="2024-12-31", published_date="2025-02-01", eps_ttm=16)]
        self.assertEqual(self.metrics(snapshots)["pe_ttm"]["value"], 44)

    def test_older_period_revision_does_not_replace_newer_period(self):
        snapshots = [financial_snapshot(), financial_snapshot(
            period_end="2024-06-30", published_date="2024-11-15", eps_ttm=8)]
        self.assertEqual(self.metrics(snapshots)["pe_ttm"]["value"], 88)

    def test_same_path_benchmark_has_zero_relative_return(self):
        benchmark = [dict(row, symbol="BENCH", close=row["close"] * 2,
                          adj_close=row["adj_close"] * 2) for row in self.prices]
        value = self.metrics(prices=self.prices + benchmark, benchmark="BENCH")["relative_return_63d"]["value"]
        self.assertAlmostEqual(value, 0)

    def test_missing_benchmark_session_is_not_forward_filled(self):
        benchmark = [dict(row, symbol="BENCH") for row in self.prices[:-1]]
        self.assertIsNone(self.metrics(prices=self.prices + benchmark, benchmark="BENCH")["relative_return_63d"]["value"])

    def test_currency_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, "币种"):
            self.metrics([financial_snapshot(currency="CNY")])
        benchmark = [dict(row, symbol="BENCH", currency="CNY") for row in self.prices]
        with self.assertRaisesRegex(ValueError, "币种"):
            self.metrics(prices=self.prices + benchmark, benchmark="BENCH")

    def test_missing_symbol_fails(self):
        with self.assertRaisesRegex(ValueError, "没有行情"):
            analyze(self.prices, [], "MISSING", self.as_of)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.csv"

    def write(self, content):
        self.path.write_text(content, encoding="utf-8")

    def test_optional_financial_numbers_remain_missing(self):
        self.write("symbol,period_end,published_date,currency,source\nTEST,2024-09-30,2024-11-01,USD,test\n")
        rows = load_fundamentals(self.path)
        self.assertIsNone(rows[0]["net_income_ttm"])
        self.assertIsNone(rows[0]["shares_outstanding"])

    def test_invalid_price_values_and_duplicates(self):
        header = "symbol,date,close,adj_close,currency,source\n"
        for raw in ("NaN", "inf", "-1", "0", "bad"):
            self.write(header + f"TEST,2024-01-01,{raw},100,USD,test\n")
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_prices(self.path)
        row = "TEST,2024-01-01,100,100,USD,test\n"
        self.write(header + row + row)
        with self.assertRaisesRegex(ValueError, "重复"):
            load_prices(self.path)

    def test_negative_capex_and_publication_before_period_are_rejected(self):
        for publication, capex in (("2024-11-01", "-100"), ("2024-01-01", "100")):
            self.write("symbol,period_end,published_date,currency,source,capex_ttm\n"
                       f"TEST,2024-09-30,{publication},USD,test,{capex}\n")
            with self.subTest(publication=publication), self.assertRaises(ValueError):
                load_fundamentals(self.path)

    def test_malformed_columns_and_duplicate_headers_are_rejected(self):
        for content in (
            "symbol,date,close,adj_close,currency,source\nTEST,2024-01-01,100,100,USD,test,extra\n",
            "symbol,date,close,adj_close,currency,source\nTEST,2024-01-01,100,100,USD\n",
            "symbol,date,close,adj_close,currency,source,close\n",
        ):
            self.write(content)
            with self.assertRaises(ValueError):
                load_prices(self.path)

    def test_cli_exports_valid_json_with_missing_financials(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            fields = ("symbol", "date", "close", "adj_close", "volume", "currency", "source")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(price_rows(range(100, 353)))
        output = Path(self.directory.name) / "nested" / "report.json"
        result = subprocess.run([sys.executable, "-m", "stock_analysis", "--prices", str(self.path),
                                 "--symbol", "TEST", "--as-of", "2024-12-31", "--output", str(output)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["metrics"]), 16)
        self.assertIsNone(report["metrics"]["pe_ttm"]["value"])
        self.assertIn("N/A", result.stdout)


if __name__ == "__main__":
    unittest.main()
