"""核对跨股票回放账户，生成固定28规则的并排比较与唐奇安明细。"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from explore_zijin_indicators import STOCKS, markdown_table, percent, save_json
from stock_analysis.swing import backtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/stock_comparison"


def check_accounts(directory, result):
    paths = pd.read_csv(directory / "account_paths.csv", parse_dates=["date"])
    trades = pd.read_csv(directory / "trades.csv")
    prices = stock_prices(directory)
    if not ((paths["shares"] >= 0) & (paths["shares"] % 100 == 0) & (paths["cash"] >= -1e-6)).all():
        raise ValueError("出现负现金或不符合整手规则的持仓")
    checks = 0
    for (mode, period, key), path in paths.groupby(["exit_mode", "period", "strategy"], sort=False):
        detail = trades.loc[(trades["exit_mode"] == mode) & (trades["period"] == period) & (trades["strategy"] == key)]
        stats = result["results"][mode][period][key]
        initial = result["base_config"]["initial_cash"]
        final = float(path["equity"].iloc[-1])
        open_pnl = (stats["open_position"] or {}).get("unrealized_net_pnl", 0)
        assert len(detail) == stats["closed_trades"]
        np.testing.assert_allclose(final, stats["ending_equity"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(final / initial - 1, stats["total_return"], atol=1e-12, rtol=0)
        drawdown = path["equity"] / path["equity"].cummax().clip(lower=initial) - 1
        np.testing.assert_allclose(drawdown.min(), stats["max_drawdown"], atol=1e-12, rtol=0)
        assert int((detail["net_pnl"] > 0).sum()) == stats["wins"]
        np.testing.assert_allclose(final - initial, detail["net_pnl"].sum() + open_pnl, atol=1e-6, rtol=0)
        np.testing.assert_allclose(path["equity"], path["cash"] + path["shares"] * prices.reindex(path["date"]).to_numpy()
                                   + path["dividend_receivable"] - path["dividend_tax_reserved"], atol=1e-6, rtol=0)
        if len(detail):
            assert (detail["signal_date"] < detail["entry_date"]).all()
            assert (detail["exit_signal_date"] < detail["exit_date"]).all()
            assert (detail["entry_date"] < detail["exit_date"]).all()
            np.testing.assert_allclose(detail["net_pnl"], detail["shares"] * detail["exit_price"] - detail["exit_cost"]
                                       + detail["gross_dividends"] - detail["dividend_tax"] - detail["entry_total"], atol=1e-6, rtol=0)
            np.testing.assert_allclose(detail["net_return"], detail["net_pnl"] / detail["entry_total"], atol=1e-12, rtol=0)
            before_cash = path.set_index("date")["cash"].shift().fillna(initial)
            caps = before_cash.reindex(pd.to_datetime(detail["entry_date"])).to_numpy() * result["base_config"]["max_position_fraction"]
            assert (detail["entry_total"].to_numpy() <= caps + 1e-6).all()
        checks += 1
    assert checks == 168
    return paths, trades, {"account_paths_checked": checks, "closed_trade_rows_checked": len(trades),
                            "reconciled": True, "nonnegative_cash_and_round_lots": True,
                            "signal_before_execution": True}


def stock_prices(directory):
    key = directory.parent.name
    raw = pd.read_csv(ROOT / f"data/{key}/processed/{key}_raw.csv", index_col="date", parse_dates=["date"])
    return raw["close"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, leaderboards, records, validations, detail_tables = {}, {}, [], {}, []
    original_catalog = json.loads((ROOT / "research/zijin/indicator_lab/strategy_catalog.json").read_text())["candidates"]
    for key, (name, code) in STOCKS.items():
        directory = ROOT / f"research/{key}/indicator_lab"
        result = json.loads((directory / "results.json").read_text())
        results[key] = result
        catalog = json.loads((directory / "strategy_catalog.json").read_text())["candidates"]
        assert catalog == original_catalog
        leaderboards[key] = pd.read_csv(directory / "leaderboard.csv").set_index("id")
        for rule in catalog:
            row = {"stock": key, "name": name, "symbol": code, "strategy": rule["id"], "strategy_name": rule["name"]}
            for period in ("full", "train", "review"):
                stats = result["results"]["native"][period][rule["id"]]
                row.update({f"{period}_{field}": stats[field] for field in (
                    "total_return", "annual_return", "ending_equity", "max_drawdown", "closed_trades", "wins", "win_rate", "average_position_weight")})
            for period in ("full", "review"):
                alternative = result["sensitivity"]["entry_cap30_only"][period][rule["id"]]
                row.update({f"{period}_entry_cap30_only_{field}": alternative[field]
                            for field in ("total_return", "max_drawdown", "closed_trades", "average_position_weight")})
            records.append(row)
        if key == "zijin":
            continue
        paths, trades, validations[key] = check_accounts(directory, result)
        features = pd.read_csv(directory / "features_and_signals.csv", index_col="date", parse_dates=["date"])
        for strategy, fraction in (("passive_30", .3), ("passive_100", 1.0)):
            run = backtest(features, result["base_config"], strategy, passive_fraction=fraction)
            np.testing.assert_allclose(run["stats"]["ending_equity"], result["benchmarks"]["full"][strategy]["ending_equity"], atol=1e-6, rtol=0)
            run["equity"].to_csv(directory / f"equity_full_{strategy}.csv", index_label="date")
            run["trades"].to_csv(directory / f"trades_full_{strategy}.csv", index=False)
        validations[key]["benchmark_paths_rebuilt_and_checked"] = 2
        full = trades.loc[(trades["period"] == "full") & (trades["exit_mode"] == "native") & (trades["strategy"] == "donchian_breakout")]
        full.to_csv(directory / "donchian_breakout_full_native_trades.csv", index=False)
        display = full[["signal_date", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "net_pnl", "net_return", "exit_reason"]].copy()
        for col in ("entry_price", "exit_price", "net_pnl"):
            display[col] = display[col].map(lambda x: f"{x:.2f}")
        display["net_return"] = display["net_return"].map(percent)
        display = display.rename(columns=dict(zip(display.columns, ["信号日", "买入日", "卖出日", "买入价", "卖出价", "股数", "净盈亏元", "单笔收益", "退出原因"])))
        detail_tables.append((name, display, result["results"]["native"]["full"]["donchian_breakout"]))
    baseline = results["zijin"]["base_config"]
    for key in ("cmb", "gigadevice"):
        for parameter, value in baseline.items():
            if parameter not in ("research_as_of", "notes"):
                assert results[key]["base_config"][parameter] == value, parameter
    save_json(OUT / "validation.json", {"fixed_catalog_matches_zijin": True, "financial_parameters_match_zijin": True,
                                        "checks": validations, "method": "reconcile each stored main/control path and closed trade; no independent audit of source prices"})
    pd.DataFrame(records).to_csv(OUT / "strategy_comparison.csv", index=False)
    overview = []
    for rule in original_catalog:
        row = {"策略": rule["name"]}
        row["紫金三年收益"] = percent(results["zijin"]["results"]["native"]["full"][rule["id"]]["total_return"])
        for key in ("cmb", "gigadevice"):
            name = STOCKS[key][0]
            stats = results[key]["results"]["native"]["full"][rule["id"]]
            row.update({name + "三年收益": percent(stats["total_return"]), name + "最大回撤": percent(stats["max_drawdown"]),
                        name + "平仓笔数": stats["closed_trades"]})
        overview.append(row)
    summary = pd.DataFrame(overview)
    sections = [("同一套规则跨股票比较", "研究日2026-10-19；回放区间2023-10-18至2026-10-16，两只新股票各726个交易日。各策略、各股票使用独立10万元账户；三年收益以全部初始资金为分母，不能将各行或各股票收益直接相加。紫金为既有结果，本次未重跑。", summary)]
    sections.append(("执行口径", "原28条规则、参数、30%入场资金上限与1%名义风险预算均保持一致，收盘发信号、下一交易日开盘执行，100股取整、T+1和涨跌停近似。初始2ATR止损、3ATR移动保护、3R目标和最长20交易日共同生效。佣金双边万2.5且最低5元，过户费双边十万分之一，卖出印花税万5，滑点双边万5；分红与红利税另计。空仓现金收益为零。期末未平仓按收盘价估值，胜率只计已平仓。", None))
    benchmark_rows = []
    for key in ("cmb", "gigadevice"):
        for label, strategy in (("唐奇安20日突破", "donchian_breakout"), ("唐奇安：仅30%入场上限", "donchian_cap30"),
                                ("被动持有：30%入场上限", "passive_30"), ("被动持有：约满仓", "passive_100")):
            if strategy == "donchian_breakout":
                stats = results[key]["results"]["native"]["full"][strategy]
            elif strategy == "donchian_cap30":
                stats = results[key]["sensitivity"]["entry_cap30_only"]["full"]["donchian_breakout"]
            else:
                stats = results[key]["benchmarks"]["full"][strategy]
            benchmark_rows.append({"股票": STOCKS[key][0], "规则": label, "三年收益": percent(stats["total_return"]),
                                   "年化": percent(stats["annual_return"]), "期末总资产元": f"{stats['ending_equity']:.2f}",
                                   "最大回撤": percent(stats["max_drawdown"]), "平均持仓权重": percent(stats["average_position_weight"])})
    sections.append(("唐奇安与被动基准", "30%被动基准仅在起始按30%资金上限买入后持有，不再平衡；100股取整后实际入场投入可能低于上限，持仓权重随价格变化。与动态风险限仓策略的暴露不同，收益和回撤需一起比较。仅30%入场上限的唐奇安对照取消1%ATR风险限仓，但仍保留ATR退出。", pd.DataFrame(benchmark_rows)))
    for name, table, stats in detail_tables:
        position = json.dumps(stats["open_position"], ensure_ascii=False)
        sections.append((name + "唐奇安逐笔操作", f"平仓{stats['closed_trades']}笔，盈利{stats['wins']}笔，胜率{percent(stats['win_rate'])}；期末未平仓信息：{position}。价格为未复权开盘参考价，滑点单独列成本；单笔收益分母为该笔含费买入金额。closing_atr_stop=ATR收盘保护，closing_3R_target=3R止盈，indicator_exit=跌破此前10日低点，holding_20_sessions=20交易日时间退出。", table))
    lot_diagnostics = json.loads((ROOT / "research/gigadevice/indicator_lab/lot_diagnostics.json").read_text())
    sections.append(("兆易创新最后一年无法开仓", f"复查段{lot_diagnostics['review_sessions']}个交易日，100股的入场前2ATR名义风险最低{lot_diagnostics['minimum_one_lot_nominal_risk']:.2f}元，全部高于10万元账户的1000元预算。28条主规则均零成交、零持仓，因此复查收益均为0%；不能把它当作信号本身的盈利验证。下表展示原先即已设定的取消1%风险限仓、仅保留30%入场资金上限的敏感性，仍使用同样ATR退出。更高资金暴露同时改变收益与风险，不据此事后替换主规则。", None))
    cap_rows = []
    for rule in original_catalog:
        row = {"策略": rule["name"]}
        for key in ("cmb", "gigadevice"):
            name = STOCKS[key][0]
            for period, label in (("full", "三年收益"), ("review", "最后一年收益")):
                row[name + label] = percent(results[key]["sensitivity"]["entry_cap30_only"][period][rule["id"]]["total_return"])
            row[name + "三年回撤"] = percent(results[key]["sensitivity"]["entry_cap30_only"]["full"][rule["id"]]["max_drawdown"])
        cap_rows.append(row)
    sections.append(("仅限30%投入敏感性完整结果", "此表与主比较使用相同入场与退出信号，差别为取消1%名义风险预算的限仓；仍按100股取整，不使用杠杆。", pd.DataFrame(cap_rows)))
    sections.append(("解释边界", "全部规则固定沿用，没有依据两只股票结果修改阈值。事后三年第一名仅为本次样本排名；前两年筛选、最后一年复查及成本/持仓期敏感性见各股报告。该日历窗口已参与此前紫金研究，本次又同时查看56个股票策略组合，不称独立样本外验证或稳定收益。MACD＋黄金板块仍照搬黄金和贵金属过滤，未替换为银行、半导体行业。兆易创新高股价与100股整数限制可能使小额账户不能按信号开仓；不能将低交易次数解读为规则稳定。", None))
    sections.append(("数据与核验", "两只股票的腾讯原始/前复权行情、新浪分红于2026-10-19联网取得；processed数据截止2026-10-16，raw响应按原样保留。黄金、贵金属和沪指来自2026-10-18已存缓存。原始与前复权日期对应，逐日加回已除息股息后的序列与腾讯前复权序列仅差固定常数；无送转股。贵金属2026-10-15缺口不填补。核对336条主比较/退出对照账户路径和全部对应已平仓明细的现金、净资产及盈亏一致性，见validation.json；这不构成行情来源的独立审计。", None))
    links = [("招商银行完整报告", "../cmb/indicator_lab/report.html"), ("招商银行文字版", "../cmb/indicator_lab/report.md"),
             ("兆易创新完整报告", "../gigadevice/indicator_lab/report.html"), ("兆易创新文字版", "../gigadevice/indicator_lab/report.md"),
             ("全部数值CSV", "strategy_comparison.csv"), ("账户核验", "validation.json")]
    title = "招商银行、兆易创新与紫金：固定28条规则三年对比"
    md, blocks = ["# " + title, ""], []
    for heading, paragraph, table in sections:
        md.extend(["## " + heading, "", paragraph, ""])
        block = f"<section><h2>{html.escape(heading)}</h2><p>{html.escape(paragraph)}</p>"
        if table is not None:
            md.extend([markdown_table(table), ""])
            block += '<div class="scroll">' + table.to_html(index=False, border=0, escape=True) + '</div>'
        blocks.append(block + "</section>")
    md.extend(["## 报告与复现", "", "```bash", ".venv/bin/python scripts/fetch_comparison_stocks.py",
               ".venv/bin/python scripts/explore_zijin_indicators.py --stock cmb",
               ".venv/bin/python scripts/explore_zijin_indicators.py --stock gigadevice",
               ".venv/bin/python scripts/compare_indicator_stocks.py", "```", ""])
    md.extend(f"- [{label}]({url})" for label, url in links)
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>body{{font:16px/1.7 system-ui;color:#223047;background:#f4f6fa;margin:0}}main{{max-width:1400px;margin:auto;padding:28px}}section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}.scroll{{overflow:auto}}a{{color:#175db2}}</style><main><h1>{title}</h1>'
                + "".join(blocks) + '<section><h2>文件</h2><ul>' + ''.join(f'<li><a href="{url}">{label}</a></li>' for label, url in links) + '</ul></section></main></html>')
    (OUT / "report.html").write_text(document)
    save_json(OUT / "reproduction.json", {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                          "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                                          "inputs": {f"research/{key}/indicator_lab/{file}": hashlib.sha256((ROOT / f"research/{key}/indicator_lab/{file}").read_bytes()).hexdigest()
                                                     for key in STOCKS for file in ("results.json", "leaderboard.csv", "strategy_catalog.json")}})
    print(summary.to_string(index=False))
    print(pd.DataFrame(benchmark_rows).to_string(index=False))
    print("Accounts reconciled:", validations)


if __name__ == "__main__":
    main()
