"""固定仓位档位与退出方式，诊断唐奇安策略的收益、回撤和趋势参与。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_zijin_indicators import STOCKS, markdown_table, percent, save_json
from stock_analysis.swing import backtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/aggressive_donchian"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(run, frame, config):
    equity, trades, stats = run["equity"], run["trades"], run["stats"]
    if not ((equity["cash"] >= -1e-6) & (equity["shares"] >= 0) & (equity["shares"] % 100 == 0)).all():
        raise ValueError("负现金或非整手持仓")
    np.testing.assert_allclose(equity["equity"], equity["cash"] + equity["shares"] * frame["close"].reindex(equity.index)
                               + equity["dividend_receivable"] - equity["dividend_tax_reserved"], atol=1e-6, rtol=0)
    if len(trades):
        assert (trades["signal_date"] < trades["entry_date"]).all()
        assert (trades["exit_signal_date"] < trades["exit_date"]).all()
        np.testing.assert_allclose(trades["net_pnl"], trades["shares"] * trades["exit_price"] - trades["exit_cost"]
                                   + trades["gross_dividends"] - trades["dividend_tax"] - trades["entry_total"], atol=1e-6, rtol=0)
    entry_mask = (equity["shares"] > 0) & (equity["shares"].shift().fillna(0) == 0)
    before_cash = equity["cash"].shift().fillna(config["initial_cash"])
    amounts = before_cash.loc[entry_mask] - equity.loc[entry_mask, "cash"]
    assert (amounts <= before_cash.loc[entry_mask] * config["max_position_fraction"] + 1e-6).all()
    risk_distances = (frame["atr14"].shift() * config["stop_atr"]).reindex(amounts.index)
    nominal_risks = equity.loc[entry_mask, "shares"] * risk_distances
    assert (nominal_risks <= before_cash.loc[entry_mask] * config["risk_per_trade"] + 1e-6).all()
    entries = pd.DataFrame({"entry_date": amounts.index.strftime("%Y-%m-%d"),
                            "entry_total": amounts.to_numpy(), "cash_before": before_cash.loc[entry_mask].to_numpy(),
                            "shares": equity.loc[entry_mask, "shares"].to_numpy(),
                            "nominal_initial_risk": nominal_risks.to_numpy()})
    assert len(entries) == len(trades) + int(stats["open_position"] is not None)
    return entries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / "config/aggressive_donchian.json"
    plan = json.loads(plan_path.read_text())
    base = json.loads((ROOT / "config/zijin_swing.json").read_text())
    provenance = {"defined_before_replay_utc": datetime.now(timezone.utc).isoformat(),
                  "plan": plan, "plan_sha256": sha(plan_path), "base_config": base,
                  "source_snapshots": {}, "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                      (Path(__file__), ROOT / "stock_analysis/swing.py", ROOT / "scripts/explore_zijin_indicators.py")},
                  "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                  "sample_status": "same previously explored historical window; no untouched out-of-sample claim"}
    for stock in plan["stocks"]:
        directory = ROOT / f"research/{stock}/indicator_lab"
        provenance["source_snapshots"][stock] = {str(p.relative_to(ROOT)): sha(p) for p in
            (directory / "features_and_signals.csv", directory / "results.json", ROOT / f"data/{stock}/manifest.json")}
        manifest = json.loads((ROOT / f"data/{stock}/manifest.json").read_text())
        for item in manifest["files"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    save_json(OUT / "reproduction.json", provenance)
    rows, all_trades, all_paths, all_entries, result, benchmark_rows = [], [], [], [], {}, []
    for stock in plan["stocks"]:
        name, code = STOCKS[stock]
        directory = ROOT / f"research/{stock}/indicator_lab"
        frame = pd.read_csv(directory / "features_and_signals.csv", index_col="date", parse_dates=["date"])
        assert frame["entry_donchian_breakout"].dtype == bool
        assert frame["exit_donchian_breakout"].dtype == bool
        previous = json.loads((directory / "results.json").read_text())
        result[stock] = {}
        for profile in plan["profiles"]:
            result[stock][profile["id"]] = {}
            for mode in plan["exit_modes"]:
                scenario = profile["id"] + "__" + mode["id"]
                config = dict(base, risk_per_trade=profile["risk_per_trade"], max_position_fraction=profile["max_position_fraction"],
                              max_holding_days=mode["max_holding_days"])
                result[stock][profile["id"]][mode["id"]] = {}
                for period, (start, end) in plan["periods"].items():
                    run = backtest(frame, config, "donchian_breakout", start, end,
                                   signals=frame["entry_donchian_breakout"], indicator_exit=frame["exit_donchian_breakout"],
                                   use_ma20_exit=False, use_profit_target=mode["use_profit_target"])
                    entries = check_run(run, frame, config)
                    stats = run["stats"]
                    result[stock][profile["id"]][mode["id"]][period] = stats
                    if profile["id"] == "original" and mode["id"] == "original_exit":
                        old_period = {"full": "full", "first_two_years": "train", "last_year_review": "review"}[period]
                        old = previous["results"]["native"][old_period]["donchian_breakout"]
                        np.testing.assert_allclose(stats["ending_equity"], old["ending_equity"], atol=1e-6, rtol=0)
                        assert stats["closed_trades"] == old["closed_trades"]
                    row = {"stock": stock, "stock_name": name, "symbol": code, "profile": profile["id"],
                           "profile_name": profile["name"], "exit_mode": mode["id"], "period": period,
                           "risk_per_trade": profile["risk_per_trade"], "max_position_fraction": profile["max_position_fraction"],
                           **{k: stats[k] for k in ("total_return", "annual_return", "max_drawdown", "ending_equity", "closed_trades", "wins", "win_rate", "average_position_weight", "exposure_days")},
                           "entry_count": len(entries), "mean_entry_total": entries["entry_total"].mean() if len(entries) else None,
                           "open_position": stats["open_position"] is not None}
                    if period == "full":
                        sample = frame.loc[start:end]
                        peak = sample["close"].idxmax()
                        row.update(stock_closing_peak_date=peak.date().isoformat(),
                                   account_return_at_stock_peak=run["equity"].loc[peak, "equity"] / base["initial_cash"] - 1,
                                   position_at_stock_peak=run["equity"].loc[peak, "position_weight"])
                    rows.append(row)
                    for table, collection in ((run["trades"].copy(), all_trades), (run["equity"].reset_index(), all_paths), (entries, all_entries)):
                        table.insert(0, "period", period)
                        table.insert(0, "scenario", scenario)
                        table.insert(0, "stock", stock)
                        collection.append(table)
                print(name, profile["name"], mode["name"], "完成", flush=True)
        # 同一资金档位的被动持有供比较，原始结果已有，保留原始字节指纹。
        result[stock]["benchmarks"] = previous["benchmarks"]
        result[stock]["matched_cap_passive_full"] = {}
        for profile in plan["profiles"]:
            fraction = profile["max_position_fraction"]
            passive = backtest(frame, base, "passive", plan["test_start"], plan["test_end"], passive_fraction=fraction)
            stats = passive["stats"]
            result[stock]["matched_cap_passive_full"][profile["id"]] = stats
            passive["equity"].to_csv(OUT / f"passive_equity_{stock}_{profile['id']}.csv", index_label="date")
            benchmark_rows.append({"stock": stock, "stock_name": name, "entry_cap": fraction,
                **{field: stats[field] for field in ("total_return", "max_drawdown", "average_position_weight", "ending_equity")}})
            if fraction in (.3, 1.0):
                reference = previous["benchmarks"]["full"]["passive_30" if fraction == .3 else "passive_100"]
                np.testing.assert_allclose(stats["ending_equity"], reference["ending_equity"], atol=1e-6, rtol=0)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat([table for table in all_trades if len(table)], ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.concat(all_paths, ignore_index=True).to_csv(OUT / "account_paths.csv", index=False)
    pd.concat(all_entries, ignore_index=True).to_csv(OUT / "entries.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(OUT / "passive_benchmarks.csv", index=False)
    save_json(OUT / "results.json", result)
    save_json(OUT / "validation.json", {"scenario_period_runs_checked": len(rows),
              "legacy_stock_period_runs_matched": len(plan["stocks"]) * len(plan["periods"]),
              "matched_cap_passive_paths_saved": len(benchmark_rows),
              "cash_equity_ledger_and_initial_risk_checked": True, "no_leverage_round_lots_and_next_day_execution_checked": True})
    sections = []
    for stock in plan["stocks"]:
        full = summary.loc[(summary.stock == stock) & (summary.period == "full")]
        review = summary.loc[(summary.stock == stock) & (summary.period == "last_year_review")].set_index(["profile", "exit_mode"])
        display_rows = []
        for row in full.to_dict("records"):
            last = review.loc[(row["profile"], row["exit_mode"])]
            display_rows.append({"仓位方案": row["profile_name"], "退出": "原短线退出" if row["exit_mode"] == "original_exit" else "趋势退出",
                "三年收益": percent(row["total_return"]), "最大回撤": percent(row["max_drawdown"]),
                "平均每次投入元": "—" if pd.isna(row["mean_entry_total"]) else f"{row['mean_entry_total']:.2f}",
                "平均持仓": percent(row["average_position_weight"]), "开仓/平仓": f"{row['entry_count']}/{row['closed_trades']}",
                "最后一年收益": percent(last["total_return"]), "最后一年开仓": int(last["entry_count"]),
                "股价最高收盘日账户收益": percent(row["account_return_at_stock_peak"])})
        sections.append((STOCKS[stock][0], pd.DataFrame(display_rows)))
        benchmark_display = pd.DataFrame([{"起始投入上限": percent(row["entry_cap"]),
            "被动持有三年收益": percent(row["total_return"]), "最大回撤": percent(row["max_drawdown"]),
            "平均持仓": percent(row["average_position_weight"])} for row in benchmark_rows if row["stock"] == stock])
        sections.append((STOCKS[stock][0] + "：买入持有基准", benchmark_display))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, stock in zip(axes, plan["stocks"]):
        subset = summary.loc[(summary.stock == stock) & (summary.period == "full")]
        for mode, marker in (("original_exit", "o"), ("trend_exit", "^")):
            group = subset.loc[subset.exit_mode == mode]
            ax.scatter(-group.max_drawdown * 100, group.total_return * 100, marker=marker, label=mode)
            for r in group.itertuples():
                ax.annotate(f"{r.risk_per_trade:.0%}/{r.max_position_fraction:.0%}", (-r.max_drawdown * 100, r.total_return * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(STOCKS[stock][1]); ax.set_xlabel("Maximum account drawdown (%)"); ax.set_ylabel("3-year account return (%)")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(OUT / "risk_return.png", dpi=150); plt.close(fig)
    paragraphs = [
        "研究日2026-10-19，历史窗口2023-10-18至2026-10-16，数据来自已归档快照；没有刷新当前行情。每股每方案独立10万元，不加杠杆，不是三股合并组合。三年收益以全部初始资金为分母，期末持仓按收盘估值且不强制卖出。",
        "固定比较风险预算1%/2%/3%/5%与投入上限30%/50%/70%/100%的四个成对档位；档位同时改变风险和投入上限，不能分别归因。各档再对比原3R止盈+20日上限与取消盈利目标/期限的趋势退出；保持同一20日上穿事件入场、10日低点指标退出、初始2ATR/移动3ATR保护，不加仓，也未改变重新入场机制。",
        "风险预算是开仓前账户资金乘比例，再按2ATR风险距离限股数，不代表实际最大亏损或账户最大回撤。高股价可能仍因100股取整或资金上限无法入场。收盘触发、次日开盘执行，跳空和跌停会扩大损失。资金占用和利润回吐需要同时看。",
        "投入上限按每次开仓时的账户资金计算，盈利滚动投入；平均开仓金额可能超过最初10万元，但没有借款。被动基准按相同30%/50%/70%/100%资金上限起始买入并持有，100股取整后可低于上限，后续权重随价格漂移；它与动态风险限仓的暴露仍不同，需结合平均持仓解释。",
        "成本沿用原佣金双边万2.5且最低5元、过户费双边十万分之一、卖出印花税万5、滑点双边万5；分红与红利税另计，空仓现金收益为0。保持A股T+1和涨跌停近似。",
        "原方案通过与此前三股、三个时期的已存结果逐项核对；所有72组账户均核验现金、净资产、逐笔盈亏、资金与初始名义风险上限及整手规则。最后一年属于已参与研究的分段复查；最高收盘日仅作事后诊断，不是可预知卖点。不根据结果调整参数，也不将事后收益最高档位自动推荐为实盘配置。"
    ]
    md = ["# 唐奇安策略：提高仓位与延长趋势持有的情景比较", "", "![收益与回撤](risk_return.png)", ""]
    md += [paragraph + "\n" for paragraph in paragraphs]
    blocks = [f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs]
    for name, table in sections:
        md += ["## " + name, "", markdown_table(table), ""]
        blocks.append(f"<h2>{name}</h2><div class='scroll'>" + table.to_html(index=False, border=0) + "</div>")
    md += ["## 文件与复现", "", "```bash", ".venv/bin/python scripts/replay_aggressive_donchian.py", "```", "",
           "参数见 `config/aggressive_donchian.json`；`summary.csv`保留全部时期统计；`trades.csv`为已平仓记录；`entries.csv`包含期末未平仓的开仓金额；`account_paths.csv`为每日账户。", "",
           "仓位与风险承受能力：[Investor.gov](https://www.investor.gov/introduction-investing/getting-started/asset-allocation)。本报告的数值档位为研究情景，不来自该来源的个人风险建议。"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    document = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>唐奇安进取情景比较</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#223047}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}img{width:100%}</style>"
                "<h1>唐奇安：提高仓位与趋势持有</h1><img src='risk_return.png'>" + "".join(blocks)
                + "<p><a href='summary.csv'>全部数值</a> · <a href='trades.csv'>逐笔交易</a> · <a href='entries.csv'>开仓金额</a> · <a href='account_paths.csv'>每日资金</a></p></html>")
    (OUT / "report.html").write_text(document)
    provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(OUT / "reproduction.json", provenance)
    print(summary.loc[summary.period == "full", ["stock_name", "profile", "exit_mode", "total_return", "max_drawdown", "mean_entry_total", "entry_count"]].to_string(index=False))
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()
