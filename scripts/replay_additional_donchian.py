"""对长城A、赛力斯H、阿里港股复用固定唐奇安仓位与退出情景。"""

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explore_zijin_indicators import markdown_table, percent, save_json
import matplotlib.pyplot as plt
from replay_aggressive_donchian import check_run
from stock_analysis.indicator_lab import add_indicators, candidates
from stock_analysis.swing import backtest, make_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/additional_donchian"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / "config/additional_donchian.json"
    plan = json.loads(plan_path.read_text())
    inherited_path = ROOT / plan["inherited_scenarios"]
    inherited = json.loads(inherited_path.read_text())
    base_path = ROOT / "config/zijin_swing.json"
    base = json.loads(base_path.read_text())
    provenance = {"defined_before_replay_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
                  "inherited_scenarios": inherited, "base": base,
                  "plan_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (plan_path, inherited_path, base_path)},
                  "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), ROOT / "stock_analysis/swing.py",
                         ROOT / "stock_analysis/indicator_lab.py", ROOT / "scripts/replay_aggressive_donchian.py")}, "input_sha256": {}}
    for stock in plan["stocks"]:
        manifest_path = ROOT / f"data/{stock['id']}/manifest.json"
        provenance["input_sha256"][str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
        for entry in json.loads(manifest_path.read_text())["files"]:
            assert sha(ROOT / entry["path"]) == entry["sha256"]
    save_json(OUT / "reproduction.json", provenance)
    rows, trades_list, paths_list, entries_list, benchmark_rows, tax_rows, quality = [], [], [], [], [], [], {}
    results = {}
    for stock in plan["stocks"]:
        key = stock["id"]
        directory = OUT / key
        directory.mkdir(exist_ok=True)
        raw = pd.read_csv(ROOT / f"data/{key}/processed/{key}_raw.csv", index_col="date", parse_dates=["date"])
        qfq = pd.read_csv(ROOT / f"data/{key}/processed/{key}_qfq.csv", index_col="date", parse_dates=["date"])
        dividends = pd.read_csv(ROOT / f"data/{key}/processed/{key}_dividends.csv", parse_dates=["ex_date"])
        assert raw.index.equals(qfq.index)
        assert dividends["ex_date"].isin(raw.index).all()
        empty = pd.DataFrame({"close": pd.Series(dtype=float)}, index=pd.DatetimeIndex([]))
        frame = add_indicators(make_features(raw, empty, empty, empty, dividends))
        rule = next(rule for rule in candidates(frame) if rule.key == "donchian_breakout")
        frame["entry_donchian_breakout"], frame["exit_donchian_breakout"] = rule.entry, rule.exit
        frame.to_csv(directory / "features_and_signals.csv", index_label="date")
        offset = frame["analysis_close"] - qfq["close"]
        assert offset.max() - offset.min() < .004
        effective_start = max(pd.Timestamp(plan["requested_start"]), raw.index[plan["new_listing_warmup_sessions"]])
        start = effective_start.date().isoformat()
        periods = {"full": (start, plan["test_end"])}
        if raw.index[0] < pd.Timestamp(plan["requested_start"]):
            periods.update({"first_two_years": (start, "2025-10-17"), "last_year_review": ("2025-10-18", plan["test_end"])})
        quality[key] = {"name": stock["name"], "symbol": stock["symbol"], "market": stock["market"], "currency": stock["currency"],
            "raw_start": raw.index[0].date().isoformat(), "raw_end": raw.index[-1].date().isoformat(), "raw_rows": len(raw),
            "effective_start": start, "end": plan["test_end"], "test_sessions": len(raw.loc[start:plan["test_end"]]),
            "entry_signals_in_test": int(rule.entry.loc[start:plan["test_end"]].sum()),
            "calendar": "provider's local-market sessions, no forward filling; HK independent exchange calendar not audited",
            "raw_qfq_dates_aligned": True, "forward_cash_vs_qfq_offset_min": float(offset.min()),
            "forward_cash_vs_qfq_offset_max": float(offset.max()), "dividend_quality": json.loads((ROOT / f"data/{key}/quality.json").read_text())}
        config = dict(base, market=stock["market"], lot_size=stock["lot_size"], initial_cash=plan["initial_cash"])
        if stock["market"] == "HK":
            config.update({field: plan["hk_execution_assumptions"][field] for field in
                ("commission_rate", "minimum_commission", "platform_fee", "slippage_rate", "dividend_tax_rate")})
        results[key] = {}
        for profile in inherited["profiles"]:
            for mode in inherited["exit_modes"]:
                scenario = profile["id"] + "__" + mode["id"]
                cfg = dict(config, risk_per_trade=profile["risk_per_trade"], max_position_fraction=profile["max_position_fraction"],
                           max_holding_days=mode["max_holding_days"])
                results[key][scenario] = {}
                for period, (period_start, end) in periods.items():
                    run = backtest(frame, cfg, "donchian_breakout", period_start, end, signals=rule.entry,
                                   indicator_exit=rule.exit, use_ma20_exit=False, use_profit_target=mode["use_profit_target"])
                    entries = check_run(run, frame, cfg)
                    stats = run["stats"]
                    results[key][scenario][period] = stats
                    rows.append({"stock": key, "stock_name": stock["name"], "symbol": stock["symbol"], "currency": stock["currency"],
                        "profile": profile["id"], "profile_name": profile["name"], "exit_mode": mode["id"], "period": period,
                        "start": run["equity"].index[0].date().isoformat(), "end": run["equity"].index[-1].date().isoformat(),
                        **{field: stats[field] for field in ("total_return", "max_drawdown", "ending_equity", "closed_trades", "wins", "win_rate", "average_position_weight", "exposure_days")},
                        "entry_count": len(entries), "mean_entry_total": entries["entry_total"].mean() if len(entries) else None,
                        "open_position": stats["open_position"] is not None})
                    for table, collection in ((run["trades"].copy(), trades_list), (run["equity"].reset_index(), paths_list), (entries, entries_list)):
                        for field, value in (("period", period), ("scenario", scenario), ("currency", stock["currency"]), ("stock", key)):
                            table.insert(0, field, value)
                        collection.append(table)
                    if period == "full" and stock["market"] == "HK":
                        tax_cfg = dict(cfg, dividend_tax_rate=plan["hk_execution_assumptions"]["dividend_tax_sensitivity"])
                        tax = backtest(frame, tax_cfg, "donchian_breakout", period_start, end, signals=rule.entry,
                                       indicator_exit=rule.exit, use_ma20_exit=False, use_profit_target=mode["use_profit_target"])
                        check_run(tax, frame, tax_cfg)
                        results[key][scenario]["dividend_tax20_sensitivity"] = tax["stats"]
                        tax_rows.append({"stock": key, "scenario": scenario, "base_return": stats["total_return"],
                            "tax20_return": tax["stats"]["total_return"], "tax20_drawdown": tax["stats"]["max_drawdown"]})
                print(stock["name"], profile["name"], mode["name"], "完成", flush=True)
        results[key]["passive"] = {}
        for profile in inherited["profiles"]:
            fraction = profile["max_position_fraction"]
            run = backtest(frame, config, "passive", start, plan["test_end"], passive_fraction=fraction)
            run["equity"].to_csv(directory / f"passive_{profile['id']}.csv", index_label="date")
            results[key]["passive"][profile["id"]] = run["stats"]
            benchmark_rows.append({"stock": key, "name": stock["name"], "currency": stock["currency"], "start": start,
                "end": plan["test_end"], "entry_cap": fraction, **{field: run["stats"][field] for field in ("total_return", "max_drawdown", "average_position_weight")}})
    # 新增港股分支不应改变既有A股默认路径，检查三个已知完整样本。
    for stock in ("zijin", "cmb", "gigadevice"):
        directory = ROOT / f"research/{stock}/indicator_lab"
        frame = pd.read_csv(directory / "features_and_signals.csv", index_col="date", parse_dates=["date"])
        previous = json.loads((directory / "results.json").read_text())["results"]["native"]["full"]["donchian_breakout"]
        run = backtest(frame, base, "donchian_breakout", signals=frame["entry_donchian_breakout"],
                       indicator_exit=frame["exit_donchian_breakout"], use_ma20_exit=False)
        np.testing.assert_allclose(run["stats"]["ending_equity"], previous["ending_equity"], atol=1e-6, rtol=0)
        assert run["stats"]["closed_trades"] == previous["closed_trades"]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat([table for table in trades_list if len(table)], ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.concat(paths_list, ignore_index=True).to_csv(OUT / "account_paths.csv", index=False)
    pd.concat(entries_list, ignore_index=True).to_csv(OUT / "entries.csv", index=False)
    pd.DataFrame(tax_rows).to_csv(OUT / "dividend_tax_sensitivity.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(OUT / "passive_benchmarks.csv", index=False)
    save_json(OUT / "results.json", results)
    save_json(OUT / "data_quality.json", quality)
    save_json(OUT / "validation.json", {"main_scenario_period_runs": len(rows), "dividend_tax_sensitivity_runs": len(tax_rows),
        "passive_full_runs": len(benchmark_rows), "legacy_a_share_runs_matched": 3,
        "cash_equity_risk_round_lot_ledger_checks_passed": True})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    combined_paths = pd.concat(paths_list, ignore_index=True)
    for ax, stock in zip(axes, plan["stocks"]):
        key = stock["id"]
        feature = pd.read_csv(OUT / key / "features_and_signals.csv", index_col="date", parse_dates=["date"])
        price = feature.loc[quality[key]["effective_start"]:plan["test_end"], "close"]
        ax.plot(price.index, price / price.iloc[0], label="Stock price / first close", color="#777777")
        for mode in ("original_exit", "trend_exit"):
            path = combined_paths.loc[(combined_paths.stock == key) & (combined_paths.period == "full") &
                                       (combined_paths.scenario == "risk3_cap70__" + mode)]
            ax.plot(pd.to_datetime(path["date"]), path["equity"] / plan["initial_cash"], label="3%/70% " + mode)
        ax.set_title(stock["symbol"] + " " + stock["currency"])
        ax.set_ylabel("Multiple of starting value")
        ax.grid(alpha=.2); ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=25)
    fig.savefig(OUT / "comparison.png", dpi=150); plt.close(fig)
    blocks, md = [], ["# 长城汽车A、赛力斯H、阿里巴巴港股：唐奇安情景回放", ""]
    paragraphs = [
        "研究日2026-10-19，行情截止2026-10-16。长城为A股601633；赛力斯为港股09927；阿里为港股09988。每股每情景独立初始10万本币，长城用人民币、港股用港元，不计人民币/港元汇兑收益，也不把三股金额相加当作组合收益。",
        "风险与投入上限、入场和退出参数完整继承上一轮：1%/30%、2%/50%、3%/70%、5%/100%，对比原3R/20日退出与无固定止盈和期限的趋势退出；保留初始2ATR、移动3ATR及10日低点退出，不加杠杆、不加仓，收盘信号下一开盘成交。",
        "赛力斯2025-11-05上市，无法做真实三年H股回放。先用20个交易日预热通道，实际比较开始日见表；被动基准也从同日开始，未使用上市前A股历史或虚构价格。该股无前两年/最后一年独立分段，短样本不年化宣传。其他两股保留原前三年及分段复查口径。",
        "港股按各100股一手，成交不套A股10%涨跌停，未建模日内交易；次日开盘执行来自信号时点，不是把港股误设为A股T+1。采用香港现金账户的HKD口径；股票日历来自提供方，没有跨市场前向填充，也没有独立审计完整港交所日历。",
        "港股佣金假设双边0.03%且最低3港元，平台费每次15港元，滑点双边0.05%；这是研究假设，实际券商费率未知。印花税双边2023-11-17前0.13%、其后0.1%并向上取整港元；交易费0.00565%、证监会征费0.0027%、会财局征费0.00015%；交收费2025-06-30前0.002%且最低2最高100港元，之后0.0042%无最低最高，适用费用按分取整。长城A股沿用原交易费用与红利税。",
        "港股主比较扣交易成本、股息按税前：个人税务身份及港股通路径未知，另提供20%股息税敏感性。港元现金股息根据行情源未复权/前复权差分推算，除净日与新浪独立日程核对；阿里实际美元股息到账港元转换率未核实，因此港股结果为该现金代理假设下的近似回放。不能将其标为用户实际税后收益。",
        "所有窗口均已在本次研究中查看，属于历史情景比较，不称独立样本外或稳定收益验证。风险比例是开仓名义预算，不是实际亏损或最大回撤上限；不据最高收益事后调整规则。"
    ]
    if quality["seres_hk"]["entry_signals_in_test"] == 0:
        paragraphs.append("赛力斯H股预热后214个交易日没有触发20日上穿入场事件，因此所有仓位/退出档均为零交易、零收益、零回撤；这一结果表示规则未参与，不构成盈利能力或低风险已验证。")
    for paragraph in paragraphs:
        md += [paragraph, ""]; blocks.append("<p>" + html.escape(paragraph) + "</p>")
    for stock in plan["stocks"]:
        key = stock["id"]; q = quality[key]
        table_rows = []
        for row in summary.loc[(summary.stock == key) & (summary.period == "full")].to_dict("records"):
            table_rows.append({"仓位": row["profile_name"], "退出": "原短线" if row["exit_mode"] == "original_exit" else "趋势",
                "期间收益": percent(row["total_return"]), "最大回撤": percent(row["max_drawdown"]),
                "开仓/平仓": f"{row['entry_count']}/{row['closed_trades']}", "胜率": percent(row["win_rate"]),
                "每次平均投入本币": "—" if pd.isna(row["mean_entry_total"]) else f"{row['mean_entry_total']:.2f}",
                "平均持仓": percent(row["average_position_weight"])})
        table = pd.DataFrame(table_rows)
        heading = f"{stock['name']} {stock['symbol']}：{q['effective_start']}至{q['end']}，{q['test_sessions']}个交易日，{stock['currency']}"
        md += ["## " + heading, "", markdown_table(table), ""]
        blocks.append("<h2>" + heading + "</h2><div class='scroll'>" + table.to_html(index=False, border=0) + "</div>")
        baseline = pd.DataFrame([{"起始投入上限": percent(row["entry_cap"]), "买入持有收益": percent(row["total_return"]),
                                "最大回撤": percent(row["max_drawdown"])} for row in benchmark_rows if row["stock"] == key])
        md += ["同区间买入持有基准（按上限买入，整手取整后可能低于上限）：", "", markdown_table(baseline), ""]
        blocks.append("<h3>同区间被动基准</h3>" + baseline.to_html(index=False, border=0))
    sources = [("港交所赛力斯上市公告", "https://www.hkexnews.hk/listedco/listconews/sehk/2025/1104/2025110402282.pdf"),
        ("港交所交易费用", "https://www.hkex.com.hk/Services/Rules-and-Forms-and-Fees/Fees/Securities-(Hong-Kong)/Trading/Transaction?sc_lang=en"),
        ("印花税调整生效日", "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/SEHK/2023/ce_SEHK_CT_158_2023.pdf"),
        ("股份交收费调整", "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/HKSCC/2025/ce_HKSCC_SET_022_2025.pdf"),
        ("阿里每手股数", "https://www.alibabagroup.com/en-US/document-1607442761408053248"),
        ("赛力斯分红日程", "https://stock.finance.sina.com.cn/hkstock/dividends/09927.html"),
        ("阿里分红日程", "https://stock.finance.sina.com.cn/hkstock/dividends/09988.html")]
    md += ["## 复现与文件", "", "```bash", ".venv/bin/python scripts/fetch_extended_stocks.py", ".venv/bin/python scripts/replay_additional_donchian.py", "```", "",
        "`summary.csv`为所有时期数值；`trades.csv`为逐笔平仓；`entries.csv`包含未平仓开仓；`account_paths.csv`为每日资金；`dividend_tax_sensitivity.csv`为港股20%股息税对照；各股子目录保留全部指标信号和被动资金曲线。", "", "## 来源", ""]
    md += [f"- [{label}]({url})" for label, url in sources]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    document = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>长城、赛力斯H、阿里唐奇安回放</title><style>body{font:16px/1.7 system-ui;color:#223047;max-width:1400px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style>"
        "<h1>长城汽车A、赛力斯H、阿里巴巴港股：唐奇安情景回放</h1><img src='comparison.png' style='max-width:100%'>" + "".join(blocks)
        + "<p><a href='summary.csv'>全部数值</a> · <a href='trades.csv'>逐笔交易</a> · <a href='entries.csv'>开仓金额</a> · <a href='dividend_tax_sensitivity.csv'>股息税敏感性</a></p></html>")
    (OUT / "report.html").write_text(document)
    provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(OUT / "reproduction.json", provenance)
    print(summary.loc[summary.period == "full", ["stock_name", "profile", "exit_mode", "start", "total_return", "max_drawdown", "entry_count", "closed_trades"]].to_string(index=False))
    print("Report:", OUT / "report.html")


if __name__ == "__main__": main()
