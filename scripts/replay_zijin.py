"""读取本地快照，生成紫金三年关联分析、交易回放、报告与图表。"""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_analysis.relationships import FACTORS, relationship_report
from stock_analysis.swing import STRATEGIES, backtest, entry_signals, make_features

OUT = ROOT / "research/zijin"


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pct(value):
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def dec(value):
    return "—" if value is None or pd.isna(value) else f"{value:.3f}"


def table_md(frame):
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows += ["| " + " | ".join(str(x).replace("|", " / ") for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config/zijin_swing.json").read_text())
    manifest = json.loads((ROOT / "data/zijin/manifest.json").read_text())
    for record in manifest["files"]:
        if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"数据文件与来源清单校验不符：{record['path']}")
    def load(name):
        return pd.read_csv(ROOT / f"data/zijin/processed/{name}.csv", index_col="date", parse_dates=["date"])
    raw, gold, sector, sse = (load(name) for name in ("zijin_raw", "gold_au9999", "precious_metals_sw801053", "sse"))
    dividends = pd.read_csv(ROOT / "data/zijin/processed/zijin_dividends.csv", parse_dates=["ex_date"])
    if ((dividends["bonus_per10"] != 0) | (dividends["split_per10"] != 0)).any():
        raise ValueError("当前回放只支持无送转股的样本")
    if raw.index.min() > pd.Timestamp(config["download_start"]) + pd.Timedelta(days=7) or raw.index.max() < pd.Timestamp(config["test_end"]):
        raise ValueError("行情未完整覆盖请求区间")
    frame = make_features(raw, gold, sector, sse, dividends)
    for strategy in STRATEGIES:
        frame[f"entry_{strategy}"] = entry_signals(frame, strategy)
    frame.to_csv(OUT / "daily_features.csv", index_label="date")
    relationships = relationship_report(frame, config)
    relationships.pop("daily_aligned").to_csv(OUT / "daily_aligned_returns.csv", index_label="date")
    rolling = relationships.pop("rolling")
    rolling.to_csv(OUT / "rolling_correlations.csv", index_label="date")
    save_json(OUT / "relationships.json", relationships)

    periods = {"full": (config["test_start"], config["test_end"]),
               "first_two_years": (config["test_start"], "2025-10-17"),
               "last_year_holdout": (config["holdout_start"], config["test_end"])}
    results, summary, full_runs = {}, [], {}
    for period, (start, end) in periods.items():
        results[period] = {}
        for strategy in [*STRATEGIES, "passive_30", "passive_100"]:
            passive = .30 if strategy == "passive_30" else 1.0 if strategy == "passive_100" else None
            run = backtest(frame, config, strategy, start, end, passive_fraction=passive)
            results[period][strategy] = run["stats"]
            summary.append({"period": period, "strategy": strategy, **{k: v for k, v in run["stats"].items() if not isinstance(v, (dict, list))}})
            run["trades"].to_csv(OUT / f"trades_{period}_{strategy}.csv", index=False)
            run["equity"].to_csv(OUT / f"equity_{period}_{strategy}.csv", index_label="date")
            if period == "full":
                full_runs[strategy] = run
    # 固定规则的成本压力检验，不选择或改变最优参数。
    stressed_config = dict(config, slippage_rate=config["slippage_rate"] * 2,
                            commission_rate=config["commission_rate"] * 2, minimum_commission=config["minimum_commission"] * 2)
    stress = {name: backtest(frame, stressed_config, name)["stats"] for name in STRATEGIES}
    save_json(OUT / "backtest_results.json", {"config": config, "periods": results, "doubled_commission_and_slippage": stress})
    pd.DataFrame(summary).to_csv(OUT / "backtest_summary.csv", index=False)

    sample = frame.loc[config["test_start"]:config["test_end"]]
    alignment = {name: [day.date().isoformat() for day in sample.index[sample[f"{name}_close"].isna()]] for name in FACTORS}
    adjusted = load("zijin_qfq")
    offsets = (frame["analysis_close"] - adjusted["close"]).dropna()
    quality = {"raw_rows": len(raw), "test_sessions": len(sample), "test_first": str(sample.index[0].date()),
               "test_last": str(sample.index[-1].date()), "missing_external_dates": alignment,
               "vendor_qfq_vs_forward_cash_adjusted_offset_min": float(offsets.min()),
               "vendor_qfq_vs_forward_cash_adjusted_offset_max": float(offsets.max()),
               "raw_vs_qfq_date_alignment": bool(raw.index.equals(adjusted.index)),
               "dividend_records": len(dividends), "data_manifest_sha256": hashlib.sha256((ROOT / "data/zijin/manifest.json").read_bytes()).hexdigest()}
    save_json(OUT / "data_quality.json", quality)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), constrained_layout=True)
    columns = {"stock_total_return_index": "Zijin gross total return", "gold_close": "SGE Au99.99", "sector_close": "SW precious metals", "sse_close": "SSE Composite"}
    for col, label in columns.items():
        series = sample[col].dropna()
        axes[0].plot(series.index, series / series.iloc[0], label=label, linewidth=1.5)
    axes[0].set_title("Market comparison (first close = 1; not strategy P&L)")
    plot_names = {"pullback_technical": "Technical pullback", "pullback_gold_sector": "Gold + sector pullback", "pullback_all": "All filters pullback", "passive_30": "Passive 30%"}
    for name, label in plot_names.items():
        nav = full_runs[name]["equity"]["equity"] / config["initial_cash"]
        axes[1].plot(nav.index, nav, label=label)
        axes[3].plot(nav.index, (nav / nav.cummax().clip(lower=1) - 1) * 100, label=label)
    axes[1].set_title("Account NAV after costs (entry cap 30%; swing risk budget 1%)")
    for name, label in FACTORS.items():
        axes[2].plot(rolling.index, rolling[f"{name}_corr60"], label={"gold": "Gold", "sector": "Precious metals", "sse": "SSE"}[name])
    axes[2].axhline(0, color="grey", linewidth=.6)
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("Rolling 60-session contemporaneous return correlation")
    axes[3].set_title("Daily closing account drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(OUT / "overview.png")
    plt.close(fig)

    full = results["full"]
    corr_table = pd.DataFrame([{"比较对象": r["name"], "日收益相关": dec(r["daily"]["pearson"]),
                               "周收益相关": dec(r["weekly"]["pearson"]), "单因素 Beta": dec(r["daily"]["beta"]),
                               "单因素 R²": pct(r["daily"]["r_squared"]),
                               "前日因素→当日紫金相关": dec(r["lag1_predictive"]["pearson"]), "日配对数": r["daily"]["n"]} for r in relationships["factors"]])
    def result_table(period):
        records = []
        for name, stats in results[period].items():
            ci = stats["win_rate_wilson95"]
            records.append({"策略": STRATEGIES.get(name, "被动买入持有 30%" if name == "passive_30" else "被动买入持有约 100%"),
                            "平仓笔数": stats["closed_trades"], "胜率": pct(stats["win_rate"]),
                            "胜率参考区间": f"{pct(ci[0])}～{pct(ci[1])}" if ci[0] is not None else "—",
                            "账户累计收益": pct(stats["total_return"]), "年化收益": pct(stats["annual_return"]),
                            "最大回撤": pct(stats["max_drawdown"]), "利润因子": dec(stats["profit_factor"]),
                            "平均持仓权重": pct(stats["average_position_weight"])})
        return pd.DataFrame(records)
    backtest_table, holdout_table = result_table("full"), result_table("last_year_holdout")
    regression = relationships["multifactor"]
    regression_table = pd.DataFrame([{"因素": FACTORS[name], "控制其他因素后 Beta": dec(regression["coefficients"][name]["beta"]),
                                     "HAC 95% 参考区间": "～".join(dec(x) for x in regression["coefficients"][name]["ci95"]),
                                     "增量 R²": pct(regression["incremental_r_squared"][name]), "VIF": dec(regression["vif"][name])} for name in FACTORS])
    stress_table = pd.DataFrame([{"策略": STRATEGIES[name], "原成本收益": pct(full[name]["total_return"]),
                                 "佣金与滑点翻倍收益": pct(stress[name]["total_return"])} for name in STRATEGIES])
    main_trades = full_runs["pullback_all"]["trades"]
    reason_names = {"closing_atr_stop": "ATR 收盘止损", "closing_3R_target": "达到 3R 目标", "close_below_ma20": "收盘跌破 MA20", "holding_20_sessions": "持满 20 个交易日"}
    trade_table = pd.DataFrame([{"信号日": trade["signal_date"], "买入日": trade["entry_date"], "卖出日": trade["exit_date"],
                               "买入价": f"{trade['entry_price']:.2f}", "卖出价": f"{trade['exit_price']:.2f}",
                               "股数": int(trade["shares"]), "持有交易日": int(trade["holding_sessions"]),
                               "交易净收益率": pct(trade["net_return"]), "账户净盈亏元": f"{trade['net_pnl']:.2f}",
                               "退出原因": reason_names[trade["exit_reason"]]} for _, trade in main_trades.iterrows()])
    feature_columns = ["close", "analysis_close", "cumulative_dividend", "raw_reference_ma20", "raw_reference_ma60", "rsi14", "atr14", "macd_dif", "macd_dea", "volume_ratio",
                       "gold_filter", "sector_filter", "market_filter", *[f"entry_{s}" for s in STRATEGIES]]
    sample[feature_columns].tail(1).to_csv(OUT / "latest_indicators.csv", index_label="date")
    latest = sample.iloc[-1]
    latest_table = pd.DataFrame([{"行情日": str(sample.index[-1].date()), "原始收盘价": f"{latest['close']:.2f}",
                                 "MA20 当日原价等效线": f"{latest['raw_reference_ma20']:.2f}",
                                 "MA60 当日原价等效线": f"{latest['raw_reference_ma60']:.2f}",
                                 "RSI14": f"{latest['rsi14']:.2f}", "ATR14 元": f"{latest['atr14']:.2f}",
                                 "全部条件回踩入场信号": "是" if latest["entry_pullback_all"] else "否"}])

    paragraphs = [
        ("范围与结论", f"研究日为 {config['research_as_of']}，完整回放区间为 {quality['test_first']} 至 {quality['test_last']}，共 {len(sample)} 个紫金交易日。研究标的是紫金矿业 A 股 601899，不是港股。下载自 2023-01-03 起的行情用于指标预热。全部数值来自本地真实数据快照。"),
        ("关联程度", f"紫金与黄金、贵金属板块、沪指的同期日收益相关系数分别为 {dec(relationships['factors'][0]['daily']['pearson'])}、{dec(relationships['factors'][1]['daily']['pearson'])}、{dec(relationships['factors'][2]['daily']['pearson'])}。本样本与板块的联动最强。前一日因素收益与当日紫金收益的相关性明显弱于同期关系；同期联动不能直接当作次日买入依据。"),
        ("三因素共同解释", f"在共同完整日期上，黄金、贵金属板块与沪指的三因素同期回归 R² 为 {pct(regression['r_squared'])}。黄金控制其他因素后的区间跨过零，不能据此断言存在稳定的独立作用。板块与黄金本身相关，不能把三份单因素解释度相加，也不能将回归视为因果或价格预测。紫金还经营铜等业务，此模型未覆盖铜价、汇率、公司事件和全部风险。"),
        ("交易结果应如何读", f"全部过滤条件的回踩策略平仓 {full['pullback_all']['closed_trades']} 笔，胜率 {pct(full['pullback_all']['win_rate'])}，账户累计收益 {pct(full['pullback_all']['total_return'])}，最大收盘回撤 {pct(full['pullback_all']['max_drawdown'])}。样本内加入沪指过滤后的收益低于只用黄金＋板块的变体，不能预设过滤越多越好。策略多数时间持有现金；收益应同时与相同初始 30% 投入上限的被动持有和接近满仓持有比较。"),
        ("成功率的定义", "胜率 = 扣除买卖费用、滑点成本和红利税后净盈利的已平仓交易数 / 全部已平仓交易数。期末未平仓交易只按市值计入账户净值，不计入胜率。利润因子 = 平仓盈利金额合计 / 平仓亏损金额绝对值合计，不是平均盈亏比。95% Wilson 区间仅作有限交易次数的参考，交易的市场状态依赖会削弱独立二项假设。样本内胜率不代表下一笔交易的成功概率。"),
        ("全部条件回踩策略的逐笔回放", "表中的交易净收益率以实际买入金额含成本为分母，账户收益则以全部账户资金为基础，两者不能混用。例如 2025 年 4 月 3 日收盘触发退出，下一交易日 4 月 7 日跌停开盘无法卖出，顺延至 4 月 8 日。2025 年 6 月的持仓跨过除息日，净盈亏包含股息和卖出时红利税，均可从交易 CSV 核对。"),
        ("规则一 回踩企稳", "收盘后同时满足：MA20 > MA60；连续价格收盘在 MA60 和 MA20 上方；当日最低价不高于 MA20 × 1.02；当日收盘上涨；RSI14 位于 45～65；成交量 / 此前 20 日平均成交量 ≥ 0.8。指标在前向现金分红连续价格上计算。"),
        ("规则二 放量突破", "收盘后同时满足：MA20 > MA60；连续价格收盘在 MA60 上方且突破此前 20 日最高价；RSI14 位于 50～75；成交量 / 此前 20 日平均成交量 ≥ 1.2。突破策略使用全部外部过滤条件。MACD 和布林带作为辅助观察输出，不加入这一版入场评分，避免重复计算同一价格信息。"),
        ("外部入场过滤", "黄金过滤：Au99.99 收盘 > MA20 且 MA20 > MA60。板块过滤：申万贵金属 801053 收盘 > MA20，且紫金 20 日含税前股息总收益高于板块 20 日价格收益。沪指过滤：000001 收盘 > MA60。依次比较无外部过滤、黄金、黄金＋板块和全部过滤；外部过滤只控制入场，不单独触发持仓退出。"),
        ("仓位与执行", "研究账户初始 10 万元。买入数量取资金约束和风险约束的较小者：投入含买入成本不超过当前账户净值的 30%；按 2 × ATR14 的初始止损距离计算的名义风险预算不超过净值 1%；股数向下取 100 股整数倍。信号日收盘后决定，下一交易日开盘成交，另扣滑点成本。入场日可以产生退出信号，最早下一交易日卖出，遵守 T+1。"),
        ("退出规则", "初始收盘止损线为入场连续价格 − 2 × 入场前 ATR14。之后止损线只上移至持仓以来最高收盘连续价格 − 3 × 当前 ATR14 的较高者。收盘触及止损、达到初始风险 3 倍的目标、跌破 MA20 或持满 20 个交易日，任一发生则下个交易日开盘退出。这里是收盘触发规则，盘中触及不退出；跳空和连续跌停可能使实际损失超过名义 1% 风险预算。"),
        ("成交约束与费用", "开盘价达到 10% 涨停时不买，达到跌停时不卖；卖出委托顺延至可执行日，买入信号失效后不延续旧买单。无成交量时不成交；同日卖出后不重新买入。日线无法证明开盘排队可成交，使用上述保守近似。佣金双边万 2.5、每笔最低 5 元；过户费双边十万分之一；卖出印花税万 5；滑点成本双边万 5。佣金、最低费率和滑点是可修改假设。"),
        ("分红与历史信息", "使用原始价格和真实股数成交；除息日将前一日持有股份应得税前股息记为应收红利，卖出时结算并按持有期计提 20% / 10% / 0% 红利税；应收红利不用于加仓。期末未卖出股息计提按当日退出的税率估值。技术连续价格 = 原始 OHLC + 从下载起点至当日累计已除息现金股息，避免未来分红影响过去信号。本样本无送转股；这不等于逐笔真实资金清算，实际派息到账日未逐笔建模。"),
        ("时点与留出段", "Au99.99 日行情可晚于 A 股 15 点收盘，全部条件在当日相关市场收盘后评估、次日 A 股开盘执行。按日期对齐，不倒填缺失板块数据；缺失日不允许依赖它开新仓。参数在首次计算策略结果前固定，不做最优胜率寻优。最后一年从 2025-10-18 起另开空仓账户进行时间留出回放，前两年结果单独保留；这是回溯验证，不能替代未来实盘或独立市场验证。"),
        ("最新已完成交易日的规则状态", "下表仅对本地截止行情计算规则状态，不是实时指令。MA20 / MA60 的当日原价等效线 = 连续价格均线 − 截至当日累计已除息股息，可以与原始收盘价直接比较；其余 CSV 中的 ma20 / ma60 字段仍为连续价格均线。"),
        ("数据质量与边界", f"保存 {len(raw)} 行紫金原始和复权行情、{len(gold)} 行黄金、{len(sector)} 行贵金属指数、{len(sse)} 行沪指和 {len(dividends)} 条现金分红记录。回测区间缺失的外部日期：{json.dumps(alignment, ensure_ascii=False)}。收益关联使用相同日期间隔，不把缺失值补成零。指数采用来源发布的历史序列，无法重建历史权重；若指数包含紫金，存在成分重叠联动，不能视为完全独立证据。未建模盘口排队、历史可用修订版本和逐笔冲击成本，收盘回撤也不包含日内最差损失。"),
        ("操作结论", "把黄金和板块作为市场状态与相对强弱的条件，再用个股技术信号决定何时观察入场；控制仓位，并严格记录触发与退出原因。本次交易次数不足以确认稳定优势，不据样本内表现挑出唯一最优策略。若要实用化，先用后续新增日期做模拟跟踪，并将铜价和重大公司事件纳入复核。"),
    ]
    tables_by_heading = {"关联程度": corr_table, "三因素共同解释": regression_table, "全部条件回踩策略的逐笔回放": trade_table,
                         "交易结果应如何读": backtest_table, "时点与留出段": holdout_table,
                         "最新已完成交易日的规则状态": latest_table}
    source_links = [
        ("腾讯紫金矿业行情", "https://gu.qq.com/sh601899/gp"),
        ("上海黄金交易所每日行情", "https://www.sge.com.cn/sjzx/mrhq"),
        ("申万贵金属指数 801053", "https://www.swsresearch.com/institute_sw/allIndex/releasedIndex/releasedetail?code=801053&name=%E8%B4%B5%E9%87%91%E5%B1%9E"),
        ("新浪分红记录", "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/601899.phtml"),
        ("上交所股票交易机制", "https://english.sse.com.cn/start/trading/mechanism/"),
        ("上交所投资税费说明", "https://one.sse.com.cn/onething/gptz/"),
        ("税务总局股息红利差别化政策", "https://www.chinatax.gov.cn/n810341/n810755/c1797427/content.html"),
        ("紫金 2024 年年度报告摘要", "https://www.zjky.cn/upload/file/2025/03/21/8cd9d4e04a5b44b08b410a5fbb2680d2.pdf"),
    ]
    markdown = ["# 紫金矿业 A 股三年波段交易回放", "", "![关联与回放图表](overview.png)", ""]
    blocks = []
    for heading, paragraph in paragraphs:
        markdown.extend([f"## {heading}", "", paragraph, ""])
        blocks.append(f"<section><h2>{html.escape(heading)}</h2><p>{html.escape(paragraph)}</p>")
        if heading in tables_by_heading:
            table = tables_by_heading[heading]
            markdown.extend([table_md(table), ""])
            blocks.append('<div class="scroll">' + table.to_html(index=False, border=0, escape=True) + "</div>")
        blocks.append("</section>")
    markdown.extend(["## 成本压力检验", "", "仅佣金和滑点成本翻倍，印花税和过户费不变。", "", table_md(stress_table), "",
                     "## 文件与复现", "", "在仓库根目录执行：", "", "```bash", ".venv/bin/python scripts/fetch_zijin_data.py", ".venv/bin/python scripts/replay_zijin.py", "```", "",
                     "原始数据与来源校验：`data/zijin/`。指标逐日数据：`daily_features.csv`；全部结果：`backtest_results.json`；各策略逐笔交易：`trades_*.csv`；每日净值：`equity_*.csv`。留出段账户从零持仓开始。图中被动 30% 为起始投入后不再平衡，后来权重可上升；策略入场上限 30% 也不是每日再平衡上限。", "", "## 来源", ""])
    markdown += [f"- [{label}]({url})" for label, url in source_links]
    (OUT / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    import base64
    image = base64.b64encode((OUT / "overview.png").read_bytes()).decode()
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>紫金矿业 A 股三年波段交易回放</title><style>'
                'body{font:16px/1.8 system-ui,sans-serif;color:#1c2838;background:#f6f8fb;margin:0}main{max-width:1120px;margin:auto;padding:40px 24px}'
                'h1{font-size:32px}h2{font-size:22px;margin-top:30px}section{background:white;border-radius:12px;padding:18px 24px;margin:18px 0}'
                'table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}td,th{border-bottom:1px solid #e1e6ed;padding:10px;text-align:right}'
                'th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{width:100%;background:white}a{color:#135ba1}</style><main>'
                '<h1>紫金矿业 A 股三年波段交易回放</h1><p>2023-10-18 至 2026-10-16 · 真实数据 · 次日执行 · 含成本与分红</p>'
                + f'<img alt="市场关联与策略净值" src="data:image/png;base64,{image}">' + "".join(blocks)
                + '<section><h2>成本压力检验</h2><p>仅佣金和滑点成本翻倍。</p><div class="scroll">' + stress_table.to_html(index=False, border=0) + '</div></section>'
                + '<section><h2>来源与复现</h2><p>原始响应、CSV 和校验值位于 data/zijin；规则位于 config/zijin_swing.json；执行 scripts/replay_zijin.py 生成本报告及逐笔明细。</p><ul>'
                + "".join(f'<li><a href="{html.escape(url)}">{html.escape(label)}</a></li>' for label, url in source_links) + '</ul></section></main></html>')
    (OUT / "report.html").write_text(document, encoding="utf-8")
    save_json(OUT / "reproduction.json", {"config": config, "data_quality": quality,
                                         "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                                         "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                                         [Path(__file__), ROOT / "stock_analysis/swing.py", ROOT / "stock_analysis/relationships.py"]}})
    print(corr_table.to_string(index=False))
    print(backtest_table.to_string(index=False))
    print("最后一年独立起始账户：")
    print(holdout_table.to_string(index=False))
    print(f"报告：{OUT / 'report.html'}")


if __name__ == "__main__":
    main()
