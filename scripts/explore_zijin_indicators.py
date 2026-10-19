"""运行固定的 28 个指标候选，完整保留成功和失败结果。"""

from __future__ import annotations

import base64
import argparse
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_analysis.indicator_lab import add_indicators, candidates
from stock_analysis.swing import backtest, make_features

OUT = ROOT / "research/zijin/indicator_lab"
STOCKS = {"zijin": ("紫金矿业", "601899"), "cmb": ("招商银行", "600036"),
          "gigadevice": ("兆易创新", "603986")}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def percent(value):
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def decimal(value):
    return "—" if value is None or pd.isna(value) else f"{value:.2f}"


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", " / ") for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def yearly_rows(run, name, initial_cash):
    output = []
    previous_equity = initial_cash
    for year, group in run["equity"].groupby(run["equity"].index.year):
        exits = run["trades"]
        exits = exits.loc[exits["exit_date"].str.startswith(str(year))] if len(exits) else exits
        peaks = group["equity"].cummax().clip(lower=previous_equity)
        output.append({"strategy": name, "year": int(year), "return": group["equity"].iloc[-1] / previous_equity - 1,
                       "drawdown_within_year": (group["equity"] / peaks - 1).min(), "closed_trades": len(exits),
                       "wins": int((exits["net_pnl"] > 0).sum()) if len(exits) else 0})
        previous_equity = group["equity"].iloc[-1]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", choices=STOCKS, default="zijin", help="复用固定规则回放指定 A 股")
    parser.add_argument("--reuse-results", action="store_true", help="验证配置和数据后复用已有回测，仅重建报告")
    args = parser.parse_args()
    stock_key = args.stock
    stock_name, stock_code = STOCKS[stock_key]
    OUT = ROOT / f"research/{stock_key}/indicator_lab"
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "config/zijin_swing.json").read_text())
    if stock_key != "zijin":
        base.update(research_as_of="2026-10-19", notes=f"复用紫金固定规则至{stock_name}A股；空仓现金收益设为0。")
    lab = json.loads((ROOT / "config/indicator_lab.json").read_text())
    manifest_path = ROOT / f"data/{stock_key}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    external_manifest_path = ROOT / "data/zijin/manifest.json"
    external_manifest = json.loads(external_manifest_path.read_text())
    external_manifest_sha = hashlib.sha256(external_manifest_path.read_bytes()).hexdigest()
    cached = cached_provenance = None
    if args.reuse_results:
        cached_provenance = json.loads((OUT / "reproduction.json").read_text())
        if cached_provenance["base_config"] != base or cached_provenance["lab_config"] != lab:
            raise ValueError("配置已变化，需完整重新执行回测")
        if cached_provenance["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError("数据清单已变化，需完整重新执行回测")
        if stock_key != "zijin" and cached_provenance.get("external_manifest_sha256") != external_manifest_sha:
            raise ValueError("外部因子数据清单已变化，需完整重新执行回测")
        for relative in ("stock_analysis/indicator_lab.py", "stock_analysis/swing.py"):
            if cached_provenance["code_sha256"][relative] != hashlib.sha256((ROOT / relative).read_bytes()).hexdigest():
                raise ValueError("指标或交易引擎已变化，需完整重新执行回测")
        cached = json.loads((OUT / "results.json").read_text())
    for record in manifest["files"] + (external_manifest["files"] if stock_key != "zijin" else []):
        if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("数据快照校验失败：" + record["path"])
    def load(name, directory="zijin"):
        return pd.read_csv(ROOT / f"data/{directory}/processed/{name}.csv", index_col="date", parse_dates=["date"])
    dividends = pd.read_csv(ROOT / f"data/{stock_key}/processed/{stock_key}_dividends.csv", parse_dates=["ex_date"])
    if ((dividends["bonus_per10"] != 0) | (dividends["split_per10"] != 0)).any():
        raise ValueError("出现送转股，需先扩展成交股数和分红模型")
    raw = load(f"{stock_key}_raw", stock_key)
    frame = add_indicators(make_features(raw, load("gold_au9999"), load("precious_metals_sw801053"), load("sse"), dividends))
    if frame.index.min() > pd.Timestamp(base["test_start"]) - pd.Timedelta(days=100) or frame.index.max() < pd.Timestamp(base["test_end"]):
        raise ValueError("缺少指标预热历史或末日行情")
    sample = frame.loc[base["test_start"]:base["test_end"]]
    if stock_key != "zijin":
        qfq = load(f"{stock_key}_qfq", stock_key)
        if not raw.index.equals(qfq.index):
            raise ValueError("原始与前复权行情交易日不一致")
        if not dividends["ex_date"].isin(raw.index).all():
            raise ValueError("除息日缺少股票行情")
        expected = load("sse").loc[base["test_start"]:base["test_end"]].index
        missing = expected.difference(sample.index)
        if len(missing):
            raise ValueError(f"股票缺少沪市交易日，需核验停牌：{missing.tolist()}")
        offset = frame["analysis_close"] - qfq["close"]
        # 腾讯现金股息前复权序列与逐日加回分红序列应只差固定常数。
        if float(offset.max() - offset.min()) > .021:
            raise ValueError("分红与前复权序列无法对齐，需核验公司行动")
        quality = {"symbol": stock_code, "test_sessions": len(sample), "raw_rows": len(raw),
                   "raw_vs_qfq_date_alignment": True, "missing_stock_sessions": [],
                   "forward_cash_vs_qfq_offset_min": float(offset.min()),
                   "forward_cash_vs_qfq_offset_max": float(offset.max()),
                   "dividend_records": len(dividends),
                   "missing_factor_dates": {name: sample.index[sample[f"{name}_close"].isna()].strftime("%Y-%m-%d").tolist()
                                            for name in ("gold", "sector", "sse")}}
        save_json(OUT / "data_quality.json", quality)
    specs = candidates(frame)
    if len(specs) != lab["candidate_count"]:
        raise ValueError("候选数与预先保存配置不符")
    catalog = [{"id": spec.key, "name": spec.name, "category": spec.category,
                "entry": spec.entry_description, "exit": spec.exit_description} for spec in specs]
    # 此文件在运行任何新规则回测之前落地，保留全部候选的固定定义。
    catalog_path = OUT / "strategy_catalog.json"
    defined_at = datetime.now(timezone.utc).isoformat()
    if catalog_path.exists():
        previous_catalog = json.loads(catalog_path.read_text())
        if previous_catalog["candidates"] == catalog and previous_catalog["lab_config"] == lab:
            defined_at = previous_catalog["created_before_replay_utc"]
    save_json(catalog_path, {"created_before_replay_utc": defined_at,
                                             "lab_config": lab, "candidates": catalog})
    for spec in specs:
        frame[f"entry_{spec.key}"] = spec.entry
        frame[f"exit_{spec.key}"] = spec.exit
    frame.to_csv(OUT / "features_and_signals.csv", index_label="date")

    periods = {"full": (base["test_start"], base["test_end"]), "train": (base["test_start"], lab["train_end"]),
               "review": (lab["review_start"], base["test_end"])}
    main_config = dict(base, max_holding_days=lab["primary_max_holding_days"])
    results = {mode: {period: {} for period in periods} for mode in ("native", "common_risk")}
    full_runs, trades, yearly, daily_paths = {}, [], [], []
    if cached is not None:
        results = cached["results"]
        stored_paths = pd.read_csv(OUT / "account_paths.csv", parse_dates=["date"])
        stored_trades = pd.read_csv(OUT / "trades.csv")
        trades = [stored_trades]
        daily_paths = [stored_paths]
        yearly = pd.read_csv(OUT / "yearly_returns.csv").to_dict("records")
        for spec in specs:
            paths = stored_paths.loc[(stored_paths["strategy"] == spec.key) & (stored_paths["period"] == "full") & (stored_paths["exit_mode"] == "native")].set_index("date")
            detail = stored_trades.loc[(stored_trades["strategy"] == spec.key) & (stored_trades["period"] == "full") & (stored_trades["exit_mode"] == "native")]
            full_runs[spec.key] = {"stats": results["native"]["full"][spec.key], "equity": paths, "trades": detail}
    def run_spec(spec, config, start, end, mode="native", classic=False):
        return backtest(frame, config, spec.key, start, end, signals=spec.entry,
                        indicator_exit=spec.exit if mode == "native" else None,
                        use_ma20_exit=False, use_atr_risk_controls=not classic)
    for number, spec in enumerate(specs if cached is None else [], 1):
        for mode in results:
            for period, (start, end) in periods.items():
                run = run_spec(spec, main_config, start, end, mode)
                results[mode][period][spec.key] = run["stats"]
                detail = run["trades"].copy()
                detail.insert(0, "strategy", spec.key)
                detail.insert(0, "exit_mode", mode)
                detail.insert(0, "period", period)
                trades.append(detail)
                path = run["equity"].copy()
                path["strategy"], path["exit_mode"], path["period"] = spec.key, mode, period
                daily_paths.append(path.reset_index())
                if mode == "native" and period == "full":
                    full_runs[spec.key] = run
                    yearly.extend(yearly_rows(run, spec.key, base["initial_cash"]))
        if number % 7 == 0:
            print(f"主比较完成 {number}/{len(specs)} 个候选（各含两种退出、三个时期）", flush=True)

    # 选择只看前两年。最后一年已在旧研究使用过，故称复查段。
    eligible = [(spec.key, results["native"]["train"][spec.key]) for spec in specs
                if results["native"]["train"][spec.key]["closed_trades"] >= lab["selection_min_train_trades"]]
    eligible.sort(key=lambda item: (-item[1][lab["selection_metric"]], item[0]))
    selected = [key for key, _ in eligible[:lab["selection_top_n"]]]
    save_json(OUT / "train_selection.json", {"selection_uses": "train only", "eligible": [key for key, _ in eligible],
                                             "selected": selected, "minimum_train_closed_trades": lab["selection_min_train_trades"],
                                             "metric": lab["selection_metric"], "review_is_untouched_holdout": False})

    variants = {"holding40": dict(main_config, max_holding_days=lab["sensitivity_max_holding_days"]),
                "double_cost": dict(main_config, commission_rate=base["commission_rate"] * 2,
                                    minimum_commission=base["minimum_commission"] * 2, slippage_rate=base["slippage_rate"] * 2),
                "entry_cap30_only": dict(main_config, risk_per_trade=1.0)}
    sensitivity = {variant: {period: {} for period in ("full", "review")} for variant in variants}
    if cached is not None:
        sensitivity = cached["sensitivity"]
    for number, spec in enumerate(specs if cached is None else [], 1):
        for variant, config in variants.items():
            for period in sensitivity[variant]:
                start, end = periods[period]
                sensitivity[variant][period][spec.key] = run_spec(spec, config, start, end)["stats"]
        if number % 7 == 0:
            print(f"持仓期、成本、固定投入敏感性完成 {number}/{len(specs)}", flush=True)
    classic = {period: {} for period in periods}
    if cached is not None:
        classic = cached["classic_indicator_only"]
    classic_config = dict(main_config, max_holding_days=100000)
    for spec in specs if cached is None else []:
        if spec.key in ("macd_cross", "macd_zero", "ema_cross"):
            for period, (start, end) in periods.items():
                run = run_spec(spec, classic_config, start, end, classic=True)
                classic[period][spec.key] = run["stats"]
                run["trades"].to_csv(OUT / f"classic_trades_{period}_{spec.key}.csv", index=False)
                run["equity"].to_csv(OUT / f"classic_equity_{period}_{spec.key}.csv", index_label="date")
    benchmarks = {period: {} for period in periods}
    for period, (start, end) in periods.items():
        for key, fraction in (("passive_30", .3), ("passive_100", 1.0)):
            run = backtest(frame, main_config, key, start, end, passive_fraction=fraction)
            benchmarks[period][key] = run["stats"]
            if period == "full":
                full_runs[key] = run
    combined_trades = pd.concat([detail for detail in trades if len(detail)], ignore_index=True)
    annual = pd.DataFrame(yearly)
    if cached is None:
        combined_trades.to_csv(OUT / "trades.csv", index=False)
        pd.concat(daily_paths, ignore_index=True).to_csv(OUT / "account_paths.csv", index=False)
        annual.to_csv(OUT / "yearly_returns.csv", index=False)
    save_json(OUT / "results.json", {"base_config": main_config, "lab_config": lab, "results": results,
                                     "sensitivity": sensitivity, "classic_indicator_only": classic, "benchmarks": benchmarks})

    rows = []
    for spec in specs:
        full, train, review = (results["native"][period][spec.key] for period in ("full", "train", "review"))
        record = {"id": spec.key, "name": spec.name, "category": spec.category, "selected_by_train": spec.key in selected}
        for prefix, stats in (("full", full), ("train", train), ("review", review)):
            record.update({f"{prefix}_{key}": stats[key] for key in (
                "total_return", "annual_return", "max_drawdown", "closed_trades", "wins", "win_rate", "profit_factor",
                "average_holding_sessions", "average_position_weight", "average_trade_return")})
        record.update({"full_common_exit_return": results["common_risk"]["full"][spec.key]["total_return"],
                       "review_common_exit_return": results["common_risk"]["review"][spec.key]["total_return"],
                       "full_win_ci_low": full["win_rate_wilson95"][0], "full_win_ci_high": full["win_rate_wilson95"][1]})
        for variant in variants:
            for period in ("full", "review"):
                record[f"{period}_{variant}_return"] = sensitivity[variant][period][spec.key]["total_return"]
        rows.append(record)
    leaderboard = pd.DataFrame(rows).sort_values(["train_total_return", "id"], ascending=[False, True]).reset_index(drop=True)
    leaderboard.to_csv(OUT / "leaderboard.csv", index=False)
    display = pd.DataFrame({"规则": leaderboard["name"], "类型": leaderboard["category"],
                            "前两年收益": leaderboard["train_total_return"].map(percent),
                            "三年收益": leaderboard["full_total_return"].map(percent),
                            "三年回撤": leaderboard["full_max_drawdown"].map(percent),
                            "平仓笔数": leaderboard["full_closed_trades"], "胜率": leaderboard["full_win_rate"].map(percent),
                            "最后一年收益": leaderboard["review_total_return"].map(percent),
                            "最后一年笔数": leaderboard["review_closed_trades"],
                            "平均持仓权重": leaderboard["full_average_position_weight"].map(percent)})
    macd_display = display.loc[leaderboard["id"].str.contains("macd")].copy()
    selection_display = display.loc[leaderboard["id"].isin(selected)].copy()
    pairs = [("macd_cross", "macd_trend"), ("macd_cross", "macd_rsi"), ("macd_cross", "macd_volume"),
             ("macd_cross", "macd_adx"), ("macd_cross", "macd_gold_sector"),
             ("kdj_rebound", "kdj_trend"), ("bollinger_reentry", "bollinger_rsi"), ("donchian_breakout", "donchian_adx_volume")]
    mixed_comparison = []
    for parent, mixed in pairs:
        parent_spec, mixed_spec = next(s for s in specs if s.key == parent), next(s for s in specs if s.key == mixed)
        parent_full, mixed_full = results["native"]["full"][parent], results["native"]["full"][mixed]
        parent_review, mixed_review = results["native"]["review"][parent], results["native"]["review"][mixed]
        mixed_comparison.append({"基础规则": parent_spec.name, "混合规则": mixed_spec.name,
                                 "基础三年收益": percent(parent_full["total_return"]), "混合三年收益": percent(mixed_full["total_return"]),
                                 "三年收益变化百分点": f"{(mixed_full['total_return'] - parent_full['total_return']) * 100:+.2f}",
                                 "基础最后一年": percent(parent_review["total_return"]), "混合最后一年": percent(mixed_review["total_return"]),
                                 "混合平仓笔数": mixed_full["closed_trades"]})
    mixed_display = pd.DataFrame(mixed_comparison)
    mixed_display.to_csv(OUT / "mixed_vs_parent.csv", index=False)
    sensitivity_display = pd.DataFrame({"规则": leaderboard["name"], "主比较三年收益": leaderboard["full_total_return"].map(percent),
                                        "统一退出收益": leaderboard["full_common_exit_return"].map(percent),
                                        "延至40日收益": leaderboard["full_holding40_return"].map(percent),
                                        "佣金滑点翻倍": leaderboard["full_double_cost_return"].map(percent),
                                        "仅限30%投入收益": leaderboard["full_entry_cap30_only_return"].map(percent)})
    classic_display = pd.DataFrame([{"规则": next(s.name for s in specs if s.key == key),
                                     "三年收益": percent(stats["total_return"]), "三年回撤": percent(stats["max_drawdown"]),
                                     "平仓笔数": stats["closed_trades"], "胜率": percent(stats["win_rate"]),
                                     "平均持有交易日": decimal(stats["average_holding_sessions"]),
                                     "平均持仓权重": percent(stats["average_position_weight"]),
                                     "最后一年收益": percent(classic["review"][key]["total_return"]),
                                     "期末未平仓": "有" if stats["open_position"] else "无"} for key, stats in classic["full"].items()])
    catalog_display = pd.DataFrame([{"规则": s.name, "类型": s.category, "入场": s.entry_description, "指标退出": s.exit_description} for s in specs])

    # 所有候选均画出；按前两年排名展示，避免只展示事后最好的几条。
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = leaderboard["id"].tolist()
    lookup = {s.key: s for s in specs}
    fig, axes = plt.subplots(1, 2, figsize=(14, 12), constrained_layout=True)
    positions = np.arange(len(order))
    axes[0].barh(positions - .18, leaderboard["full_total_return"] * 100, height=.35, label="Full 3 years")
    axes[0].barh(positions + .18, leaderboard["review_total_return"] * 100, height=.35, label="Last-year review")
    axes[0].set_yticks(positions, order, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="grey", linewidth=.7)
    axes[0].set_title("Account return after costs (%)\nOrdered by first-two-year performance")
    axes[0].legend()
    years = sorted(annual["year"].unique())
    matrix = annual.pivot(index="strategy", columns="year", values="return").reindex(order)[years]
    limit = max(.05, float(matrix.abs().max().max()))
    image = axes[1].imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(len(years)), [f"{year}" + (" (partial)" if year in (2023, 2026) else "") for year in years], rotation=30)
    axes[1].set_yticks(positions, order, fontsize=8)
    for i in range(len(order)):
        for j in range(len(years)):
            axes[1].text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center", fontsize=7)
    axes[1].set_title("Calendar-period account return\n2023 and 2026 are partial years")
    fig.colorbar(image, ax=axes[1], fraction=.025)
    fig.savefig(OUT / "comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    sample = frame.loc[base["test_start"]:base["test_end"]]
    axes[0].plot(sample.index, sample["close"], color="#34495e", linewidth=1, label=f"{stock_code} raw close")
    macd_trades = full_runs["macd_cross"]["trades"]
    axes[0].scatter(pd.to_datetime(macd_trades["entry_date"]), macd_trades["entry_price"], marker="^", s=35, color="green", label="MACD entries (next open)")
    axes[0].scatter(pd.to_datetime(macd_trades["exit_date"]), macd_trades["exit_price"], marker="v", s=35, color="red", label="MACD exits (next open)")
    axes[0].set_title("MACD cross strategy with ATR controls and 20-session cap")
    axes[1].plot(sample.index, sample["macd_dif"], label="DIF EMA12 - EMA26")
    axes[1].plot(sample.index, sample["macd_dea"], label="DEA EMA9(DIF)")
    axes[1].bar(sample.index, sample["macd_hist"], width=1.5, color=np.where(sample["macd_hist"] >= 0, "#68aa87", "#c67973"), alpha=.5)
    axes[1].axhline(0, color="grey", linewidth=.7)
    plotted = list(dict.fromkeys(["macd_cross", "macd_rsi", *selected, "passive_30"]))
    for key in plotted:
        nav = full_runs[key]["equity"]["equity"] / base["initial_cash"]
        axes[2].plot(nav.index, nav, label=key, linewidth=1.2)
    axes[2].set_title("Account NAV: MACD variants, train-selected candidates, passive 30%")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(OUT / "macd_detail.png")
    plt.close(fig)

    macd = results["native"]["full"]["macd_cross"]
    bb = results["native"]["full"]["bollinger_reentry"]
    bb_review = results["native"]["review"]["bollinger_reentry"]
    bb_common = results["common_risk"]["full"]["bollinger_reentry"]
    classic_zero = classic["full"]["macd_zero"]
    hist_native = results["native"]["full"]["macd_hist_turn"]
    hist_common = results["common_risk"]["full"]["macd_hist_turn"]
    improved_pairs = [next(s.name for s in specs if s.key == mixed) for parent, mixed in pairs
                      if results["native"]["full"][mixed]["total_return"] > results["native"]["full"][parent]["total_return"]]
    top_full = leaderboard.sort_values("full_total_return", ascending=False).iloc[0]
    profitable_full = int((leaderboard["full_total_return"] > 0).sum())
    profitable_review = int((leaderboard["review_total_return"] > 0).sum())
    improvements = int((leaderboard["full_total_return"] > leaderboard["full_common_exit_return"]).sum())
    selected_text = "、".join(lookup[key].name for key in selected) if selected else "没有规则满足最少交易次数"
    def image_html(filename):
        encoded = base64.b64encode((OUT / filename).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(filename)}">'
    sections = [
        ("研究范围", f"回放 {base['test_start']} 至 {base['test_end']} 的{stock_name} A 股 {stock_code} 历史行情，共 {len(sample)} 个交易日，预热历史从 2023 年初开始。固定 28 个候选，其中 16 个单指标规则和 12 个混合规则，没有对阈值作网格寻优。初始账户 10 万元，只做多、不加杠杆。全部账户收益以初始10万元为分母，期末未平仓按收盘价估值并保留，不强制卖出。", None),
        ("比较口径", "主比较使用指标自身退出，另叠加初始 2ATR 收盘止损、最高收盘价减 3ATR 的移动保护、3R 目标和最长 20 个交易日；去掉上轮统一的跌破 MA20 退出，除非某个规则自身要求该条件。对照比较使用同样入场、相同 ATR/时间规则，移除指标自身退出。所有交易都是收盘产生信号、次日开盘执行，遵守 T+1、100 股整数倍和涨跌停近似。", None),
        ("总体结果", f"28 个主比较候选中，三年账户收益为正的有 {profitable_full} 个，最后一年为正的有 {profitable_review} 个。三年事后最高收益的候选为“{top_full['name']}”，收益 {percent(top_full['full_total_return'])}，平仓 {int(top_full['full_closed_trades'])} 笔；这是已做多次探索后的样本内排名，不能直接当作最优策略或未来成功率。完整表按前两年收益排序，所有失败结果也保留。", display),
        ("MACD 单独与混合的表现", f"MACD 12/26/9 金叉规则在共同风控下，三年账户收益 {percent(macd['total_return'])}、最大收盘回撤 {percent(macd['max_drawdown'])}、平仓 {macd['closed_trades']} 笔、胜率 {percent(macd['win_rate'])}。其胜率 Wilson 参考区间为 {percent(macd['win_rate_wilson95'][0])}～{percent(macd['win_rate_wilson95'][1])}。以下列出全部含 MACD 的候选，便于比较均线、RSI、放量、ADX、黄金板块等附加条件是否改善样本结果。", macd_display),
        ("混合指标是否改善结果", f"逐一与基础规则比较，八组混合比较中三年账户收益增加的有 {len(improved_pairs)} 组，分别为：{'、'.join(improved_pairs) or '无'}。这些差异同时受交易次数、投入股数及个别大波段影响；添加条件会减少交易，不能将一次收益改善解释为新增指标具有独立预测能力。", mixed_display),
        ("高胜率与收益的区别", f"布林下轨回归主比较为 {bb['closed_trades']} 笔平仓中 {bb['wins']} 笔盈利，胜率 {percent(bb['win_rate'])}，三年账户收益 {percent(bb['total_return'])}，最后一年为 {percent(bb_review['total_return'])}。本样本平均盈利约 {decimal(bb['average_win'])} 元，平均亏损绝对值约 {decimal(abs(bb['average_loss']) if bb['average_loss'] is not None else None)} 元。MACD 零轴经典对照的胜率为 {percent(classic_zero['win_rate'])}，三年收益为 {percent(classic_zero['total_return'])}，且仓位更大、持有更久。这说明评估时需要同时看胜率、盈亏幅度、交易次数、资金暴露和分段表现。", None),
        ("退出方式的重要性", f"MACD 绿柱拐点信号使用原生退出时三年收益为 {percent(hist_native['total_return'])}，改为仅共同风控后为 {percent(hist_common['total_return'])}，对应回撤分别为 {percent(hist_native['max_drawdown'])}、{percent(hist_common['max_drawdown'])}；最后一年对应收益为 {percent(results['native']['review']['macd_hist_turn']['total_return'])}、{percent(results['common_risk']['review']['macd_hist_turn']['total_return'])}。布林下轨回归使用达到中轨退出时收益为 {percent(bb['total_return'])}，改为共同风控后为 {percent(bb_common['total_return'])}。共同风控的保护和持有时间改变了盈亏分布，三年结果尚不能确认为稳定优势。", None),
        ("经典金叉买死叉卖对照", "另做三组纯指标退出对照：取消 ATR 风险限仓、ATR 止损、移动保护、3R 目标及 20 日上限，仅在指标反向时退出，每次投入含成本最多占账户 30%。长持仓期间权重会漂移，平均资金暴露可能明显高于主比较；收益差异同时来自持仓时间、仓位和退出逻辑，不能仅归因于指标优劣。期末未平仓仅估值，不计入胜率。", classic_display),
        ("仅按前两年挑选后的最后一年复查", f"先限定前两年至少平仓 {lab['selection_min_train_trades']} 笔，按前两年账户净收益选择前 {lab['selection_top_n']} 个，选出：{selected_text}。然后在最后一年从空仓重新回放。规则源于此前紫金研究，同一日历窗口已参与历史探索；本轮完整查看所有标的和候选结果，将最后一年标为时间分段复查，不宣称独立未触碰样本外或已验证可交易优势。", selection_display),
        ("退出方式和参数敏感性", f"主比较中有 {improvements} 个候选的三年收益高于统一退出对照。下表另比较最长持仓改为 40 日、佣金和滑点翻倍，以及取消 1% ATR 风险限仓、改为仅使用 30% 入场资金上限的情况。固定投入对照仍保留 ATR 退出。敏感性结果用于理解规则依赖，不据此再挑选新的参数。", sensitivity_display),
        ("资金与成本", "主比较和统一退出对照都取 30% 入场资金上限与 1% 名义风险预算约束的较小者，风险距离为入场前的 2ATR14。空仓现金收益按零计算。佣金双边万 2.5、最低 5 元；过户费双边十万分之一；卖出印花税万 5；滑点成本双边万 5。分红和红利税单独记账。ATR 收盘止损不能保证实际亏损不超过预算，隔夜跳空和跌停延迟可能扩大损失。最大回撤仅按日收盘账户净值计算。", None),
        ("被动基准", f"同区间，起始按30%资金上限买入后持有不再平衡（100股取整后实际投入可能低于30%），账户收益 {percent(benchmarks['full']['passive_30']['total_return'])}、回撤 {percent(benchmarks['full']['passive_30']['max_drawdown'])}、平均持仓权重 {percent(benchmarks['full']['passive_30']['average_position_weight'])}；接近满仓被动持有收益 {percent(benchmarks['full']['passive_100']['total_return'])}、回撤 {percent(benchmarks['full']['passive_100']['max_drawdown'])}。波段候选平均权重见表；低回撤可能主要来自低暴露，不能据此宣称风险调整后优势。", None),
        ("指标公式与数据时点", "MACD：DIF = EMA12 − EMA26，DEA = EMA9(DIF)，柱值 = 2×(DIF−DEA)。KDJ 使用 9 日 RSV、K/D 从 50 开始按 SMA(X,3,1) 递推，J=3K−2D。CCI20 使用典型价的平均绝对偏差；MFI14 使用典型价和成交股数；ADX14 使用 Wilder 指数平滑与方向移动；Supertrend 使用 ATR10、3 倍带宽和首次有效状态为多。不同平台的初始化方法可能产生细微差别。技术价格只加回截至当日已除息现金股息，日线回放从不回写未来分红。", None),
        ("探索结果的边界", "胜率仅为扣费后盈利的已平仓交易比例，未平仓不参与；Wilson 区间的独立二项假设只作近似。28 个候选、两种退出和附加敏感性都使用同一股票和相同历史，存在多重尝试偏差，候选之间也不独立。MACD、均线、RSI 等共享价格信息，多票通过不等于多个独立证据。更可信的下一步是冻结少量规则，记录后续新增交易日的模拟执行，不继续在同一三年样本上追逐最高胜率。", None),
        ("全部入场和指标退出规则", "下列指标退出仅在原生退出比较中启用；主比较中的共同 ATR 风控和时间上限仍然有效。信号是事件触发，需要发生上穿或下穿；止损退出后不会仅因状态仍为多头就自动重新买入。ADX 状态混合规则按当日 ADX 决定退出分支，未锁定入场时的状态。", catalog_display),
    ]
    if stock_key != "zijin":
        sections.append(("跨标的复用与数据来源", "27条技术规则直接复用；MACD＋黄金板块仍使用原上海金Au99.99及申万贵金属801053过滤，未替换为银行或半导体行业指数，这一条应作为原规则照搬实验解释。股票与分红为2026-10-19联网取得的腾讯、新浪快照；黄金、贵金属、沪指复用原紫金研究缓存，抓取时间以原清单为准。贵金属2026-10-15缺失时依赖该条件的入场关闭，未填造数据。两只股票分别使用独立10万元，结果不能直接相加为组合收益。股价较高时100股取整和1%风险预算可能导致无法入场。", None))
        review_frame = frame.loc[lab["review_start"]:base["test_end"]]
        one_lot_risk = (frame["atr14"].shift() * base["stop_atr"] * 100).reindex(review_frame.index)
        review_trades = sum(stats["closed_trades"] for stats in results["native"]["review"].values())
        lot_diagnostics = {"review_sessions": len(review_frame), "initial_account_risk_budget": base["initial_cash"] * base["risk_per_trade"],
                           "minimum_one_lot_nominal_risk": float(one_lot_risk.min()), "maximum_one_lot_nominal_risk": float(one_lot_risk.max()),
                           "sessions_one_lot_risk_exceeds_initial_budget": int((one_lot_risk > base["initial_cash"] * base["risk_per_trade"]).sum()),
                           "all_candidates_review_closed_trades": review_trades,
                           "all_candidates_review_open_positions": sum(stats["open_position"] is not None for stats in results["native"]["review"].values())}
        save_json(OUT / "lot_diagnostics.json", lot_diagnostics)
        sections.append(("100股最小交易单位与风险预算", f"最后一年共{len(review_frame)}个交易日。按入场前2ATR计算，100股的名义风险范围为{one_lot_risk.min():.2f}～{one_lot_risk.max():.2f}元，其中{lot_diagnostics['sessions_one_lot_risk_exceeds_initial_budget']}天高于期初10万元账户的1000元预算。28个主规则在该复查段合计平仓{review_trades}笔，期末持仓{lot_diagnostics['all_candidates_review_open_positions']}个。若整段无法开仓，0%收益体现的是仓位约束，不能据此评价该年度信号盈利能力；请并列查看预设的仅限30%投入敏感性，后者仍有ATR退出，但承担更高名义风险。", None))
    sources = [("Fidelity MACD 定义与 12/26/9 参数", "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd"),
               ("Fidelity ADX 趋势强度说明", "https://www.fidelity.com/viewpoints/active-investor/average-directional-index-ADX"),
               ("Fidelity CCI 定义", "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cci"),
               ("Fidelity MFI 定义", "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/mfi"),
               ("原始数据与来源", f"../../../data/{stock_key}/README.md")]
    title = f"{stock_name}三年多指标与混合规则探索"
    command = ".venv/bin/python scripts/explore_zijin_indicators.py" + (f" --stock {stock_key}" if stock_key != "zijin" else "")
    md = [f"# {title}", "", "![全部候选对比](comparison.png)", "", "![MACD 回放](macd_detail.png)", ""]
    blocks = []
    for number, (heading, paragraph, table) in enumerate(sections):
        md.extend([f"## {heading}", "", paragraph, ""])
        blocks.append(f'<section><h2>{html.escape(heading)}</h2><p>{html.escape(paragraph)}</p>')
        if table is not None:
            md.extend([markdown_table(table), ""])
            identifier = ' id="leaderboard"' if number == 2 else ""
            if number == 2:
                blocks.append('<label>筛选规则 <input id="filter" placeholder="例如 MACD、布林、混合"></label><p>点击表头可排序；默认按前两年收益排序。</p>')
            blocks.append(f'<div class="scroll"><div{identifier}>' + table.to_html(index=False, border=0, escape=True) + '</div></div>')
        blocks.append("</section>")
    md.extend(["## 文件和复现", "", "在仓库根目录运行：", "", "```bash", command, "```", "",
               "`leaderboard.csv` 保留全部候选数值；`results.json` 保存所有退出和敏感性结果；`trades.csv` 保存逐笔交易；`account_paths.csv` 保存每日账户；`features_and_signals.csv` 保存全部指标和每日入场/退出信号；`train_selection.json` 保存前两年挑选逻辑。", "", "## 参考来源", ""])
    md.extend(f"- [{label}]({url})" for label, url in sources)
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    style = ('body{font:16px/1.8 system-ui,sans-serif;color:#223047;background:#f4f6fa;margin:0}main{max-width:1250px;margin:auto;padding:35px 24px}'
             'h1{font-size:30px}h2{font-size:22px}section{padding:20px 24px;background:white;border-radius:12px;margin:18px 0}'
             'table{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}td,th{padding:9px;border-bottom:1px solid #e0e5ee;text-align:right}'
             'th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{width:100%;background:white}a{color:#175db2}input{padding:8px;width:260px;border:1px solid #aab5c8;border-radius:5px}'
             '#leaderboard th{cursor:pointer;position:sticky;top:0;background:#eaf0f8}#leaderboard tr:hover{background:#f4f7fc}')
    javascript = """const table=document.querySelector('#leaderboard table');
document.querySelector('#filter').addEventListener('input',e=>{for(const row of table.tBodies[0].rows)row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())});
for(const [column,header] of [...table.tHead.rows[0].cells].entries()){let descending=true;header.addEventListener('click',()=>{const rows=[...table.tBodies[0].rows];rows.sort((a,b)=>{const x=a.cells[column].textContent.trim(),y=b.cells[column].textContent.trim();if(x==='—')return y==='—'?0:1;if(y==='—')return -1;const nx=Number(x.replace('%','')),ny=Number(y.replace('%',''));const comparison=Number.isNaN(nx)||Number.isNaN(ny)?x.localeCompare(y):nx-ny;return descending?-comparison:comparison});for(const row of rows)table.tBodies[0].appendChild(row);descending=!descending})}
"""
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>' + style + f'</style><main><h1>{title}</h1>'
                '<p>28 个固定候选 · 真实三年数据 · 全部结果保留 · 最后一年为复查段</p>' + image_html("comparison.png") + image_html("macd_detail.png")
                + "".join(blocks) + f'<section><h2>文件与复现</h2><p>执行 {command} 可从本地数据重建全部结果。</p>'
                '<p><a href="leaderboard.csv">全部数值 CSV</a> · <a href="trades.csv">逐笔交易 CSV</a> · <a href="results.json">完整结果 JSON</a></p><ul>'
                + "".join(f'<li><a href="{html.escape(url)}">{html.escape(label)}</a></li>' for label, url in sources) + '</ul></section></main><script>' + javascript + '</script></html>')
    (OUT / "report.html").write_text(document, encoding="utf-8")
    save_json(OUT / "reproduction.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "base_config": base,
                                          "stock": stock_key, "external_manifest_sha256": external_manifest_sha,
                                          "lab_config": lab, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                          "used_cached_results": cached is not None,
                                          "calculation_script_sha256": (cached_provenance.get("calculation_script_sha256", cached_provenance["code_sha256"]["scripts/explore_zijin_indicators.py"]) if cached is not None else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                                          "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                                          [Path(__file__), ROOT / "stock_analysis/indicator_lab.py", ROOT / "stock_analysis/swing.py"]}})
    print("MACD 对比：\n" + macd_display.to_string(index=False), flush=True)
    print("前两年选出的规则：\n" + selection_display.to_string(index=False), flush=True)
    print("经典指标对照：\n" + classic_display.to_string(index=False), flush=True)
    print(f"报告：{OUT / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
