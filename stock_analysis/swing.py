"""紫金 A 股单标的波段回放：只做多、不加杠杆、收盘信号次日开盘执行。"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd


STRATEGIES = {
    "pullback_technical": "回踩企稳：仅技术面",
    "pullback_gold": "回踩企稳＋黄金",
    "pullback_gold_sector": "回踩企稳＋黄金＋板块",
    "pullback_all": "回踩企稳＋黄金＋板块＋沪指",
    "breakout_all": "放量突破＋黄金＋板块＋沪指",
}


def make_features(raw, gold, sector, sse, dividends):
    """输入按日期索引。现金分红只在实际除息日起加回，不使用未来复权因子。"""
    frame = raw.copy().sort_index()
    cash = dividends.groupby("ex_date")["cash_per10"].sum() / 10
    frame["cash_dividend"] = cash.reindex(frame.index).fillna(0.0)
    frame["cumulative_dividend"] = frame["cash_dividend"].cumsum()
    for col in ("open", "high", "low", "close"):
        frame[f"analysis_{col}"] = frame[col] + frame["cumulative_dividend"]
    close, high, low = (frame[f"analysis_{col}"] for col in ("close", "high", "low"))
    frame["stock_return"] = (frame["close"] + frame["cash_dividend"]) / frame["close"].shift() - 1
    frame["stock_total_return_index"] = (1 + frame["stock_return"].fillna(0)).cumprod()
    for window in (10, 20, 60):
        frame[f"ma{window}"] = close.rolling(window).mean()
        frame[f"raw_reference_ma{window}"] = frame[f"ma{window}"] - frame["cumulative_dividend"]
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["rsi14"] = 100 - 100 / (1 + up / down)
    frame.loc[(up == 0) & (down == 0), "rsi14"] = 50
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    frame["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["macd_dif"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    frame["macd_dea"] = frame["macd_dif"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = 2 * (frame["macd_dif"] - frame["macd_dea"])
    deviation = close.rolling(20).std(ddof=0)
    frame["bollinger_upper"] = frame["ma20"] + 2 * deviation
    frame["bollinger_lower"] = frame["ma20"] - 2 * deviation
    frame["volume_ratio"] = frame["volume"] / frame["volume"].rolling(20).mean().shift()
    frame["prior_high20"] = high.rolling(20).max().shift()
    frame["stock_return20"] = frame["stock_total_return_index"].pct_change(20, fill_method=None)
    for name, dataset in (("gold", gold), ("sector", sector), ("sse", sse)):
        dataset = dataset.sort_index()
        frame[f"{name}_close"] = dataset["close"].reindex(frame.index)
        for window in (20, 60):
            frame[f"{name}_ma{window}"] = dataset["close"].rolling(window).mean().reindex(frame.index)
        frame[f"{name}_return20"] = dataset["close"].pct_change(20, fill_method=None).reindex(frame.index)
    frame["gold_filter"] = (frame["gold_close"] > frame["gold_ma20"]) & (frame["gold_ma20"] > frame["gold_ma60"])
    frame["sector_filter"] = (frame["sector_close"] > frame["sector_ma20"]) & (frame["stock_return20"] > frame["sector_return20"])
    frame["market_filter"] = frame["sse_close"] > frame["sse_ma60"]
    trend = (frame["ma20"] > frame["ma60"]) & (close > frame["ma60"])
    frame["pullback_signal"] = (trend & (low <= frame["ma20"] * 1.02) & (close >= frame["ma20"]) &
                                (close > close.shift()) & frame["rsi14"].between(45, 65) & (frame["volume_ratio"] >= .8))
    frame["breakout_signal"] = (trend & (close > frame["prior_high20"]) & frame["rsi14"].between(50, 75) &
                                (frame["volume_ratio"] >= 1.2))
    return frame


def entry_signals(frame, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略：{strategy}")
    signal = frame["breakout_signal" if strategy.startswith("breakout") else "pullback_signal"].copy()
    if strategy != "pullback_technical":
        signal &= frame["gold_filter"]
    if strategy in ("pullback_gold_sector", "pullback_all", "breakout_all"):
        signal &= frame["sector_filter"]
    if strategy in ("pullback_all", "breakout_all"):
        signal &= frame["market_filter"]
    return signal.fillna(False)


def fees(notional, side, config):
    return {"commission": max(config["minimum_commission"], notional * config["commission_rate"]),
            "transfer": notional * config["transfer_rate"],
            "stamp": notional * config["sell_stamp_rate"] if side == "sell" else 0.0,
            "slippage": notional * config["slippage_rate"]}


def dividend_tax_rate(entry_date, exit_date):
    # 持股期按自然月/年；阈值按含一个月、含一年处理。
    if exit_date <= entry_date + pd.DateOffset(months=1):
        return .2
    if exit_date <= entry_date + pd.DateOffset(years=1):
        return .1
    return 0.0


def price_limit(reference, multiple):
    return float((Decimal(str(reference)) * Decimal(str(multiple))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def can_fill(row, previous, side):
    if row["volume"] <= 0:
        return False
    reference = previous["close"] - row["cash_dividend"]
    if side == "buy":
        return row["open"] < price_limit(reference, 1.1) - 1e-8
    return row["open"] > price_limit(reference, .9) + 1e-8


def wilson_interval(wins, count):
    if count == 0:
        return [None, None]
    z = 1.959963984540054
    p = wins / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def performance(equity, trades, initial_cash):
    nav = equity["equity"] / initial_cash
    previous = nav.shift().fillna(1.0)
    returns = nav / previous - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252)
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    profits = trades["net_pnl"] if len(trades) else pd.Series(dtype=float)
    wins = int((profits > 0).sum())
    gains, losses = profits[profits > 0], profits[profits < 0]
    count = len(profits)
    return {"total_return": float(nav.iloc[-1] - 1), "annual_return": float(nav.iloc[-1] ** (1 / years) - 1),
            "max_drawdown": float(drawdown.min()), "annual_volatility": float(returns.std(ddof=1) * math.sqrt(252)),
            "sharpe_zero_rf": float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if returns.std(ddof=1) > 0 else None,
            "closed_trades": count, "wins": wins, "losses": int((profits < 0).sum()),
            "win_rate": wins / count if count else None, "win_rate_wilson95": wilson_interval(wins, count),
            "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else None,
            "average_trade_return": float(trades["net_return"].mean()) if count else None,
            "average_holding_sessions": float(trades["holding_sessions"].mean()) if count else None,
            "average_win": float(gains.mean()) if len(gains) else None,
            "average_loss": float(losses.mean()) if len(losses) else None,
            "exposure_days": float((equity["shares"] > 0).mean()), "average_position_weight": float(equity["position_weight"].mean()),
            "total_cost": float(equity["cumulative_cost"].iloc[-1]), "ending_equity": float(equity["equity"].iloc[-1])}


def backtest(frame, config, strategy, start=None, end=None, passive_fraction=None,
             signals=None, indicator_exit=None, use_ma20_exit=True, use_atr_risk_controls=True,
             use_profit_target=True):
    """首日空仓，首个信号在区间内首日收盘后产生；不利用区间前信号开仓。"""
    start, end = pd.Timestamp(start or config["test_start"]), pd.Timestamp(end or config["test_end"])
    sample = frame.loc[start:end]
    if sample.empty:
        raise ValueError("回测区间没有行情")
    if passive_fraction is not None:
        signal = pd.Series(False, index=frame.index)
    elif signals is not None:
        signal = signals.reindex(frame.index).fillna(False).astype(bool)
    else:
        signal = entry_signals(frame, strategy)
    exits = indicator_exit.reindex(frame.index).fillna(False).astype(bool) if indicator_exit is not None else None
    cash = float(config["initial_cash"])
    position = None
    pending_exit = None
    cumulative_cost = 0.0
    trades, equity_rows = [], []
    blocked_buys = blocked_sells = 0
    position_was_bought = False
    for session, (day, row) in enumerate(sample.iterrows()):
        location = frame.index.get_loc(day)
        previous_day = frame.index[location - 1] if location > 0 else None
        previous = frame.iloc[location - 1] if location > 0 else None
        exited = False
        # 除息权利属于前一交易日收盘持有人；先记红利，再处理当日卖出。
        if position is not None:
            position["gross_dividends"] += position["shares"] * row["cash_dividend"]
            if pending_exit is not None and session > position["entry_session"]:
                if can_fill(row, previous, "sell"):
                    notional = position["shares"] * row["open"]
                    selling_cost = sum(fees(notional, "sell", config).values())
                    dividend_tax = position["gross_dividends"] * dividend_tax_rate(position["entry_date"], day)
                    proceeds = notional - selling_cost + position["gross_dividends"] - dividend_tax
                    cash += proceeds
                    cumulative_cost += selling_cost + dividend_tax
                    net_pnl = proceeds - position["entry_total"]
                    trades.append({"signal_date": position["signal_date"].date().isoformat(),
                                   "entry_date": position["entry_date"].date().isoformat(), "exit_date": day.date().isoformat(),
                                   "entry_price": position["entry_price"], "exit_price": float(row["open"]), "shares": position["shares"],
                                   "holding_sessions": session - position["entry_session"], "entry_total": position["entry_total"],
                                   "entry_cost": position["entry_cost"], "exit_cost": selling_cost, "gross_dividends": position["gross_dividends"],
                                   "dividend_tax": dividend_tax, "net_pnl": net_pnl, "net_return": net_pnl / position["entry_total"],
                                   "exit_signal_date": position["exit_signal_date"].date().isoformat(), "exit_reason": pending_exit})
                    position, pending_exit, exited = None, None, True
                else:
                    blocked_sells += 1
        wants_buy = (session > 0 and bool(signal.loc[previous_day])) if passive_fraction is None else not position_was_bought
        if position is None and not exited and wants_buy and previous is not None:
            if can_fill(row, previous, "buy"):
                max_fraction = config["max_position_fraction"] if passive_fraction is None else passive_fraction
                budget = cash * max_fraction
                atr = float(previous["atr14"])
                risk_distance = config["stop_atr"] * atr
                shares = budget / row["open"]
                if passive_fraction is None and use_atr_risk_controls:
                    if not np.isfinite(risk_distance) or risk_distance <= 0:
                        shares = 0
                    else:
                        shares = min(shares, cash * config["risk_per_trade"] / risk_distance)
                shares = int(shares // 100) * 100
                while shares > 0:
                    notional = shares * row["open"]
                    buying_cost = sum(fees(notional, "buy", config).values())
                    if notional + buying_cost <= min(cash, budget):
                        break
                    shares -= 100
                if shares > 0:
                    cash -= notional + buying_cost
                    cumulative_cost += buying_cost
                    position_was_bought = True
                    position = {"entry_session": session, "entry_date": day, "signal_date": previous_day,
                                "entry_price": float(row["open"]), "shares": shares, "entry_total": notional + buying_cost,
                                "entry_cost": buying_cost, "gross_dividends": 0.0,
                                "stop": row["analysis_open"] - risk_distance,
                                "target": row["analysis_open"] + config["target_r"] * risk_distance,
                                "highest_close": row["analysis_open"]}
            else:
                blocked_buys += 1
        if position is not None and passive_fraction is None:
            position["highest_close"] = max(position["highest_close"], row["analysis_close"])
            if use_atr_risk_controls:
                position["stop"] = max(position["stop"], position["highest_close"] - config["trailing_atr"] * row["atr14"])
            if pending_exit is None:
                reason = None
                if use_atr_risk_controls and row["analysis_close"] <= position["stop"]:
                    reason = "closing_atr_stop"
                elif use_atr_risk_controls and use_profit_target and row["analysis_close"] >= position["target"]:
                    reason = "closing_3R_target"
                elif exits is not None and bool(exits.loc[day]):
                    reason = "indicator_exit"
                elif use_ma20_exit and row["analysis_close"] < row["ma20"]:
                    reason = "close_below_ma20"
                elif config["max_holding_days"] is not None and session - position["entry_session"] + 1 >= config["max_holding_days"]:
                    reason = f"holding_{config['max_holding_days']}_sessions"
                if reason:
                    pending_exit = reason
                    position["exit_signal_date"] = day
        shares = position["shares"] if position else 0
        receivable = position["gross_dividends"] if position else 0.0
        reserved_tax = receivable * dividend_tax_rate(position["entry_date"], day) if position else 0.0
        equity = cash + shares * row["close"] + receivable - reserved_tax
        equity_rows.append({"date": day, "equity": equity, "cash": cash, "shares": shares,
                            "dividend_receivable": receivable, "dividend_tax_reserved": reserved_tax,
                            "position_weight": shares * row["close"] / equity, "cumulative_cost": cumulative_cost})
    equity = pd.DataFrame(equity_rows).set_index("date")
    trade_columns = ["signal_date", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "holding_sessions",
                     "entry_total", "entry_cost", "exit_cost", "gross_dividends", "dividend_tax", "net_pnl", "net_return", "exit_signal_date", "exit_reason"]
    trades = pd.DataFrame(trades, columns=trade_columns)
    stats = performance(equity, trades, config["initial_cash"])
    stats.update({"blocked_buys": blocked_buys, "blocked_sells": blocked_sells, "open_position": None})
    if position:
        stats["open_position"] = {"entry_date": position["entry_date"].date().isoformat(), "shares": position["shares"],
                                  "entry_price": position["entry_price"], "last_close": float(sample["close"].iloc[-1]),
                                  "gross_dividends": position["gross_dividends"], "reserved_dividend_tax": reserved_tax,
                                  "unrealized_net_pnl": shares * sample["close"].iloc[-1] + receivable - reserved_tax - position["entry_total"],
                                  "pending_exit": pending_exit}
    realized = float(trades["net_pnl"].sum()) if len(trades) else 0.0
    unrealized = stats["open_position"]["unrealized_net_pnl"] if position else 0.0
    if not math.isclose(stats["ending_equity"] - config["initial_cash"], realized + unrealized, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("账户净值与逐笔盈亏无法对账")
    return {"stats": stats, "equity": equity, "trades": trades}
