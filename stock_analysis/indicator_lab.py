"""固定候选指标规则。先构造信号，后运行回放，不按历史结果调整阈值。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    key: str
    name: str
    category: str
    entry_description: str
    exit_description: str
    entry: pd.Series
    exit: pd.Series


def cross_up(left, right):
    return (left > right) & (left.shift() <= (right.shift() if isinstance(right, pd.Series) else right))


def cross_down(left, right):
    return (left < right) & (left.shift() >= (right.shift() if isinstance(right, pd.Series) else right))


def chinese_sma(series, seed=50.0):
    """SMA(X,3,1)：首个有效 RSV 之前保留空值，从 K/D=50 开始递推。"""
    output = []
    state = seed
    for value in series:
        if pd.isna(value):
            output.append(np.nan)
        else:
            state = (2 * state + value) / 3
            output.append(state)
    return pd.Series(output, index=series.index)


def add_indicators(frame):
    frame = frame.copy()
    close, high, low = (frame[f"analysis_{col}"] for col in ("close", "high", "low"))
    for window in (10, 30):
        frame[f"ema{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()
    low9, high9 = low.rolling(9).min(), high.rolling(9).max()
    rsv = 100 * (close - low9) / (high9 - low9).replace(0, np.nan)
    rsv = rsv.mask((high9 == low9) & high9.notna(), 50)
    frame["kdj_k"] = chinese_sma(rsv)
    frame["kdj_d"] = chinese_sma(frame["kdj_k"])
    frame["kdj_j"] = 3 * frame["kdj_k"] - 2 * frame["kdj_d"]
    typical = (high + low + close) / 3
    mean_deviation = typical.rolling(20).apply(lambda values: np.abs(values - values.mean()).mean(), raw=True)
    frame["cci20"] = (typical - typical.rolling(20).mean()) / (.015 * mean_deviation.replace(0, np.nan))
    flow = typical * frame["volume"]
    positive = flow.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = flow.where(typical.diff() < 0, 0).rolling(14).sum()
    frame["mfi14"] = 100 * positive / (positive + negative).replace(0, np.nan)
    frame.loc[(positive == 0) & (negative == 0), "mfi14"] = 50
    frame["obv"] = (np.sign(close.diff()).fillna(0) * frame["volume"]).cumsum()
    frame["obv_ema20"] = frame["obv"].ewm(span=20, adjust=False, min_periods=20).mean()
    frame["prior_obv_high20"] = frame["obv"].rolling(20).max().shift()
    frame["prior_low10"] = low.rolling(10).min().shift()
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    frame["williams_r14"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)
    frame.loc[(high14 == low14) & high14.notna(), "williams_r14"] = -50

    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    smooth = lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    denominator = smooth(true_range).replace(0, np.nan)
    frame["di_plus14"] = 100 * smooth(plus_dm) / denominator
    frame["di_minus14"] = 100 * smooth(minus_dm) / denominator
    di_sum = frame["di_plus14"] + frame["di_minus14"]
    dx = 100 * (frame["di_plus14"] - frame["di_minus14"]).abs() / di_sum.replace(0, np.nan)
    dx = dx.mask((di_sum == 0) & di_sum.notna(), 0)
    frame["adx14"] = smooth(dx)

    # Supertrend(10,3)：当前收盘仅影响当前状态，整段计算与前缀计算一致。
    atr10 = true_range.ewm(alpha=1 / 10, adjust=False, min_periods=10).mean()
    base = (high + low) / 2
    upper, lower = base + 3 * atr10, base - 3 * atr10
    final_upper, final_lower, direction, line = [], [], [], []
    old_upper = old_lower = np.nan
    old_direction = 1
    previous_close = np.nan
    for day in frame.index:
        if pd.isna(atr10.loc[day]):
            final_upper.append(np.nan); final_lower.append(np.nan); direction.append(np.nan); line.append(np.nan)
            previous_close = close.loc[day]
            continue
        current_upper = upper.loc[day] if pd.isna(old_upper) or upper.loc[day] < old_upper or previous_close > old_upper else old_upper
        current_lower = lower.loc[day] if pd.isna(old_lower) or lower.loc[day] > old_lower or previous_close < old_lower else old_lower
        current_direction = old_direction
        if old_direction == -1 and close.loc[day] > current_upper:
            current_direction = 1
        elif old_direction == 1 and close.loc[day] < current_lower:
            current_direction = -1
        final_upper.append(current_upper); final_lower.append(current_lower); direction.append(current_direction)
        line.append(current_lower if current_direction == 1 else current_upper)
        old_upper, old_lower, old_direction, previous_close = current_upper, current_lower, current_direction, close.loc[day]
    frame["supertrend_upper"] = final_upper
    frame["supertrend_lower"] = final_lower
    frame["supertrend_direction"] = direction
    frame["supertrend"] = line
    frame["trend_vote"] = ((close > frame["ma60"]).astype(int) + (frame["macd_dif"] > frame["macd_dea"]).astype(int) +
                            frame["rsi14"].between(50, 70).astype(int) +
                            ((frame["adx14"] > 20) & (frame["di_plus14"] > frame["di_minus14"])).astype(int) +
                            (frame["obv"] > frame["obv_ema20"]).astype(int))
    return frame


def candidates(frame):
    close = frame["analysis_close"]
    dif, dea, rsi = frame["macd_dif"], frame["macd_dea"], frame["rsi14"]
    macd_up, macd_down = cross_up(dif, dea), cross_down(dif, dea)
    kdj_up, kdj_down = cross_up(frame["kdj_k"], frame["kdj_d"]), cross_down(frame["kdj_k"], frame["kdj_d"])
    band_reentry = cross_up(close, frame["bollinger_lower"])
    band_exit = close >= frame["ma20"]
    breakout = cross_up(close, frame["prior_high20"])
    donchian_exit = close < frame["prior_low10"]
    super_up = cross_up(frame["supertrend_direction"], 0)
    super_down = cross_down(frame["supertrend_direction"], 0)
    trend = (frame["ma20"] > frame["ma60"]) & (close > frame["ma60"])
    adx_positive = (frame["adx14"] > 20) & (frame["di_plus14"] > frame["di_minus14"])
    specs = []
    def add(key, name, category, entry_description, exit_description, entry, exit):
        specs.append(Candidate(key, name, category, entry_description, exit_description,
                               entry.fillna(False).astype(bool), exit.fillna(False).astype(bool)))
    add("macd_cross", "MACD 金叉", "单指标", "DIF 上穿 DEA，参数 12/26/9", "DIF 下穿 DEA", macd_up, macd_down)
    add("macd_zero", "MACD 零轴突破", "单指标", "DIF 上穿零轴", "DIF 下穿零轴", cross_up(dif, 0), cross_down(dif, 0))
    hist = frame["macd_hist"]
    add("macd_hist_turn", "MACD 绿柱缩短拐点", "单指标", "柱值在零轴下方，由下降转为上升", "柱值在零轴上方，由上升转为下降",
        (hist < 0) & (hist > hist.shift()) & (hist.shift() <= hist.shift(2)),
        (hist > 0) & (hist < hist.shift()) & (hist.shift() >= hist.shift(2)))
    add("ema_cross", "EMA10/30 金叉", "单指标", "EMA10 上穿 EMA30", "EMA10 下穿 EMA30",
        cross_up(frame["ema10"], frame["ema30"]), cross_down(frame["ema10"], frame["ema30"]))
    add("rsi_rebound", "RSI30 超卖回升", "单指标", "RSI14 上穿 30", "RSI14 ≥ 60 或下穿 25", cross_up(rsi, 30), (rsi >= 60) | cross_down(rsi, 25))
    add("rsi_momentum", "RSI50 动量", "单指标", "RSI14 上穿 50", "RSI14 下穿 50", cross_up(rsi, 50), cross_down(rsi, 50))
    add("kdj_rebound", "KDJ 低位金叉", "单指标", "K 上穿 D 且前日 K < 30，参数 9/3/3", "K 下穿 D", kdj_up & (frame["kdj_k"].shift() < 30), kdj_down)
    add("bollinger_reentry", "布林下轨回归", "单指标", "收盘从下方重返布林下轨之上，参数 20/2", "收盘达到布林中轨", band_reentry, band_exit)
    add("bollinger_breakout", "布林上轨突破", "单指标", "收盘上穿布林上轨，参数 20/2", "收盘跌破布林中轨",
        cross_up(close, frame["bollinger_upper"]), close < frame["ma20"])
    add("donchian_breakout", "唐奇安20日突破", "单指标", "收盘上穿此前 20 日最高价", "收盘跌破此前 10 日最低价", breakout, donchian_exit)
    add("cci_rebound", "CCI 超卖回升", "单指标", "CCI20 上穿 −100", "CCI20 ≥ 100 或下穿 −100", cross_up(frame["cci20"], -100),
        (frame["cci20"] >= 100) | cross_down(frame["cci20"], -100))
    add("mfi_rebound", "MFI 资金流回升", "单指标", "MFI14 上穿 20", "MFI14 ≥ 70", cross_up(frame["mfi14"], 20), frame["mfi14"] >= 70)
    add("obv_breakout", "OBV 量能突破", "单指标", "OBV 上穿此前 20 日最高 OBV 且收盘 > MA20", "OBV < 其 EMA20 或收盘 < MA20",
        cross_up(frame["obv"], frame["prior_obv_high20"]) & (close > frame["ma20"]),
        (frame["obv"] < frame["obv_ema20"]) | (close < frame["ma20"]))
    add("adx_di_cross", "ADX 趋势金叉", "单指标", "+DI 上穿 −DI 且 ADX14 > 20", "+DI 下穿 −DI",
        cross_up(frame["di_plus14"], frame["di_minus14"]) & (frame["adx14"] > 20), cross_down(frame["di_plus14"], frame["di_minus14"]))
    add("supertrend_flip", "Supertrend 翻多", "单指标", "Supertrend(10,3) 从空翻多", "Supertrend 从多翻空", super_up, super_down)
    add("williams_rebound", "威廉指标超卖回升", "单指标", "Williams %R14 上穿 −80", "Williams %R14 ≥ −20",
        cross_up(frame["williams_r14"], -80), frame["williams_r14"] >= -20)

    add("macd_trend", "MACD＋均线趋势", "混合", "MACD 金叉且 MA20 > MA60、收盘 > MA60", "MACD 死叉", macd_up & trend, macd_down)
    add("macd_rsi", "MACD＋RSI", "混合", "MACD 金叉且 RSI14 位于 45～70", "MACD 死叉", macd_up & rsi.between(45, 70), macd_down)
    add("macd_volume", "MACD＋放量", "混合", "MACD 金叉且量比 ≥ 1.2（基于此前 20 日均量）", "MACD 死叉", macd_up & (frame["volume_ratio"] >= 1.2), macd_down)
    add("macd_adx", "MACD＋ADX", "混合", "MACD 金叉且 ADX14 > 20、+DI > −DI", "MACD 死叉或 +DI < −DI", macd_up & adx_positive, macd_down | (frame["di_plus14"] < frame["di_minus14"]))
    add("macd_gold_sector", "MACD＋黄金板块", "混合", "MACD 金叉且通过既有黄金与板块过滤", "MACD 死叉", macd_up & frame["gold_filter"] & frame["sector_filter"], macd_down)
    add("rsi_macd_trend", "RSI50＋MACD＋均线", "混合", "RSI14 上穿 50、DIF > DEA、通过均线趋势", "RSI14 下穿 50 或 MACD 死叉",
        cross_up(rsi, 50) & (dif > dea) & trend, cross_down(rsi, 50) | macd_down)
    add("kdj_trend", "KDJ＋均线趋势", "混合", "K 上穿 D、前日 K < 50、通过均线趋势", "K 下穿 D", kdj_up & (frame["kdj_k"].shift() < 50) & trend, kdj_down)
    add("bollinger_rsi", "布林下轨＋RSI", "混合", "收盘重返下轨之上且 RSI14 < 45", "收盘达到中轨", band_reentry & (rsi < 45), band_exit)
    add("donchian_adx_volume", "突破＋ADX＋放量", "混合", "20 日突破、ADX14 > 20、+DI > −DI、量比 ≥ 1.2", "收盘跌破此前 10 日最低价", breakout & adx_positive & (frame["volume_ratio"] >= 1.2), donchian_exit)
    add("supertrend_macd", "Supertrend＋MACD", "混合", "MACD 金叉且 Supertrend 状态为多", "Supertrend 翻空或 MACD 死叉",
        macd_up & (frame["supertrend_direction"] == 1), super_down | macd_down)
    add("regime_adaptive", "ADX 区分趋势与震荡", "混合", "ADX≥25 时 MACD 金叉且+DI>−DI；ADX<20 时布林回归且RSI<45；中间区不入场",
        "当前 ADX≥25 时 MACD 死叉；当前 ADX<25 时收盘达到 MA20",
        ((frame["adx14"] >= 25) & macd_up & (frame["di_plus14"] > frame["di_minus14"])) |
        ((frame["adx14"] < 20) & band_reentry & (rsi < 45)),
        ((frame["adx14"] >= 25) & macd_down) | ((frame["adx14"] < 25) & band_exit))
    add("vote3of5", "五项条件三票通过", "混合", "收盘>MA60、MACD多头、RSI50～70、ADX多头、OBV>EMA20，满足数从<3升至≥3",
        "上述条件满足数降至 ≤2", cross_up(frame["trend_vote"], 2.5), frame["trend_vote"] <= 2)
    return specs
