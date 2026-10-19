"""纯计算层；所有比率用小数表示，不推导买卖评级。"""

from __future__ import annotations

import math
from datetime import date
from statistics import stdev


def divide(numerator, denominator, *, positive_denominator=True):
    if numerator is None or denominator is None or denominator == 0:
        return None
    if positive_denominator and denominator < 0:
        return None
    return numerator / denominator


def growth(current, previous):
    ratio = divide(current, previous)
    return ratio - 1 if ratio is not None else None


def max_drawdown(values):
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def analyze(prices, fundamentals, symbol, as_of, benchmark=None):
    """as_of 是交易日收盘后研究日；输入已由 io 层校验。"""
    rows = sorted(
        (r for r in prices if r["symbol"] == symbol and r["date"] <= as_of),
        key=lambda r: r["date"],
    )
    if not rows:
        raise ValueError(f"{symbol} 在 {as_of} 及以前没有行情")
    currencies = {r["currency"] for r in rows}
    if len(currencies) != 1:
        raise ValueError(f"{symbol} 的行情混用了币种")
    latest = rows[-1]
    currency = latest["currency"]
    values = [r["adj_close"] for r in rows]
    metrics = {}
    warnings = []

    def put(key, name, value, unit, note):
        if value is not None and not math.isfinite(value):
            value = None
            note += "；运算结果超出数值范围"
        metrics[key] = {"name": name, "value": value, "unit": unit, "note": note}

    for window in (50, 200):
        gap = values[-1] / (sum(values[-window:]) / window) - 1 if len(values) >= window else None
        put(f"ma{window}_gap", f"距 {window} 日均线", gap, "ratio",
            f"复权收盘价 / 最近 {window} 个交易日复权收盘均值 − 1；需要 {window} 行")
    momentum = values[-22] / values[-253] - 1 if len(values) >= 253 else None
    put("momentum_12_1", "12−1 月动量", momentum, "ratio",
        "P[t−21] / P[t−252] − 1；跳过最近 21 个交易日；需要 253 行")
    returns = [b / a - 1 for a, b in zip(values[-61:-1], values[-60:])] if len(values) >= 61 else []
    put("volatility_60d", "60 日年化波动率", stdev(returns) * math.sqrt(252) if returns else None,
        "ratio", "最近 60 个日简单收益率的样本标准差 × √252；需要 61 行")
    put("max_drawdown_252d", "252 日最大回撤", max_drawdown(values[-252:]) if len(values) >= 252 else None,
        "ratio", "最近 252 个收盘观察值内最差的 价格 / 此前峰值 − 1；负数；不含日内跌幅")
    volume_rows = rows[-20:]
    liquid = len(volume_rows) == 20 and all(r["volume"] is not None for r in volume_rows)
    put("turnover_20d_proxy", "20 日成交额代理", sum(r["close"] * r["volume"] for r in volume_rows) / 20 if liquid else None,
        currency, "未复权收盘价 × 原始成交股数的 20 日均值；不是真实逐笔成交额；需要 20 行完整成交量")

    relative = None
    if benchmark:
        if benchmark == symbol:
            raise ValueError("基准和被分析股票不能相同")
        dates = [r["date"] for r in rows[-64:]]
        reference = {r["date"]: r for r in prices if r["symbol"] == benchmark and r["date"] <= as_of}
        if len(dates) == 64 and all(d in reference for d in dates):
            if any(reference[d]["currency"] != currency for d in dates):
                raise ValueError("基准与股票币种不同；请先统一货币及汇率口径")
            relative = (values[-1] / values[-64] - 1) - (
                reference[dates[-1]]["adj_close"] / reference[dates[0]]["adj_close"] - 1)
        else:
            warnings.append("基准缺少股票最近 64 个交易日中的行情，未计算相对收益")
    put("relative_return_63d", "63 日相对基准收益", relative, "ratio",
        "股票 63 日复权收益 − 基准同起止日复权收益；差值为百分点；需指定基准且日期完整对齐")

    # 先按财务期末选最新期间，再选截至研究日已披露的最新版本。
    eligible = [r for r in fundamentals if r["symbol"] == symbol and r["published_date"] <= as_of]
    financial = max(eligible, key=lambda r: (r["period_end"], r["published_date"])) if eligible else None
    f = financial or {}
    if financial and f["currency"] != currency:
        raise ValueError("行情与财务币种不同；请先统一币种")
    field = lambda key: f.get(key)
    fcf = None
    if field("operating_cash_flow_ttm") is not None and field("capex_ttm") is not None:
        fcf = field("operating_cash_flow_ttm") - field("capex_ttm")
    # 当期发行在外普通股总数只用于市值；每股盈利使用财报披露的稀释 EPS。
    market_cap = latest["close"] * field("shares_outstanding") if field("shares_outstanding") is not None else None
    put("revenue_growth", "TTM 营收同比", growth(field("revenue_ttm"), field("revenue_prior_ttm")), "ratio",
        "最近四季度营收 / 上年同期四季度营收 − 1；基期须为正")
    put("eps_growth", "TTM 稀释 EPS 同比", growth(field("eps_ttm"), field("eps_prior_ttm")), "ratio",
        "TTM 稀释 EPS / 上年同期 TTM 稀释 EPS − 1；亏损基期不计算百分比")
    put("roe", "ROE", divide(field("net_income_ttm"), field("average_equity")), "ratio",
        "归属普通股股东 TTM 净利润 / 对应 TTM 平均普通股权益；权益须为正")
    put("operating_margin", "营业利润率", divide(field("operating_income_ttm"), field("revenue_ttm")), "ratio",
        "TTM 营业利润 / TTM 营收")
    put("cash_conversion", "经营现金流 / 净利润", divide(field("operating_cash_flow_ttm"), field("net_income_ttm")), "multiple",
        "TTM 经营现金流 / TTM 净利润；净利润须为正；需核查营运资本波动")
    put("fcf_margin", "自由现金流率", divide(fcf, field("revenue_ttm")), "ratio",
        "(TTM 经营现金流 − TTM 资本开支) / TTM 营收")
    put("pe_ttm", "TTM 市盈率", divide(latest["close"], field("eps_ttm")), "multiple",
        "未复权收盘价 / TTM 稀释 EPS；EPS 须为正且与股价拆股口径一致")
    put("fcf_yield", "自由现金流收益率", divide(fcf, market_cap), "ratio",
        "TTM 自由现金流 / (未复权收盘价 × 当期发行在外普通股总数)；不扣除股权薪酬经济成本")
    net_debt = field("debt") - field("cash") if field("debt") is not None and field("cash") is not None else None
    put("net_debt_ebitda", "净债务 / EBITDA", divide(net_debt, field("ebitda_ttm")), "multiple",
        "(有息债务 − 现金及现金等价物) / TTM EBITDA；EBITDA 须为正；负值可表示净现金")

    as_of_date = date.fromisoformat(as_of)
    price_age = (as_of_date - date.fromisoformat(latest["date"])).days
    if price_age > 7:
        warnings.append(f"最近行情距研究日 {price_age} 个自然日；检查停牌、休市或数据缺失")
    if not financial:
        warnings.append("截至研究日没有已披露的财务快照，财务指标不可用")
    elif (as_of_date - date.fromisoformat(f["period_end"])).days > 180:
        warnings.append("财务期末距研究日超过 180 个自然日，请检查是否有更新披露")
    if financial and f["published_date"] > latest["date"]:
        warnings.append("财务披露晚于最新行情日期，估值使用了披露前股价")
    if field("net_income_ttm") is not None and field("net_income_ttm") <= 0:
        warnings.append("TTM 净利润非正，盈利质量和常规盈利估值的适用性受限")
    if field("average_equity") is not None and field("average_equity") <= 0:
        warnings.append("平均普通股权益非正，需核查累计亏损、回购或资本结构")
    if fcf is not None and fcf < 0:
        warnings.append("TTM 自由现金流为负，需区分扩张投入与经营现金流恶化")
    missing = [key for key, metric in metrics.items() if metric["value"] is None]
    if missing:
        warnings.append("以下指标数据不足或分母不适用：" + ", ".join(missing))
    warnings.append("复权因子、拆股后 EPS/股本一致性及历史数据修订需由数据提供方保证；日期过滤不等于完整的时点回测")
    return {
        "symbol": symbol, "as_of": as_of, "currency": currency, "benchmark": benchmark,
        "price_date": latest["date"], "close": latest["close"], "price_rows": len(rows),
        "price_sources": sorted({r["source"] for r in rows}),
        "financial_period_end": f.get("period_end"), "financial_published_date": f.get("published_date"),
        "financial_source": f.get("source"), "metrics": metrics, "warnings": warnings,
    }
