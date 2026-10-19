"""CSV 输入校验，缺失数值保留为 None。"""

import csv
import math
from datetime import date
from pathlib import Path


FINANCIAL_FIELDS = (
    "revenue_ttm", "revenue_prior_ttm", "eps_ttm", "eps_prior_ttm",
    "net_income_ttm", "average_equity", "operating_income_ttm",
    "operating_cash_flow_ttm", "capex_ttm", "shares_outstanding", "debt", "cash", "ebitda_ttm",
)


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path} 存在重复列名")
        absent = set(required) - set(headers)
        if absent:
            raise ValueError(f"{path} 缺少列：{', '.join(sorted(absent))}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line} 列数与表头不匹配")
            row = {key: value.strip() for key, value in row.items()}
            if any(not row[key] for key in required):
                raise ValueError(f"{path}:{line} 必填字段为空")
            rows.append(row)
        return rows


def iso_date(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"日期必须为 YYYY-MM-DD：{value}")
    return value


def number(value, name, *, minimum=None, strictly_positive=False):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是数字或空值：{value}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} 不允许 NaN 或无穷大")
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} 必须大于 0")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return result


def load_prices(path):
    rows = read_csv(path, ("symbol", "date", "close", "adj_close", "currency", "source"))
    seen = set()
    for row in rows:
        iso_date(row["date"])
        key = row["symbol"], row["date"]
        if key in seen:
            raise ValueError(f"行情存在重复 symbol/date：{key}")
        seen.add(key)
        for field in ("close", "adj_close"):
            row[field] = number(row[field], field, strictly_positive=True)
        row["volume"] = number(row.get("volume"), "volume", minimum=0)
    return rows


def load_fundamentals(path):
    if path is None:
        return []
    rows = read_csv(path, ("symbol", "period_end", "published_date", "currency", "source"))
    seen = set()
    for row in rows:
        for field in ("period_end", "published_date"):
            iso_date(row[field])
        if row["published_date"] < row["period_end"]:
            raise ValueError("财务披露日期不能早于财务期末")
        key = row["symbol"], row["period_end"], row["published_date"]
        if key in seen:
            raise ValueError(f"财务快照存在重复 symbol/period_end/published_date：{key}")
        seen.add(key)
        for field in FINANCIAL_FIELDS:
            minimum = 0 if field in ("capex_ttm", "debt", "cash") else None
            row[field] = number(row.get(field), field, minimum=minimum,
                                strictly_positive=field == "shares_outstanding")
    return rows
