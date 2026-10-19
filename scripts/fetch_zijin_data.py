"""下载并缓存真实行情。无凭证，原始响应、CSV 与来源清单一并保存。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/zijin/raw"
PROCESSED = ROOT / "data/zijin/processed"


def get_cached(name, url, *, method="GET", data=None, refresh=False, verify=True):
    path = RAW / name
    if path.exists() and not refresh:
        return path
    error = None
    for attempt in range(3):
        try:
            response = requests.request(method, url, data=data, timeout=35, verify=verify,
                                        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.sge.com.cn/sjzx/mrhq"})
            response.raise_for_status()
            if name.endswith(".json"):
                content = json.dumps(response.json(), ensure_ascii=False).encode("utf-8")
            else:
                content = response.content
            path.write_bytes(content)
            return path
        except (requests.RequestException, ValueError) as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"获取 {name} 失败：{error}")


def read_tencent(path, symbol, adjusted):
    payload = json.loads(path.read_text(encoding="utf-8"))
    obj = payload["data"][symbol]
    bars = obj.get("qfqday") if adjusted else obj.get("day")
    if bars is None:
        bars = obj.get("day")
    if not bars:
        raise ValueError(f"{path} 没有日线")
    frame = pd.DataFrame([bar[:6] for bar in bars], columns=["date", "open", "close", "high", "low", "volume"])
    for col in frame.columns[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if symbol == "sh601899":
        frame["volume"] *= 100  # 腾讯主板股票以手为单位，输出统一为股。
    return frame


def validate(frame, name):
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"{name} 日期重复")
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{name} 有缺失行情")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{name} 有非正价格")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
            (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{name} OHLC 不一致")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="重新下载覆盖本地缓存")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/zijin_swing.json").read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = {"manifest_created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": [],
                "request_start": config["download_start"], "request_end": config["research_as_of"]}
    for name, symbol, adjusted in (("zijin_qfq", "sh601899", True),
                                    ("zijin_raw", "sh601899", False), ("sse", "sh000001", False)):
        frames, urls = [], []
        for year in range(2023, 2027):
            adjustment = "qfq" if adjusted else ""
            url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
                   f"{symbol},day,{year}-01-01,{year}-12-31,640,{adjustment}")
            path = get_cached(f"tencent_{name}_{year}.json", url, refresh=args.refresh)
            frames.append(read_tencent(path, symbol, adjusted))
            urls.append(url)
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.loc[frame["date"].between(config["download_start"], config["research_as_of"])].copy()
        frame = validate(frame, name)
        frame.to_csv(PROCESSED / f"{name}.csv", index=False)
        manifest["datasets"].append({"name": name, "provider": "Tencent Finance", "symbol": symbol,
                                      "adjustment": "qfq" if adjusted else "none", "urls": urls,
                                      "rows": len(frame), "first": frame["date"].iloc[0], "last": frame["date"].iloc[-1]})
        print(name, len(frame), frame["date"].iloc[0], frame["date"].iloc[-1], flush=True)

    gold_url = "https://www.sge.com.cn/graph/Dailyhq"
    path = get_cached("gold_au9999.json", gold_url, method="POST", data={"instid": "Au99.99"}, refresh=args.refresh)
    gold = pd.DataFrame(json.loads(path.read_text())["time"], columns=["date", "open", "close", "low", "high"])
    gold = gold.loc[gold["date"].between(config["download_start"], config["research_as_of"])].copy()
    gold = validate(gold, "gold_au9999")
    gold.to_csv(PROCESSED / "gold_au9999.csv", index=False)
    manifest["datasets"].append({"name": "gold_au9999", "provider": "Shanghai Gold Exchange", "symbol": "Au99.99",
                                  "unit": "CNY/gram", "urls": [gold_url], "post_data": {"instid": "Au99.99"},
                                  "rows": len(gold), "first": gold["date"].iloc[0], "last": gold["date"].iloc[-1]})

    sector_url = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/?swindexcode=801053&period=DAY"
    path = get_cached("sw_sector.json", sector_url, refresh=args.refresh)
    sector = pd.DataFrame(json.loads(path.read_text())["data"]).rename(columns={
        "bargaindate": "date", "openindex": "open", "maxindex": "high", "minindex": "low", "closeindex": "close"})
    sector["date"] = sector["date"].str[:10]
    sector = sector.loc[sector["date"].between(config["download_start"], config["research_as_of"]),
                        ["date", "open", "high", "low", "close"]].copy()
    sector = validate(sector, "precious_metals_sw801053")
    sector.to_csv(PROCESSED / "precious_metals_sw801053.csv", index=False)
    manifest["datasets"].append({"name": "precious_metals_sw801053", "provider": "SWS Research", "symbol": "801053",
                                  "urls": [sector_url], "rows": len(sector), "first": sector["date"].iloc[0], "last": sector["date"].iloc[-1],
                                  "initial_snapshot_note": "初次取数该站点证书链验证失败，公开数据快照经关闭验证取得；常规重新下载启用验证。"})

    dividend_url = "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/601899.phtml"
    path = get_cached("zijin_dividends.html", dividend_url, refresh=args.refresh)
    text = path.read_bytes().decode("gb18030", errors="replace")
    tables = pd.read_html(io.StringIO(text))
    table = next(t for t in tables if "除权除息日" in str(t.columns) and "派息" in str(t.columns))
    table.columns = [col[-1] if isinstance(col, tuple) else col for col in table.columns]
    table = table.rename(columns={"公告日期": "announcement_date", "送股(股)": "bonus_per10", "转增(股)": "split_per10",
                                   "派息(税前)(元)": "cash_per10", "进度": "status", "除权除息日": "ex_date", "股权登记日": "record_date"})
    table["ex_date"] = pd.to_datetime(table["ex_date"], errors="coerce")
    table = table.loc[table["ex_date"].between(config["download_start"], config["research_as_of"])].copy()
    for col in ("cash_per10", "bonus_per10", "split_per10"):
        table[col] = pd.to_numeric(table[col], errors="raise")
    if ((table["bonus_per10"] != 0) | (table["split_per10"] != 0)).any():
        raise ValueError("样本含送转股，需要扩展持仓股数处理后再回测")
    table["ex_date"] = table["ex_date"].dt.strftime("%Y-%m-%d")
    table = table[["announcement_date", "record_date", "ex_date", "cash_per10", "bonus_per10", "split_per10", "status"]].sort_values("ex_date")
    table.to_csv(PROCESSED / "zijin_dividends.csv", index=False)
    manifest["datasets"].append({"name": "zijin_dividends", "provider": "Sina Finance", "urls": [dividend_url], "rows": len(table)})
    manifest["files"] = []
    for path in sorted(list(RAW.glob("*")) + list(PROCESSED.glob("*.csv"))):
        manifest["files"].append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    (ROOT / "data/zijin/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print("本地行情、分红和来源清单已保存。")


if __name__ == "__main__":
    main()
