"""缓存招商银行、兆易创新原始行情与分红，保留抓取时间和来源。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

from fetch_zijin_data import read_tencent, validate

ROOT = Path(__file__).resolve().parents[1]
STOCKS = {"cmb": ("600036", "招商银行"), "gigadevice": ("603986", "兆易创新")}
START, END = "2023-01-01", "2026-10-16"


def fetch(path, url, refresh=False):
    receipt = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and receipt.exists() and not refresh:
        metadata = json.loads(receipt.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"原始缓存校验失败：{path}")
        return metadata
    response = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    if path.suffix == ".json":
        response.json()  # 拒绝 HTML 错误响应。
    path.write_bytes(response.content)
    metadata = {"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code, "sha256": hashlib.sha256(response.content).hexdigest()}
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def dividends_from_html(path):
    tables = pd.read_html(io.StringIO(path.read_bytes().decode("gb18030", errors="replace")))
    table = next(t for t in tables if "除权除息日" in str(t.columns) and "派息" in str(t.columns))
    table.columns = [col[-1] if isinstance(col, tuple) else col for col in table.columns]
    table = table.rename(columns={"公告日期": "announcement_date", "送股(股)": "bonus_per10",
                                 "转增(股)": "split_per10", "派息(税前)(元)": "cash_per10",
                                 "进度": "status", "除权除息日": "ex_date", "股权登记日": "record_date"})
    table["ex_date"] = pd.to_datetime(table["ex_date"], errors="coerce")
    table = table.loc[table["ex_date"].between(START, END)].copy()
    for col in ("cash_per10", "bonus_per10", "split_per10"):
        table[col] = pd.to_numeric(table[col], errors="raise")
    if not table["status"].eq("实施").all():
        raise ValueError("存在未实施分红，需人工核验")
    if ((table["bonus_per10"] != 0) | (table["split_per10"] != 0)).any():
        raise ValueError("存在送转股，当前现金分红模型不能直接回放")
    table["ex_date"] = table["ex_date"].dt.strftime("%Y-%m-%d")
    return table[["announcement_date", "record_date", "ex_date", "cash_per10", "bonus_per10", "split_per10", "status"]].sort_values("ex_date")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    jobs = []
    for key, (code, name) in STOCKS.items():
        directory = ROOT / "data" / key
        (directory / "raw").mkdir(parents=True, exist_ok=True)
        (directory / "processed").mkdir(exist_ok=True)
        for kind, adjustment in (("raw", ""), ("qfq", "qfq")):
            for year in range(2023, 2027):
                url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{code},day,{year}-01-01,{year}-12-31,640,{adjustment}"
                jobs.append((directory / f"raw/tencent_{kind}_{year}.json", url))
        jobs.append((directory / "raw/dividends.html", f"https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda job: fetch(*job, refresh=args.refresh), jobs))
    for key, (code, name) in STOCKS.items():
        directory = ROOT / "data" / key
        datasets = []
        for kind in ("raw", "qfq"):
            frames = [read_tencent(directory / f"raw/tencent_{kind}_{year}.json", f"sh{code}", kind == "qfq") for year in range(2023, 2027)]
            frame = pd.concat(frames, ignore_index=True)
            frame = frame.loc[frame["date"].between(START, END)].copy()
            frame["volume"] *= 100  # 腾讯沪市主板股票量为手，统一为股。
            frame = validate(frame, name + kind)
            if frame["date"].iloc[0] > "2023-01-10" or frame["date"].iloc[-1] != END:
                raise ValueError(f"{name} 数据区间不足：{frame['date'].iloc[0]} ~ {frame['date'].iloc[-1]}")
            frame.to_csv(directory / f"processed/{key}_{kind}.csv", index=False)
            datasets.append({"name": kind, "provider": "Tencent Finance", "adjustment": kind,
                             "rows": len(frame), "first": frame["date"].iloc[0], "last": frame["date"].iloc[-1]})
        dividends = dividends_from_html(directory / "raw/dividends.html")
        dividends.to_csv(directory / f"processed/{key}_dividends.csv", index=False)
        files = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"]
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stock": name,
                    "symbol": f"sh{code}", "currency": "CNY", "price_unit": "CNY/share", "volume_unit": "shares",
                    "request_start": START, "request_end": END, "datasets": datasets,
                    "dividends_provider": "Sina Finance", "dividends_rows": len(dividends), "files": files}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(name, datasets, flush=True)
        print(dividends.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
