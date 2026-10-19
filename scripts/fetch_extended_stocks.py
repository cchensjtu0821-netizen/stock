"""归档长城A股、赛力斯H股和阿里港股日线、分红与来源。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_comparison_stocks import fetch, dividends_from_html
from fetch_zijin_data import read_tencent, validate

ROOT = Path(__file__).resolve().parents[1]
STOCKS = {"gwm_a": ("sh601633", "长城汽车A", "CNY"),
          "seres_hk": ("hk09927", "赛力斯H股", "HKD"),
          "alibaba_hk": ("hk09988", "阿里巴巴港股", "HKD")}
END = "2026-10-16"


def main():
    jobs = []
    for key, (symbol, name, currency) in STOCKS.items():
        directory = ROOT / f"data/{key}"
        (directory / "raw").mkdir(parents=True, exist_ok=True)
        (directory / "processed").mkdir(exist_ok=True)
        for kind in ("raw", "qfq"):
            if currency == "CNY":
                for year in range(2023, 2027):
                    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{year}-01-01,{year}-12-31,640,{'qfq' if kind == 'qfq' else ''}"
                    jobs.append((directory / f"raw/tencent_{kind}_{year}.json", url))
            elif kind == "raw":
                jobs.append((directory / "raw/tencent_raw.json", f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,2023-01-01,{END},1500,"))
            else:
                for year in range(2023, 2027):
                    jobs.append((directory / f"raw/tencent_qfq_{year}.json", f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param={symbol},day,,{year}-12-31,640,qfq"))
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{symbol[2:]}.phtml" if currency == "CNY" else f"https://stock.finance.sina.com.cn/hkstock/dividends/{symbol[2:]}.html"
        jobs.append((directory / "raw/dividends.html", url))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda job: fetch(*job), jobs))
    for key, (symbol, name, currency) in STOCKS.items():
        directory = ROOT / f"data/{key}"
        frames = {}
        for kind in ("raw", "qfq"):
            paths = sorted((directory / "raw").glob(f"tencent_{kind}*.json"))
            paths = [path for path in paths if not path.name.endswith(".source.json")]
            parts = []
            for path in paths:
                payload = json.loads(path.read_text())
                if key == "seres_hk" and ("2023" in path.name or "2024" in path.name):
                    continue  # 上市前年度的空响应不是价格数据。
                if payload.get("code") != 0:
                    # 赛力斯上市前年度无数据，仅保留错误原文与抓取记录。
                    if key == "seres_hk" and ("2023" in path.name or "2024" in path.name):
                        continue
                    raise ValueError(f"数据源返回错误：{path} {payload.get('msg')}")
                parts.append(read_tencent(path, symbol, kind == "qfq"))
            frame = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last")
            frame = frame.loc[frame["date"].between("2023-01-01", END)].copy()
            if currency == "CNY": frame["volume"] *= 100
            frame = validate(frame, key + kind)
            frame.to_csv(directory / f"processed/{key}_{kind}.csv", index=False)
            frames[kind] = frame.set_index("date")
        raw, qfq = frames["raw"], frames["qfq"]
        if not raw.index.equals(qfq.index):
            raise ValueError(f"{key}原始与复权日期不一致：{len(raw)} / {len(qfq)}")
        if currency == "CNY":
            dividends = dividends_from_html(directory / "raw/dividends.html")
            quality = {"dividend_method": "Sina implemented cash dividends", "dividend_currency": "CNY"}
        else:
            # 仅现金调整时，各OHLC的 raw-qfq 差相同。先验证，避免把拆股当作派息。
            offsets = raw[["open", "high", "low", "close"]] - qfq[["open", "high", "low", "close"]]
            assert ((offsets.max(axis=1) - offsets.min(axis=1)) < .0031).all()
            offset = offsets["close"].round(3)
            change = (-offset.diff()).round(3)
            assert not (change < -.003).any(), "出现非现金调整，需人工核查"
            events = change.loc[change > .003]
            dividends = pd.DataFrame({"ex_date": events.index, "cash_per10": events.values * 10,
                                      "bonus_per10": 0, "split_per10": 0, "status": "行情源现金调整推算"})
            # 来源独立的除净日用于核对；金额为提供方HKD除息调整代理，非实际到账汇率。
            tables = pd.read_html(io.StringIO((directory / "raw/dividends.html").read_bytes().decode("gb18030", errors="replace")), header=0)
            table = next(table for table in tables if "除净日" in str(table.columns))
            table.columns = [col[-1] if isinstance(col, tuple) else col for col in table.columns]
            schedule_dates = pd.to_datetime(table["除净日"], format="%Y-%m-%d", errors="coerce").dropna()
            expected = sorted(schedule_dates.loc[schedule_dates.between("2023-01-01", END)].dt.strftime("%Y-%m-%d").unique())
            if events.index.tolist() != expected:
                raise ValueError(f"{key} 除净日不一致：行情源{events.index.tolist()}，独立日程{expected}")
            quality = {"dividend_method": "provider raw-minus-qfq difference changes; cash-adjustment proxy, not verified broker payout",
                       "dividend_currency": "HKD", "ex_dates_from_provider": events.index.tolist(),
                       "independent_schedule_expected_dates": expected,
                       "schedule_dates_match": events.index.tolist() == expected,
                       "cash_amount_source_precision": "HKD 0.001 per share", "actual_USD_HKD_payout_conversion": "unknown"}
        dividends.to_csv(directory / f"processed/{key}_dividends.csv", index=False)
        (directory / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
        files = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(list((directory / "raw").glob("*")) + list((directory / "processed").glob("*")) + [directory / "quality.json"])]
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stock": name, "symbol": symbol,
                    "currency": currency, "price_unit": currency + "/share", "volume_unit": "shares",
                    "raw_first": raw.index[0], "raw_last": raw.index[-1], "rows": len(raw), "files": files}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(key, manifest["rows"], manifest["raw_first"], manifest["raw_last"], flush=True)
        print(dividends.to_string(index=False), quality, flush=True)


if __name__ == "__main__": main()
