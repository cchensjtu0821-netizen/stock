"""生成确定性的虚构数据，包含工作日但不模拟真实交易所节假日。"""

import csv
import math
from datetime import date, timedelta
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    dates = []
    day = date(2025, 12, 31)
    while len(dates) < 320:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day -= timedelta(days=1)
    with (root / "demo_prices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "symbol", "date", "close", "adj_close", "volume", "currency", "source"))
        writer.writeheader()
        for i, day in enumerate(reversed(dates)):
            for symbol, start, drift, amplitude in (
                ("DEMO", 80, 0.0008, 0.05), ("DEMO_BENCH", 100, 0.0003, 0.025)
            ):
                price = round(start * math.exp(drift * i) * (1 + amplitude * math.sin(i / 13)), 4)
                writer.writerow({
                    "symbol": symbol, "date": day, "close": price, "adj_close": price,
                    "volume": 1_000_000 + i * 100, "currency": "USD", "source": "synthetic-demo-not-market-data",
                })
    print(f"虚构行情：{root / 'demo_prices.csv'}；研究日 2025-12-31")


if __name__ == "__main__":
    main()
