import argparse
import csv
import json
from datetime import date
from pathlib import Path

from .engine import analyze
from .io import iso_date, load_fundamentals, load_prices


def main():
    parser = argparse.ArgumentParser(description="从 CSV 计算股票决策研究指标，不生成买卖评级")
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--fundamentals", type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--as-of", default=date.today().isoformat(), help="研究日 YYYY-MM-DD，按当日收盘后计算")
    parser.add_argument("--benchmark", help="prices CSV 中的基准代码，需与股票同币种")
    parser.add_argument("--output", type=Path, help="另存完整 JSON 报告")
    args = parser.parse_args()
    try:
        iso_date(args.as_of)
        result = analyze(load_prices(args.prices), load_fundamentals(args.fundamentals),
                         args.symbol, args.as_of, args.benchmark)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
    print(f"{result['symbol']} | 研究日 {result['as_of']} | 行情日 {result['price_date']}")
    print(f"收盘价 {result['close']:.2f} {result['currency']} | 财务期末 {result['financial_period_end'] or '无'}")
    for key, metric in result["metrics"].items():
        value = metric["value"]
        if value is None:
            label = "N/A"
        elif key == "relative_return_63d":
            label = f"{value * 100:.2f} 个百分点"
        elif metric["unit"] == "ratio":
            label = f"{value:.2%}"
        elif metric["unit"] == "multiple":
            label = f"{value:.2f} 倍"
        else:
            label = f"{value:,.2f} {metric['unit']}"
        print(f"  {metric['name']}: {label}")
    for warning in result["warnings"]:
        print(f"提示：{warning}")
    if args.output:
        print(f"JSON 报告：{args.output}")


if __name__ == "__main__":
    main()
