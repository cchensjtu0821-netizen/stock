# 股票决策研究

用一组定义清楚、能追溯数据来源的指标，支持选股、估值和持仓复盘。第一版以美股中长期研究为起点，提供 **16 项指标的离线计算**，保留现有 `research/` 中的 AVGO 历史研究。

研究顺序是：**公司经营是否改善 → 价格是否合理 → 市场走势如何 → 这笔投资可能造成多大损失**。指标用于提出和检验投资假设；当前版本没有经过历史收益验证的买卖策略。

## A 股紫金矿业波段研究

已增加真实数据的三年回放，区间为 **2023-10-18 至 2026-10-16**。包括紫金 A 股、上海金 Au99.99、申万贵金属指数 801053 和上证综指的关联分析，以及五组固定波段规则的交易记录、净值、胜率和时间留出回放。策略回放已完成，有限样本尚不足以确认稳定的投资优势。

- [完整可视化报告](research/zijin/report.html) / [文字报告](research/zijin/report.md)
- [数据目录与来源](data/zijin/README.md)
- [规则参数](config/zijin_swing.json)
- [指标和交易执行代码](stock_analysis/swing.py) / [关联分析代码](stock_analysis/relationships.py)

研究功能需要额外依赖；可使用项目中的 `.venv/bin/python`。新环境安装和复现：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python scripts/fetch_zijin_data.py
.venv/bin/python scripts/replay_zijin.py
.venv/bin/python -m unittest discover -s tests -v
```

下载脚本优先使用已有原始缓存；`--refresh` 重新访问数据源。申万站点初次取数时存在证书链校验问题，已在来源清单注明；重新下载默认保留证书校验。回放仅访问本地数据并校验 SHA256，不需要联网。

### 多指标与混合规则探索

扩展研究覆盖 MACD、EMA、RSI、KDJ、布林带、唐奇安通道、CCI、MFI、OBV、ADX、Supertrend 和威廉指标，共 **28 条固定候选规则：16 条单指标规则、12 条混合规则**。

- [多指标可视化报告](research/zijin/indicator_lab/report.html) / [文字报告](research/zijin/indicator_lab/report.md)
- [全部策略数值](research/zijin/indicator_lab/leaderboard.csv) / [逐笔交易](research/zijin/indicator_lab/trades.csv)
- [指标与规则实现](stock_analysis/indicator_lab.py) / [实验设置](config/indicator_lab.json)

每条规则比较“指标自身退出＋共同 ATR 风控”和“仅共同风控”两种方式；另保留经典 MACD/EMA 金叉买死叉卖对照，以及持仓 40 日、成本翻倍和固定投入上限的敏感性结果。主比较按前两年收益排序，保留所有候选结果；最后一年已在前轮研究中使用，只标作复查段。

```bash
.venv/bin/python scripts/explore_zijin_indicators.py
```

### 招商银行、兆易创新跨标的回放

沿用同一套28条候选和交易参数，研究窗口仍为 **2023-10-18 至 2026-10-16**，每只股票的每个策略独立使用10万元。未根据两只股票结果调整规则；黄金板块过滤仍为原黄金和贵金属条件，未替换为银行或半导体指数。

- [三股并排对比](research/stock_comparison/report.html) / [文字版](research/stock_comparison/report.md) / [全部数值](research/stock_comparison/strategy_comparison.csv)
- [招商银行报告](research/cmb/indicator_lab/report.html) / [文字版](research/cmb/indicator_lab/report.md) / [逐笔交易](research/cmb/indicator_lab/trades.csv) / [数据来源](data/cmb/README.md)
- [兆易创新报告](research/gigadevice/indicator_lab/report.html) / [文字版](research/gigadevice/indicator_lab/report.md) / [逐笔交易](research/gigadevice/indicator_lab/trades.csv) / [数据来源](data/gigadevice/README.md)

```bash
.venv/bin/python scripts/fetch_comparison_stocks.py
.venv/bin/python scripts/explore_zijin_indicators.py --stock cmb
.venv/bin/python scripts/explore_zijin_indicators.py --stock gigadevice
.venv/bin/python scripts/compare_indicator_stocks.py
```

下载脚本保留原始响应及每次抓取时间，默认复用缓存；股票以未复权价格成交，技术指标只加回当日已除息股息。三年账户收益以初始10万元为分母，未平仓按末日收盘估值；低资金暴露和100股整数交易限制需结合持仓权重解释。分段结果属于历史复查，不能作为独立样本外或实盘验证。

## 从这些指标开始

| 维度 | 优先关注 | 帮助回答的问题 |
| --- | --- | --- |
| 增长 | TTM 营收同比、稀释 EPS 同比 | 业务在增长吗？增长是否落实到每股盈利？ |
| 盈利质量 | ROE、营业利润率、经营现金流 / 净利润、自由现金流率 | 利润有现金支持吗？回报是否依赖负债或回购？ |
| 估值 | TTM PE、自由现金流收益率 | 当前价格需要多强的盈利与现金流来支撑？ |
| 趋势 | 距 50 / 200 日均线、12−1 月动量、63 日相对基准收益 | 股价和相对市场表现是否支持当前判断？ |
| 风险 | 净债务 / EBITDA、60 日波动率、252 日最大回撤、20 日成交额代理 | 偿债、价格波动和流动性风险有多大？ |

如果刚开始，可以先看营收增长、EPS 增长、现金利润比、FCF 收益率、净债务倍数、200 日均线偏离和最大回撤。再用财报解释异常值，不必同时给所有指标设阈值。

PE 和营业利润率的基础定义可参考 [SEC 财务报表入门](https://www.sec.gov/about/reports-publications/beginners-guide-financial-statements)。指标应结合行业、会计口径和业务周期比较。完整公式、适用边界和研究流程见 [指标字典](docs/indicators.md)。

## 运行一个完整示例

需要 Python 3.10 或更新版本，仅使用标准库，无需安装依赖。在仓库根目录运行：

```bash
python3 examples/generate_demo.py
python3 -m stock_analysis \
  --prices examples/demo_prices.csv \
  --fundamentals examples/demo_fundamentals.csv \
  --symbol DEMO \
  --benchmark DEMO_BENCH \
  --as-of 2025-12-31 \
  --output output/DEMO.json
```

`DEMO` 和 `DEMO_BENCH` 都是虚构代码，所有示例数值均为模拟数据。命令在终端展示结果，JSON 保存原始数值、公式说明、数据日期、来源与缺失提示。JSON 中百分比使用小数，例如 `0.2` 表示 `20%`。

使用自己的 CSV 时替换路径和代码即可。`--fundamentals` 与 `--benchmark` 都可省略，缺少的指标显示 `N/A`。默认研究日为运行机器本地日期；可复现的研究应明确指定 `--as-of`。

## 数据约定

行情 CSV 必填列：

```csv
symbol,date,close,adj_close,currency,source
```

可选 `volume` 为当日实际成交股数。`close` 是未复权收盘价，用于估值和成交额代理；`adj_close` 用于收益、均线和回撤，同一数据集应统一拆股和分红调整方式。每行代表一个完整交易日，不加入周末填充值；每个 `symbol,date` 只能出现一次。代码区分大小写。相对基准收益要求同币种、对应日期完整对齐，且两条序列采用相同的收益调整方式。

财务 CSV 必填列及可选数值列见 [数据字段说明](docs/data.md)，示例见 [demo_fundamentals.csv](examples/demo_fundamentals.csv)。财务数值可以留空；空值不会被替换成零。

**财务金额使用基础货币单位，股数使用实际股数**，例如 `1200000000` 美元与 `100000000` 股；EPS 单位为货币 / 股。若来源以百万或亿元展示，导入前先换算。股价、EPS、股数、股权类别和 ADR 比例必须一致。

`--as-of` 按研究日收盘后解释，使用当日及以前的行情和披露日期不晚于研究日的财务快照。仅有日期，无法区分盘前与盘后；**当日盘后财报不能据此模拟在当日收盘价成交**。严格回测还需要披露时间、历史数据版本和下一可交易时点。

## 验证与下一步

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖公式、非正分母、缺失数据、拆股价格用途、财务披露过滤、基准对齐与 CSV 校验。

下一阶段先接入确定市场的数据源，保存原始数据与抓取时间；之后做同行业比较、历史估值分位和样本外验证。验证时记录交易成本、退市股票、基准收益与最大回撤，再决定是否形成组合规则。当前工具不自动获取实时数据、排名或下单。

现有 [AVGO 研究](research/AVGO_2026-09-09.md) 是带历史截止日期的个股分析；本次没有重新核验其中数值，也没有将其作为当前行情导入。
