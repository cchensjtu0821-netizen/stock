# 兆易创新 A 股三年回放数据

标的为兆易创新 A 股 603986，币种人民币，价格单位元/股，成交量单位股。2026-10-19 联网取得原始行情及分红；逐请求的 UTC 抓取时间、URL、HTTP 状态和 SHA256 见 `raw/*.source.json`。处理后数据截止 2026-10-16，研究窗口为 2023-10-18 至 2026-10-16；2023 年初起的数据用于指标预热。

| 文件 | 来源与口径 | 行数 |
| --- | --- | --- |
| `processed/gigadevice_raw.csv` | 腾讯未复权 OHLC，成交量由手乘100转为股 | 915 |
| `processed/gigadevice_qfq.csv` | 腾讯前复权 OHLC，仅核对分红和日期，不用于成交 | 915 |
| `processed/gigadevice_dividends.csv` | 新浪已实施分红，派息单位为每10股税前元 | 3 |

腾讯请求接口为 `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get`，逐年查询沪市 `sh603986`；完整查询参数保存在原始文件旁的来源记录中。[新浪分红来源](https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/603986.phtml)。原响应保留原始字节，年度请求可能返回研究末日以后的内容；只使用处理后 CSV 中不晚于截止日的数据。

2024年公告的分红记录为“不分配”，无实际除息日，不作为现金分红入账。处理后列表包括2023、2025、2026年已实施分红，样本内无送转股。726个股票研究交易日与原沪指日历对应，无缺失股票交易日。技术指标使用截至当日已除息股息的前向累计加回价格；该收盘序列与腾讯前复权序列差固定1.71元，见 [核对结果](../../research/gigadevice/indicator_lab/data_quality.json)。原始价格用于成交，股息与红利税单独记账。此核对不代表所有来源数据已经独立审计。

28条原始规则完整照搬紫金研究；其中 `macd_gold_sector` 仍采用上海金 Au99.99 和申万贵金属801053，未替换成半导体板块。黄金、贵金属、沪指来自 [原紫金数据快照](../zijin/README.md)，清单生成于2026-10-18；不是本次重新下载。贵金属2026-10-15缺失时依赖该条件的入场关闭，不填补价格。股价较高时100股最小交易单位和1%名义风险预算会限制开仓，需结合实际持仓权重解释结果。

本次实际诊断发现：最后一年复查段241个交易日，100股对应的入场前2ATR名义风险均超过期初10万元账户的1000元预算（最低约1935.39元），28条主规则均无交易、无持仓。0%复查收益不能用于评价该年度信号的盈利能力；另有预先定义的取消1%风险限仓、仅保留30%入场上限的敏感性结果，见完整报告及 `lot_diagnostics.json`。

复现：`.venv/bin/python scripts/fetch_comparison_stocks.py` 优先使用已校验的原始缓存；`--refresh` 会重新获取。随后运行 `.venv/bin/python scripts/explore_zijin_indicators.py --stock gigadevice`。文件哈希和来源见 `manifest.json`。
