# 招商银行 A 股三年回放数据

标的为招商银行 A 股 600036，币种人民币，价格单位元/股，成交量单位股。2026-10-19 联网取得原始行情及分红；逐请求的 UTC 抓取时间、URL、HTTP 状态和 SHA256 见 `raw/*.source.json`。处理后数据截止 2026-10-16，研究窗口为 2023-10-18 至 2026-10-16；2023 年初起的数据用于指标预热。

| 文件 | 来源与口径 | 行数 |
| --- | --- | --- |
| `processed/cmb_raw.csv` | 腾讯未复权 OHLC，成交量由手乘100转为股 | 915 |
| `processed/cmb_qfq.csv` | 腾讯前复权 OHLC，仅核对分红和日期，不用于成交 | 915 |
| `processed/cmb_dividends.csv` | 新浪已实施分红，派息单位为每10股税前元 | 5 |

腾讯请求接口为 `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get`，逐年查询沪市 `sh600036`；完整查询参数保存在原始文件旁的来源记录中。[新浪分红来源](https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/600036.phtml)。原响应保留原始字节，年度请求可能返回研究末日以后的内容；只使用处理后 CSV 中不晚于截止日的数据。

样本内无送转股。726个股票研究交易日与原沪指日历对应，无缺失股票交易日。技术指标使用截至当日已除息股息的前向累计加回价格；该收盘序列与腾讯前复权序列差固定7.726元，见 [核对结果](../../research/cmb/indicator_lab/data_quality.json)。原始价格用于成交，股息与红利税单独记账。此核对不代表所有来源数据已经独立审计。

28条原始规则完整照搬紫金研究；其中 `macd_gold_sector` 仍采用上海金 Au99.99 和申万贵金属801053，未替换成银行板块。黄金、贵金属、沪指来自 [原紫金数据快照](../zijin/README.md)，清单生成于2026-10-18；不是本次重新下载。贵金属2026-10-15缺失时依赖该条件的入场关闭，不填补价格。

复现：`.venv/bin/python scripts/fetch_comparison_stocks.py` 优先使用已校验的原始缓存；`--refresh` 会重新获取。随后运行 `.venv/bin/python scripts/explore_zijin_indicators.py --stock cmb`。文件哈希和来源见 `manifest.json`。
