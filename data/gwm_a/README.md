# 长城汽车A历史回放数据

标的 `sh601633`，币种CNY，价格为CNY/股，成交量为股。原始行情来自腾讯财经，分红日程来自新浪财经，均于2026-10-19抓取；逐请求URL、UTC抓取时间、HTTP状态和SHA256见 `raw/*.source.json`。原始响应按字节保存，处理后CSV截止2026-10-16，原始及前复权日期已核对一致。

- `processed/gwm_a_raw.csv`：未复权开高低收及成交量，用于成交。
- `processed/gwm_a_qfq.csv`：提供方前复权，用于公司行动核对。
- `processed/gwm_a_dividends.csv`：除息日、每10股现金金额、送转股字段。
- `quality.json`：分红来源、核验方法及未知信息；`manifest.json`：文件哈希。

行情范围为2023-01-03至2026-10-16，共915行；历史原始响应可能包含研究窗口之外的内容，回放只使用截至2026-10-16的数据。日历为行情提供方股票交易日，不跨A/H股日历前向填充。研究结果及指标信号位于 `research/additional_donchian/`。

腾讯A股成交量由手乘100转换为股；现金分红使用新浪已实施记录，每10股税前人民币元。技术指标只前向加回截至当日已除息现金股息，实际成交使用未复权价格。

复现：`.venv/bin/python scripts/fetch_extended_stocks.py`（默认复用并校验缓存），随后执行 `.venv/bin/python scripts/replay_additional_donchian.py`。
