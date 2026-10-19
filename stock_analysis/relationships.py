"""收益关联研究：同期相关、滞后相关与多因子描述性回归。"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FACTORS = {"gold": "上海金 Au99.99", "sector": "申万贵金属 801053", "sse": "上证综指 000001"}


def pair_metrics(pair):
    pair = pair.dropna()
    if len(pair) < 10 or (pair.std() == 0).any():
        return {"n": len(pair), "pearson": None, "spearman": None, "beta": None, "r_squared": None}
    y, x = pair.iloc[:, 0], pair.iloc[:, 1]
    correlation = float(pearsonr(y, x).statistic)
    return {"n": len(pair), "pearson": correlation, "spearman": float(spearmanr(y, x).statistic),
            "beta": float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)), "r_squared": correlation ** 2}


def regression(returns):
    data = returns[["zijin", *FACTORS]].dropna()
    y = data["zijin"].to_numpy()
    x = np.column_stack([np.ones(len(data)), data[list(FACTORS)].to_numpy()])
    n, k = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, 6):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / 6) * (cross + cross.T)
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ meat @ bread * n / (n - k)
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    total = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (residual ** 2).sum() / total
    coefficients = {}
    for i, name in enumerate(["intercept", *FACTORS]):
        coefficients[name] = {"beta": float(beta[i]), "hac5_standard_error": float(errors[i]),
                              "ci95": [float(beta[i] - 1.96 * errors[i]), float(beta[i] + 1.96 * errors[i])]}
    incremental_r2, vif = {}, {}
    for index, name in enumerate(FACTORS, 1):
        reduced = np.delete(x, index, axis=1)
        reduced_residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        incremental_r2[name] = float(((reduced_residual ** 2).sum() - (residual ** 2).sum()) / total)
        factor = x[:, index]
        factor_residual = factor - reduced @ np.linalg.lstsq(reduced, factor, rcond=None)[0]
        vif[name] = float(((factor - factor.mean()) ** 2).sum() / (factor_residual ** 2).sum())
    return {"n": n, "r_squared": float(r_squared), "adjusted_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k)),
            "coefficients": coefficients, "incremental_r_squared": incremental_r2, "vif": vif,
            "factor_correlation": data[list(FACTORS)].corr().to_dict()}


def relationship_report(frame, config):
    # 先在股票交易日历上对齐价格，再计算收益；缺失行情和其后一天不补零。
    prices = frame[["stock_total_return_index", "gold_close", "sector_close", "sse_close"]].rename(
        columns={"stock_total_return_index": "zijin", "gold_close": "gold", "sector_close": "sector", "sse_close": "sse"})
    daily = prices.pct_change(fill_method=None)
    daily["zijin"] = frame["stock_return"]
    sample = daily.loc[config["test_start"]:config["test_end"]]
    weekly = prices.resample("W-FRI").last().pct_change(fill_method=None).loc[config["test_start"]:config["test_end"]]
    rows, rolling = [], pd.DataFrame(index=sample.index)
    for name in FACTORS:
        pair = sample[["zijin", name]]
        record = {"factor": name, "name": FACTORS[name], "daily": pair_metrics(pair),
                  "weekly": pair_metrics(weekly[["zijin", name]]),
                  "lag1_predictive": pair_metrics(pd.concat([sample["zijin"], daily[name].shift().reindex(sample.index)], axis=1))}
        for window in (60, 120):
            values = daily["zijin"].rolling(window, min_periods=int(window * .9)).corr(daily[name]).reindex(sample.index)
            rolling[f"{name}_corr{window}"] = values
            valid = values.dropna()
            record[f"rolling_{window}"] = {"latest": float(valid.iloc[-1]), "minimum": float(valid.min()),
                                            "median": float(valid.median()), "maximum": float(valid.max())}
        record["by_calendar_year"] = {str(year): pair_metrics(group) for year, group in pair.groupby(pair.index.year)}
        rows.append(record)
    return {"factors": rows, "multifactor": regression(sample),
            "daily_aligned": sample, "rolling": rolling}
