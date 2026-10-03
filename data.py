"""Data layer for Momentum Lab.

Three sources: Yahoo Finance (NSE .NS tickers), user-uploaded CSV, or a synthetic
demo universe (so the app runs offline and you can sanity-check the engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Current Nifty 50-style universe. NOTE: using today's constituents over a long
# history introduces SURVIVORSHIP BIAS. Swap in a point-in-time universe via CSV
# upload for publication-grade work.
NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "NESTLEIND", "ULTRACEMCO", "WIPRO", "HCLTECH", "NTPC",
    "POWERGRID", "ONGC", "TATASTEEL", "M&M", "TECHM", "BAJAJFINSV", "ADANIENT",
    "COALINDIA", "JSWSTEEL", "GRASIM", "HINDALCO", "DRREDDY", "CIPLA", "EICHERMOT",
    "BRITANNIA", "DIVISLAB", "APOLLOHOSP", "HEROMOTOCO", "BPCL", "TATACONSUM",
    "SBILIFE", "INDUSINDBK", "BAJAJ-AUTO", "ADANIPORTS",
]


def load_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily adjusted close from Yahoo. Plain NSE symbols get '.NS' appended;
    index symbols (starting with '^') are used as-is. Columns are returned
    without the '.NS' suffix."""
    import yfinance as yf

    yf_syms = [t if t.startswith("^") or "." in t else f"{t}.NS" for t in tickers]
    raw = yf.download(
        yf_syms, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no data (check tickers / connection).")
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(yf_syms[0])
    close.columns = [str(c).replace(".NS", "") for c in close.columns]
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def parse_csv(file) -> pd.DataFrame:
    """CSV with Date in the first column and one price column per ticker.
    An optional column named BENCHMARK is used as the market index."""
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()
    return df


def equal_weight_benchmark(px: pd.DataFrame) -> pd.Series:
    """Fallback market proxy: daily-rebalanced equal-weight index of the universe."""
    r = px.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    return (1 + r).cumprod().rename("EW Universe")


def synthetic_prices(n_stocks: int = 45, end: str = "2026-09-30", seed: int = 7) -> pd.DataFrame:
    """Offline demo data: one-factor market + persistent stock-specific drift
    (so momentum genuinely exists) + a few crash/rebound episodes. NOT real."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", end)
    T = len(idx)
    mkt = rng.normal(0.0004, 0.010, T)
    # crash then rebound episodes
    for s in (int(T * 0.15), int(T * 0.45), int(T * 0.75)):
        mkt[s:s + 60] -= 0.004
        mkt[s + 60:s + 100] += 0.006
    betas = rng.uniform(0.6, 1.4, n_stocks)
    drift = np.zeros((T, n_stocks))
    d = rng.normal(0, 0.0004, n_stocks)
    for t in range(T):  # slow AR(1) drift => medium-term persistence
        d = 0.995 * d + rng.normal(0, 0.00012, n_stocks)
        drift[t] = d
    idio = rng.normal(0, 0.014, (T, n_stocks))
    rets = mkt[:, None] * betas + drift + idio
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                      columns=[f"DEMO{i:02d}" for i in range(n_stocks)])
    # late listers / early delisters to exercise the NaN handling
    px.iloc[: T // 3, :5] = np.nan
    px.iloc[int(T * 0.8):, 5:8] = np.nan
    px["BENCHMARK"] = 100 * np.exp(np.cumsum(mkt))
    return px