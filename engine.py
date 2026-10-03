"""Momentum factor research engine.

Conventions (no look-ahead):
  * Signal at month-end t uses prices up to t-skip (classic 12-1: lookback=12, skip=1).
  * Portfolio formed at end of t is held during month t+1.
  * hold>1 uses Jegadeesh-Titman overlapping cohorts (each month 1/hold of the
    book is re-formed).
  * Costs = one-way cost (bps) x turnover of gross weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PPY = 12  # monthly periods per year


# --------------------------------------------------------------------------- #
# Prep
# --------------------------------------------------------------------------- #
def to_monthly(px):
    try:
        return px.resample("ME").last()
    except ValueError:  # older pandas
        return px.resample("M").last()


def momentum_signal(pm: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return pm.shift(skip) / pm.shift(lookback) - 1


def cohort_weights(sig: pd.DataFrame, n_q: int, min_names: int):
    """Top quantile (winners) / bottom quantile (losers), equal-weighted legs."""
    valid = sig.notna()
    cnt = valid.sum(axis=1)
    pct = sig.rank(axis=1, method="first", pct=True)
    q = np.ceil(pct * n_q).clip(1, n_q)
    win = (q == n_q) & valid
    los = (q == 1) & valid
    ok = (cnt >= max(min_names, n_q * 2)).astype(float)
    wn = win.sum(axis=1).replace(0, np.nan)
    ln = los.sum(axis=1).replace(0, np.nan)
    long_w = win.astype(float).div(wn, axis=0).fillna(0).mul(ok, axis=0)
    short_w = -los.astype(float).div(ln, axis=0).fillna(0).mul(ok, axis=0)
    return long_w, short_w


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_backtest(px: pd.DataFrame, lookback=12, skip=1, hold=1, n_q=5,
                 mode="long_short", cost_bps=20.0, min_names=10) -> pd.DataFrame:
    pm = to_monthly(px)
    rets = pm.pct_change(fill_method=None).fillna(0.0)
    sig = momentum_signal(pm, lookback, skip)
    long_w, short_w = cohort_weights(sig, n_q, min_names)

    def held(w):
        eff = sum(w.shift(k).fillna(0.0) for k in range(hold)) / hold
        return eff.shift(1).fillna(0.0)  # formed at t, earned in t+1

    hl, hs = held(long_w), held(short_w)
    w = hl if mode == "long_only" else hl + hs

    long_ret = (hl * rets).sum(axis=1)
    short_ret = (hs * rets).sum(axis=1)  # already signed (negative weights)
    gross = (w * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4
    net = gross - cost

    out = pd.DataFrame({"gross": gross, "net": net, "long": long_ret,
                        "short": short_ret, "turnover": turnover, "cost": cost})
    live = w.abs().sum(axis=1) > 0
    if not live.any():
        return out.iloc[0:0]
    return out.loc[live.idxmax():]


def quantile_returns(px: pd.DataFrame, lookback=12, skip=1, n_q=5, min_names=10):
    """Equal-weight next-month return of each momentum quantile (Q1 losers ... Qn winners)."""
    pm = to_monthly(px)
    nxt = pm.pct_change(fill_method=None).shift(-1)
    sig = momentum_signal(pm, lookback, skip)
    cnt = sig.notna().sum(axis=1)
    q = np.ceil(sig.rank(axis=1, method="first", pct=True) * n_q).clip(1, n_q)
    cols = {}
    for k in range(1, n_q + 1):
        cols[f"Q{k}"] = nxt.where(q == k).mean(axis=1)
    out = pd.DataFrame(cols)
    out = out[cnt >= max(min_names, n_q * 2)].dropna(how="all")
    return out


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def perf_stats(r: pd.Series, rf: float = 0.0) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 3:
        return {}
    ex = r - rf / PPY
    sd = r.std(ddof=1)
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    cagr = cum.iloc[-1] ** (PPY / n) - 1
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(PPY)
    return {
        "CAGR": cagr,
        "Ann. vol": sd * np.sqrt(PPY),
        "Sharpe": ex.mean() / sd * np.sqrt(PPY) if sd > 0 else np.nan,
        "Sortino": ex.mean() * PPY / downside if downside > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Hit rate": (r > 0).mean(),
        "Skew": stats.skew(r),
        "Excess kurtosis": stats.kurtosis(r),
        "Worst month": r.min(),
        "Months": n,
    }


def drawdown(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


def rolling_sharpe(r: pd.Series, window=36) -> pd.Series:
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(PPY)


# --------------------------------------------------------------------------- #
# Statistical significance
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def nw_tstat(r: pd.Series) -> dict:
    """t-stat of mean return with Newey-West (HAC) standard errors."""
    r = r.dropna()
    n = len(r)
    lags = max(_nw_lags(n), 1)
    m = sm.OLS(r.values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean_monthly": m.params[0], "t": m.tvalues[0], "p": m.pvalues[0], "lags": lags}


def capm_alpha(r: pd.Series, bench: pd.Series, rf: float = 0.0) -> dict:
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    if len(df) < 12:
        return {}
    y = df.iloc[:, 0] - rf / PPY
    x = df.iloc[:, 1] - rf / PPY
    n = len(df)
    m = sm.OLS(y.values, sm.add_constant(x.values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(_nw_lags(n), 1)})
    a = m.params[0]
    return {"alpha_ann": (1 + a) ** PPY - 1, "alpha_t": m.tvalues[0], "alpha_p": m.pvalues[0],
            "beta": m.params[1], "beta_t": m.tvalues[1], "r2": m.rsquared, "n": n}


def block_bootstrap_sharpe(r: pd.Series, block=6, n_boot=5000, seed=42) -> dict:
    """Circular block bootstrap of the annualised Sharpe ratio."""
    x = r.dropna().values
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    s = x[idx]
    sd = s.std(axis=1, ddof=1)
    sh = np.where(sd > 0, s.mean(axis=1) / sd * np.sqrt(PPY), np.nan)
    sh = sh[~np.isnan(sh)]
    return {"dist": sh, "ci_low": np.percentile(sh, 2.5), "ci_high": np.percentile(sh, 97.5),
            "p_le_zero": float((sh <= 0).mean()), "observed": x.mean() / x.std(ddof=1) * np.sqrt(PPY)}


def deflated_sharpe(r: pd.Series, trial_sharpes_monthly: np.ndarray) -> dict:
    """Bailey & Lopez de Prado (2014). Probability the true Sharpe > 0 after
    correcting for the number of strategy variants tried (selection bias) and
    for non-normal returns. trial_sharpes_monthly: per-period (non-annualised)
    Sharpes of ALL variants tried."""
    r = r.dropna()
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    n = len(trial_sharpes_monthly)
    if n < 2 or T < 12:
        return {}
    emc = 0.5772156649
    sr0 = np.sqrt(np.var(trial_sharpes_monthly, ddof=1)) * (
        (1 - emc) * stats.norm.ppf(1 - 1 / n) + emc * stats.norm.ppf(1 - 1 / (n * np.e)))
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-9))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"dsr": float(dsr), "sr0_annual": float(sr0 * np.sqrt(PPY)),
            "sr_annual": float(sr * np.sqrt(PPY)), "n_trials": n}


# --------------------------------------------------------------------------- #
# Robustness grid
# --------------------------------------------------------------------------- #
def robustness_grid(px, lookbacks, holds, skip, n_q, mode, cost_bps, min_names, rf=0.0):
    rows, series = [], {}
    for L in lookbacks:
        if L <= skip:
            continue
        for H in holds:
            bt = run_backtest(px, L, skip, H, n_q, mode, cost_bps, min_names)
            if bt.empty:
                continue
            r = bt["net"]
            s = perf_stats(r, rf)
            if not s:
                continue
            rows.append({"lookback": L, "hold": H, "Sharpe": s["Sharpe"], "CAGR": s["CAGR"],
                         "Max drawdown": s["Max drawdown"], "t-stat": nw_tstat(r)["t"],
                         "Turnover (mo.)": bt["turnover"].mean()})
            series[(L, H)] = r
    return pd.DataFrame(rows), series


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def regime_labels(bench_daily: pd.Series) -> pd.DataFrame:
    """Regimes known at the START of each month (lagged one month => no look-ahead).
      market : Bull / Bear  (trailing 12m benchmark return >0 / <=0)
      vol    : High / Low   (trailing 6m realised vol above / below expanding median)
      rebound: month where the benchmark rose while in a Bear state (momentum-crash setup)
    """
    bm = to_monthly(bench_daily.dropna())
    bret = bm.pct_change(fill_method=None)
    trail = bm / bm.shift(12) - 1
    vol = bret.rolling(6).std()
    bull = (trail > 0).shift(1)
    hv = (vol > vol.expanding(12).median()).shift(1)
    out = pd.DataFrame(index=bm.index)
    out["Market"] = np.where(bull.isna(), None, np.where(bull.astype(bool), "Bull", "Bear"))
    out["Volatility"] = np.where(hv.isna(), None, np.where(hv.astype(bool), "High vol", "Low vol"))
    bear = out["Market"] == "Bear"
    out["Bear rebound"] = np.where(bear & (bret > 0), "Rebound month",
                                   np.where(bear, "Bear, still falling", None))
    return out


def regime_table(r: pd.Series, labels: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    df = pd.concat([r.rename("r"), labels.rename("g")], axis=1, join="inner").dropna()
    rows = []
    for g, d in df.groupby("g"):
        x = d["r"]
        sd = x.std(ddof=1) if len(x) > 1 else np.nan
        rows.append({"Regime": g, "Months": len(x), "Mean / month": x.mean(),
                     "Ann. vol": sd * np.sqrt(PPY),
                     "Sharpe": (x.mean() - rf / PPY) / sd * np.sqrt(PPY) if sd and sd > 0 else np.nan,
                     "Hit rate": (x > 0).mean(), "Worst month": x.min()})
    return pd.DataFrame(rows).set_index("Regime")


def welch_between(r: pd.Series, labels: pd.Series, a: str, b: str) -> dict:
    df = pd.concat([r.rename("r"), labels.rename("g")], axis=1, join="inner").dropna()
    xa, xb = df.loc[df.g == a, "r"], df.loc[df.g == b, "r"]
    if len(xa) < 3 or len(xb) < 3:
        return {}
    t, p = stats.ttest_ind(xa, xb, equal_var=False)
    return {"diff": xa.mean() - xb.mean(), "t": t, "p": p, "n_a": len(xa), "n_b": len(xb)}


def subperiods(r: pd.Series, k: int = 3, rf: float = 0.0) -> pd.DataFrame:
    r = r.dropna()
    rows = {}
    for pos in np.array_split(np.arange(len(r)), k):
        chunk = r.iloc[pos]
        s = perf_stats(chunk, rf)
        if s:
            rows[f"{chunk.index[0]:%b %Y} – {chunk.index[-1]:%b %Y}"] = s
    return pd.DataFrame(rows).T
