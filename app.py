"""Momentum Lab — quant research platform for stock analysis + momentum factor backtesting.

Run:  python -m streamlit run app.py
"""
import warnings

import numpy as np
import pandas as pd
import plotly.express as pex
import plotly.graph_objects as go
import streamlit as st

import data as D
import engine as E

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Momentum Lab", page_icon="📈", layout="wide")

try:
    _t = st.context.theme.type
except Exception:  # older Streamlit
    _t = None
DARK = (_t or st.get_option("theme.base") or "light") == "dark"
if DARK:
    GOOD, BAD, WARN, ACCENT = "#2ee6a6", "#ff7a93", "#ffcf66", "#5cc8ff"
    GRID = "rgba(140,155,185,.20)"
    CARD, BORDER, TXT, MUTED = "#16213a", "#2a3a5c", "#f1f5fb", "#a3b4d2"
else:
    GOOD, BAD, WARN, ACCENT = "#0a8f63", "#d02f55", "#b97800", "#2563eb"
    GRID = "rgba(100,115,145,.18)"
    CARD, BORDER, TXT, MUTED = "#ffffff", "#d9e1ee", "#0f172a", "#52627e"
COLORS = [ACCENT, GOOD, WARN, "#8e5bd6", BAD, "#7b8aa6"]

_CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1400px}
header[data-testid="stHeader"]{background:transparent}
.hero{background:linear-gradient(120deg,#0f2f5a 0%,#0d4a55 60%,#12306a 100%);
  border-radius:18px;padding:24px 30px;margin-bottom:16px}
.hero h1{margin:0;font-size:2.2rem;letter-spacing:-.5px;padding:0;color:#ffffff}
.hero p{margin:.25rem 0 0;color:#d6e4f7;font-size:1rem}
.chip{display:inline-block;padding:3px 11px;border-radius:999px;background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.30);color:#ffffff;font-size:.78rem;margin:12px 6px 0 0}
.kpi,.sc{background:__CARD__;border:1px solid __BORDER__;border-radius:14px;padding:13px 16px;
  height:100%;box-shadow:0 1px 3px rgba(15,23,42,.08)}
.sc{border-left:5px solid __MUTED__;border-radius:12px}
.kpi-l,.sc-t{color:__MUTED__;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.kpi-v{color:__TXT__;font-size:1.6rem;font-weight:700;margin-top:2px;line-height:1.2}
.kpi-s,.sc-d{color:__MUTED__;font-size:.8rem;margin-top:2px}
.sc-v{color:__TXT__;font-weight:700;font-size:1.15rem;margin:2px 0}
.kpi.good .kpi-v,.sc.good .sc-v{color:__GOOD__}
.kpi.bad .kpi-v,.sc.bad .sc-v{color:__BAD__}
.kpi.warn .kpi-v,.sc.warn .sc-v{color:__WARN__}
.sc.good{border-left-color:__GOOD__}.sc.bad{border-left-color:__BAD__}.sc.warn{border-left-color:__WARN__}
h3{margin-top:.6rem}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:rgba(128,140,170,.10);border-radius:10px 10px 0 0;padding:8px 16px}
.stTabs [aria-selected="true"]{background:rgba(128,140,170,.24)}
</style>
"""
_rep = {"__GOOD__": GOOD, "__BAD__": BAD, "__WARN__": WARN, "__CARD__": CARD,
        "__BORDER__": BORDER, "__TXT__": TXT, "__MUTED__": MUTED}
for _k, _v in _rep.items():
    _CSS = _CSS.replace(_k, _v)
st.markdown(_CSS, unsafe_allow_html=True)

PCT_KEYS = {"CAGR", "Ann. vol", "Max drawdown", "Hit rate", "Worst month", "Mean / month"}


# ------------------------------- helpers ----------------------------------- #
def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi {tone}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


def kpis(items):
    for c, it in zip(st.columns(len(items)), items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def style(fig, h=340, title=None):
    fig.update_layout(template="plotly_dark" if DARK else "plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(t=50, b=10, l=10, r=10), colorway=COLORS,
                      legend=dict(orientation="h", y=1.14, x=0, title=""))
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in out.columns:
        for i in out.index:
            v = df.loc[i, c]
            if pd.isna(v):
                out.loc[i, c] = "–"
            elif c == "Months":
                out.loc[i, c] = f"{int(v)}"
            elif c in PCT_KEYS:
                out.loc[i, c] = f"{v:.1%}"
            else:
                out.loc[i, c] = f"{v:.2f}"
    return out


def fmt_stats(d: dict) -> pd.Series:
    return fmt_df(pd.DataFrame({"x": d}))["x"]


def segments(mask: pd.Series):
    segs, start, prev = [], None, None
    for d, v in mask.items():
        if v and start is None:
            start = d
        if (not v) and start is not None:
            segs.append((start, prev))
            start = None
        prev = d
    if start is not None:
        segs.append((start, prev))
    return segs


# ------------------------------- sidebar ----------------------------------- #
st.sidebar.markdown("## 📈 Momentum Lab")
with st.sidebar.expander("Data", expanded=False):
    universe = st.radio("Universe", ["Nifty 50", "Custom list"], horizontal=True)
    tickers_txt = "\n".join(D.NIFTY50)
    if universe == "Custom list":
        tickers_txt = st.text_area("NSE symbols (one per line)", tickers_txt, height=140)
    _today = pd.Timestamp.today().date()
    _min = pd.Timestamp("2000-01-01").date()
    start = str(st.date_input("Start date", pd.Timestamp("2010-01-01").date(), min_value=_min, max_value=_today))
    end = pd.Timestamp(st.date_input("End date", _today, min_value=_min, max_value=_today))
    force_demo = st.checkbox("Use demo data (offline)", value=False)

st.sidebar.markdown("### Strategy")
lookback = st.sidebar.slider("Lookback (months)", 3, 24, 12)
skip = st.sidebar.slider("Skip latest (months)", 0, 3, 1)
hold = st.sidebar.slider("Holding period (months)", 1, 12, 1)
n_q = st.sidebar.select_slider("Quantiles", [3, 5, 10], value=5)
mode = st.sidebar.radio("Portfolio", ["long_short", "long_only"], horizontal=True,
                        format_func=lambda x: "Long–short" if x == "long_short" else "Long-only")
with st.sidebar.expander("Costs & settings"):
    cost_bps = st.number_input("One-way cost (bps)", 0.0, 200.0, 20.0, 5.0)
    rf = st.number_input("Risk-free (annual %)", 0.0, 15.0, 6.5, 0.25) / 100
    min_names = st.number_input("Min. stocks with signal", 5, 100, 15)
rf_eff = rf if mode == "long_only" else 0.0


# -------------------------------- data ------------------------------------- #
@st.cache_data(show_spinner="Pulling NSE prices from Yahoo Finance…", ttl=86400)
def get_yahoo(tickers: tuple, start: str, end: str):
    px = D.load_prices(list(tickers), start, end)
    try:
        b = D.load_prices(["^NSEI"], start, end).iloc[:, 0]
    except Exception:
        b = None
    return px, b


@st.cache_data(show_spinner=False)
def get_demo():
    df = D.synthetic_prices()
    return df.drop(columns="BENCHMARK"), df["BENCHMARK"]


is_demo, fell_back = force_demo, False
if not force_demo:
    try:
        tks = tuple(t.strip().upper() for t in tickers_txt.splitlines() if t.strip())
        px, bench = get_yahoo(tks, start, str((end + pd.Timedelta(days=1)).date()))
        bench_label = "Nifty 50 (^NSEI)"
    except Exception:  # noqa: BLE001
        is_demo, fell_back = True, True
if is_demo:
    px, bench = get_demo()
    bench_label = "Synthetic market"

px = px.loc[:, px.notna().sum() > 260]
if px.shape[1] < max(min_names, 2 * n_q):
    st.error(f"Only {px.shape[1]} usable stocks; need at least {max(min_names, 2 * n_q)}.")
    st.stop()
if bench is None:
    bench, bench_label = D.equal_weight_benchmark(px), "Equal-weight universe"
bench_m = E.to_monthly(bench).pct_change(fill_method=None)

mode_name = "Long–short" if mode == "long_short" else "Long-only"
chips = (f'<span class="chip">{px.shape[1]} stocks</span>'
         f'<span class="chip">data: {px.index[0]:%d %b %Y} → {px.index[-1]:%d %b %Y}</span>'
         f'<span class="chip">{lookback}-{skip} momentum · hold {hold}m</span>'
         f'<span class="chip">{mode_name} · {cost_bps:.0f} bps</span>'
         f'<span class="chip">vs {bench_label}</span>')
st.markdown(f'<div class="hero"><h1>Momentum Lab</h1>'
            f'<p>Stock analysis and a backtesting framework that stress-tests momentum for robustness, risk, '
            f'statistical significance and regime dependence.</p>{chips}</div>', unsafe_allow_html=True)
if (not is_demo) and (pd.Timestamp.today() - px.index[-1]).days > 10:
    st.warning(f"Your data ends on **{px.index[-1]:%d %b %Y}**. Set the End date in the sidebar (Data) to today and reload.")
if fell_back:
    st.warning("Couldn't reach Yahoo Finance, so this is **synthetic demo data**. Check your internet and refresh for real NSE data.")
elif is_demo:
    st.info("Demo mode: synthetic data, for testing the platform only.")

bt = E.run_backtest(px, lookback, skip, hold, n_q, mode, cost_bps, min_names)
if bt.empty or len(bt) < 24:
    st.error("Not enough history for this setup. Lower the lookback or move the start date earlier.")
    st.stop()
net = bt["net"]


@st.cache_data(show_spinner="Running parameter grid…")
def get_grid(px, lookbacks, holds, skip, n_q, mode, cost_bps, min_names, rf):
    return E.robustness_grid(px, lookbacks, holds, skip, n_q, mode, cost_bps, min_names, rf)


# ------------------------------ scorecard ---------------------------------- #
nw = E.nw_tstat(net)
grid, gser = get_grid(px, (3, 6, 9, 12, 15, 18), (1, 3, 6, 12), skip, n_q, mode, cost_bps, min_names, rf_eff)
trials = np.array([r.mean() / r.std(ddof=1) for r in gser.values()])
dsr = E.deflated_sharpe(net, trials)
hi_cost = E.run_backtest(px, lookback, skip, hold, n_q, mode, max(cost_bps * 2, 40), min_names)["net"]
sh_hi = E.perf_stats(hi_cost, rf_eff).get("Sharpe", np.nan)
lab = E.regime_labels(bench)
wb = E.welch_between(net, lab["Market"], "Bull", "Bear")
rob = (grid["Sharpe"] > 0).mean() if len(grid) else np.nan


def card(title, value, detail, tone):
    st.markdown(f'<div class="sc {tone}"><div class="sc-t">{title}</div><div class="sc-v">{value}</div>'
                f'<div class="sc-d">{detail}</div></div>', unsafe_allow_html=True)


st.markdown("#### Verdict at a glance")
c = st.columns(5)
with c[0]:
    card("Significant?", "Yes" if nw["t"] >= 2 else ("Borderline" if nw["t"] >= 1.65 else "No"),
         f"Newey–West t = {nw['t']:.2f}", "good" if nw["t"] >= 2 else ("warn" if nw["t"] >= 1.65 else "bad"))
with c[1]:
    card("Robust?", "Yes" if rob >= .8 else ("Partly" if rob >= .5 else "No"),
         f"{rob:.0%} of {len(grid)} parameter variants profitable",
         "good" if rob >= .8 else ("warn" if rob >= .5 else "bad"))
with c[2]:
    card("Survives costs?", "Yes" if sh_hi > .3 else ("Barely" if sh_hi > 0 else "No"),
         f"Sharpe {sh_hi:.2f} at {max(cost_bps * 2, 40):.0f} bps", "good" if sh_hi > .3 else ("warn" if sh_hi > 0 else "bad"))
with c[3]:
    d_ = dsr.get("dsr", np.nan)
    card("Beats data-mining?", "Yes" if d_ >= .95 else ("Maybe" if d_ >= .8 else "No"),
         f"Deflated Sharpe {d_:.0%}", "good" if d_ >= .95 else ("warn" if d_ >= .8 else "bad"))
with c[4]:
    if wb:
        dep = wb["p"] < .10
        card("Regime-dependent?", "Yes" if dep else "Not clearly",
             f"Bull vs Bear gap {wb['diff']:.2%}/mo (p={wb['p']:.2f})", "warn" if dep else "good")
    else:
        card("Regime-dependent?", "n/a", "Not enough data", "")
st.write("")

tab_stock, tab_bt, tab_rob, tab_sig, tab_reg, tab_method = st.tabs(
    ["🔎 Stock explorer", "🧪 Backtest", "🧱 Robustness", "📐 Significance", "🌦️ Regimes", "📝 Method"])

# ---------------------------- Stock explorer ------------------------------- #
with tab_stock:
    pm = E.to_monthly(px)
    sig = E.momentum_signal(pm, lookback, skip)
    rank_hist = sig.rank(axis=1, pct=True)
    t = st.selectbox("Search a stock", list(px.columns))
    s = px[t].dropna()
    ls = sig[t].dropna()
    cur_sig = ls.iloc[-1] if len(ls) else np.nan
    cur_rank = rank_hist[t].dropna().iloc[-1] if len(ls) else np.nan
    dr = s.pct_change().dropna()
    bucket = ("Winner bucket 🟢" if cur_rank >= 1 - 1 / n_q else "Loser bucket 🔴" if cur_rank <= 1 / n_q else "Middle") \
        if pd.notna(cur_rank) else ""
    kpis([("Last price", f"{s.iloc[-1]:,.2f}", f"{s.index[-1]:%d %b %Y}"),
          (f"{lookback}-{skip} momentum", f"{cur_sig:.1%}" if pd.notna(cur_sig) else "–", "vs. universe below",
           "good" if pd.notna(cur_sig) and cur_sig > 0 else "bad"),
          ("Momentum rank", f"{cur_rank:.0%}" if pd.notna(cur_rank) else "–", bucket),
          ("1y volatility", f"{dr.tail(252).std() * np.sqrt(252):.1%}", "annualised"),
          ("Max drawdown", f"{(s / s.cummax() - 1).min():.1%}", "full history", "bad")])
    st.write("")
    f = go.Figure()
    f.add_scatter(x=s.index, y=s, name="Price", line=dict(width=2))
    f.add_scatter(x=s.index, y=s.rolling(50).mean(), name="50 DMA", line=dict(width=1, dash="dot"))
    f.add_scatter(x=s.index, y=s.rolling(200).mean(), name="200 DMA", line=dict(width=1, dash="dot"))
    show(style(f, 360, f"{t} — price & moving averages"))
    c1, c2 = st.columns(2)
    with c1:
        rh = rank_hist[t].dropna()
        f2 = go.Figure(go.Scatter(x=rh.index, y=rh, fill="tozeroy", line=dict(color=ACCENT)))
        f2.add_hline(y=1 - 1 / n_q, line_dash="dash", line_color=GOOD)
        f2.add_hline(y=1 / n_q, line_dash="dash", line_color=BAD)
        f2.update_yaxes(tickformat=".0%")
        show(style(f2, 300, "Momentum percentile vs universe"))
    with c2:
        f3 = go.Figure(go.Scatter(x=s.index, y=s / s.cummax() - 1, fill="tozeroy", line=dict(color=BAD)))
        f3.update_yaxes(tickformat=".0%")
        show(style(f3, 300, "Drawdown"))
    snap = sig.iloc[-1].dropna().sort_values(ascending=False)
    a, b = st.columns(2)
    a.markdown("**🟢 Current winners**")
    a.dataframe(snap.head(10).map(lambda v: f"{v:.1%}").rename("Momentum"), width="stretch")
    b.markdown("**🔴 Current losers**")
    b.dataframe(snap.tail(10).iloc[::-1].map(lambda v: f"{v:.1%}").rename("Momentum"), width="stretch")

# -------------------------------- Backtest --------------------------------- #
with tab_bt:
    sn, sg = E.perf_stats(net, rf_eff), E.perf_stats(bt["gross"], rf_eff)
    kpis([("CAGR (net)", f"{sn['CAGR']:.1%}", f"gross {sg['CAGR']:.1%}", "good" if sn["CAGR"] > 0 else "bad"),
          ("Sharpe (net)", f"{sn['Sharpe']:.2f}", f"Sortino {sn['Sortino']:.2f}", "good" if sn["Sharpe"] > .5 else "warn"),
          ("Volatility", f"{sn['Ann. vol']:.1%}", "annualised"),
          ("Max drawdown", f"{sn['Max drawdown']:.1%}", f"Calmar {sn['Calmar']:.2f}", "bad"),
          ("Cost drag", f"{bt['cost'].mean() * 12:.1%}/yr", f"turnover {bt['turnover'].mean():.0%}/mo", "warn")])
    st.write("")
    eq = pd.DataFrame({f"{mode_name} (net)": (1 + net).cumprod(), f"{mode_name} (gross)": (1 + bt["gross"]).cumprod(),
                       "Winners leg": (1 + bt["long"]).cumprod(),
                       bench_label: (1 + bench_m.reindex(net.index).fillna(0)).cumprod()}) * 100
    fe = pex.line(eq, log_y=True)
    show(style(fe, 380, "Growth of 100 (log scale)"))
    c1, c2 = st.columns(2)
    with c1:
        qr = E.quantile_returns(px, lookback, skip, n_q, min_names)
        ann = qr.mean() * 12
        cols = pex.colors.sample_colorscale("RdYlGn", [i / max(n_q - 1, 1) for i in range(n_q)])
        fq = go.Figure(go.Bar(x=ann.index, y=ann.values, marker_color=cols))
        fq.update_yaxes(tickformat=".0%")
        show(style(fq, 320, "Annualised return by momentum quantile (Q1 = losers)"))
        mono = pd.Series(ann.values).corr(pd.Series(range(len(ann))), method="spearman")
        st.caption(f"Monotonicity: **{mono:.2f}** (1.0 = returns rise perfectly from losers to winners).")
    with c2:
        dd = E.drawdown(net)
        fd = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=BAD)))
        fd.update_yaxes(tickformat=".0%")
        show(style(fd, 320, "Strategy drawdown (net)"))
    hm = net.to_frame("r")
    hm["Year"], hm["Month"] = hm.index.year, hm.index.month
    pv = hm.pivot(index="Year", columns="Month", values="r")
    fh = go.Figure(go.Heatmap(z=pv.values, x=[pd.Timestamp(2000, i, 1).strftime("%b") for i in pv.columns],
                              y=pv.index.astype(str), zmid=0, colorscale="RdYlGn",
                              text=np.round(pv.values * 100, 1), texttemplate="%{text}", hoverongaps=False))
    fh.update_yaxes(autorange="reversed")
    show(style(fh, max(300, 24 * len(pv)), "Monthly net returns (%)"))
    st.markdown("**Full comparison**")
    st.dataframe(pd.DataFrame({f"{mode_name} (net)": fmt_stats(sn), f"{mode_name} (gross)": fmt_stats(sg),
                               bench_label: fmt_stats(E.perf_stats(bench_m.reindex(net.index), rf))}), width="stretch")
    st.download_button("⬇️ Download monthly returns (CSV)", bt.to_csv().encode(), "momentum_backtest.csv")

# ------------------------------- Robustness -------------------------------- #
with tab_rob:
    st.caption("Plateau = robust. One lone hot cell = probably overfit. Every cell is a full backtest with your settings.")
    lbs = tuple(st.multiselect("Lookbacks (months)", [3, 4, 6, 9, 12, 15, 18, 24], default=[3, 6, 9, 12, 15, 18]))
    hds = tuple(st.multiselect("Holding periods (months)", [1, 2, 3, 6, 9, 12], default=[1, 3, 6, 12]))
    if lbs and hds:
        g, _ = get_grid(px, lbs, hds, skip, n_q, mode, cost_bps, min_names, rf_eff)
        metric = st.radio("Metric", ["Sharpe", "CAGR", "Max drawdown", "t-stat"], horizontal=True)
        pv = g.pivot(index="lookback", columns="hold", values=metric)
        fmt = ".0%" if metric in ("CAGR", "Max drawdown") else ".2f"
        fg = go.Figure(go.Heatmap(z=pv.values, x=[f"{h}m" for h in pv.columns], y=[f"{l}m" for l in pv.index],
                                  colorscale="RdYlGn", zmid=0 if metric != "Max drawdown" else None,
                                  text=pv.values, texttemplate="%{text:" + fmt + "}"))
        fg.update_layout(xaxis_title="Holding period", yaxis_title="Lookback")
        fg.update_yaxes(autorange="reversed")
        show(style(fg, 420, f"{metric}: lookback × holding period"))
        kpis([("Variants profitable", f"{(g['Sharpe'] > 0).mean():.0%}", f"{len(g)} tested"),
              ("Variants with t > 2", f"{(g['t-stat'] > 2).mean():.0%}", "Newey–West"),
              ("Sharpe range", f"{g['Sharpe'].min():.2f} → {g['Sharpe'].max():.2f}", "min → max")])
        st.markdown("**Cost sensitivity**")
        rows = []
        for cb in [0, 10, 20, 40, 80]:
            s_ = E.perf_stats(E.run_backtest(px, lookback, skip, hold, n_q, mode, cb, min_names)["net"], rf_eff)
            rows.append({"One-way cost (bps)": cb, "Sharpe": s_["Sharpe"], "CAGR": s_["CAGR"]})
        st.dataframe(fmt_df(pd.DataFrame(rows).set_index("One-way cost (bps)")), width="stretch")
    else:
        st.info("Pick at least one lookback and one holding period.")

# ------------------------------- Significance ------------------------------ #
with tab_sig:
    st.caption("Does the return survive proper statistics? From basic to strict.")
    ca = E.capm_alpha(net, bench_m, rf_eff)
    kpis([("Mean / month", f"{nw['mean_monthly']:.2%}", "net of costs"),
          ("Newey–West t-stat", f"{nw['t']:.2f}", f"{nw['lags']} lags · need >2 (>3 for new factors)",
           "good" if nw["t"] >= 2 else "warn"),
          ("p-value", f"{nw['p']:.4f}", "H0: mean return = 0"),
          ("Alpha vs benchmark", f"{ca['alpha_ann']:.1%}" if ca else "–", f"t = {ca['alpha_t']:.2f}" if ca else ""),
          ("Market beta", f"{ca['beta']:.2f}" if ca else "–", f"R² {ca['r2']:.2f}" if ca else "")])
    st.write("")
    st.markdown("##### Bootstrap confidence interval for Sharpe")
    block = st.slider("Block length (months)", 1, 12, 6, help="Preserves autocorrelation in returns.")
    bs = E.block_bootstrap_sharpe(net, block=block)
    fb = go.Figure(go.Histogram(x=bs["dist"], nbinsx=60, marker_color=ACCENT))
    fb.add_vline(x=0, line_color=BAD)
    fb.add_vline(x=bs["observed"], line_dash="dash", line_color=GOOD, annotation_text="observed")
    show(style(fb, 300, "Bootstrap distribution of annualised Sharpe"))
    st.write(f"95% CI **[{bs['ci_low']:.2f}, {bs['ci_high']:.2f}]** · resamples with Sharpe ≤ 0: **{bs['p_le_zero']:.1%}**")
    st.markdown("##### Deflated Sharpe (adjusts for data-snooping)")
    if dsr:
        kpis([("Deflated Sharpe", f"{dsr['dsr']:.1%}", "prob. true Sharpe > 0", "good" if dsr["dsr"] >= .95 else "warn"),
              ("Variants tried", f"{dsr['n_trials']}", "lookback × holding grid"),
              ("Sharpe from luck alone", f"{dsr['sr0_annual']:.2f}", f"yours: {dsr['sr_annual']:.2f}")])
    st.markdown("##### Stability over time")
    sp = E.subperiods(net, k=3, rf=rf_eff)[["CAGR", "Ann. vol", "Sharpe", "Max drawdown", "Months"]]
    st.dataframe(fmt_df(sp), width="stretch")
    rs = E.rolling_sharpe(net, 36).dropna()
    fr = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color=ACCENT)))
    fr.add_hline(y=0, line_color=BAD)
    show(style(fr, 280, "Rolling 36-month Sharpe"))

# --------------------------------- Regimes --------------------------------- #
with tab_reg:
    st.caption("Momentum tends to crash in sharp rebounds after bear markets (Daniel & Moskowitz, 2016). "
               "Regimes are measured at the start of each month, so there is no look-ahead.")
    pick = st.radio("Lens", ["Market", "Volatility", "Bear rebound"], horizontal=True,
                    format_func=lambda x: {"Market": "Bull vs Bear", "Volatility": "High vs Low vol",
                                           "Bear rebound": "Bear-market rebounds"}[x])
    st.dataframe(fmt_df(E.regime_table(net, lab[pick], rf_eff)), width="stretch")
    pair = {"Market": ("Bull", "Bear"), "Volatility": ("Low vol", "High vol")}.get(pick)
    if pair:
        w = E.welch_between(net, lab[pick], *pair)
        if w:
            st.write(f"{pair[0]} − {pair[1]} mean monthly return: **{w['diff']:.2%}** (Welch t = {w['t']:.2f}, p = {w['p']:.3f}). "
                     "A high p-value means no strong evidence of regime dependence.")
    cum = (1 + net).cumprod() * 100
    fr_ = go.Figure(go.Scatter(x=cum.index, y=cum, name="Strategy (net)", line=dict(color=ACCENT, width=2)))
    for a_, b_ in segments((lab["Market"] == "Bear").reindex(cum.index).fillna(False)):
        fr_.add_vrect(x0=a_, x1=b_, fillcolor=BAD, opacity=0.12, line_width=0)
    fr_.update_yaxes(type="log")
    show(style(fr_, 360, "Equity curve · bear-market periods shaded"))
    worst = net.nsmallest(10).to_frame("Return").join(lab[["Market", "Volatility", "Bear rebound"]]).fillna("–")
    worst["Return"] = worst["Return"].map(lambda v: f"{v:.1%}")
    worst["Benchmark"] = bench_m.reindex(worst.index).map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
    worst.index = worst.index.strftime("%b %Y")
    st.markdown("**10 worst months: what was the market doing?**")
    st.dataframe(worst, width="stretch")

# ---------------------------------- Method --------------------------------- #
with tab_method:
    st.markdown("""
### Methodology
- **Signal:** return from `t − lookback` to `t − skip` (default 12-1, Jegadeesh–Titman). Skipping the latest month avoids short-term reversal.
- **Portfolios:** stocks ranked each month-end. Top quantile = winners, bottom = losers, equal-weighted. Long–short = winners − losers.
- **Timing:** formed at month-end *t*, earned in *t+1* (no look-ahead, verified by perturbing future prices).
- **Holding > 1 month:** overlapping cohorts; each month 1/*h* of the book is re-formed.
- **Costs:** one-way bps × turnover of gross weights.
- **Significance:** Newey–West t-stat, CAPM alpha (HAC), block-bootstrap Sharpe CI, Deflated Sharpe (Bailey & López de Prado, 2014).
- **Regimes:** lagged trailing-12m benchmark return, trailing 6m vol vs expanding median, bear-market rebound months.

### Limitations
1. **Survivorship bias.** A static list of today's index members flatters results.
2. **Data.** Yahoo adjusted closes can contain errors; no delisting returns.
3. **Liquidity.** No market-impact model.
4. **Shorting.** Single-stock shorting in India is limited, so treat long–short as a factor measurement and long-only as the realistic version.
5. **Turnover** ignores intra-month weight drift.
6. **Multiple testing.** Every parameter you tweak is another trial. Trust the Deflated Sharpe, not the best cell.
""")