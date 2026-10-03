# Momentum Lab 📈
A quant research platform for stock analysis + rigorous momentum-factor backtesting (NSE focus).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Pick **Yahoo Finance (NSE)** for real data, **Upload CSV** for your own (point-in-time) universe, or **Demo** to test offline.

## What's inside
| Tab | What it answers |
|---|---|
| Stock explorer | Price/MAs, momentum score + percentile vs universe, drawdown, current winners/losers |
| Backtest | Net vs gross, costs, quantile monotonicity, drawdowns, monthly heatmap |
| Robustness | Lookback × holding-period heatmaps (Sharpe/CAGR/DD/t-stat), cost sensitivity |
| Significance | Newey-West t-stat, CAPM alpha, bootstrap Sharpe CI, Deflated Sharpe, rolling Sharpe, subperiods |
| Regimes | Bull/bear, high/low vol, bear-market rebounds (momentum-crash setup), worst months |
| Method & caveats | Exactly what the code does, and its limits |

## Files
- `engine.py` – signal, backtest, stats (pure pandas/numpy, importable in notebooks)
- `data.py` – Yahoo loader, CSV parser, synthetic demo data
- `app.py` – Streamlit UI

## CSV format
`Date` in first column, one daily price column per ticker. Optional `BENCHMARK` column (else equal-weight universe is used).

## Honest caveats
Static current-constituent universe = survivorship bias. Use a point-in-time universe before quoting numbers anywhere serious.
