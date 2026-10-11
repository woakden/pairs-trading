# Pairs trading: Eli Lilly vs Novo Nordisk

A Python backtest of a pairs trading strategy on Eli Lilly (LLY) and Novo Nordisk (NVO).

**Pairs trading** looks for two stocks whose prices usually move together. When the gap between them stretches further than usual, it buys the cheaper one and sells the dearer one, betting that the gap will close. Because it is long one stock and short the other, it is mostly protected from the market as a whole moving up or down.

## Why this pair

Lilly and Novo Nordisk lead the market for GLP-1 drugs for diabetes and obesity (Lilly's Mounjaro and Zepbound, Novo's Ozempic and Wegovy). They respond to much of the same news: demand for weight-loss drugs, pricing pressure, regulation and supply.

They are also direct rivals, so when one wins market share the other can lose it, and that can break the link between their prices. The project tests whether the relationship is stable enough to trade, rather than assuming it.

Novo Nordisk's New York listing (NVO, in US dollars) is used so that both stocks are in the same currency.

## Results (2 Jan 2018 to 9 Oct 2026)

**The pair did not hold together, and the strategy lost money.**

| | |
|---|---|
| Correlation of daily returns | 0.40 |
| Engle-Granger p-value, whole period | 0.99 (not cointegrated) |
| Rolling one-year windows cointegrated | 10% of 94 |
| Half-life of the spread | none (the spread did not revert) |
| Strategy total return, after costs | -55.5% (-10.2% a year) |
| Sharpe ratio | -0.57 |
| Maximum drawdown | -58.8% |
| Trades | 21, of which 40% of the closed ones made money |

The two stocks moved together day to day, but not over the long run. Lilly rose from $75 to $1,179 over the period, while Novo Nordisk peaked at $136 in June 2024 and then fell to $39, as Lilly's Mounjaro and Zepbound took market share from Ozempic and Wegovy. So the gap between them kept widening instead of closing, and a strategy that bets on it closing kept losing: it lost money in 6 of the 7 full years it traded (2019 to 2025).

The lesson is that a shared industry is not enough. A pairs trade needs a stable relationship, and direct rivals in a market that one of them is winning are a poor candidate. The rolling cointegration chart shows the test would have warned against trading this pair for most of the period, so a natural next step is to trade only while the pair passes that test.

Charts: [prices](results/1_prices.png), [z-score and trades](results/2_zscore_and_trades.png), [strategy value](results/3_equity_curve.png), [rolling cointegration](results/4_rolling_cointegration.png).

## Method

1. **Prices:** daily closing prices adjusted for splits and dividends, from Yahoo Finance, since 2018.
2. **Cointegration test:** an Engle-Granger test checks whether the two log prices share a long-run relationship (p-value below 0.05). It is run over the whole period and on rolling one-year windows, to show when the relationship held and when it broke.
3. **Spread:** `spread = log(LLY) - beta * log(NVO)`, where the hedge ratio `beta` comes from a rolling one-year regression, so it adapts as the relationship changes.
4. **Z-score:** how many standard deviations the spread is from its 60-day average.
5. **Trading rules:**

| Rule | Default |
|---|---|
| Short the spread (sell LLY, buy NVO) | z-score above +2.0 |
| Buy the spread (buy LLY, sell NVO) | z-score below -2.0 |
| Close the trade | z-score back within ±0.5 |
| Stop-loss | z-score beyond ±4.0, then wait until it is back inside ±2.0 |
| Trading costs | 5 basis points per trade |

6. **Performance:** total and annual return, volatility, Sharpe ratio, maximum drawdown, win rate and returns by year, after costs.

Each day's position is decided using only prices up to that day, and it starts earning from the next day, so the backtest never uses future information. A test checks this.

## Running it

Needs Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run.py                               # download prices and run with the defaults
python3 run.py --entry 1.5 --stop 3.5        # try different rules
python3 run.py --csv data/LLY_NVO_<date>.csv # rerun on saved prices without downloading
python3 run.py --y KO --x PEP                # any other pair
python3 -m unittest discover tests           # tests (use simulated prices, no download)
```

All settings are in `config.py` and can be changed from the command line (`python3 run.py --help`).

## Output

Saved to `results/`:

- `summary.txt`: the relationship statistics and the strategy's performance
- `trades.csv`: every trade with its dates, direction, length and return
- `1_prices.png`: both stocks rebased to 100
- `2_zscore_and_trades.png`: the z-score with the entry, exit and stop-loss levels, and when the strategy was in a trade
- `3_equity_curve.png`: the value of the strategy over time
- `4_rolling_cointegration.png`: when the pair was and wasn't cointegrated

## Limitations

- It assumes trades happen at the closing price used to make the decision (`--lag 2` waits a day).
- The hedge is rebalanced every day without charging costs for it, and the cost of borrowing stock to short it is ignored.
- One pair, tested over one period. Changing the rules after seeing the results would overfit them to the past.
- Cointegration found in the past doesn't guarantee it will continue, especially between rivals.
