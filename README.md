# Pairs trading: Shell vs BP

A Python backtest of a pairs trading strategy on Shell (SHEL) and BP (BP).

**Pairs trading** looks for two stocks whose prices usually move together. When the gap between them stretches further than usual, it buys the cheaper one and sells the dearer one, betting that the gap will close. Because it is long one stock and short the other, it is mostly protected from the market as a whole moving up or down.

## Why this pair

Shell and BP are the two London-headquartered oil majors. They produce, refine and sell the same products in many of the same markets, so their profits respond to the same things: oil and gas prices, refining margins, windfall taxes and the pace of the energy transition. That gives a reason for their share prices to move together over the long run, which is what a pairs trade relies on.

They can still drift apart when one company's strategy or execution differs from the other's, for example BP's shift towards renewables from 2020 and its strategy reset back towards oil and gas in February 2025. The project tests whether the relationship is stable enough to trade, rather than assuming it.

Both stocks use their New York listings (in US dollars) so the two legs are in the same currency.

## Results (2 Jan 2018 to 16 Oct 2026)

**The two stocks moved closely together and the strategy made money, but the relationship was not stable enough to be statistically reliable.**

| | |
|---|---|
| Correlation of daily returns | 0.88 |
| Engle-Granger p-value, whole period | 0.16 (not cointegrated at 5%) |
| Rolling one-year windows cointegrated | 2% of 94 |
| Half-life of the spread | 97 trading days |
| Strategy total return, after costs | +28.9% (+3.4% a year) |
| Annual volatility | 7.8% |
| Sharpe ratio | 0.47 |
| Maximum drawdown | -13.7% |
| Trades | 16, of which 60% of the closed ones made money |

Day to day, Shell and BP moved together very closely (a correlation of 0.88). The gap between them also tended to close, but slowly: the half-life of 97 trading days means a typical gap took around four to five months to halve. Over the whole period, though, the cointegration test does not pass at the 5% level, because Shell steadily pulled ahead from 2023 onwards: Shell's price divided by BP's rose from 1.71 at the end of 2022 to 2.18 at the end of 2024.

The strategy made +28.9% after costs. Its best year was 2024 (+12.9%), and it lost less than 1% in each of its losing years (2020, 2021 and 2026 so far). That is a modest return with low volatility, but it comes from 16 trades over almost nine years, which is too few to show the strategy has a real edge rather than luck.

The lesson is that a strong correlation is not the same as a stable long-run relationship. A natural next step is to trade only while the rolling cointegration test passes, and to test the same rules on other pairs from the same industry before trusting them.

Charts: [prices](results/1_prices.png), [z-score and trades](results/2_zscore_and_trades.png), [strategy value](results/3_equity_curve.png), [rolling cointegration](results/4_rolling_cointegration.png).

## Method

1. **Prices:** daily closing prices adjusted for splits and dividends, from Yahoo Finance, since 2018.
2. **Cointegration test:** an Engle-Granger test checks whether the two log prices share a long-run relationship (p-value below 0.05). It is run over the whole period and on rolling one-year windows, to show when the relationship held and when it broke.
3. **Spread:** `spread = log(SHEL) - beta * log(BP)`, where the hedge ratio `beta` comes from a rolling one-year regression, so it adapts as the relationship changes.
4. **Z-score:** how many standard deviations the spread is from its 60-day average.
5. **Trading rules:**

| Rule | Default |
|---|---|
| Short the spread (sell SHEL, buy BP) | z-score above +2.0 |
| Buy the spread (buy SHEL, sell BP) | z-score below -2.0 |
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
python3 run.py --csv data/SHEL_BP_<date>.csv # rerun on saved prices without downloading
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
- Cointegration found in the past doesn't guarantee it will continue, and with only 16 trades the results could be down to luck.
