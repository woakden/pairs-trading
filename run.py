"""Pairs trading backtest. By default: Eli Lilly (LLY) against Novo Nordisk (NVO).

Usage:
    python3 run.py                      # download prices and run with the defaults in config.py
    python3 run.py --start 2020-01-01 --entry 1.5
    python3 run.py --csv data/prices.csv   # rerun on saved prices without downloading

Steps:
  1. Download daily prices (adjusted for splits and dividends) and save them to data/.
  2. Test whether the two stocks are cointegrated (move together long term), over the whole
     period and on rolling one-year windows.
  3. Build the spread log(Y) - beta * log(X) with a rolling hedge ratio, and its z-score.
  4. Trade the spread when it stretches: short it above +entry, buy it below -entry, close it
     near the mean, and stop out if it keeps widening. Charge trading costs.
  5. Print the results and save charts, a trade list and a summary to results/.
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np

import config
import data
import pairs
from backtest import list_trades, performance, returns_by_year, run_backtest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--y", default=config.TICKER_Y, help="first stock (regressed on the second)")
    p.add_argument("--x", default=config.TICKER_X, help="second stock")
    p.add_argument("--start", default=config.START)
    p.add_argument("--end", default=None, help="last date (default: today)")
    p.add_argument("--csv", help="use prices saved in this CSV instead of downloading")
    p.add_argument("--hedge-window", type=int, default=config.HEDGE_WINDOW)
    p.add_argument("--z-window", type=int, default=config.Z_WINDOW)
    p.add_argument("--entry", type=float, default=config.ENTRY_Z)
    p.add_argument("--exit", type=float, default=config.EXIT_Z)
    p.add_argument("--stop", type=float, default=config.STOP_Z)
    p.add_argument("--cost-bps", type=float, default=config.COST_BPS)
    p.add_argument("--lag", type=int, default=1, help="days between signal and position (default 1)")
    p.add_argument("--out", default="results", help="folder for charts and tables")
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args(argv)


def analyse(prices, args):
    """Run every step on a price table with columns args.y and args.x. Returns a dict of results."""
    log_y, log_x = np.log(prices[args.y]), np.log(prices[args.x])
    t_stat, p_value = pairs.engle_granger(log_y, log_x)
    intercept, slope = pairs.static_hedge_ratio(log_y, log_x)
    frame = pairs.spread_and_zscore(log_y, log_x, args.hedge_window, args.z_window)
    result = run_backtest(prices, frame, args.y, args.x, args.entry, args.exit, args.stop,
                          args.cost_bps, args.lag)
    traded = result.iloc[args.hedge_window + args.z_window:]   # skip the warm-up with no signal
    trades = list_trades(result, args.lag)
    return {
        "frame": frame,
        "result": result,
        "traded": traded,
        "trades": trades,
        "coint_t": t_stat,
        "coint_p": p_value,
        "static_beta": slope,
        "half_life": pairs.half_life(log_y - slope * log_x - intercept),
        "return_correlation": float(prices[args.y].pct_change().corr(prices[args.x].pct_change())),
        "rolling_coint": pairs.rolling_cointegration(log_y, log_x, config.COINT_WINDOW, config.COINT_STEP),
        "performance": performance(traded["strategy_return"], config.TRADING_DAYS),
        "by_year": returns_by_year(traded["strategy_return"]),
    }


def summary_text(prices, out, args):
    perf, trades, roll = out["performance"], out["trades"], out["rolling_coint"]
    closed = trades[trades["closed"] != "still open"]
    in_market = float((out["traded"]["position"] != 0).mean())
    share_coint = float((roll < 0.05).mean()) if len(roll) else float("nan")
    lines = [
        f"Pair: {args.y} vs {args.x}, {prices.index[0].date()} to {prices.index[-1].date()} "
        f"({len(prices)} trading days)",
        "",
        "Relationship",
        f"  Correlation of daily returns:          {out['return_correlation']:.2f}",
        f"  Engle-Granger cointegration p-value:   {out['coint_p']:.3f} "
        f"({'cointegrated' if out['coint_p'] < 0.05 else 'not cointegrated'} at 5% over the whole period)",
        f"  Rolling 1-year windows cointegrated:   {share_coint:.0%} of {len(roll)} windows",
        f"  Hedge ratio (whole period):            {out['static_beta']:.2f}",
        f"  Half-life of the spread:               {out['half_life']:.0f} trading days",
        "",
        f"Strategy (entry {args.entry}, exit {args.exit}, stop {args.stop} standard deviations; "
        f"costs {args.cost_bps:g} bps per trade)",
        f"  Total return:          {perf['total_return']:+.1%}",
        f"  Annual return:         {perf['annual_return']:+.1%}",
        f"  Annual volatility:     {perf['annual_volatility']:.1%}",
        f"  Sharpe ratio:          {perf['sharpe']:.2f}",
        f"  Maximum drawdown:      {perf['max_drawdown']:.1%}",
        f"  Trades:                {len(trades)} ({len(closed)} closed, win rate "
        f"{(closed['return'] > 0).mean():.0%})" if len(closed) else f"  Trades:                {len(trades)}",
        f"  Time in the market:    {in_market:.0%}",
        "",
        "Return by year",
    ]
    lines += [f"  {year}: {r:+.1%}" for year, r in out["by_year"].items()]
    return "\n".join(lines)


def save_charts(prices, out, args, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    frame, result, roll = out["frame"], out["result"], out["rolling_coint"]

    fig, ax = plt.subplots(figsize=(10, 4))
    (prices / prices.iloc[0] * 100).plot(ax=ax)
    ax.set_title(f"{args.y} and {args.x}, rebased to 100")
    ax.set_ylabel("Price (start = 100)")
    fig.tight_layout(); fig.savefig(folder / "1_prices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    frame["z"].plot(ax=ax, color="black", linewidth=0.8, label="z-score of the spread")
    for level, style in [(args.entry, "--"), (args.exit, ":"), (args.stop, "-.")]:
        ax.axhline(level, color="grey", linestyle=style, linewidth=0.8)
        ax.axhline(-level, color="grey", linestyle=style, linewidth=0.8)
    ax.fill_between(result.index, -args.stop, args.stop, where=result["position"] > 0,
                    color="tab:green", alpha=0.15, label="long the spread")
    ax.fill_between(result.index, -args.stop, args.stop, where=result["position"] < 0,
                    color="tab:red", alpha=0.15, label="short the spread")
    ax.set_title("Spread z-score with entry (--), exit (:) and stop-loss (-.) levels")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(folder / "2_zscore_and_trades.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ((1 + out["traded"]["strategy_return"]).cumprod() * 100).plot(ax=ax, color="tab:blue")
    ax.set_title("Strategy value (start = 100, after trading costs)")
    fig.tight_layout(); fig.savefig(folder / "3_equity_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    roll.plot(ax=ax, marker=".", color="tab:purple")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="5% significance")
    ax.set_ylim(0, 1)
    ax.set_title("Cointegration p-value on rolling 1-year windows (below the red line = cointegrated)")
    ax.legend()
    fig.tight_layout(); fig.savefig(folder / "4_rolling_cointegration.png", dpi=150); plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.csv:
        prices = data.load_csv(args.csv)[[args.y, args.x]]
    else:
        prices = data.download([args.y, args.x], args.start, args.end)
        saved = data.save_csv(prices, Path("data") / f"{args.y}_{args.x}_{date.today().isoformat()}.csv")
        print(f"Saved prices to {saved}")

    out = analyse(prices, args)
    text = summary_text(prices, out, args)
    print(text)

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text(text + "\n")
    out["trades"].to_csv(folder / "trades.csv", index=False)
    if not args.no_charts:
        save_charts(prices, out, args, folder)
    print(f"\nCharts, trades.csv and summary.txt saved in {folder}/")


if __name__ == "__main__":
    main()
