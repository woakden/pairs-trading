"""Turn z-scores into trades and measure how the strategy would have performed."""
import numpy as np
import pandas as pd


def positions_from_z(z, entry, exit_, stop):
    """+1 = long the spread (buy Y, sell X), -1 = short the spread, 0 = flat.

    Opens when |z| > entry, closes when |z| falls back below exit_, and stops out when |z| > stop.
    After a stop-loss it waits until |z| is back inside the entry band before trading again,
    so it doesn't jump straight back into a spread that is still blowing out.
    """
    out = np.zeros(len(z))
    pos, cooling_off = 0, False
    for i, value in enumerate(np.asarray(z, dtype=float)):
        if np.isnan(value):
            pos = 0
        elif pos == 0:
            if cooling_off and abs(value) < entry:
                cooling_off = False
            if not cooling_off:
                if value > entry:
                    pos = -1
                elif value < -entry:
                    pos = 1
        elif abs(value) > stop:
            pos, cooling_off = 0, True
        elif (pos == 1 and value > -exit_) or (pos == -1 and value < exit_):
            pos = 0
        out[i] = pos
    return pd.Series(out, index=getattr(z, "index", None), name="signal")


def run_backtest(prices, frame, y, x, entry, exit_, stop, cost_bps, lag=1):
    """Daily returns of the strategy.

    The signal is worked out from each day's closing prices. The position starts earning `lag` days
    later (lag=1: you trade at that same close; lag=2: you trade a day later), so a day's return is
    never used to decide that day's position. Long the spread = $1 of Y bought and $beta of X sold;
    returns are per dollar of gross exposure (1 + |beta|). Each change in position pays cost_bps on
    the money traded.
    """
    signal = positions_from_z(frame["z"], entry, exit_, stop)
    held = signal.shift(lag).fillna(0.0)
    beta = frame["beta"].shift(lag)
    r_y = prices[y].pct_change()
    r_x = prices[x].pct_change()
    gross = (r_y - beta * r_x) / (1 + beta.abs())
    costs = held.diff().abs().fillna(held.abs()) * cost_bps / 10_000
    strategy = (held * gross).fillna(0.0) - costs
    return pd.DataFrame({"signal": signal, "position": held, "strategy_return": strategy})


def list_trades(result, lag=1):
    """One row per trade: the days the signal opened and closed it, direction, days held and return."""
    trades, start, side = [], None, 0
    position = result["position"].to_numpy()
    returns = result["strategy_return"].to_numpy()
    dates = result.index
    for i in range(len(position)):
        if side == 0 and position[i] != 0:
            start, side = i, position[i]
        elif side != 0 and position[i] != side:
            # Day i still pays the cost of closing, so it belongs to this trade.
            trades.append(_trade(dates, returns, start, i, side, lag, closed=True))
            start, side = (i, position[i]) if position[i] != 0 else (None, 0)
    if side != 0:
        trades.append(_trade(dates, returns, start, len(position) - 1, side, lag, closed=False))
    return pd.DataFrame(trades, columns=["opened", "closed", "direction", "days", "return"])


def _trade(dates, returns, start, end, side, lag, closed):
    trade_return = float(np.prod(1 + returns[start:end + 1]) - 1)
    days = end - start if closed else end - start + 1
    return [dates[start - lag].date(), dates[end - lag].date() if closed else "still open",
            "long spread" if side > 0 else "short spread", days, trade_return]


def performance(returns, trading_days=252):
    """Headline statistics for a series of daily returns."""
    r = returns.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / trading_days
    total = float(equity.iloc[-1] - 1) if len(r) else 0.0
    vol = float(r.std() * np.sqrt(trading_days))
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total,
        "annual_return": float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0,
        "annual_volatility": vol,
        "sharpe": float(r.mean() / r.std() * np.sqrt(trading_days)) if r.std() > 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(r) else 0.0,
    }


def returns_by_year(returns):
    return (1 + returns).groupby(returns.index.year).prod() - 1
