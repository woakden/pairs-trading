"""Statistics for a pair of stocks: hedge ratio, spread, z-score, cointegration and half-life."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def rolling_hedge_ratio(log_y, log_x, window):
    """Slope of a rolling least-squares regression of log(Y) on log(X).

    Each value uses only the current day and the days before it, so it never looks ahead.
    """
    return log_y.rolling(window).cov(log_x) / log_x.rolling(window).var()


def spread_and_zscore(log_y, log_x, hedge_window, z_window):
    """Spread = log(Y) - beta * log(X), and how many standard deviations it is from its recent mean."""
    beta = rolling_hedge_ratio(log_y, log_x, hedge_window)
    spread = log_y - beta * log_x
    mean = spread.rolling(z_window).mean()
    std = spread.rolling(z_window).std()
    z = (spread - mean) / std
    return pd.DataFrame({"beta": beta, "spread": spread, "z": z})


def engle_granger(log_y, log_x):
    """Engle-Granger cointegration test. A p-value below 0.05 suggests the pair moves together long term."""
    t_stat, p_value, _ = coint(log_y, log_x)
    return float(t_stat), float(p_value)


def static_hedge_ratio(log_y, log_x):
    """Intercept and slope of one regression over the whole period (used for the half-life)."""
    slope, intercept = np.polyfit(log_x.to_numpy(), log_y.to_numpy(), 1)
    return float(intercept), float(slope)


def half_life(spread):
    """Days for a gap in the spread to halve, from an AR(1) fit: change = a + b * yesterday's level.

    Returns infinity when the spread does not pull back towards its mean (b >= 0).
    """
    s = pd.Series(spread).dropna()
    lagged = s.shift(1).iloc[1:]
    change = s.diff().iloc[1:]
    b, _ = np.polyfit(lagged.to_numpy(), change.to_numpy(), 1)
    return float("inf") if b >= 0 else float(-np.log(2) / b)


def rolling_cointegration(log_y, log_x, window, step):
    """Engle-Granger p-value on rolling windows, to show when the relationship held and when it broke."""
    dates, p_values = [], []
    for end in range(window, len(log_y) + 1, step):
        _, p = engle_granger(log_y.iloc[end - window:end], log_x.iloc[end - window:end])
        dates.append(log_y.index[end - 1])
        p_values.append(p)
    return pd.Series(p_values, index=pd.DatetimeIndex(dates), name="coint_p_value")
