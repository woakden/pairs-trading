"""Default settings. Every one can be overridden from the command line (see run.py --help)."""

# The pair: Y is regressed on X, so the spread is log(Y) - beta * log(X).
# Both use their New York listings (in US dollars) so the two legs are in the same currency.
TICKER_Y = "SHEL"  # Shell (NYSE)
TICKER_X = "BP"    # BP (NYSE ADR)
START = "2018-01-01"

# Hedge ratio: rolling least-squares regression of log prices over this many trading days (~1 year).
HEDGE_WINDOW = 252
# The z-score compares today's spread with its mean and standard deviation over this many days.
Z_WINDOW = 60

# Trading rules, in standard deviations of the spread.
ENTRY_Z = 2.0   # open a trade when the spread is this far from its mean
EXIT_Z = 0.5    # close it when the spread has come back within this distance
STOP_Z = 4.0    # stop-loss: close it if the spread keeps widening to this distance

# Trading cost per trade, in basis points of the money traded (commission plus half the bid-ask spread).
COST_BPS = 5.0

# Rolling cointegration check: window length and how often (in trading days) to re-test.
COINT_WINDOW = 252
COINT_STEP = 21

TRADING_DAYS = 252
