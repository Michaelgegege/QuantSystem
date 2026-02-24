"""Backtesting engine."""

import pandas as pd
import numpy as np
from typing import Optional

from quantsystem.strategy.base import BaseStrategy


class BacktestResult:
    """Container for backtest results and performance metrics."""

    def __init__(self, trades: pd.DataFrame, equity: pd.Series, initial_capital: float):
        self.trades = trades
        self.equity = equity
        self.initial_capital = initial_capital

    @property
    def total_return(self) -> float:
        """Total return as a fraction (e.g. 0.15 for +15%)."""
        if self.equity.empty:
            return 0.0
        return (self.equity.iloc[-1] - self.initial_capital) / self.initial_capital

    @property
    def annualized_return(self) -> float:
        """Annualized return assuming 252 trading days per year."""
        n = len(self.equity)
        if n < 2:
            return 0.0
        years = n / 252
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (risk-free rate = 0)."""
        daily_returns = self.equity.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction."""
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max
        return float(drawdown.min())

    @property
    def num_trades(self) -> int:
        """Total number of completed (round-trip) trades."""
        return len(self.trades)

    def summary(self) -> dict:
        """Return a dictionary of key performance metrics."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(self.equity.iloc[-1]) if not self.equity.empty else self.initial_capital,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BacktestResult("
            f"total_return={s['total_return']:.2%}, "
            f"sharpe={s['sharpe_ratio']:.2f}, "
            f"max_drawdown={s['max_drawdown']:.2%}, "
            f"num_trades={s['num_trades']})"
        )


class BacktestEngine:
    """Event-driven backtesting engine.

    Supports a simple long-only strategy driven by signals (+1 / -1 / 0).

    Args:
        initial_capital: Starting cash in account currency.
        commission: Per-trade commission as a fraction of trade value (e.g. 0.001).
    """

    def __init__(self, initial_capital: float = 100_000.0, commission: float = 0.001):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not (0.0 <= commission < 1.0):
            raise ValueError("commission must be in [0, 1)")
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """Run a backtest.

        Args:
            data: OHLCV DataFrame with a 'Close' column.
            strategy: A :class:`~quantsystem.strategy.base.BaseStrategy` instance.

        Returns:
            :class:`BacktestResult` with equity curve and trade log.
        """
        signals = strategy.generate_signals(data)
        close = data["Close"]

        cash = self.initial_capital
        position = 0  # number of shares held
        entry_price = 0.0
        equity = pd.Series(index=data.index, dtype=float)
        trades = []

        for i, (date, price) in enumerate(close.items()):
            signal = signals.iloc[i]

            if signal == 1 and position == 0:
                # Buy: use all available cash
                shares = int(cash / (price * (1 + self.commission)))
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    cash -= cost
                    position = shares
                    entry_price = price

            elif signal == -1 and position > 0:
                # Sell: close entire position
                proceeds = position * price * (1 - self.commission)
                pnl = proceeds - position * entry_price * (1 + self.commission)
                trades.append(
                    {
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "shares": position,
                        "pnl": pnl,
                    }
                )
                cash += proceeds
                position = 0
                entry_price = 0.0

            equity.iloc[i] = cash + position * price

        trades_df = pd.DataFrame(
            trades,
            columns=["exit_date", "entry_price", "exit_price", "shares", "pnl"],
        )
        return BacktestResult(trades_df, equity, self.initial_capital)
