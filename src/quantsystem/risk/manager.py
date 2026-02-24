"""Risk management module."""

import pandas as pd
import numpy as np


class RiskManager:
    """Portfolio risk management utilities.

    Args:
        max_position_pct: Maximum fraction of capital to allocate to one position (default 0.1).
        max_drawdown_pct: Maximum tolerated drawdown before halting trading (default 0.2).
    """

    def __init__(
        self,
        max_position_pct: float = 0.1,
        max_drawdown_pct: float = 0.2,
    ):
        if not (0 < max_position_pct <= 1.0):
            raise ValueError("max_position_pct must be in (0, 1]")
        if not (0 < max_drawdown_pct <= 1.0):
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct

    def position_size(self, capital: float, price: float) -> int:
        """Calculate the maximum allowed position size (number of shares).

        Args:
            capital: Available capital.
            price: Current asset price.

        Returns:
            Number of shares to buy (floored to whole shares).
        """
        if price <= 0:
            raise ValueError("price must be positive")
        max_value = capital * self.max_position_pct
        return int(max_value / price)

    def is_drawdown_exceeded(self, equity: pd.Series) -> bool:
        """Return True if the current drawdown exceeds the configured maximum.

        Args:
            equity: Equity curve series.

        Returns:
            True if drawdown threshold breached, False otherwise.
        """
        if equity.empty:
            return False
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return bool(drawdown.iloc[-1] < -self.max_drawdown_pct)

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value-at-Risk.

        Args:
            returns: Daily return series.
            confidence: Confidence level (default 0.95 → 95% VaR).

        Returns:
            VaR as a positive number (loss expressed as a fraction).
        """
        if returns.empty:
            return 0.0
        return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

    @staticmethod
    def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
        """Kelly criterion fraction to wager.

        Args:
            win_rate: Probability of a winning trade (0–1).
            win_loss_ratio: Average win size divided by average loss size.

        Returns:
            Optimal bet fraction (clamped to [0, 1]).
        """
        if win_loss_ratio <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0.0, min(1.0, kelly))
