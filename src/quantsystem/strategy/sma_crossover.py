"""SMA crossover strategy."""

import pandas as pd
from quantsystem.strategy.base import BaseStrategy
from quantsystem.indicators.technical import TechnicalIndicators


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy.

    Generates a buy signal (1) when the fast SMA crosses above the slow SMA,
    and a sell signal (-1) when it crosses below.  A hold signal (0) is
    emitted otherwise.

    Args:
        fast_window: Period for the fast (short) SMA.
        slow_window: Period for the slow (long) SMA.
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 50):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        super().__init__(name=f"SMA({fast_window},{slow_window})")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate crossover signals.

        Args:
            data: OHLCV DataFrame with a 'Close' column.

        Returns:
            Signal series aligned with *data*.
        """
        close = data["Close"]
        fast = TechnicalIndicators.sma(close, self.fast_window)
        slow = TechnicalIndicators.sma(close, self.slow_window)

        position = (fast > slow).astype(int).diff()
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[position == 1] = 1
        signals[position == -1] = -1
        return signals
