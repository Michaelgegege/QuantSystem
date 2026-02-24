"""Strategy base class."""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement :meth:`generate_signals`.
    """

    def __init__(self, name: str = "Strategy"):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data.

        Args:
            data: OHLCV DataFrame with at least a 'Close' column.

        Returns:
            Series of signals: 1 (buy), -1 (sell), 0 (hold).
            Index must match *data*.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
