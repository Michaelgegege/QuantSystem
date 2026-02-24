"""Data fetching and management module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class DataFetcher:
    """Fetches and manages market data for quantitative analysis."""

    def __init__(self):
        self._cache: dict = {}

    def fetch(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a given symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL').
            start: Start date string 'YYYY-MM-DD'. Overrides period if provided.
            end: End date string 'YYYY-MM-DD'. Defaults to today.
            period: Period string used when start/end not provided (e.g. '1y').

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required: pip install yfinance") from exc

        ticker = yf.Ticker(symbol)
        if start:
            df = ticker.history(start=start, end=end)
        else:
            df = ticker.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        self._cache[symbol] = df
        return df

    def from_dataframe(self, df: pd.DataFrame, symbol: str = "CUSTOM") -> pd.DataFrame:
        """Load data from an existing DataFrame.

        Args:
            df: DataFrame with at least a 'Close' column.
            symbol: Logical name for the data.

        Returns:
            Validated and stored DataFrame.
        """
        required = {"Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        self._cache[symbol] = df
        return df

    def get_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return cached data for a symbol, or None if not available."""
        return self._cache.get(symbol)
