"""Technical indicators module."""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Collection of common technical indicators."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average.

        Args:
            series: Price series (typically 'Close').
            window: Rolling window size.

        Returns:
            SMA series.
        """
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average.

        Args:
            series: Price series.
            window: Span (period) for the EMA.

        Returns:
            EMA series.
        """
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.

        Args:
            series: Price series.
            window: RSI period (default 14).

        Returns:
            RSI series (0–100).
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence.

        Args:
            series: Price series.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).
            signal: Signal line EMA period (default 9).

        Returns:
            DataFrame with columns: macd, signal, histogram.
        """
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        )

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands.

        Args:
            series: Price series.
            window: Rolling window size (default 20).
            num_std: Number of standard deviations for the bands (default 2).

        Returns:
            DataFrame with columns: upper, middle, lower.
        """
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range.

        Args:
            df: OHLCV DataFrame with 'High', 'Low', 'Close' columns.
            window: ATR period (default 14).

        Returns:
            ATR series.
        """
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(com=window - 1, adjust=False).mean()
