"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from quantsystem.indicators.technical import TechnicalIndicators


@pytest.fixture
def price_series():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200))
    return pd.Series(prices, name="Close")


@pytest.fixture
def ohlcv_df(price_series):
    np.random.seed(42)
    close = price_series
    high = close + np.abs(np.random.randn(len(close))) * 0.5
    low = close - np.abs(np.random.randn(len(close))) * 0.5
    return pd.DataFrame({"High": high, "Low": low, "Close": close})


class TestSMA:
    def test_length(self, price_series):
        sma = TechnicalIndicators.sma(price_series, window=10)
        assert len(sma) == len(price_series)

    def test_first_values_nan(self, price_series):
        sma = TechnicalIndicators.sma(price_series, window=10)
        assert sma.iloc[:9].isna().all()

    def test_value_correctness(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = TechnicalIndicators.sma(s, window=3)
        assert sma.iloc[2] == pytest.approx(2.0)
        assert sma.iloc[4] == pytest.approx(4.0)


class TestEMA:
    def test_length(self, price_series):
        ema = TechnicalIndicators.ema(price_series, window=10)
        assert len(ema) == len(price_series)

    def test_no_nan_after_warmup(self, price_series):
        ema = TechnicalIndicators.ema(price_series, window=10)
        assert not ema.iloc[9:].isna().any()


class TestRSI:
    def test_range(self, price_series):
        rsi = TechnicalIndicators.rsi(price_series, window=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self, price_series):
        rsi = TechnicalIndicators.rsi(price_series, window=14)
        assert len(rsi) == len(price_series)


class TestMACD:
    def test_columns(self, price_series):
        macd_df = TechnicalIndicators.macd(price_series)
        assert set(macd_df.columns) == {"macd", "signal", "histogram"}

    def test_histogram_formula(self, price_series):
        macd_df = TechnicalIndicators.macd(price_series)
        expected = macd_df["macd"] - macd_df["signal"]
        pd.testing.assert_series_equal(
            macd_df["histogram"], expected, check_names=False
        )


class TestBollingerBands:
    def test_columns(self, price_series):
        bb = TechnicalIndicators.bollinger_bands(price_series)
        assert set(bb.columns) == {"upper", "middle", "lower"}

    def test_upper_above_lower(self, price_series):
        bb = TechnicalIndicators.bollinger_bands(price_series).dropna()
        assert (bb["upper"] >= bb["lower"]).all()

    def test_middle_is_sma(self, price_series):
        window = 20
        bb = TechnicalIndicators.bollinger_bands(price_series, window=window)
        sma = TechnicalIndicators.sma(price_series, window=window)
        pd.testing.assert_series_equal(bb["middle"], sma, check_names=False)


class TestATR:
    def test_length(self, ohlcv_df):
        atr = TechnicalIndicators.atr(ohlcv_df)
        assert len(atr) == len(ohlcv_df)

    def test_positive(self, ohlcv_df):
        atr = TechnicalIndicators.atr(ohlcv_df).dropna()
        assert (atr >= 0).all()
