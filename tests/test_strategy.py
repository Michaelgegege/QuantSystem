"""Tests for strategies."""

import numpy as np
import pandas as pd
import pytest

from quantsystem.strategy.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def trending_data():
    """A clearly upward-trending price series."""
    np.random.seed(0)
    prices = np.linspace(100, 200, 300) + np.random.randn(300) * 0.5
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)


class TestSMACrossoverStrategy:
    def test_signal_values(self, trending_data):
        strategy = SMACrossoverStrategy(fast_window=10, slow_window=50)
        signals = strategy.generate_signals(trending_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signal_index_matches_data(self, trending_data):
        strategy = SMACrossoverStrategy(fast_window=10, slow_window=50)
        signals = strategy.generate_signals(trending_data)
        assert signals.index.equals(trending_data.index)

    def test_at_least_one_buy_signal_in_uptrend(self, trending_data):
        strategy = SMACrossoverStrategy(fast_window=10, slow_window=50)
        signals = strategy.generate_signals(trending_data)
        assert (signals == 1).any(), "Expected at least one buy signal in an uptrend"

    def test_invalid_windows_raises(self):
        with pytest.raises(ValueError):
            SMACrossoverStrategy(fast_window=50, slow_window=10)

    def test_repr(self):
        s = SMACrossoverStrategy(10, 50)
        assert "SMA" in repr(s)
