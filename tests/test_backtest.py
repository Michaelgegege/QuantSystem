"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from quantsystem.backtest.engine import BacktestEngine, BacktestResult
from quantsystem.strategy.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def market_data():
    np.random.seed(7)
    prices = np.linspace(100, 150, 400) + np.random.randn(400) * 2
    dates = pd.date_range("2019-01-01", periods=400, freq="B")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, 400),
        },
        index=dates,
    )


class TestBacktestEngine:
    def test_equity_length_matches_data(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        strategy = SMACrossoverStrategy(10, 50)
        result = engine.run(market_data, strategy)
        assert len(result.equity) == len(market_data)

    def test_equity_starts_near_capital(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        strategy = SMACrossoverStrategy(10, 50)
        result = engine.run(market_data, strategy)
        assert result.equity.iloc[0] == pytest.approx(10_000, rel=0.05)

    def test_total_return_type(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        strategy = SMACrossoverStrategy(10, 50)
        result = engine.run(market_data, strategy)
        assert isinstance(result.total_return, float)

    def test_max_drawdown_non_positive(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        strategy = SMACrossoverStrategy(10, 50)
        result = engine.run(market_data, strategy)
        assert result.max_drawdown <= 0

    def test_uptrend_positive_return(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        strategy = SMACrossoverStrategy(10, 50)
        result = engine.run(market_data, strategy)
        # In a steady uptrend the strategy should make money
        assert result.total_return > 0

    def test_summary_keys(self, market_data):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(market_data, SMACrossoverStrategy(10, 50))
        summary = result.summary()
        for key in [
            "initial_capital",
            "final_equity",
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
        ]:
            assert key in summary

    def test_invalid_capital_raises(self):
        with pytest.raises(ValueError):
            BacktestEngine(initial_capital=-1000)

    def test_invalid_commission_raises(self):
        with pytest.raises(ValueError):
            BacktestEngine(commission=1.5)

    def test_repr(self, market_data):
        result = BacktestEngine(initial_capital=10_000).run(
            market_data, SMACrossoverStrategy(10, 50)
        )
        assert "BacktestResult" in repr(result)
