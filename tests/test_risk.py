"""Tests for the risk manager."""

import numpy as np
import pandas as pd
import pytest

from quantsystem.risk.manager import RiskManager


class TestRiskManager:
    def test_position_size_basic(self):
        rm = RiskManager(max_position_pct=0.1)
        # 10% of 10_000 = 1_000 / 50 = 20 shares
        assert rm.position_size(10_000, 50.0) == 20

    def test_position_size_zero_price_raises(self):
        rm = RiskManager()
        with pytest.raises(ValueError):
            rm.position_size(10_000, 0)

    def test_drawdown_not_exceeded(self):
        rm = RiskManager(max_drawdown_pct=0.2)
        equity = pd.Series([100.0, 105.0, 102.0, 108.0])
        assert not rm.is_drawdown_exceeded(equity)

    def test_drawdown_exceeded(self):
        rm = RiskManager(max_drawdown_pct=0.1)
        # drops 30% from peak
        equity = pd.Series([100.0, 110.0, 77.0])
        assert rm.is_drawdown_exceeded(equity)

    def test_var_positive(self):
        np.random.seed(1)
        returns = pd.Series(np.random.randn(500) * 0.01)
        var = RiskManager.value_at_risk(returns, confidence=0.95)
        assert var > 0

    def test_var_empty(self):
        assert RiskManager.value_at_risk(pd.Series(dtype=float)) == 0.0

    def test_kelly_fraction_range(self):
        kf = RiskManager.kelly_fraction(win_rate=0.6, win_loss_ratio=1.5)
        assert 0.0 <= kf <= 1.0

    def test_kelly_fraction_zero_ratio(self):
        assert RiskManager.kelly_fraction(0.6, 0.0) == 0.0

    def test_invalid_max_position_pct(self):
        with pytest.raises(ValueError):
            RiskManager(max_position_pct=0.0)

    def test_invalid_max_drawdown_pct(self):
        with pytest.raises(ValueError):
            RiskManager(max_drawdown_pct=1.5)
