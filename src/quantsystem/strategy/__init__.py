"""Strategy module."""

from quantsystem.strategy.base import BaseStrategy
from quantsystem.strategy.sma_crossover import SMACrossoverStrategy

__all__ = ["BaseStrategy", "SMACrossoverStrategy"]
