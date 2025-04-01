"""
Trading strategies package
"""

from .trend_following import TrendFollowingStrategy
from .moving_average import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .harmonizing import HarmonizingStrategy

# Export strategy classes
__all__ = [
    'TrendFollowingStrategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'HarmonizingStrategy'
]