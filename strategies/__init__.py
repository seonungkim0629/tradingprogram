"""
Trading strategies package
"""

from .trend_following import TrendFollowingStrategy
from .moving_average import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .mixed_strategy import MixedTimeFrameStrategy

# Export strategy classes
__all__ = [
    'TrendFollowingStrategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'MixedTimeFrameStrategy'
]