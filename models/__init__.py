"""
Models package

Contains model implementations for trading systems.
"""

from .base import (
    ModelBase,
    ClassificationModel,
    RegressionModel,
    TimeSeriesModel,
    ReinforcementLearningModel,
    EnsembleModel,
    Signal,
    GPTAnalysisModel
)

from .signal import (
    TradingSignal,
    standardize_signal,
    ModelOutput,
    standardize_model_output
)

# Add legacy support for Signal to TradingSignal conversions
def legacy_signal_to_trading_signal(signal: Signal) -> TradingSignal:
    """Convert legacy Signal to new TradingSignal format"""
    from utils.constants import SignalType
    
    return TradingSignal(
        signal_type=SignalType.standardize(signal.signal_type),
        confidence=signal.confidence,
        reason=signal.reason,
        metadata=signal.metadata.copy() if signal.metadata else {},
        timestamp=signal.timestamp
    )
    
# Export common symbols
__all__ = [
    'ModelBase',
    'ClassificationModel',
    'RegressionModel',
    'TimeSeriesModel',
    'ReinforcementLearningModel',
    'EnsembleModel',
    'Signal',
    'TradingSignal',
    'standardize_signal',
    'ModelOutput',
    'standardize_model_output',
    'legacy_signal_to_trading_signal',
    'GPTAnalysisModel'
] 