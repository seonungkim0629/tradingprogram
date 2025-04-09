"""
Moving Average Crossover Strategy for Bitcoin Trading Bot

This module implements a moving average crossover strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from strategies.base import BaseStrategy
from utils.logging import get_logger, log_execution

# Initialize logger
logger = get_logger(__name__)


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Generates buy signals when the fast moving average crosses above the slow moving average,
    and sell signals when the fast moving average crosses below the slow moving average.
    """
    
    def __init__(self, 
                market: str = "KRW-BTC",
                name: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Moving Average Crossover strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
                - fast_period (int): Fast moving average period
                - slow_period (int): Slow moving average period
                - ma_type (str): Moving average type ('sma', 'ema', 'wma')
                - signal_threshold (float): Minimum distance between MAs to generate signal
                - position_size (float): Position size (0.0-1.0)
        """
        # Set default parameters
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'ema',
            'signal_threshold': 0.0,
            'position_size': 0.5
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(market, name or "MovingAverageCrossover", default_params)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized {self.name} with fast_period={self.parameters['fast_period']}, "
                    f"slow_period={self.parameters['slow_period']}, "
                    f"ma_type={self.parameters['ma_type']}")
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters"""
        # Ensure fast period is shorter than slow period
        if self.parameters['fast_period'] >= self.parameters['slow_period']:
            logger.warning(f"Fast period ({self.parameters['fast_period']}) should be "
                          f"shorter than slow period ({self.parameters['slow_period']})")
            self.parameters['fast_period'] = min(self.parameters['fast_period'], 
                                                self.parameters['slow_period'] - 5)
            logger.info(f"Adjusted fast period to {self.parameters['fast_period']}")
        
        # Validate MA type
        valid_ma_types = ['sma', 'ema', 'wma']
        if self.parameters['ma_type'].lower() not in valid_ma_types:
            logger.warning(f"Invalid MA type: {self.parameters['ma_type']}, using 'ema' instead")
            self.parameters['ma_type'] = 'ema'
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate moving averages
        
        Args:
            data (pd.DataFrame): Price data with 'close' column
            
        Returns:
            Tuple[pd.Series, pd.Series]: Fast and slow moving averages
        """
        if len(data) < self.parameters['slow_period']:
            logger.warning(f"Insufficient data for MA calculation. Need at least "
                          f"{self.parameters['slow_period']} data points, got {len(data)}")
            # Return empty series
            return pd.Series(), pd.Series()
        
        # Get parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type'].lower()
        
        # Calculate moving averages based on type
        if ma_type == 'sma':
            fast_ma = data['close'].rolling(window=fast_period).mean()
            slow_ma = data['close'].rolling(window=slow_period).mean()
        elif ma_type == 'ema':
            fast_ma = data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ma = data['close'].ewm(span=slow_period, adjust=False).mean()
        elif ma_type == 'wma':
            # Weighted moving average
            weights_fast = np.arange(1, fast_period + 1)
            weights_slow = np.arange(1, slow_period + 1)
            
            fast_ma = data['close'].rolling(window=fast_period).apply(
                lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True)
            slow_ma = data['close'].rolling(window=slow_period).apply(
                lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True)
        else:
            # Default to EMA
            fast_ma = data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ma = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        return fast_ma, slow_ma
    
    def _check_crossover(self, 
                       fast_ma: pd.Series, 
                       slow_ma: pd.Series) -> Optional[str]:
        """
        Check for moving average crossover
        
        Args:
            fast_ma (pd.Series): Fast moving average
            slow_ma (pd.Series): Slow moving average
            
        Returns:
            Optional[str]: 'BUY', 'SELL', or None
        """
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return None
        
        # Get current and previous values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Calculate signal threshold as percentage of price
        threshold = self.parameters['signal_threshold'] * current_slow
        
        # Check for crossover with threshold
        if prev_fast <= prev_slow and current_fast > current_slow + threshold:
            return "BUY"
        elif prev_fast >= prev_slow and current_fast < current_slow - threshold:
            return "SELL"
        
        # 추가: 이동평균선이 가까워지는 경우에도 신호 발생
        if current_fast > current_slow * 0.98 and current_fast < current_slow:
            # 빠른 이동평균선이 느린 이동평균선에 거의 접근했을 때 매수 신호
            return "BUY"
        elif current_fast < current_slow * 1.02 and current_fast > current_slow:
            # 빠른 이동평균선이 느린 이동평균선을 조금 넘었을 때 매도 신호
            return "SELL"
        
        return None
    
    def _calculate_signal_strength(self, 
                                fast_ma: pd.Series, 
                                slow_ma: pd.Series) -> float:
        """
        Calculate signal strength based on moving averages
        
        Args:
            fast_ma (pd.Series): Fast moving average
            slow_ma (pd.Series): Slow moving average
            
        Returns:
            float: Signal strength (0.0-1.0)
        """
        if len(fast_ma) < 1 or len(slow_ma) < 1:
            return 0.5
        
        # Calculate distance between MAs
        fast_val = fast_ma.iloc[-1]
        slow_val = slow_ma.iloc[-1]
        
        # Percentage difference
        diff_pct = abs((fast_val - slow_val) / slow_val)
        
        # Map to 0.5-1.0 range
        # 0% difference -> 0.5 confidence
        # 5%+ difference -> 1.0 confidence
        strength = min(0.5 + diff_pct * 10, 1.0)
        
        return strength
    
    @log_execution
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Generate trading signal based on moving average crossover
        
        Args:
            data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): Market data with OHLCV,
                either a single DataFrame or a dictionary of DataFrames
            
        Returns:
            Dict[str, Any]: Signal dictionary
        """
        # 데이터 전처리: DataFrame이나 Dict 모두 처리 가능하도록
        if isinstance(data, dict):
            # 딕셔너리에서 첫 번째 데이터프레임 사용
            # 일봉 데이터가 있으면 우선 사용
            if 'daily' in data:
                data_df = data['daily']
            else:
                # 첫 번째 사용 가능한 데이터프레임 사용
                data_df = next(iter(data.values()))
        else:
            # 단일 데이터프레임인 경우 그대로 사용
            data_df = data
            
        # Default signal is HOLD
        signal = {
            'signal': 'HOLD',
            'reason': 'No crossover detected',
            'confidence': 0.5,
            'metadata': {},
            'position_size': self.parameters['position_size']
        }
        
        # Not enough data
        if len(data_df) < self.parameters['slow_period']:
            signal['reason'] = f"Insufficient data (need {self.parameters['slow_period']} points)"
            return signal
        
        # Calculate moving averages
        fast_ma, slow_ma = self._calculate_moving_averages(data_df)
        
        # Add to metadata
        signal['metadata']['fast_ma'] = fast_ma.iloc[-1] if not fast_ma.empty else None
        signal['metadata']['slow_ma'] = slow_ma.iloc[-1] if not slow_ma.empty else None
        
        # Check for crossover
        crossover = self._check_crossover(fast_ma, slow_ma)
        
        if crossover:
            signal['signal'] = crossover
            signal['reason'] = f"{self.parameters['ma_type'].upper()} {self.parameters['fast_period']}/{self.parameters['slow_period']} crossover"
            signal['confidence'] = self._calculate_signal_strength(fast_ma, slow_ma)
        
        # Log signal
        logger.info(f"Generated signal: {signal['signal']} with confidence {signal['confidence']:.2f}")
        
        return signal
    
    def apply_risk_management(self, 
                           signal: Dict[str, Any], 
                           portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply risk management rules to the signal
        
        Args:
            signal (Dict[str, Any]): Trading signal
            portfolio (Dict[str, Any]): Current portfolio state
            
        Returns:
            Dict[str, Any]: Signal after risk management
        """
        # Call parent method first
        signal = super().apply_risk_management(signal, portfolio)
        
        # Apply strategy-specific risk management rules
        if signal['signal'] == 'BUY':
            # Only buy if confidence is high enough
            min_confidence = 0.6  # Minimum confidence for buy
            if signal['confidence'] < min_confidence:
                signal['signal'] = 'HOLD'
                signal['reason'] = f"Low confidence ({signal['confidence']:.2f} < {min_confidence})"
                logger.info(f"Signal changed to HOLD: {signal['reason']}")
        
        return signal
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any], 
                              available_balance: float) -> float:
        """
        Calculate position size based on signal confidence
        
        Args:
            signal (Dict[str, Any]): Trading signal
            available_balance (float): Available balance
            
        Returns:
            float: Position size in base currency
        """
        # Base position size from parameters
        base_size = self.parameters['position_size']
        
        # Adjust based on confidence
        confidence_factor = signal['confidence']
        adjusted_size = base_size * confidence_factor
        
        # Calculate final position size
        position_size = available_balance * adjusted_size
        
        logger.info(f"Calculated position size: {position_size} (confidence: {confidence_factor:.2f})")
        
        return position_size 