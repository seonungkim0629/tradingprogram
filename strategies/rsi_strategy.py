"""
RSI (Relative Strength Index) Strategy for Bitcoin Trading Bot

This module implements a RSI-based trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union

from strategies.base import BaseStrategy
from utils.logging import get_logger, log_execution

# Initialize logger
logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI Trading Strategy
    
    Generates buy signals when RSI drops below oversold threshold,
    and sell signals when RSI rises above overbought threshold.
    """
    
    def __init__(self, 
                market: str = "KRW-BTC",
                name: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize RSI strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
                - rsi_period (int): RSI calculation period
                - oversold_threshold (float): Oversold threshold (0-100)
                - overbought_threshold (float): Overbought threshold (0-100)
                - confirmation_period (int): Periods to confirm signal
                - exit_rsi (float): RSI level to exit positions
                - position_size (float): Position size (0.0-1.0)
        """
        # Set default parameters
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'confirmation_period': 1,
            'exit_rsi': 50,
            'position_size': 0.5
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(market, name or "RSIStrategy", default_params)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized {self.name} with period={self.parameters['rsi_period']}, "
                    f"oversold={self.parameters['oversold_threshold']}, "
                    f"overbought={self.parameters['overbought_threshold']}")
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters"""
        # Ensure RSI period is reasonable
        if self.parameters['rsi_period'] < 2:
            logger.warning(f"RSI period too short: {self.parameters['rsi_period']}, setting to 14")
            self.parameters['rsi_period'] = 14
        
        # Ensure thresholds are in valid range
        for param in ['oversold_threshold', 'overbought_threshold', 'exit_rsi']:
            if self.parameters[param] < 0 or self.parameters[param] > 100:
                logger.warning(f"Invalid {param}: {self.parameters[param]}, must be between 0-100")
                # Set to default values
                defaults = {'oversold_threshold': 30, 'overbought_threshold': 70, 'exit_rsi': 50}
                self.parameters[param] = defaults[param]
        
        # Ensure proper ordering of thresholds
        if self.parameters['oversold_threshold'] >= self.parameters['overbought_threshold']:
            logger.warning(f"Oversold threshold ({self.parameters['oversold_threshold']}) must be "
                          f"less than overbought threshold ({self.parameters['overbought_threshold']})")
            self.parameters['oversold_threshold'] = 30
            self.parameters['overbought_threshold'] = 70
            logger.info(f"Reset thresholds to oversold={self.parameters['oversold_threshold']}, "
                       f"overbought={self.parameters['overbought_threshold']}")
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI indicator
        
        Args:
            data (pd.DataFrame): Price data with 'close' column
            
        Returns:
            pd.Series: RSI values
        """
        if len(data) < self.parameters['rsi_period'] + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need at least "
                          f"{self.parameters['rsi_period'] + 1} data points, got {len(data)}")
            return pd.Series()
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Get gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        period = self.parameters['rsi_period']
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _check_conditions(self, rsi: pd.Series) -> Dict[str, Any]:
        """
        Check RSI conditions for trading signals
        
        Args:
            rsi (pd.Series): RSI values
            
        Returns:
            Dict[str, Any]: Signal information
        """
        if len(rsi) < self.parameters['confirmation_period'] + 1:
            return {'signal': 'HOLD', 'reason': 'Insufficient data for confirmation'}
        
        # Get parameters
        oversold = self.parameters['oversold_threshold']
        overbought = self.parameters['overbought_threshold']
        confirmation = self.parameters['confirmation_period']
        exit_level = self.parameters['exit_rsi']
        
        # Get current and previous RSI values
        current_rsi = rsi.iloc[-1]
        
        # Check for confirmation periods
        oversold_confirmed = True
        overbought_confirmed = True
        
        for i in range(1, confirmation + 1):
            if i < len(rsi):
                if rsi.iloc[-i] > oversold:
                    oversold_confirmed = False
                if rsi.iloc[-i] < overbought:
                    overbought_confirmed = False
        
        # Determine signal
        if current_rsi < oversold and oversold_confirmed:
            return {
                'signal': 'BUY',
                'reason': f'RSI ({current_rsi:.2f}) below oversold threshold ({oversold})',
                'confidence': (oversold - current_rsi) / oversold * 0.5 + 0.5  # Scale to 0.5-1.0
            }
        elif current_rsi > overbought and overbought_confirmed:
            return {
                'signal': 'SELL',
                'reason': f'RSI ({current_rsi:.2f}) above overbought threshold ({overbought})',
                'confidence': (current_rsi - overbought) / (100 - overbought) * 0.5 + 0.5  # Scale to 0.5-1.0
            }
        elif current_rsi > exit_level:
            return {
                'signal': 'SELL',
                'reason': f'RSI ({current_rsi:.2f}) above exit level ({exit_level})',
                'confidence': 0.5 + (current_rsi - exit_level) / 100
            }
        elif current_rsi < 45 and current_rsi > oversold:  # 추가: RSI가 45 미만이면 매수 신호 발생
            return {
                'signal': 'BUY',
                'reason': f'RSI ({current_rsi:.2f}) below neutral (45)',
                'confidence': 0.5 - (current_rsi - oversold) / (45 - oversold) * 0.3  # 낮은 신뢰도로 신호 발생
            }
        elif current_rsi > 55 and current_rsi < overbought:  # 추가: RSI가 55 초과면 매도 신호 발생
            return {
                'signal': 'SELL',
                'reason': f'RSI ({current_rsi:.2f}) above neutral (55)',
                'confidence': 0.5 - (overbought - current_rsi) / (overbought - 55) * 0.3  # 낮은 신뢰도로 신호 발생
            }
        
        return {'signal': 'HOLD', 'reason': f'No RSI signal triggered (current: {current_rsi:.2f})', 'confidence': 0.5}
    
    @log_execution
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on RSI values
        
        Args:
            data (pd.DataFrame): Market data with OHLCV
            
        Returns:
            Dict[str, Any]: Signal dictionary
        """
        # Default signal is HOLD
        signal = {
            'signal': 'HOLD',
            'reason': 'No RSI signal',
            'confidence': 0.5,
            'metadata': {},
            'position_size': self.parameters['position_size']
        }
        
        # Not enough data
        if len(data) < self.parameters['rsi_period'] + self.parameters['confirmation_period']:
            signal['reason'] = f"Insufficient data (need {self.parameters['rsi_period'] + self.parameters['confirmation_period']} points)"
            return signal
        
        # Calculate RSI
        rsi_values = self._calculate_rsi(data)
        
        # Add to metadata
        if not rsi_values.empty:
            signal['metadata']['rsi'] = rsi_values.iloc[-1]
        
        # Check conditions
        if not rsi_values.empty:
            rsi_signal = self._check_conditions(rsi_values)
            signal.update(rsi_signal)
        
        # Log signal
        logger.info(f"Generated signal: {signal['signal']} with confidence {signal.get('confidence', 0.5):.2f}")
        
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
        
        # Don't sell at a loss unless stop loss is triggered
        if signal['signal'] == 'SELL' and portfolio.get('position', 0) > 0:
            entry_price = portfolio.get('entry_price', 0)
            current_price = portfolio.get('current_price', 0)
            
            if entry_price > 0 and current_price < entry_price:
                # Calculate loss percentage
                loss_pct = (current_price / entry_price - 1) * 100
                
                # Only allow selling at a loss if RSI is very high or loss is significant
                if loss_pct > -5 and signal.get('confidence', 0) < 0.8:
                    signal['signal'] = 'HOLD'
                    signal['reason'] = f"Avoiding small loss ({loss_pct:.2f}%) with moderate signal confidence"
                    logger.info(f"Signal changed to HOLD: {signal['reason']}")
        
        return signal
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any], 
                              available_balance: float) -> float:
        """
        Calculate position size based on RSI extremes
        
        Args:
            signal (Dict[str, Any]): Trading signal
            available_balance (float): Available balance
            
        Returns:
            float: Position size in base currency
        """
        # Base position size from parameters
        base_size = self.parameters['position_size']
        
        # Adjust based on signal confidence
        confidence_factor = signal.get('confidence', 0.5)
        
        # RSI extreme values increase position size
        rsi_value = signal.get('metadata', {}).get('rsi', 50)
        
        # RSI extremes = larger positions
        rsi_factor = 1.0
        if rsi_value < 20:  # Very oversold
            rsi_factor = 1.2
        elif rsi_value < 15:  # Extremely oversold
            rsi_factor = 1.5
        elif rsi_value > 80:  # Very overbought
            rsi_factor = 1.2
        elif rsi_value > 85:  # Extremely overbought
            rsi_factor = 1.5
        
        # Calculate final position size
        adjusted_size = base_size * confidence_factor * rsi_factor
        position_size = available_balance * min(adjusted_size, 1.0)
        
        logger.info(f"Calculated position size: {position_size} (confidence: {confidence_factor:.2f}, rsi_factor: {rsi_factor:.2f})")
        
        return position_size 