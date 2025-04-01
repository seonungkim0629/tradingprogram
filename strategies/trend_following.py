"""
Trend Following Strategy for Bitcoin Trading Bot

This module implements a trend following strategy based on multiple indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

from strategies.base import BaseStrategy
from utils.logging import get_logger, log_execution

# Initialize logger
logger = get_logger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy
    
    Uses a combination of moving averages, MACD, and ADX to identify and follow trends.
    """
    
    def __init__(self, 
                market: str = "KRW-BTC",
                name: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Trend Following strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
                - ema_short (int): Short EMA period
                - ema_medium (int): Medium EMA period
                - ema_long (int): Long EMA period
                - macd_fast (int): MACD fast period
                - macd_slow (int): MACD slow period
                - macd_signal (int): MACD signal period
                - adx_period (int): ADX period
                - adx_threshold (float): ADX threshold for strong trend
                - stop_loss (float): Stop loss percentage
                - take_profit (float): Take profit percentage
                - position_size (float): Position size (0.0-1.0)
        """
        # Set default parameters
        default_params = {
            'ema_short': 10,
            'ema_medium': 21,
            'ema_long': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'position_size': 0.5
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(market, name or "TrendFollowing", default_params)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized {self.name} strategy with EMA periods: "
                    f"{self.parameters['ema_short']}/{self.parameters['ema_medium']}/{self.parameters['ema_long']}")
        logger.info(f"MACD parameters: {self.parameters['macd_fast']}/{self.parameters['macd_slow']}/"
                    f"{self.parameters['macd_signal']}, ADX threshold: {self.parameters['adx_threshold']}")
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters"""
        # Ensure proper ordering of EMA periods
        if not (self.parameters['ema_short'] < self.parameters['ema_medium'] < self.parameters['ema_long']):
            logger.warning("EMA periods not in ascending order, adjusting...")
            self.parameters['ema_short'] = 10
            self.parameters['ema_medium'] = 21
            self.parameters['ema_long'] = 50
        
        # Ensure proper ordering of MACD periods
        if self.parameters['macd_fast'] >= self.parameters['macd_slow']:
            logger.warning("MACD fast period should be smaller than slow period, adjusting...")
            self.parameters['macd_fast'] = 12
            self.parameters['macd_slow'] = 26
        
        # Ensure reasonable ADX threshold
        if self.parameters['adx_threshold'] < 10 or self.parameters['adx_threshold'] > 50:
            logger.warning(f"ADX threshold {self.parameters['adx_threshold']} outside reasonable range, setting to 25")
            self.parameters['adx_threshold'] = 25
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, Any]]:
        """
        Calculate required indicators
        
        Args:
            data (pd.DataFrame): Price data with OHLCV
            
        Returns:
            Tuple[Dict[str, pd.Series], Dict[str, Any]]: Indicators and trend information
        """
        indicators = {}
        trend_info = {}
        
        # Calculate EMAs
        ema_short = data['close'].ewm(span=self.parameters['ema_short'], adjust=False).mean()
        ema_medium = data['close'].ewm(span=self.parameters['ema_medium'], adjust=False).mean()
        ema_long = data['close'].ewm(span=self.parameters['ema_long'], adjust=False).mean()
        
        indicators['ema_short'] = ema_short
        indicators['ema_medium'] = ema_medium
        indicators['ema_long'] = ema_long
        
        # Calculate MACD
        ema_fast = data['close'].ewm(span=self.parameters['macd_fast'], adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.parameters['macd_slow'], adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.parameters['macd_signal'], adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        indicators['macd_line'] = macd_line
        indicators['signal_line'] = signal_line
        indicators['macd_histogram'] = macd_histogram
        
        # Calculate ADX (Average Directional Index)
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.parameters['adx_period']).mean()
        
        # Calculate Directional Movement
        pos_dm = high - high.shift(1)
        neg_dm = low.shift(1) - low
        
        pos_dm[pos_dm < 0] = 0
        neg_dm[neg_dm < 0] = 0
        
        pos_dm[(pos_dm > 0) & (neg_dm > 0) & (pos_dm < neg_dm)] = 0
        neg_dm[(pos_dm > 0) & (neg_dm > 0) & (pos_dm > neg_dm)] = 0
        
        # Calculate Directional Indicators
        adx_period = self.parameters['adx_period']
        pos_di = 100 * (pos_dm.rolling(window=adx_period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=adx_period).mean() / atr)
        
        # Calculate Directional Index
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        
        # Calculate ADX
        adx = dx.rolling(window=adx_period).mean()
        
        indicators['pos_di'] = pos_di
        indicators['neg_di'] = neg_di
        indicators['adx'] = adx
        
        # Determine current trend based on EMAs
        if len(ema_short) > 0 and len(ema_medium) > 0 and len(ema_long) > 0:
            current_short = ema_short.iloc[-1]
            current_medium = ema_medium.iloc[-1]
            current_long = ema_long.iloc[-1]
            
            if current_short > current_medium > current_long:
                trend_info['ema_trend'] = 'UP'
            elif current_short < current_medium < current_long:
                trend_info['ema_trend'] = 'DOWN'
            else:
                trend_info['ema_trend'] = 'NEUTRAL'
        else:
            trend_info['ema_trend'] = 'UNKNOWN'
        
        # Determine MACD trend
        if len(macd_line) > 1 and len(signal_line) > 1:
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_hist = macd_histogram.iloc[-1]
            prev_hist = macd_histogram.iloc[-2] if len(macd_histogram) > 2 else 0
            
            if current_macd > current_signal:
                trend_info['macd_trend'] = 'UP'
            else:
                trend_info['macd_trend'] = 'DOWN'
            
            if current_hist > 0 and prev_hist < 0:
                trend_info['macd_cross'] = 'BULLISH'
            elif current_hist < 0 and prev_hist > 0:
                trend_info['macd_cross'] = 'BEARISH'
            else:
                trend_info['macd_cross'] = 'NONE'
        else:
            trend_info['macd_trend'] = 'UNKNOWN'
            trend_info['macd_cross'] = 'UNKNOWN'
        
        # Determine ADX trend strength
        if len(adx) > 0 and len(pos_di) > 0 and len(neg_di) > 0:
            current_adx = adx.iloc[-1]
            current_pos_di = pos_di.iloc[-1]
            current_neg_di = neg_di.iloc[-1]
            
            trend_info['adx_value'] = current_adx
            
            if current_adx > self.parameters['adx_threshold']:
                trend_info['adx_strong'] = True
                if current_pos_di > current_neg_di:
                    trend_info['adx_trend'] = 'UP'
                else:
                    trend_info['adx_trend'] = 'DOWN'
            else:
                trend_info['adx_strong'] = False
                trend_info['adx_trend'] = 'WEAK'
        else:
            trend_info['adx_value'] = 0
            trend_info['adx_strong'] = False
            trend_info['adx_trend'] = 'UNKNOWN'
        
        return indicators, trend_info
    
    def _analyze_trend(self, trend_info: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Analyze trend information to generate signals
        
        Args:
            trend_info (Dict[str, Any]): Trend information
            
        Returns:
            Tuple[str, float, str]: Signal type, confidence, and reason
        """
        signal = 'HOLD'
        confidence = 0.5
        reason = ''
        
        # Count supporting factors for each trend direction
        up_factors = 0
        down_factors = 0
        total_factors = 3  # EMA, MACD, ADX
        
        # Check EMA trend
        if trend_info.get('ema_trend') == 'UP':
            up_factors += 1
        elif trend_info.get('ema_trend') == 'DOWN':
            down_factors += 1
        
        # Check MACD trend
        if trend_info.get('macd_trend') == 'UP':
            up_factors += 1
        elif trend_info.get('macd_trend') == 'DOWN':
            down_factors += 1
        
        # Check ADX trend
        if trend_info.get('adx_strong', False) and trend_info.get('adx_trend') == 'UP':
            up_factors += 1
        elif trend_info.get('adx_strong', False) and trend_info.get('adx_trend') == 'DOWN':
            down_factors += 1
        
        # Generate signal based on trend agreement
        if up_factors >= 2:
            signal = 'BUY'
            confidence = 0.5 + (up_factors / total_factors) * 0.5
            
            factors_list = []
            if trend_info.get('ema_trend') == 'UP':
                factors_list.append('EMA alignment bullish')
            if trend_info.get('macd_trend') == 'UP':
                factors_list.append('MACD bullish')
            if trend_info.get('adx_strong', False) and trend_info.get('adx_trend') == 'UP':
                factors_list.append(f"Strong uptrend (ADX: {trend_info.get('adx_value', 0):.1f})")
            
            reason = f"Bullish trend: {', '.join(factors_list)}"
            
        elif down_factors >= 2:
            signal = 'SELL'
            confidence = 0.5 + (down_factors / total_factors) * 0.5
            
            factors_list = []
            if trend_info.get('ema_trend') == 'DOWN':
                factors_list.append('EMA alignment bearish')
            if trend_info.get('macd_trend') == 'DOWN':
                factors_list.append('MACD bearish')
            if trend_info.get('adx_strong', False) and trend_info.get('adx_trend') == 'DOWN':
                factors_list.append(f"Strong downtrend (ADX: {trend_info.get('adx_value', 0):.1f})")
            
            reason = f"Bearish trend: {', '.join(factors_list)}"
            
        else:
            # Look for MACD crosses for entry signals
            if trend_info.get('macd_cross') == 'BULLISH' and trend_info.get('ema_trend') != 'DOWN':
                signal = 'BUY'
                confidence = 0.6
                reason = f"MACD bullish cross with EMA trend {trend_info.get('ema_trend')}"
                
            elif trend_info.get('macd_cross') == 'BEARISH' and trend_info.get('ema_trend') != 'UP':
                signal = 'SELL'
                confidence = 0.6
                reason = f"MACD bearish cross with EMA trend {trend_info.get('ema_trend')}"
                
            else:
                reason = f"No clear trend direction (UP: {up_factors}, DOWN: {down_factors})"
        
        return signal, confidence, reason
    
    @log_execution
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on trend analysis
        
        Args:
            data (pd.DataFrame): Market data with OHLCV
            
        Returns:
            Dict[str, Any]: Signal dictionary
        """
        # Default signal is HOLD
        signal = {
            'signal': 'HOLD',
            'reason': 'Insufficient data for trend analysis',
            'confidence': 0.5,
            'metadata': {},
            'position_size': self.parameters['position_size']
        }
        
        # Check if we have enough data
        min_periods = max(
            self.parameters['ema_long'],
            self.parameters['macd_slow'] + self.parameters['macd_signal'],
            self.parameters['adx_period'] * 2
        )
        
        if len(data) < min_periods:
            return signal
        
        # Calculate indicators and analyze trend
        indicators, trend_info = self._calculate_indicators(data)
        
        # Add indicators to metadata
        for name, indicator in indicators.items():
            if not indicator.empty:
                signal['metadata'][name] = indicator.iloc[-1]
        
        # Add trend info to metadata
        signal['metadata']['trend_info'] = trend_info
        
        # Generate signal based on trend analysis
        signal_type, confidence, reason = self._analyze_trend(trend_info)
        
        signal['signal'] = signal_type
        signal['confidence'] = confidence
        signal['reason'] = reason
        
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
        
        # Check stop loss and take profit conditions
        if portfolio.get('position', 0) > 0:
            entry_price = portfolio.get('entry_price', 0)
            current_price = portfolio.get('current_price', 0)
            
            if entry_price > 0:
                price_change = (current_price / entry_price - 1)
                
                # Stop loss
                if price_change <= -self.parameters['stop_loss']:
                    signal['signal'] = 'SELL'
                    signal['reason'] = f"Stop loss triggered: {price_change*100:.2f}% loss"
                    signal['confidence'] = 1.0
                    logger.info(f"Signal changed to SELL: {signal['reason']}")
                
                # Take profit
                elif price_change >= self.parameters['take_profit']:
                    signal['signal'] = 'SELL'
                    signal['reason'] = f"Take profit triggered: {price_change*100:.2f}% gain"
                    signal['confidence'] = 1.0
                    logger.info(f"Signal changed to SELL: {signal['reason']}")
        
        # Don't trade in weak trends
        trend_info = signal.get('metadata', {}).get('trend_info', {})
        if signal['signal'] != 'HOLD' and not trend_info.get('adx_strong', False):
            if signal['confidence'] < 0.7:  # Allow high-confidence signals even in weak trends
                signal['signal'] = 'HOLD'
                signal['reason'] = f"Weak trend (ADX: {trend_info.get('adx_value', 0):.1f}), waiting for stronger trend"
                logger.info(f"Signal changed to HOLD: {signal['reason']}")
        
        return signal
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any], 
                              available_balance: float) -> float:
        """
        Calculate position size based on trend strength
        
        Args:
            signal (Dict[str, Any]): Trading signal
            available_balance (float): Available balance
            
        Returns:
            float: Position size in base currency
        """
        # Base position size from parameters
        base_size = self.parameters['position_size']
        
        # Adjust based on confidence and trend strength
        confidence_factor = signal.get('confidence', 0.5)
        
        # Get ADX value from metadata to gauge trend strength
        trend_info = signal.get('metadata', {}).get('trend_info', {})
        adx_value = trend_info.get('adx_value', 0)
        
        # Scale position size based on ADX (stronger trend = larger position)
        adx_factor = 1.0
        if adx_value > 40:  # Very strong trend
            adx_factor = 1.3
        elif adx_value > 30:  # Strong trend
            adx_factor = 1.2
        elif adx_value > 25:  # Moderate trend
            adx_factor = 1.1
        elif adx_value < 15:  # Very weak trend
            adx_factor = 0.8
        
        # Calculate final position size
        adjusted_size = base_size * confidence_factor * adx_factor
        position_size = available_balance * min(adjusted_size, 1.0)
        
        logger.info(f"Calculated position size: {position_size} "
                   f"(confidence: {confidence_factor:.2f}, ADX factor: {adx_factor:.2f})")
        
        return position_size 