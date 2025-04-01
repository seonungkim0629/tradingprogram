"""
Stop Loss Module for Bitcoin Trading Bot

This module implements various stop loss and take profit strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class StopLossStrategy:
    """Base class for stop loss strategies"""
    
    def __init__(self, name: str):
        """
        Initialize stop loss strategy
        
        Args:
            name (str): Name of the stop loss strategy
        """
        self.name = name
        self.logger = logger
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> Optional[float]:
        """
        Calculate stop loss price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
            
        Returns:
            Optional[float]: Stop loss price, None if not triggered
        """
        raise NotImplementedError("Subclasses must implement calculate_stop_loss")
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the stop loss has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
            
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        raise NotImplementedError("Subclasses must implement should_exit")


class FixedPercentageStop(StopLossStrategy):
    """Fixed percentage stop loss strategy"""
    
    def __init__(self, 
                stop_loss_pct: float = 0.05,
                take_profit_pct: Optional[float] = None):
        """
        Initialize fixed percentage stop loss strategy
        
        Args:
            stop_loss_pct (float): Stop loss percentage (0.0-1.0)
            take_profit_pct (Optional[float]): Take profit percentage (0.0-1.0)
        """
        super().__init__("Fixed Percentage")
        self.stop_loss_pct = max(0.0, min(1.0, stop_loss_pct))
        self.take_profit_pct = take_profit_pct
        
        self.logger.info(f"Initialized Fixed Percentage stop loss with "
                       f"stop_loss={self.stop_loss_pct:.1%}" +
                       (f", take_profit={self.take_profit_pct:.1%}" if self.take_profit_pct else ""))
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            float: Stop loss price
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        if market_direction == 'long':
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        else:  # short
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
        
        return stop_loss_price
    
    def calculate_take_profit(self, 
                            entry_price: float, 
                            current_price: float,
                            **kwargs) -> Optional[float]:
        """
        Calculate take profit price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Optional[float]: Take profit price, None if not set
        """
        if not self.take_profit_pct:
            return None
            
        market_direction = kwargs.get('market_direction', 'long')
        
        if market_direction == 'long':
            take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:  # short
            take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        return take_profit_price
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the stop loss or take profit has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate stop loss and take profit prices
        stop_loss_price = self.calculate_stop_loss(entry_price, current_price, **kwargs)
        take_profit_price = self.calculate_take_profit(entry_price, current_price, **kwargs)
        
        # Check if stop loss or take profit has been triggered
        if market_direction == 'long':
            if current_price <= stop_loss_price:
                return True, f"Stop loss triggered at {current_price:.2f} (threshold: {stop_loss_price:.2f})"
            elif take_profit_price and current_price >= take_profit_price:
                return True, f"Take profit triggered at {current_price:.2f} (threshold: {take_profit_price:.2f})"
        else:  # short
            if current_price >= stop_loss_price:
                return True, f"Stop loss triggered at {current_price:.2f} (threshold: {stop_loss_price:.2f})"
            elif take_profit_price and current_price <= take_profit_price:
                return True, f"Take profit triggered at {current_price:.2f} (threshold: {take_profit_price:.2f})"
        
        return False, "No exit triggered"


class TrailingStop(StopLossStrategy):
    """Trailing stop loss strategy"""
    
    def __init__(self, 
                initial_stop_pct: float = 0.05,
                trail_pct: float = 0.03):
        """
        Initialize trailing stop loss strategy
        
        Args:
            initial_stop_pct (float): Initial stop loss percentage (0.0-1.0)
            trail_pct (float): Trailing percentage (0.0-1.0)
        """
        super().__init__("Trailing Stop")
        self.initial_stop_pct = max(0.0, min(1.0, initial_stop_pct))
        self.trail_pct = max(0.0, min(1.0, trail_pct))
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        
        self.logger.info(f"Initialized Trailing Stop with "
                       f"initial_stop={self.initial_stop_pct:.1%}, "
                       f"trail={self.trail_pct:.1%}")
    
    def update_extremes(self, current_price: float, market_direction: str) -> None:
        """
        Update the highest/lowest price seen
        
        Args:
            current_price (float): Current price
            market_direction (str): 'long' or 'short'
        """
        if market_direction == 'long':
            self.highest_price = max(self.highest_price, current_price)
        else:  # short
            self.lowest_price = min(self.lowest_price, current_price)
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> float:
        """
        Calculate trailing stop loss price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            float: Stop loss price
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Update highest/lowest price
        self.update_extremes(current_price, market_direction)
        
        if market_direction == 'long':
            # Initial stop loss
            initial_stop = entry_price * (1 - self.initial_stop_pct)
            
            # Trailing stop loss
            if self.highest_price > entry_price:
                trailing_stop = self.highest_price * (1 - self.trail_pct)
                return max(initial_stop, trailing_stop)
            else:
                return initial_stop
        else:  # short
            # Initial stop loss
            initial_stop = entry_price * (1 + self.initial_stop_pct)
            
            # Trailing stop loss
            if self.lowest_price < entry_price:
                trailing_stop = self.lowest_price * (1 + self.trail_pct)
                return min(initial_stop, trailing_stop)
            else:
                return initial_stop
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the trailing stop loss has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate stop loss price
        stop_loss_price = self.calculate_stop_loss(entry_price, current_price, **kwargs)
        
        # Check if stop loss has been triggered
        if market_direction == 'long':
            if current_price <= stop_loss_price:
                trailing_amount = (self.highest_price - stop_loss_price) / self.highest_price * 100
                return True, f"Trailing stop triggered at {current_price:.2f} " \
                             f"(stop: {stop_loss_price:.2f}, {trailing_amount:.1f}% below peak)"
        else:  # short
            if current_price >= stop_loss_price:
                trailing_amount = (stop_loss_price - self.lowest_price) / self.lowest_price * 100
                return True, f"Trailing stop triggered at {current_price:.2f} " \
                             f"(stop: {stop_loss_price:.2f}, {trailing_amount:.1f}% above trough)"
        
        return False, "No exit triggered"
    
    def reset(self) -> None:
        """Reset the trailing stop (e.g., after exiting a position)"""
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        self.logger.info("Reset trailing stop")


class ATRStop(StopLossStrategy):
    """ATR-based stop loss strategy"""
    
    def __init__(self, 
                atr_multiplier: float = 3.0,
                take_profit_multiplier: Optional[float] = None):
        """
        Initialize ATR-based stop loss strategy
        
        Args:
            atr_multiplier (float): ATR multiplier for stop loss
            take_profit_multiplier (Optional[float]): ATR multiplier for take profit
        """
        super().__init__("ATR Stop")
        self.atr_multiplier = max(0.5, atr_multiplier)
        self.take_profit_multiplier = take_profit_multiplier
        
        self.logger.info(f"Initialized ATR Stop with "
                       f"multiplier={self.atr_multiplier:.1f}" +
                       (f", take_profit_multiplier={self.take_profit_multiplier:.1f}" 
                        if self.take_profit_multiplier else ""))
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> Optional[float]:
        """
        Calculate ATR-based stop loss price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - atr (float): Current ATR value
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Optional[float]: Stop loss price, None if ATR is not provided
        """
        atr = kwargs.get('atr')
        if atr is None:
            self.logger.warning("ATR value missing for ATR-based stop loss")
            return None
            
        market_direction = kwargs.get('market_direction', 'long')
        
        if market_direction == 'long':
            stop_loss_price = entry_price - (atr * self.atr_multiplier)
        else:  # short
            stop_loss_price = entry_price + (atr * self.atr_multiplier)
        
        return stop_loss_price
    
    def calculate_take_profit(self, 
                            entry_price: float, 
                            current_price: float,
                            **kwargs) -> Optional[float]:
        """
        Calculate ATR-based take profit price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - atr (float): Current ATR value
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Optional[float]: Take profit price, None if take_profit_multiplier not set or ATR missing
        """
        if not self.take_profit_multiplier:
            return None
            
        atr = kwargs.get('atr')
        if atr is None:
            return None
            
        market_direction = kwargs.get('market_direction', 'long')
        
        if market_direction == 'long':
            take_profit_price = entry_price + (atr * self.take_profit_multiplier)
        else:  # short
            take_profit_price = entry_price - (atr * self.take_profit_multiplier)
        
        return take_profit_price
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the ATR-based stop loss or take profit has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - atr (float): Current ATR value
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate stop loss and take profit prices
        stop_loss_price = self.calculate_stop_loss(entry_price, current_price, **kwargs)
        take_profit_price = self.calculate_take_profit(entry_price, current_price, **kwargs)
        
        # If ATR is missing, we can't check the stops
        if stop_loss_price is None:
            return False, "Missing ATR for stop calculation"
        
        # Check if stop loss or take profit has been triggered
        if market_direction == 'long':
            if current_price <= stop_loss_price:
                return True, f"ATR stop loss triggered at {current_price:.2f} (threshold: {stop_loss_price:.2f})"
            elif take_profit_price and current_price >= take_profit_price:
                return True, f"ATR take profit triggered at {current_price:.2f} (threshold: {take_profit_price:.2f})"
        else:  # short
            if current_price >= stop_loss_price:
                return True, f"ATR stop loss triggered at {current_price:.2f} (threshold: {stop_loss_price:.2f})"
            elif take_profit_price and current_price <= take_profit_price:
                return True, f"ATR take profit triggered at {current_price:.2f} (threshold: {take_profit_price:.2f})"
        
        return False, "No exit triggered"


class TimeBasedStop(StopLossStrategy):
    """Time-based stop loss strategy"""
    
    def __init__(self, 
                max_hold_time_hours: float = 48.0,
                min_profit_pct: float = 0.0):
        """
        Initialize time-based stop loss strategy
        
        Args:
            max_hold_time_hours (float): Maximum hold time in hours
            min_profit_pct (float): Minimum profit percentage to exit (0.0-1.0)
        """
        super().__init__("Time-Based")
        self.max_hold_time_hours = max(1.0, max_hold_time_hours)
        self.min_profit_pct = max(0.0, min_profit_pct)
        
        self.logger.info(f"Initialized Time-Based stop with "
                       f"max_hold_time={self.max_hold_time_hours:.1f} hours, "
                       f"min_profit={self.min_profit_pct:.1%}")
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the time-based stop has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - entry_time (datetime): Entry time
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        entry_time = kwargs.get('entry_time')
        if entry_time is None:
            return False, "Missing entry time for time-based stop"
            
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate hold time
        current_time = datetime.now()
        hold_time_hours = (current_time - entry_time).total_seconds() / 3600
        
        # Calculate profit percentage
        if market_direction == 'long':
            profit_pct = (current_price / entry_price) - 1
        else:  # short
            profit_pct = 1 - (current_price / entry_price)
        
        # Check if max hold time has elapsed and minimum profit achieved
        if hold_time_hours >= self.max_hold_time_hours:
            if profit_pct >= self.min_profit_pct:
                return True, f"Time-based exit after {hold_time_hours:.1f} hours with {profit_pct:.1%} profit"
            else:
                # If in profit but below min_profit, wait longer
                if profit_pct > 0:
                    return False, f"Holding position despite time limit ({hold_time_hours:.1f} hours)" \
                                 f" as profit ({profit_pct:.1%}) is below target ({self.min_profit_pct:.1%})"
                else:
                    return True, f"Time-based exit after {hold_time_hours:.1f} hours despite {profit_pct:.1%} loss"
        
        return False, "No exit triggered"


class IndicatorBasedStop(StopLossStrategy):
    """Indicator-based stop loss strategy"""
    
    def __init__(self, indicator_name: str):
        """
        Initialize indicator-based stop loss strategy
        
        Args:
            indicator_name (str): Name of the indicator to use
        """
        super().__init__("Indicator-Based")
        self.indicator_name = indicator_name
        
        self.logger.info(f"Initialized Indicator-Based stop using {self.indicator_name}")
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the indicator-based stop has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - indicators (Dict[str, Any]): Dictionary of indicator values
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        indicators = kwargs.get('indicators', {})
        if not indicators or self.indicator_name not in indicators:
            return False, f"Missing {self.indicator_name} indicator for stop calculation"
            
        market_direction = kwargs.get('market_direction', 'long')
        indicator_value = indicators[self.indicator_name]
        
        # Implement indicator-specific exit logic
        # For RSI
        if self.indicator_name == 'rsi':
            if market_direction == 'long' and indicator_value > 70:
                return True, f"RSI exit signal: {indicator_value:.1f} > 70 (overbought)"
            elif market_direction == 'short' and indicator_value < 30:
                return True, f"RSI exit signal: {indicator_value:.1f} < 30 (oversold)"
        
        # For MACD
        elif self.indicator_name == 'macd':
            macd_line = indicator_value.get('macd_line')
            signal_line = indicator_value.get('signal_line')
            if not macd_line or not signal_line:
                return False, "Missing MACD components"
                
            if market_direction == 'long' and macd_line < signal_line:
                return True, f"MACD bearish crossover (MACD: {macd_line:.2f}, Signal: {signal_line:.2f})"
            elif market_direction == 'short' and macd_line > signal_line:
                return True, f"MACD bullish crossover (MACD: {macd_line:.2f}, Signal: {signal_line:.2f})"
        
        # For Bollinger Bands
        elif self.indicator_name == 'bollinger_bands':
            upper_band = indicator_value.get('upper')
            lower_band = indicator_value.get('lower')
            if not upper_band or not lower_band:
                return False, "Missing Bollinger Band components"
                
            if market_direction == 'long' and current_price >= upper_band:
                return True, f"Price at upper Bollinger Band: {current_price:.2f} >= {upper_band:.2f}"
            elif market_direction == 'short' and current_price <= lower_band:
                return True, f"Price at lower Bollinger Band: {current_price:.2f} <= {lower_band:.2f}"
        
        # Add other indicators as needed
        
        return False, "No indicator-based exit triggered"


class CompositeStopLoss(StopLossStrategy):
    """Composite stop loss strategy that combines multiple strategies"""
    
    def __init__(self, strategies: List[StopLossStrategy]):
        """
        Initialize composite stop loss strategy
        
        Args:
            strategies (List[StopLossStrategy]): List of stop loss strategies
        """
        super().__init__("Composite")
        self.strategies = strategies
        
        strategy_names = [strategy.name for strategy in strategies]
        self.logger.info(f"Initialized Composite stop with strategies: {', '.join(strategy_names)}")
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> Optional[float]:
        """
        Calculate most conservative stop loss price from all strategies
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
            
        Returns:
            Optional[float]: Most conservative stop loss price
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Collect all valid stop loss prices
        stop_prices = []
        for strategy in self.strategies:
            stop_price = strategy.calculate_stop_loss(entry_price, current_price, **kwargs)
            if stop_price is not None:
                stop_prices.append(stop_price)
        
        if not stop_prices:
            return None
        
        # Return the most conservative stop loss price
        if market_direction == 'long':
            # For long positions, higher stop loss is more conservative
            return max(stop_prices)
        else:
            # For short positions, lower stop loss is more conservative
            return min(stop_prices)
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if any of the stop loss strategies has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
            
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        for strategy in self.strategies:
            should_exit, reason = strategy.should_exit(entry_price, current_price, **kwargs)
            if should_exit:
                return True, f"{strategy.name}: {reason}"
        
        return False, "No exit triggered"
    
    def add_strategy(self, strategy: StopLossStrategy) -> None:
        """
        Add a new strategy to the composite
        
        Args:
            strategy (StopLossStrategy): Strategy to add
        """
        self.strategies.append(strategy)
        self.logger.info(f"Added {strategy.name} to composite stop loss")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from the composite
        
        Args:
            strategy_name (str): Name of the strategy to remove
            
        Returns:
            bool: True if strategy was removed, False otherwise
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                self.strategies.pop(i)
                self.logger.info(f"Removed {strategy_name} from composite stop loss")
                return True
        
        self.logger.warning(f"Strategy {strategy_name} not found in composite stop loss")
        return False


class ChannelBasedStop(StopLossStrategy):
    """Channel-based stop loss (uses support/resistance levels)"""
    
    def __init__(self, 
                channel_type: str = 'donchian',
                period: int = 20,
                buffer_pct: float = 0.01):
        """
        Initialize channel-based stop loss strategy
        
        Args:
            channel_type (str): Channel type ('donchian', 'keltner', 'support_resistance')
            period (int): Channel calculation period
            buffer_pct (float): Additional buffer percentage (0.0-1.0)
        """
        super().__init__("Channel-Based")
        self.channel_type = channel_type
        self.period = max(5, period)
        self.buffer_pct = max(0.0, min(0.1, buffer_pct))
        
        self.logger.info(f"Initialized Channel-Based stop with type={self.channel_type}, "
                       f"period={self.period}, buffer={self.buffer_pct:.1%}")
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          current_price: float,
                          **kwargs) -> Optional[float]:
        """
        Calculate channel-based stop loss price
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                - high_prices (pd.Series): Series of high prices
                - low_prices (pd.Series): Series of low prices
                - close_prices (pd.Series): Series of close prices
                - atr (float): Current ATR value (for Keltner channels)
                - support_levels (List[float]): Support levels (for S/R based stops)
                - resistance_levels (List[float]): Resistance levels (for S/R based stops)
                
        Returns:
            Optional[float]: Stop loss price
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate channel-based stop loss based on channel type
        if self.channel_type == 'donchian':
            high_prices = kwargs.get('high_prices')
            low_prices = kwargs.get('low_prices')
            
            if high_prices is None or low_prices is None:
                self.logger.warning("Missing price data for Donchian channel calculation")
                return None
                
            if len(high_prices) < self.period or len(low_prices) < self.period:
                self.logger.warning(f"Insufficient data for Donchian channel (need {self.period} points)")
                return None
                
            # Calculate Donchian channel
            upper_band = high_prices[-self.period:].max()
            lower_band = low_prices[-self.period:].min()
            
            # Apply buffer
            upper_band *= (1 + self.buffer_pct)
            lower_band *= (1 - self.buffer_pct)
            
            if market_direction == 'long':
                return lower_band
            else:  # short
                return upper_band
                
        elif self.channel_type == 'keltner':
            close_prices = kwargs.get('close_prices')
            atr = kwargs.get('atr')
            
            if close_prices is None or atr is None:
                self.logger.warning("Missing data for Keltner channel calculation")
                return None
                
            if len(close_prices) < self.period:
                self.logger.warning(f"Insufficient data for Keltner channel (need {self.period} points)")
                return None
                
            # Calculate Keltner channel
            middle_band = close_prices[-self.period:].mean()
            upper_band = middle_band + (2 * atr)
            lower_band = middle_band - (2 * atr)
            
            # Apply buffer
            upper_band *= (1 + self.buffer_pct)
            lower_band *= (1 - self.buffer_pct)
            
            if market_direction == 'long':
                return lower_band
            else:  # short
                return upper_band
                
        elif self.channel_type == 'support_resistance':
            support_levels = kwargs.get('support_levels', [])
            resistance_levels = kwargs.get('resistance_levels', [])
            
            if not support_levels and not resistance_levels:
                self.logger.warning("Missing support/resistance levels")
                return None
                
            if market_direction == 'long':
                # Find the highest support level below current price
                relevant_supports = [s for s in support_levels if s < current_price]
                if relevant_supports:
                    return max(relevant_supports) * (1 - self.buffer_pct)
            else:  # short
                # Find the lowest resistance level above current price
                relevant_resistances = [r for r in resistance_levels if r > current_price]
                if relevant_resistances:
                    return min(relevant_resistances) * (1 + self.buffer_pct)
            
            return None
        
        self.logger.warning(f"Unknown channel type: {self.channel_type}")
        return None
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the channel-based stop has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
            
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate stop loss price
        stop_loss_price = self.calculate_stop_loss(entry_price, current_price, **kwargs)
        
        if stop_loss_price is None:
            return False, f"Could not calculate {self.channel_type} channel stop"
        
        # Check if stop loss has been triggered
        if market_direction == 'long':
            if current_price <= stop_loss_price:
                return True, f"{self.channel_type.capitalize()} channel stop triggered at {current_price:.2f} (level: {stop_loss_price:.2f})"
        else:  # short
            if current_price >= stop_loss_price:
                return True, f"{self.channel_type.capitalize()} channel stop triggered at {current_price:.2f} (level: {stop_loss_price:.2f})"
        
        return False, "No exit triggered"


class VolatilityExpansionStop(StopLossStrategy):
    """Stop loss based on volatility expansion (for breakout fading)"""
    
    def __init__(self, 
                volatility_period: int = 10,
                volatility_multiplier: float = 2.0,
                lookback_period: int = 3):
        """
        Initialize volatility expansion stop loss
        
        Args:
            volatility_period (int): Period for volatility calculation
            volatility_multiplier (float): Multiplier for volatility threshold
            lookback_period (int): Period to look back for volatility expansion
        """
        super().__init__("Volatility Expansion")
        self.volatility_period = max(5, volatility_period)
        self.volatility_multiplier = max(1.0, volatility_multiplier)
        self.lookback_period = max(1, lookback_period)
        
        self.logger.info(f"Initialized Volatility Expansion stop with "
                       f"period={self.volatility_period}, "
                       f"multiplier={self.volatility_multiplier:.1f}")
    
    def _is_volatility_expanding(self, 
                              close_prices: pd.Series, 
                              high_prices: pd.Series, 
                              low_prices: pd.Series) -> bool:
        """
        Check if volatility is expanding
        
        Args:
            close_prices (pd.Series): Series of close prices
            high_prices (pd.Series): Series of high prices
            low_prices (pd.Series): Series of low prices
            
        Returns:
            bool: True if volatility is expanding, False otherwise
        """
        if len(close_prices) < self.volatility_period + self.lookback_period:
            return False
        
        # Calculate true ranges
        tr_values = []
        for i in range(1, len(close_prices)):
            high = high_prices.iloc[i]
            low = low_prices.iloc[i]
            prev_close = close_prices.iloc[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr_values.append(max(tr1, tr2, tr3))
        
        tr_series = pd.Series(tr_values)
        
        # Calculate current and previous average true ranges
        current_atr = tr_series[-self.volatility_period:].mean()
        prev_atr = tr_series[-(self.volatility_period + self.lookback_period):-self.lookback_period].mean()
        
        # Check if volatility is expanding
        return current_atr > prev_atr * self.volatility_multiplier
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if the volatility expansion stop has been triggered
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - close_prices (pd.Series): Series of close prices
                - high_prices (pd.Series): Series of high prices
                - low_prices (pd.Series): Series of low prices
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        close_prices = kwargs.get('close_prices')
        high_prices = kwargs.get('high_prices')
        low_prices = kwargs.get('low_prices')
        
        if close_prices is None or high_prices is None or low_prices is None:
            return False, "Missing price data for volatility expansion calculation"
        
        # Check if volatility is expanding
        is_expanding = self._is_volatility_expanding(close_prices, high_prices, low_prices)
        
        if is_expanding:
            return True, f"Volatility expansion detected, exiting position at {current_price:.2f}"
        
        return False, "No exit triggered"


class ProfitTargetLadder(StopLossStrategy):
    """
    Profit target ladder for scaling out of positions
    
    Exits a portion of the position at different profit targets.
    """
    
    def __init__(self, 
                profit_targets: List[float],
                exit_portions: List[float]):
        """
        Initialize profit target ladder
        
        Args:
            profit_targets (List[float]): List of profit targets (in ascending order)
            exit_portions (List[float]): List of position portions to exit at each target (0.0-1.0)
        """
        super().__init__("Profit Target Ladder")
        
        # Validate inputs
        if len(profit_targets) != len(exit_portions):
            raise ValueError("profit_targets and exit_portions must have the same length")
            
        # Ensure profit targets are in ascending order
        if not all(profit_targets[i] <= profit_targets[i+1] for i in range(len(profit_targets)-1)):
            raise ValueError("profit_targets must be in ascending order")
            
        # Ensure exit portions are valid
        if not all(0 < portion <= 1 for portion in exit_portions):
            raise ValueError("exit_portions must be between 0 and 1")
            
        # Ensure total exit portion doesn't exceed 1
        if sum(exit_portions) > 1:
            raise ValueError("sum of exit_portions cannot exceed 1")
        
        self.profit_targets = profit_targets
        self.exit_portions = exit_portions
        self.reached_targets = [False] * len(profit_targets)
        
        self.logger.info(f"Initialized Profit Target Ladder with targets: {profit_targets} "
                       f"and portions: {[f'{p:.1%}' for p in exit_portions]}")
    
    def should_exit(self, 
                  entry_price: float, 
                  current_price: float,
                  **kwargs) -> Tuple[bool, str]:
        """
        Check if any profit target has been reached
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        market_direction = kwargs.get('market_direction', 'long')
        
        # Calculate profit percentage
        if market_direction == 'long':
            profit_pct = (current_price / entry_price) - 1
        else:  # short
            profit_pct = 1 - (current_price / entry_price)
        
        # Check each profit target
        for i, target in enumerate(self.profit_targets):
            if not self.reached_targets[i] and profit_pct >= target:
                self.reached_targets[i] = True
                portion = self.exit_portions[i]
                
                return True, f"Profit target {i+1} reached: {profit_pct:.1%} >= {target:.1%}, exiting {portion:.1%} of position"
        
        return False, "No profit target reached"
    
    def get_exit_portion(self, 
                       entry_price: float, 
                       current_price: float,
                       **kwargs) -> float:
        """
        Get the portion of the position to exit
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            **kwargs: Additional parameters
                - market_direction (str): 'long' or 'short'
                
        Returns:
            float: Portion of the position to exit (0.0-1.0)
        """
        should_exit, reason = self.should_exit(entry_price, current_price, **kwargs)
        
        if should_exit:
            for i, reached in enumerate(self.reached_targets):
                if reached and i == self.reached_targets.index(True):
                    return self.exit_portions[i]
        
        return 0.0
    
    def reset(self) -> None:
        """Reset reached targets"""
        self.reached_targets = [False] * len(self.profit_targets)
        self.logger.info("Reset profit target ladder")


# Update the factory function to include new strategies
def get_stop_loss_strategy(method: str, **kwargs) -> StopLossStrategy:
    """
    Factory function to get a stop loss strategy by name
    
    Args:
        method (str): Stop loss method
                     ('fixed', 'trailing', 'atr', 'time', 'indicator',
                      'composite', 'channel', 'volatility', 'profit_ladder')
        **kwargs: Additional parameters for the stop loss strategy
        
    Returns:
        StopLossStrategy: Stop loss strategy instance
    """
    method = method.lower()
    
    if method == 'fixed':
        return FixedPercentageStop(
            stop_loss_pct=kwargs.get('stop_loss_pct', 0.05),
            take_profit_pct=kwargs.get('take_profit_pct')
        )
    elif method == 'trailing':
        return TrailingStop(
            initial_stop_pct=kwargs.get('initial_stop_pct', 0.05),
            trail_pct=kwargs.get('trail_pct', 0.03)
        )
    elif method == 'atr':
        return ATRStop(
            atr_multiplier=kwargs.get('atr_multiplier', 3.0),
            take_profit_multiplier=kwargs.get('take_profit_multiplier')
        )
    elif method == 'time':
        return TimeBasedStop(
            max_hold_time_hours=kwargs.get('max_hold_time_hours', 48.0),
            min_profit_pct=kwargs.get('min_profit_pct', 0.0)
        )
    elif method == 'indicator':
        return IndicatorBasedStop(
            indicator_name=kwargs.get('indicator_name', 'rsi')
        )
    elif method == 'composite':
        # Create component strategies
        strategies = []
        for strategy_config in kwargs.get('strategies', []):
            if isinstance(strategy_config, dict) and 'method' in strategy_config:
                sub_method = strategy_config.pop('method')
                strategies.append(get_stop_loss_strategy(sub_method, **strategy_config))
            elif isinstance(strategy_config, StopLossStrategy):
                strategies.append(strategy_config)
        
        return CompositeStopLoss(strategies=strategies or [])
    elif method == 'channel':
        return ChannelBasedStop(
            channel_type=kwargs.get('channel_type', 'donchian'),
            period=kwargs.get('period', 20),
            buffer_pct=kwargs.get('buffer_pct', 0.01)
        )
    elif method == 'volatility':
        return VolatilityExpansionStop(
            volatility_period=kwargs.get('volatility_period', 10),
            volatility_multiplier=kwargs.get('volatility_multiplier', 2.0),
            lookback_period=kwargs.get('lookback_period', 3)
        )
    elif method == 'profit_ladder':
        return ProfitTargetLadder(
            profit_targets=kwargs.get('profit_targets', [0.05, 0.1, 0.2]),
            exit_portions=kwargs.get('exit_portions', [0.3, 0.3, 0.4])
        )
    else:
        logger.warning(f"Unknown stop loss method '{method}', defaulting to fixed percentage")
        return FixedPercentageStop(stop_loss_pct=kwargs.get('stop_loss_pct', 0.05)) 