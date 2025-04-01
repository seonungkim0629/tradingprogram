"""
Position Sizing Module for Bitcoin Trading Bot

This module provides position sizing strategies to manage risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import math

from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class PositionSizer:
    """Base class for position sizing strategies"""
    
    def __init__(self, name: str):
        """
        Initialize position sizer
        
        Args:
            name (str): Name of the position sizing strategy
        """
        self.name = name
        self.logger = logger
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
            
        Returns:
            float: Position size in base currency
        """
        raise NotImplementedError("Subclasses must implement calculate_position_size")


class FixedPercentage(PositionSizer):
    """Fixed percentage of balance position sizing"""
    
    def __init__(self, percentage: float = 0.1):
        """
        Initialize fixed percentage position sizer
        
        Args:
            percentage (float): Percentage of balance to use (0.0-1.0)
        """
        super().__init__("Fixed Percentage")
        self.percentage = max(0.0, min(1.0, percentage))
        self.logger.info(f"Initialized Fixed Percentage position sizer with {self.percentage:.1%}")
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size as a fixed percentage of available balance
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
                - confidence (float): Signal confidence (0.0-1.0) to adjust position size
                
        Returns:
            float: Position size in base currency
        """
        # Adjust percentage by confidence if provided
        confidence = kwargs.get('confidence', 1.0)
        adjusted_percentage = self.percentage * confidence
        
        position_size = available_balance * adjusted_percentage
        
        self.logger.info(f"Calculated position size: {position_size:.2f} "
                       f"({adjusted_percentage:.1%} of {available_balance:.2f})")
        
        return position_size


class KellyFormula(PositionSizer):
    """Kelly formula position sizing"""
    
    def __init__(self, 
                win_rate: float = 0.5,
                win_loss_ratio: float = 2.0,
                max_percentage: float = 0.25):
        """
        Initialize Kelly formula position sizer
        
        Args:
            win_rate (float): Winning rate (0.0-1.0)
            win_loss_ratio (float): Average win/loss ratio
            max_percentage (float): Maximum percentage of balance to risk
        """
        super().__init__("Kelly Formula")
        self.win_rate = max(0.0, min(1.0, win_rate))
        self.win_loss_ratio = max(0.1, win_loss_ratio)
        self.max_percentage = max(0.0, min(1.0, max_percentage))
        
        self.logger.info(f"Initialized Kelly Formula position sizer with "
                       f"win_rate={self.win_rate:.2f}, "
                       f"win_loss_ratio={self.win_loss_ratio:.2f}, "
                       f"max_percentage={self.max_percentage:.1%}")
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size using Kelly formula
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
                - win_rate (float): Override win rate
                - win_loss_ratio (float): Override win/loss ratio
                - confidence (float): Signal confidence to adjust position size
                
        Returns:
            float: Position size in base currency
        """
        # Override parameters if provided
        win_rate = kwargs.get('win_rate', self.win_rate)
        win_loss_ratio = kwargs.get('win_loss_ratio', self.win_loss_ratio)
        confidence = kwargs.get('confidence', 1.0)
        
        # Calculate Kelly percentage
        # f* = (p*b - q) / b where p = win rate, q = 1-p, b = win/loss ratio
        q = 1 - win_rate
        kelly_percentage = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Cap at maximum percentage and ensure non-negative
        kelly_percentage = max(0.0, min(self.max_percentage, kelly_percentage))
        
        # Adjust by confidence
        adjusted_percentage = kelly_percentage * confidence
        
        position_size = available_balance * adjusted_percentage
        
        self.logger.info(f"Calculated Kelly position size: {position_size:.2f} "
                       f"({adjusted_percentage:.1%} of {available_balance:.2f})")
        
        return position_size


class ATRBasedPositionSizer(PositionSizer):
    """Position sizing based on Average True Range (ATR)"""
    
    def __init__(self, 
                risk_percentage: float = 0.01,
                atr_multiplier: float = 2.0,
                max_percentage: float = 0.25):
        """
        Initialize ATR-based position sizer
        
        Args:
            risk_percentage (float): Percentage of balance to risk per trade
            atr_multiplier (float): Multiplier for ATR to set stop loss distance
            max_percentage (float): Maximum percentage of balance to use
        """
        super().__init__("ATR-Based")
        self.risk_percentage = max(0.0, min(0.05, risk_percentage))
        self.atr_multiplier = max(0.5, atr_multiplier)
        self.max_percentage = max(0.0, min(1.0, max_percentage))
        
        self.logger.info(f"Initialized ATR-Based position sizer with "
                       f"risk_percentage={self.risk_percentage:.1%}, "
                       f"atr_multiplier={self.atr_multiplier:.1f}, "
                       f"max_percentage={self.max_percentage:.1%}")
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size based on ATR
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
                - atr (float): ATR value
                - risk_percentage (float): Override risk percentage
                - atr_multiplier (float): Override ATR multiplier
                - confidence (float): Signal confidence to adjust position size
                
        Returns:
            float: Position size in base currency
        """
        # Get ATR value
        atr = kwargs.get('atr')
        if atr is None:
            self.logger.warning("ATR value missing, using fixed percentage fallback")
            return FixedPercentage(self.max_percentage).calculate_position_size(
                available_balance, current_price, **kwargs)
        
        # Override parameters if provided
        risk_percentage = kwargs.get('risk_percentage', self.risk_percentage)
        atr_multiplier = kwargs.get('atr_multiplier', self.atr_multiplier)
        confidence = kwargs.get('confidence', 1.0)
        
        # Calculate stop loss distance in currency
        stop_loss_distance = atr * atr_multiplier
        
        # Calculate position size
        risk_amount = available_balance * risk_percentage
        position_size = risk_amount / (stop_loss_distance / current_price)
        
        # Cap at maximum percentage
        max_position_size = available_balance * self.max_percentage
        position_size = min(position_size, max_position_size)
        
        # Adjust by confidence
        position_size *= confidence
        
        self.logger.info(f"Calculated ATR-based position size: {position_size:.2f} "
                       f"(ATR: {atr:.2f}, stop distance: {stop_loss_distance:.2f})")
        
        return position_size


class VolatilityAdjustedPositionSizer(PositionSizer):
    """Position sizing adjusted by market volatility"""
    
    def __init__(self, 
                base_percentage: float = 0.2,
                volatility_period: int = 20,
                reference_volatility: float = 0.03,
                max_percentage: float = 0.5):
        """
        Initialize volatility-adjusted position sizer
        
        Args:
            base_percentage (float): Base percentage of balance to use
            volatility_period (int): Period for calculating volatility
            reference_volatility (float): Reference volatility level
            max_percentage (float): Maximum percentage of balance to use
        """
        super().__init__("Volatility-Adjusted")
        self.base_percentage = max(0.0, min(1.0, base_percentage))
        self.volatility_period = max(5, volatility_period)
        self.reference_volatility = max(0.001, reference_volatility)
        self.max_percentage = max(0.0, min(1.0, max_percentage))
        
        self.logger.info(f"Initialized Volatility-Adjusted position sizer with "
                       f"base_percentage={self.base_percentage:.1%}, "
                       f"reference_volatility={self.reference_volatility:.1%}")
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size adjusted by volatility
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
                - volatility (float): Current volatility (std dev of returns)
                - returns (List[float]): List of recent returns to calculate volatility
                - confidence (float): Signal confidence to adjust position size
                
        Returns:
            float: Position size in base currency
        """
        # Get volatility
        volatility = kwargs.get('volatility')
        returns = kwargs.get('returns')
        
        if volatility is None and returns is not None:
            # Calculate volatility from returns
            volatility = np.std(returns, ddof=1)
        
        if volatility is None:
            self.logger.warning("Volatility missing, using fixed percentage fallback")
            return FixedPercentage(self.base_percentage).calculate_position_size(
                available_balance, current_price, **kwargs)
        
        # Get confidence factor
        confidence = kwargs.get('confidence', 1.0)
        
        # Calculate volatility ratio
        volatility_ratio = self.reference_volatility / max(0.0001, volatility)
        
        # Adjust position size based on volatility ratio
        # Lower volatility -> larger position, higher volatility -> smaller position
        adjusted_percentage = self.base_percentage * volatility_ratio * confidence
        
        # Cap at maximum percentage
        adjusted_percentage = min(adjusted_percentage, self.max_percentage)
        
        position_size = available_balance * adjusted_percentage
        
        self.logger.info(f"Calculated volatility-adjusted position size: {position_size:.2f} "
                       f"(volatility: {volatility:.2%}, adjustment: {volatility_ratio:.2f})")
        
        return position_size


class RiskOfRuin(PositionSizer):
    """Position sizing based on risk of ruin equation"""
    
    def __init__(self, 
                win_rate: float = 0.5,
                loss_percentage: float = 0.02,
                risk_of_ruin_target: float = 0.05,
                max_percentage: float = 0.25):
        """
        Initialize risk of ruin position sizer
        
        Args:
            win_rate (float): Winning rate (0.0-1.0)
            loss_percentage (float): Percentage of balance to risk per trade
            risk_of_ruin_target (float): Target risk of ruin probability
            max_percentage (float): Maximum percentage of balance to use
        """
        super().__init__("Risk of Ruin")
        self.win_rate = max(0.0, min(1.0, win_rate))
        self.loss_percentage = max(0.0, min(0.1, loss_percentage))
        self.risk_of_ruin_target = max(0.0001, min(0.2, risk_of_ruin_target))
        self.max_percentage = max(0.0, min(1.0, max_percentage))
        
        self.logger.info(f"Initialized Risk of Ruin position sizer with "
                       f"win_rate={self.win_rate:.2f}, "
                       f"loss_percentage={self.loss_percentage:.1%}, "
                       f"risk_of_ruin_target={self.risk_of_ruin_target:.1%}")
    
    def calculate_position_size(self, 
                              available_balance: float, 
                              current_price: float,
                              **kwargs) -> float:
        """
        Calculate position size based on risk of ruin
        
        Args:
            available_balance (float): Available balance in base currency
            current_price (float): Current price of the asset
            **kwargs: Additional parameters
                - win_rate (float): Override win rate
                - loss_percentage (float): Override loss percentage
                - confidence (float): Signal confidence to adjust position size
                
        Returns:
            float: Position size in base currency
        """
        # Override parameters if provided
        win_rate = kwargs.get('win_rate', self.win_rate)
        loss_percentage = kwargs.get('loss_percentage', self.loss_percentage)
        confidence = kwargs.get('confidence', 1.0)
        
        # Calculate risk-adjusted position size
        if win_rate <= 0.5:
            # Higher risk scenario, use more conservative sizing
            adjusted_percentage = loss_percentage * 0.5
        else:
            # For win rates > 0.5, calculate safe trading size
            # This is an approximation of solving for f in the risk of ruin equation:
            # R = ((1-E)/E)^n where E is expected value and n is drawdown capacity
            expected_value = (win_rate * 2) - 1  # Simplified EV calculation
            drawdown_capacity = math.log(self.risk_of_ruin_target) / math.log((1 - expected_value) / (1 + expected_value))
            adjusted_percentage = min(loss_percentage, 0.2 / drawdown_capacity)
        
        # Apply confidence adjustment
        adjusted_percentage *= confidence
        
        # Cap at maximum percentage
        adjusted_percentage = min(adjusted_percentage, self.max_percentage)
        
        position_size = available_balance * adjusted_percentage
        
        self.logger.info(f"Calculated risk of ruin position size: {position_size:.2f} "
                       f"({adjusted_percentage:.1%} of {available_balance:.2f})")
        
        return position_size


def get_position_sizer(method: str, **kwargs) -> PositionSizer:
    """
    Factory function to get a position sizer by name
    
    Args:
        method (str): Position sizing method 
                     ('fixed', 'kelly', 'atr', 'volatility', 'risk_of_ruin')
        **kwargs: Additional parameters for the position sizer
        
    Returns:
        PositionSizer: Position sizer instance
    """
    method = method.lower()
    
    if method == 'fixed':
        return FixedPercentage(percentage=kwargs.get('percentage', 0.1))
    elif method == 'kelly':
        return KellyFormula(
            win_rate=kwargs.get('win_rate', 0.5),
            win_loss_ratio=kwargs.get('win_loss_ratio', 2.0),
            max_percentage=kwargs.get('max_percentage', 0.25)
        )
    elif method == 'atr':
        return ATRBasedPositionSizer(
            risk_percentage=kwargs.get('risk_percentage', 0.01),
            atr_multiplier=kwargs.get('atr_multiplier', 2.0),
            max_percentage=kwargs.get('max_percentage', 0.25)
        )
    elif method == 'volatility':
        return VolatilityAdjustedPositionSizer(
            base_percentage=kwargs.get('base_percentage', 0.2),
            volatility_period=kwargs.get('volatility_period', 20),
            reference_volatility=kwargs.get('reference_volatility', 0.03),
            max_percentage=kwargs.get('max_percentage', 0.5)
        )
    elif method == 'risk_of_ruin':
        return RiskOfRuin(
            win_rate=kwargs.get('win_rate', 0.5),
            loss_percentage=kwargs.get('loss_percentage', 0.02),
            risk_of_ruin_target=kwargs.get('risk_of_ruin_target', 0.05),
            max_percentage=kwargs.get('max_percentage', 0.25)
        )
    else:
        logger.warning(f"Unknown position sizing method '{method}', defaulting to fixed percentage")
        return FixedPercentage(percentage=kwargs.get('percentage', 0.1)) 