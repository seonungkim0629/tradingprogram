"""
Base Strategy Abstract Class for Bitcoin Trading Bot

This module provides an abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import json
import logging
import pandas as pd
# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/strategies.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                market: str = "KRW-BTC",
                name: Optional[str] = None, 
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
        """
        self.market = market
        self.name = name or self.__class__.__name__
        self.parameters = parameters or {}
        self.logger = logger
        self.hourly_data = None  # 시간봉 데이터를 저장할 속성 추가
        
        self.logger.info(f"Initialized {self.name} strategy for {market}")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on market data
        
        Args:
            data (pd.DataFrame): Market data with OHLCV and indicators
            
        Returns:
            Dict[str, Any]: Signal dictionary with keys:
                - signal (str): 'BUY', 'SELL', or 'HOLD'
                - reason (str): Reason for the signal
                - confidence (float): Signal confidence (0.0-1.0)
                - metadata (Dict): Additional signal metadata
        """
        pass
    
    def set_hourly_data(self, hourly_data: pd.DataFrame) -> None:
        """
        시간봉 데이터 설정
        
        Args:
            hourly_data (pd.DataFrame): 시간봉 마켓 데이터
        """
        self.hourly_data = hourly_data
        self.logger.info(f"{self.name} 전략에 {len(hourly_data)}개의 시간봉 데이터 설정 완료")
    
    def get_hourly_data(self) -> Optional[pd.DataFrame]:
        """
        시간봉 데이터 반환
        
        Returns:
            Optional[pd.DataFrame]: 시간봉 데이터 또는 없을 경우 None
        """
        return self.hourly_data
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on market data
        
        Args:
            data (pd.DataFrame): Market data with OHLCV and indicators
            
        Returns:
            Dict[str, Any]: Signal dictionary with keys:
                - signal (str): 'BUY', 'SELL', or 'HOLD'
                - reason (str): Reason for the signal
                - confidence (float): Signal confidence (0.0-1.0)
                - metadata (Dict): Additional signal metadata
        """
        pass
    
    def apply_risk_management(self, 
                            signal: Dict[str, Any], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply risk management to the generated signal
        
        Args:
            signal (Dict[str, Any]): Generated signal
            portfolio (Dict[str, Any]): Current portfolio state
            
        Returns:
            Dict[str, Any]: Modified signal with risk management applied
        """
        # Default implementation just passes through the signal
        # Subclasses can override to implement risk management
        return signal
    
    def calculate_position_size(self, 
                               signal: Dict[str, Any], 
                               available_balance: float) -> float:
        """
        Calculate position size for a trade
        
        Args:
            signal (Dict[str, Any]): Generated signal
            available_balance (float): Available balance for trading
            
        Returns:
            float: Position size in base currency
        """
        # Default implementation: use position sizing from signal or full balance
        position_size = signal.get("position_size", 1.0)  # Default to 100%
        
        # If position_size is a percentage (0.0-1.0), convert to absolute value
        if 0.0 <= position_size <= 1.0:
            position_size = available_balance * position_size
        
        return min(position_size, available_balance)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
        """
        self.parameters.update(parameters)
        self.logger.info(f"Updated {self.name} parameters: {parameters}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters
        
        Returns:
            Dict[str, Any]: Strategy parameters
        """
        return self.parameters.copy()
    
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization
        
        Returns:
            Dict[str, List[Any]]: Parameter grid for grid search
        """
        # Default implementation returns empty grid
        # Subclasses should override this method with their specific parameters
        return {}
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save strategy configuration to file
        
        Args:
            filepath (Optional[str]): File path to save to
            
        Returns:
            str: Path where the strategy was saved
        """
        if filepath is None:
            # Create default filepath
            strategy_dir = "strategies/saved"
            os.makedirs(strategy_dir, exist_ok=True)
            filepath = os.path.join(strategy_dir, f"{self.name}.json")
        
        # Create strategy config
        config = {
            "name": self.name,
            "class": self.__class__.__name__,
            "market": self.market,
            "parameters": self.parameters
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved {self.name} strategy to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'BaseStrategy':
        """
        Load strategy from file
        
        Args:
            filepath (str): Path to strategy configuration file
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            BaseStrategy: Loaded strategy instance
        """
        # Load configuration
        with open(filepath, "r") as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            market=config.get("market", "KRW-BTC"),
            name=config.get("name"),
            parameters=config.get("parameters", {}),
            **kwargs
        )
        
        logger.info(f"Loaded {instance.name} strategy from {filepath}")
        return instance 