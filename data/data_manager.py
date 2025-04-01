"""
Data Manager for Live Trading

This module provides functionality to manage data for live trading,
combining daily and hourly data for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import os

from config import settings
from utils.logging import get_logger, log_execution
from data.collectors import upbit_collector, get_historical_data
from data.indicators import add_indicators
from data.storage import stack_hourly_data, save_ohlcv_data

# Initialize logger
logger = get_logger(__name__)

class TradingDataManager:
    """Data manager for live trading that handles daily and hourly data"""
    
    def __init__(self, ticker: str = None):
        """
        Initialize trading data manager
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        """
        self.ticker = ticker or settings.DEFAULT_MARKET
        self.daily_data = None
        self.hourly_data = None
        self.combined_signals = {}
        self.last_update_daily = None
        self.last_update_hourly = None
        
        logger.info(f"Initialized trading data manager for {self.ticker}")
    
    @log_execution
    def load_initial_data(self) -> bool:
        """
        Load initial daily and hourly data
        
        Returns:
            bool: Success status
        """
        try:
            # Load daily data (100 days should be enough for most strategies)
            self.daily_data = get_historical_data(ticker=self.ticker, days=100)
            if self.daily_data is None:
                logger.error(f"Failed to load daily data for {self.ticker}")
                return False
            
            # Load hourly data (last 500 hours ~ 3 weeks)
            self.hourly_data = upbit_collector.get_hourly_ohlcv(ticker=self.ticker, count=500)
            if self.hourly_data is None:
                logger.error(f"Failed to load hourly data for {self.ticker}")
                return False
            
            # Add indicators
            self.daily_data = add_indicators(self.daily_data)
            self.hourly_data = add_indicators(self.hourly_data)
            
            # Save the data
            save_ohlcv_data(self.daily_data, self.ticker, 'day')
            hourly_file = stack_hourly_data(self.ticker, self.hourly_data)
            
            # Update last update time
            self.last_update_daily = datetime.now()
            self.last_update_hourly = datetime.now()
            
            logger.info(f"Loaded initial data: {len(self.daily_data)} daily candles, {len(self.hourly_data)} hourly candles")
            return True
            
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")
            return False
    
    @log_execution
    def update_data(self, force_daily: bool = False) -> Tuple[bool, bool]:
        """
        Update daily and hourly data based on time passed
        
        Args:
            force_daily (bool): Force update of daily data regardless of time
            
        Returns:
            Tuple[bool, bool]: (daily_updated, hourly_updated)
        """
        now = datetime.now()
        daily_updated = False
        hourly_updated = False
        
        # Update daily data if a day has passed or forced
        if (force_daily or 
            self.last_update_daily is None or 
            now.date() != self.last_update_daily.date()):
            try:
                logger.info("Updating daily data")
                new_daily = upbit_collector.get_ohlcv(ticker=self.ticker, interval="day", count=5)
                
                if new_daily is not None and not new_daily.empty:
                    # Combine with existing data
                    combined = pd.concat([self.daily_data, new_daily])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    
                    # Keep last 100 days for efficiency
                    self.daily_data = combined.tail(100).copy()
                    
                    # Add indicators
                    self.daily_data = add_indicators(self.daily_data)
                    
                    # Save the updated data
                    save_ohlcv_data(self.daily_data, self.ticker, 'day')
                    
                    self.last_update_daily = now
                    daily_updated = True
                    logger.info(f"Daily data updated, now have {len(self.daily_data)} candles")
            except Exception as e:
                logger.error(f"Error updating daily data: {str(e)}")
        
        # Update hourly data every hour or if None
        if (self.last_update_hourly is None or 
            (now - self.last_update_hourly).seconds >= 3600):
            try:
                logger.info("Updating hourly data")
                new_hourly = upbit_collector.get_ohlcv(ticker=self.ticker, interval="minute60", count=24)
                
                if new_hourly is not None and not new_hourly.empty:
                    # Combine with existing data
                    combined = pd.concat([self.hourly_data, new_hourly])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    
                    # Keep last 500 hours for efficiency
                    self.hourly_data = combined.tail(500).copy()
                    
                    # Add indicators
                    self.hourly_data = add_indicators(self.hourly_data)
                    
                    # Stack hourly data
                    stack_hourly_data(self.ticker, new_hourly)
                    
                    self.last_update_hourly = now
                    hourly_updated = True
                    logger.info(f"Hourly data updated, now have {len(self.hourly_data)} candles")
            except Exception as e:
                logger.error(f"Error updating hourly data: {str(e)}")
        
        return daily_updated, hourly_updated
    
    @log_execution
    def get_current_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get current data for strategy generation
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with 'daily' and 'hourly' DataFrames
        """
        # Make sure data is up to date
        self.update_data()
        
        return {
            'daily': self.daily_data.copy() if self.daily_data is not None else None,
            'hourly': self.hourly_data.copy() if self.hourly_data is not None else None
        }
    
    @log_execution
    def get_latest_price(self) -> float:
        """
        Get latest price
        
        Returns:
            float: Latest price or 0 if error
        """
        try:
            current_price = upbit_collector.get_current_price(self.ticker)
            return current_price or 0
        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}")
            return 0