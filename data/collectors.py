"""
Data collection module for Bitcoin Trading Bot

This module provides functionality to collect market data from Upbit exchange API.
"""

import os
import time
import json
import traceback
import numpy as np
import pandas as pd
import requests
import pyupbit
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
import concurrent.futures
from dateutil.relativedelta import relativedelta

# 설정 및 유틸리티 모듈
from config import settings
from utils.logging import get_logger, log_execution
from utils.monitoring import track_api_call

# 데이터 관련 모듈
from data.indicators import calculate_all_indicators as add_indicators
from models.base import ClassificationModel

# 로거 설정
logger = get_logger(__name__)

class UpbitDataCollector:
    """Data collector for Upbit exchange"""
    
    def __init__(self, access_key: str = None, secret_key: str = None):
        """
        Initialize Upbit data collector
        
        Args:
            access_key (str, optional): Upbit API access key. Defaults to settings.UPBIT_ACCESS_KEY.
            secret_key (str, optional): Upbit API secret key. Defaults to settings.UPBIT_SECRET_KEY.
        """
        self.access_key = access_key or settings.UPBIT_ACCESS_KEY
        self.secret_key = secret_key or settings.UPBIT_SECRET_KEY
        self.upbit = None
        
        # 타임프레임별 API 엔드포인트 매핑
        self._interval_mappings = {
            'minute1': 'minute1', 
            'minute3': 'minute3', 
            'minute5': 'minute5', 
            'minute10': 'minute10',
            'minute15': 'minute15', 
            'minute30': 'minute30', 
            'minute60': 'minute60', 
            'minute240': 'minute240',
            'day': 'day',
            'week': 'week',
            'month': 'month'
        }
        
        # API 호출 간격 제한
        self._api_call_intervals = {
            'day': 0.2,        # 일봉 데이터는 빠르게 가져올 수 있음
            'minute60': 0.2,   # 시간봉
            'minute5': 0.3,    # 5분봉
            'minute1': 0.3,    # 1분봉
            'default': 0.1     # 기타
        }
        
        # 데이터 저장 경로 정의
        self.data_dir = settings.DATA_DIR
        self.ohlcv_dir = settings.OHLCV_DIR
        
        if self.access_key and self.secret_key:
            try:
                self.upbit = pyupbit.Upbit(self.access_key, self.secret_key)
                logger.info("Upbit API authenticated successfully")
            except Exception as e:
                logger.error(f"Error authenticating with Upbit API: {str(e)}")
                self.upbit = None
    
    @log_execution
    def get_tickers(self, fiat: str = "KRW") -> List[str]:
        """
        Get list of all available tickers
        
        Args:
            fiat (str, optional): Fiat currency to filter by. Defaults to "KRW".
            
        Returns:
            List[str]: List of ticker symbols
        """
        try:
            track_api_call("upbit", "ticker_list")
            tickers = pyupbit.get_tickers(fiat=fiat)
            return tickers
        except Exception as e:
            logger.error(f"Error getting tickers: {str(e)}")
            return []
    
    @log_execution
    def get_current_price(self, ticker: str = None) -> Union[float, Dict[str, float], None]:
        """
        Get current price for a ticker or all tickers
        
        Args:
            ticker (str, optional): Ticker symbol. If None, returns all prices. Defaults to None.
            
        Returns:
            Union[float, Dict[str, float], None]: Current price or dict of prices
        """
        try:
            track_api_call("upbit", "current_price")
            ticker = ticker or settings.DEFAULT_MARKET
            price = pyupbit.get_current_price(ticker)
            return price
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return None
    
    @log_execution
    def get_ohlcv(self, ticker: str = None, interval: str = "day", count: int = 200, to: str = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from Upbit API
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            interval (str, optional): Time interval ('day', 'minute1', 'minute3', 'minute5', etc). 
                                      Defaults to "day".
            count (int, optional): Number of candles to retrieve (max 200). Defaults to 200.
            to (str, optional): End date in format 'YYYY-MM-DD HH:MM:SS'. Defaults to None.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing OHLCV data or None if error
        """
        try:
            # 기본값 설정
            ticker = ticker or settings.DEFAULT_MARKET
            
            # 지원되는 타임프레임인지 확인
            if interval not in self._interval_mappings:
                logger.warning(f"Unsupported interval: {interval}. Using 'day' instead.")
                interval = 'day'
                
            # API 호출 시작 시간 기록
            start_time = time.time()
                
            # API 요청 추적
            track_api_call("upbit", f"ohlcv_{interval}")
            
            # API 요청 수행
            if to:
                df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count, to=to)
            else:
                df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)
            
            # API 호출 결과 확인
            if df is None or df.empty:
                logger.warning(f"No OHLCV data returned for {ticker} ({interval})")
                return None
            
            # 필요한 메타데이터 추가
            df['ticker'] = ticker
            
            # 컬럼명 일관성 확인 및 필요시 조정
            expected_columns = ['open', 'high', 'low', 'close', 'volume', 'value']
            for col in expected_columns:
                if col not in df.columns and col != 'value':  # value는 없을 수 있음
                    logger.warning(f"Missing column '{col}' in OHLCV data for {ticker}")
            
            # API 호출 간격 유지 (레이트 리밋 방지)
            elapsed = time.time() - start_time
            wait_time = self._api_call_intervals.get(interval, self._api_call_intervals['default']) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {ticker} ({interval}): {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    @log_execution
    def get_orderbook(self, ticker: str = None) -> Optional[pd.DataFrame]:
        """
        Get current orderbook
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing orderbook or None if error
        """
        try:
            ticker = ticker or settings.DEFAULT_MARKET
            track_api_call("upbit", "orderbook")
            
            orderbook = pyupbit.get_orderbook(ticker)
            if not orderbook:
                logger.warning(f"No orderbook data returned for {ticker}")
                return None
            
            # Convert to DataFrame for easier manipulation
            df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
            df_orderbook['timestamp'] = pd.Timestamp.now()
            df_orderbook['ticker'] = ticker
            
            return df_orderbook
        
        except Exception as e:
            logger.error(f"Error getting orderbook for {ticker}: {str(e)}")
            return None
    
    @log_execution
    def get_daily_ohlcv(self, ticker: str = None, since: str = None, to: str = None) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV data for an extended period by making multiple API calls
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            since (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1000 days ago.
            to (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing daily OHLCV data or None if error
        """
        ticker = ticker or settings.DEFAULT_MARKET
        
        # Set default dates if not provided
        if not to:
            to = datetime.now().strftime('%Y-%m-%d')
        
        if not since:
            since_date = datetime.now() - timedelta(days=1000)  # Maximum allowed by Upbit
            since = since_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching daily OHLCV data for {ticker} from {since} to {to}")
        
        try:
            # Convert dates to datetime objects for comparison
            since_dt = datetime.strptime(since, '%Y-%m-%d')
            to_dt = datetime.strptime(to, '%Y-%m-%d')
            
            # Calculate days between dates
            days_difference = (to_dt - since_dt).days
            
            # Check if dates are valid
            if days_difference <= 0:
                logger.error(f"Invalid date range: {since} to {to}")
                return None
            
            # If the range is less than 200 days, we can get it in one call
            if days_difference <= 200:
                return self.get_ohlcv(ticker=ticker, interval="day", count=days_difference, to=to)
            
            # Otherwise, we need to make multiple calls
            all_df = []
            
            # Calculate number of iterations (200 days per call)
            iterations = (days_difference // 200) + (1 if days_difference % 200 > 0 else 0)
            
            current_to = to_dt
            
            for i in range(iterations):
                # Format current_to as string
                current_to_str = current_to.strftime('%Y-%m-%d %H:%M:%S')
                
                # Get data for this interval
                curr_count = min(200, (current_to - since_dt).days)
                logger.debug(f"Getting daily data batch {i+1}/{iterations}, to={current_to_str}, count={curr_count}")
                
                df = self.get_ohlcv(ticker=ticker, interval="day", count=curr_count, to=current_to_str)
                
                if df is not None and not df.empty:
                    all_df.append(df)
                    
                    # Update current_to to be one day before the earliest date in df
                    if len(df) > 0:
                        earliest_date = df.index.min()
                        current_to = earliest_date.to_pydatetime() - timedelta(days=1)
                    else:
                        # Not enough data, break the loop
                        break
                else:
                    logger.warning(f"No data returned for interval {i+1}/{iterations}")
                    break
                
                # Check if we've reached or passed the start date
                if current_to <= since_dt:
                    break
                
                # Add delay between API calls to avoid rate limiting
                time.sleep(0.5)
            
            # Combine all data frames
            if not all_df:
                logger.warning("No data collected for any interval")
                return None
                
            combined_df = pd.concat(all_df)
            
            # Remove duplicates and sort by date
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()
            
            # Filter to match the original date range
            mask = (combined_df.index >= since_dt) & (combined_df.index <= to_dt)
            combined_df = combined_df.loc[mask]
            
            logger.info(f"Collected {len(combined_df)} days of OHLCV data for {ticker}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error getting daily OHLCV data for {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    @log_execution
    def get_extended_timeframe_data(self, ticker: str = None, interval: str = "minute60", count: int = 200,
                                   max_iterations: int = 10) -> Optional[pd.DataFrame]:
        """
        특정 타임프레임의 데이터를 여러 API 호출을 통해 더 많이 가져옴
        
        Args:
            ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
            interval (str, optional): 시간 간격. Defaults to "minute60".
            count (int, optional): 각 API 호출당 가져올 캔들 수 (최대 200). Defaults to 200.
            max_iterations (int, optional): 최대 API 호출 횟수. Defaults to 10.
            
        Returns:
            Optional[pd.DataFrame]: OHLCV 데이터를 포함하는 DataFrame 또는 오류 시 None
        """
        ticker = ticker or settings.DEFAULT_MARKET
        interval_map = {
            'hour': 'minute60',
            'minute60': 'minute60',
            'minute5': 'minute5',
            'minute1': 'minute1'
        }
        
        # 간격 매핑 확인 및 변환
        if interval in interval_map:
            interval = interval_map[interval]
        elif interval not in self._interval_mappings:
            logger.warning(f"Unsupported interval: {interval}. Using 'minute60' instead.")
            interval = 'minute60'
            
        logger.info(f"Fetching extended {interval} data for {ticker}, max {count*max_iterations} candles")
        
        all_df = []
        current_to = None
        
        try:
            for i in range(max_iterations):
                # Get data for this batch
                df = self.get_ohlcv(ticker=ticker, interval=interval, count=count, to=current_to)
                
                if df is not None and not df.empty:
                    all_df.append(df)
                    
                    # Update 'to' parameter to the earliest timestamp minus one minute
                    earliest_time = df.index.min()
                    current_to = (earliest_time - pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
                    
                    logger.debug(f"Batch {i+1}/{max_iterations}: Got {len(df)} candles, next to={current_to}")
                else:
                    logger.warning(f"No data returned for batch {i+1}/{max_iterations}")
                    break
                
                # Delay between API calls
                time.sleep(self._api_call_intervals.get(interval, 0.2))
                
                # If we got less than requested, we've reached the limit
                if len(df) < count:
                    logger.debug(f"Got {len(df)} < {count} candles, reached the limit")
                    break
            
            # Combine all data frames
            if not all_df:
                logger.warning(f"No {interval} data collected for {ticker}")
                return None
                
            combined_df = pd.concat(all_df)
            
            # Remove duplicates and sort by date
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()
            
            logger.info(f"Collected {len(combined_df)} {interval} candles for {ticker}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error getting extended {interval} data for {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    @log_execution
    def get_hourly_ohlcv(self, ticker: str = None, count: int = 2000) -> Optional[pd.DataFrame]:
        """
        시간봉 데이터 가져오기 (최대 2000개)
        
        Args:
            ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
            count (int, optional): 가져올 캔들 수. Defaults to 2000.
            
        Returns:
            Optional[pd.DataFrame]: 시간봉 데이터를 포함하는 DataFrame 또는 오류 시 None
        """
        # 최대 API 호출 횟수 계산 (200개씩 가져올 때)
        max_iterations = min(10, (count + 199) // 200)  # 최대 10회 API 호출 (2000개)
        
        return self.get_extended_timeframe_data(
            ticker=ticker,
            interval="minute60",
            count=min(200, count),  # 한 번에 최대 200개 가능
            max_iterations=max_iterations
        )
    
    @log_execution
    def get_minute5_ohlcv(self, ticker: str = None, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        5분봉 데이터 가져오기 (최대 1000개)
        
        Args:
            ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
            count (int, optional): 가져올 캔들 수. Defaults to 1000.
            
        Returns:
            Optional[pd.DataFrame]: 5분봉 데이터를 포함하는 DataFrame 또는 오류 시 None
        """
        # 최대 API 호출 횟수 계산 (200개씩 가져올 때)
        max_iterations = min(5, (count + 199) // 200)  # 최대 5회 API 호출 (1000개)
        
        return self.get_extended_timeframe_data(
            ticker=ticker,
            interval="minute5",
            count=min(200, count),  # 한 번에 최대 200개 가능
            max_iterations=max_iterations
        )
    
    @log_execution
    def get_minute1_ohlcv(self, ticker: str = None, count: int = 500) -> Optional[pd.DataFrame]:
        """
        1분봉 데이터 가져오기 (최대 500개)
        
        Args:
            ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
            count (int, optional): 가져올 캔들 수. Defaults to 500.
            
        Returns:
            Optional[pd.DataFrame]: 1분봉 데이터를 포함하는 DataFrame 또는 오류 시 None
        """
        # 최대 API 호출 횟수 계산 (200개씩 가져올 때)
        max_iterations = min(3, (count + 199) // 200)  # 최대 3회 API 호출 (500개)
        
        return self.get_extended_timeframe_data(
            ticker=ticker,
            interval="minute1",
            count=min(200, count),  # 한 번에 최대 200개 가능
            max_iterations=max_iterations
        )
    
    @log_execution
    def get_account_balance(self) -> Optional[List[Dict[str, Any]]]:
        """
        계정 잔고 정보 가져오기
        
        Returns:
            Optional[List[Dict[str, Any]]]: 잔고 정보 리스트 또는 오류 시 None
        """
        if not self.upbit:
            logger.error("Upbit API not authenticated. Cannot get account balance.")
            return None
            
        try:
            track_api_call("upbit", "account_balance")
            balance = self.upbit.get_balances()
            return balance
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return None
    
    @log_execution
    def get_ticker_info(self, ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        특정 티커의 상세 정보 가져오기
        
        Args:
            ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
            
        Returns:
            Optional[Dict[str, Any]]: 티커 정보 또는 오류 시 None
        """
        ticker = ticker or settings.DEFAULT_MARKET
        
        # API에서 티커 정보 가져오기
        try:
            track_api_call("upbit", "ticker_info")
            
            # 모든 마켓 정보 가져오기
            url = "https://api.upbit.com/v1/market/all"
            response = requests.get(url)
            response.raise_for_status()
            markets = response.json()
            
            # 해당 티커 찾기
            ticker_info = next((market for market in markets if market['market'] == ticker), None)
            
            if not ticker_info:
                logger.warning(f"Ticker {ticker} not found in markets")
                return None
            
            # 추가 정보 가져오기 (24시간 동안의 가격 정보)
            current_info = pyupbit.get_current_price(ticker, verbose=True)
            
            if not current_info or not isinstance(current_info, list) or len(current_info) == 0:
                logger.warning(f"Failed to get current info for {ticker}")
                return ticker_info  # 티커 기본 정보만 반환
            
            # 현재 가격 정보 추가
            ticker_info.update(current_info[0])
            
            return ticker_info
            
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            return None


# Create a global instance for convenience
upbit_collector = UpbitDataCollector()


@log_execution
def get_market_data(ticker: str = None, timeframe: str = "day", count: int = 200) -> Optional[pd.DataFrame]:
    """
    Upbit에서 시장 데이터를 가져와 DataFrame으로 반환
    
    Args:
        ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
        timeframe (str, optional): 시간 프레임 ('day', 'hour', 'minute5', 'minute1').
                                  Defaults to "day".
        count (int, optional): 가져올 캔들 개수. Defaults to 200.
        
    Returns:
        Optional[pd.DataFrame]: OHLCV 데이터를 포함하는 DataFrame 또는 오류 시 None
    """
    ticker = ticker or settings.DEFAULT_MARKET
    logger.info(f"Getting {timeframe} market data for {ticker}, count={count}")
    
    # 시간 프레임 매핑
    timeframe_mapping = {
        'day': 'day',
        'hour': 'minute60',
        'minute60': 'minute60',
        'minute5': 'minute5',
        'minute1': 'minute1'
    }
    
    # 업비트 API의 interval로 변환
    interval = timeframe_mapping.get(timeframe, 'day')
    
    try:
        # UpbitDataCollector 인스턴스 생성
        collector = UpbitDataCollector()
        
        # 타임프레임별 다른 메소드 호출
        if timeframe == 'day':
            df = collector.get_daily_ohlcv(ticker=ticker, to=None, since=None)
            if df is None or df.empty:
                # 일별 데이터가 없으면 기본 API로 시도
                df = collector.get_ohlcv(ticker=ticker, interval=interval, count=min(count, 200))
        elif timeframe == 'hour' or timeframe == 'minute60':
            df = collector.get_hourly_ohlcv(ticker=ticker, count=count)
        elif timeframe == 'minute5':
            df = collector.get_minute5_ohlcv(ticker=ticker, count=count)
        elif timeframe == 'minute1':
            df = collector.get_minute1_ohlcv(ticker=ticker, count=count)
        else:
            # 지원되지 않는 타임프레임은 기본 API 사용
            df = collector.get_ohlcv(ticker=ticker, interval=interval, count=min(count, 200))
        
        if df is None or df.empty:
            logger.warning(f"No {timeframe} data available for {ticker}")
            return None
            
        logger.info(f"Got {len(df)} {timeframe} candles for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error getting {timeframe} market data for {ticker}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


@log_execution
def get_historical_data(ticker: str = None, 
                      days: int = 100, 
                      indicators: bool = True,
                      verbose: bool = True,
                      source: str = "upbit",
                      split: bool = False,
                      train_days: int = 650,
                      validation_days: int = 150,
                      test_days: int = 200,
                      extend_with_synthetic: bool = False,
                      train_models: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Get historical market data with optional indicators and splits for ML
    
    Args:
        ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        days (int, optional): Number of days to retrieve. Defaults to 100.
        indicators (bool, optional): Whether to add technical indicators. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        source (str, optional): Data source ("upbit", "file"). Defaults to "upbit".
        split (bool, optional): Whether to split data for ML. Defaults to False.
        train_days (int, optional): Days for training set. Defaults to 650.
        validation_days (int, optional): Days for validation set. Defaults to 150.
        test_days (int, optional): Days for test set. Defaults to 200.
        extend_with_synthetic (bool, optional): Whether to add synthetic data. Defaults to False.
        train_models (bool, optional): Whether to train models with the data. Defaults to False.
        
    Returns:
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]: DataFrame or dict of DataFrames if split=True
    """
    ticker = ticker or settings.DEFAULT_MARKET
    
    # 최소 필요 일수 계산 (split=True인 경우 train+validation+test 일수)
    min_required_days = days
    if split:
        min_required_days = max(days, train_days + validation_days + test_days)
        
    if verbose:
        logger.info(f"Getting {min_required_days} days of historical data for {ticker} from {source}")
    
    df = None
    
    # 데이터 소스에 따른 처리
    if source.lower() == "upbit":
        # Upbit API에서 데이터 가져오기
        collector = UpbitDataCollector()
        df = collector.get_daily_ohlcv(ticker=ticker, since=None, to=None)
        
        if df is None or df.empty:
            logger.warning(f"No data returned from Upbit for {ticker}")
            return None
            
        # 필요 일수만큼 데이터가 충분한지 확인
        if len(df) < min_required_days:
            logger.warning(f"Only got {len(df)} days, need {min_required_days} days. Using available data.")
            
    elif source.lower() == "file":
        # 파일에서 데이터 읽기
        try:
            # 가장 최근 파일 찾기
            pattern = f"{ticker}_day_*.csv"
            matching_files = []
            
            for file in os.listdir(settings.OHLCV_DIR):
                if file.startswith(f"{ticker}_day_") and file.endswith(".csv"):
                    matching_files.append(os.path.join(settings.OHLCV_DIR, file))
            
            if not matching_files:
                logger.warning(f"No saved data files found for {ticker}, trying Upbit API")
                return get_historical_data(ticker, days, indicators, verbose, "upbit", 
                                         split, train_days, validation_days, test_days)
            
            # 가장 최근 파일 사용
            latest_file = max(matching_files, key=os.path.getmtime)
            
            if verbose:
                logger.info(f"Loading data from file: {latest_file}")
                
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            # 필요 일수만큼 데이터가 충분한지 확인
            if len(df) < min_required_days:
                logger.warning(f"Saved data has only {len(df)} days, need {min_required_days} days. Using available data.")
                
        except Exception as e:
            logger.error(f"Error loading data from file: {str(e)}")
            logger.warning("Falling back to Upbit API")
            return get_historical_data(ticker, days, indicators, verbose, "upbit", 
                                     split, train_days, validation_days, test_days)
    else:
        logger.error(f"Unknown data source: {source}")
        return None
    
    # 중복 제거 및 정렬
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    # 타임스탬프 열 추가
    df['timestamp'] = df.index
    
    # 필요한 일수만큼 제한
    if len(df) > days and not split:
        if verbose:
            logger.info(f"Limiting to the most recent {days} days")
        df = df.iloc[-days:]
    
    # 기술적 지표 추가
    if indicators:
        if verbose:
            logger.info("Adding technical indicators")
        df = add_indicators(df)
    
    # 데이터 저장
    try:
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{ticker}_day_{timestamp}.csv"
        filepath = os.path.join(settings.OHLCV_DIR, filename)
        
        df.to_csv(filepath)
        
        if verbose:
            logger.info(f"Saved historical data to {filepath}")
    except Exception as e:
        logger.warning(f"Error saving historical data: {str(e)}")
    
    # ML을 위한 데이터 분할
    if split:
        if verbose:
            logger.info(f"Splitting data for ML: train={train_days}, val={validation_days}, test={test_days}")
        
        data_splits = _split_data(df, train_days, validation_days, test_days)
        
        # 데이터 확장 옵션
        if extend_with_synthetic and 'train' in data_splits:
            if verbose:
                logger.info("Extending training data with synthetic samples")
            
            # 확장 로직은 별도 함수로 구현 가능
            # data_splits['train'] = extend_with_synthetic_data(data_splits['train'])
        
        # 모델 훈련 옵션
        if train_models:
            if verbose:
                logger.info("Training models with the data")
            _train_models_with_data(data_splits, ticker)
        
        return data_splits
    
    return df

def _split_data(df: pd.DataFrame, train_days: int, validation_days: int, test_days: int) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, validation, and test sets for machine learning
    
    Args:
        df (pd.DataFrame): DataFrame to split
        train_days (int): Days for training set
        validation_days (int): Days for validation set
        test_days (int): Days for test set
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'train', 'val', and 'test' DataFrames
    """
    # 데이터가 충분한지 확인
    total_days = train_days + validation_days + test_days
    
    if len(df) < total_days:
        logger.warning(f"Not enough data for requested split. Adjusting split sizes. {len(df)} available, {total_days} requested.")
        
        # 데이터 비율 유지하며 크기 조정
        available_days = len(df)
        total_ratio = train_days + validation_days + test_days
        
        train_days = int(available_days * (train_days / total_ratio))
        validation_days = int(available_days * (validation_days / total_ratio))
        test_days = available_days - train_days - validation_days
        
        logger.info(f"Adjusted split: train={train_days}, val={validation_days}, test={test_days}")
    
    # 데이터 정렬
    df = df.sort_index()
    
    # 분할 인덱스 계산
    test_start_idx = len(df) - test_days
    val_start_idx = test_start_idx - validation_days
    
    # 분할 실행
    if test_start_idx > 0 and val_start_idx > 0:
        test_df = df.iloc[test_start_idx:]
        val_df = df.iloc[val_start_idx:test_start_idx]
        train_df = df.iloc[:val_start_idx]
        
        # 각 세트에 설명 열 추가
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['set'] = 'train'
        val_df['set'] = 'validation'
        test_df['set'] = 'test'
        
        # 결과 반환
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'all': df
        }
    else:
        logger.error(f"Invalid split: train_days={train_days}, val_days={validation_days}, test_days={test_days}, data_len={len(df)}")
        # 분할 실패 시 전체 데이터를 모든 세트에 사용
        return {
            'train': df.copy(),
            'val': df.copy(),
            'test': df.copy(),
            'all': df
        }

def _train_models_with_data(data_splits: Dict[str, pd.DataFrame], ticker: str = "KRW-BTC") -> None:
    """
    분할된 데이터를 사용하여 모델 훈련
    
    Args:
        data_splits (Dict[str, pd.DataFrame]): 학습/검증/테스트 데이터
        ticker (str): 티커
    """
    try:
        from models.random_forest import RandomForestDirectionModel
        from models.gru import GRUDirectionModel
        from ensemble.ensemble_core import VotingEnsemble
        from data.processors import prepare_data_for_training
        
        logger.info("모델 훈련을 위한 데이터 준비 중...")
        
        # 데이터 준비 (특성 및 타겟 변수 생성, 정규화 등)
        train_data = data_splits['train']
        val_data = data_splits['validation']
        test_data = data_splits['test']
        
        # 방향 예측을 위한 타겟 컬럼 추가 (1=상승, 0=하락)
        # 모든 데이터프레임에 방향 컬럼 추가
        for df in [train_data, val_data, test_data]:
            if df is not None and not df.empty:
                # 다음 날의 종가가 오늘보다 높으면 1, 아니면 0
                df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
                # NaN 제거
                df.dropna(inplace=True)
        
        # 훈련, 검증 및 테스트 데이터 준비
        prepared_data = prepare_data_for_training(
            pd.concat([train_data, val_data]), 
            window_size=30,
            prediction_steps=1,
            target_column='direction',  # 방향 예측 타깃으로 변경
            price_change_periods=[1, 3, 7],
            train_ratio=len(train_data) / (len(train_data) + len(val_data)),
            val_ratio=0,  # 이미 분할되어 있으므로 검증 데이터 비율은 0
            scaler_type='minmax',
            shuffle=False
        )
        
        # 테스트 데이터 준비 (스케일러는 훈련/검증 데이터에서 생성된 것 사용)
        from data.processors import prepare_latest_data_for_prediction
        X_test = None
        y_test = None
        
        if test_data is not None and not test_data.empty:
            try:
                X_test = prepare_latest_data_for_prediction(
                    test_data,
                    window_size=30,
                    scalers=prepared_data['scalers']
                )
                # 테스트 타깃 (방향)
                y_test = test_data['direction'].values
            except Exception as e:
                logger.error(f"테스트 데이터 처리 중 오류: {str(e)}")
                y_test = np.array([])
        else:
            # 테스트 데이터가 충분하지 않은 경우의 처리
            logger.warning("테스트 데이터가 부족하여 타깃 변수를 생성할 수 없습니다")
            y_test = np.array([])
        
        # 훈련 및 검증 데이터
        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_val, y_val = prepared_data['X_val'], prepared_data['y_val']
        
        # 확인: y_train이 이진값(0, 1)을 가지고 있는지 확인
        if y_train.size > 0:
            unique_values = np.unique(y_train)
            logger.info(f"y_train 고유값: {unique_values}")
            
            # 연속형 값이면 이진 분류를 위해 변환
            if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
                logger.warning("y_train이 이진값이 아닙니다. 이진 분류로 변환합니다.")
                # 양수이면 1, 그렇지 않으면 0
                y_train = (y_train > 0).astype(int)
                if y_val.size > 0:
                    y_val = (y_val > 0).astype(int)
                if y_test is not None and y_test.size > 0:
                    y_test = (y_test > 0).astype(int)
                
                # 변환 후 고유값 다시 확인
                logger.info(f"변환 후 y_train 고유값: {np.unique(y_train)}")
        
        # 데이터 차원 확인
        logger.info(f"원본 데이터 형태: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # RandomForest를 위한 2D 데이터 준비 (3D -> 2D 변환)
        X_train_2d = X_train
        X_val_2d = X_val
        X_test_2d = X_test
        
        # 3D 데이터의 경우 2D로 변환 (각 시퀀스를 단일 벡터로 평탄화)
        if len(X_train.shape) == 3:  # (samples, time_steps, features)
            samples, time_steps, features = X_train.shape
            X_train_2d = X_train.reshape(samples, time_steps * features)
            
            if X_val.size > 0:
                X_val_2d = X_val.reshape(X_val.shape[0], time_steps * features)
            
            if X_test is not None and X_test.size > 0:
                X_test_2d = X_test.reshape(X_test.shape[0], time_steps * features)
                
            logger.info(f"RandomForest용 변환 데이터 형태: X_train_2d={X_train_2d.shape}")
        
        # 랜덤 포레스트 모델 초기화 및 훈련
        logger.info("Random Forest 모델 훈련 중...")
        rf_model = RandomForestDirectionModel(
            name="RF_Direction",
            version="1.0.0",
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=10,
            class_weight='balanced'  # 클래스 균형을 위해 추가
        )
        
        # 데이터가 충분한지 확인
        if X_train_2d.size > 0 and y_train.size > 0:
            rf_metrics = rf_model.train(X_train_2d, y_train)
            logger.info(f"Random Forest 훈련 완료. 훈련 정확도: {rf_metrics.get('train_accuracy', 0):.4f}")
            
            # 검증 데이터로 평가 (데이터가 충분한 경우)
            if X_val_2d.size > 0 and y_val.size > 0:
                rf_val_metrics = rf_model.evaluate(X_val_2d, y_val)
                logger.info(f"Random Forest 검증 성능: 정확도={rf_val_metrics.get('test_accuracy', 0):.4f}")
            
            # 테스트 데이터로 평가 (데이터가 충분한 경우)
            if X_test_2d is not None and X_test_2d.size > 0 and y_test is not None and y_test.size > 0 and len(X_test_2d) == len(y_test):
                rf_test_metrics = rf_model.evaluate(X_test_2d, y_test)
                logger.info(f"Random Forest 테스트 성능: 정확도={rf_test_metrics.get('test_accuracy', 0):.4f}")
            
            # 모델 저장
            rf_model.save()
            logger.info("Random Forest 모델 저장 완료")
        else:
            logger.warning("훈련 데이터가 부족하여 Random Forest 모델을 훈련할 수 없습니다")
        
        # GRU 모델 초기화 및 훈련
        try:
            logger.info("GRU 모델 훈련 중...")
            gru_model = GRUDirectionModel(
                name="GRU_Direction",
                version="1.0.0",
                sequence_length=30,
                units=[64, 32],  # 유닛 수 감소 (128, 64 -> 64, 32)
                dropout_rate=0.4,  # 드롭아웃 증가 (0.2 -> 0.4)
                learning_rate=0.001,  # 적절한 학습률 설정
                batch_size=32,
                epochs=50  # 에포크 수 조정
            )
            
            # 모델 빌드 (명시적으로 호출)
            input_shape = (30, X_train.shape[2]) if len(X_train.shape) == 3 else None
            if input_shape:
                gru_model.build_model(input_shape=input_shape)
                logger.info(f"GRU 모델 빌드 완료: {input_shape}")
            
            # 3D 형태로 데이터 변환 (GRU 입력 형식)
            # 이미 3D인 경우 그대로 사용, 2D인 경우 3D로 변환
            X_train_3d = X_train
            if len(X_train.shape) == 2:
                # 시퀀스 크기와 피처 수 계산
                if X_train.shape[1] % 30 == 0:  # 30은 window_size
                    features = X_train.shape[1] // 30
                    X_train_3d = X_train.reshape(X_train.shape[0], 30, features)
            
            X_val_3d = X_val
            if len(X_val.shape) == 2 and X_val.size > 0:
                if X_val.shape[1] % 30 == 0:
                    features = X_val.shape[1] // 30
                    X_val_3d = X_val.reshape(X_val.shape[0], 30, features)
            
            X_test_3d = X_test
            if X_test is not None and len(X_test.shape) == 2 and X_test.size > 0:
                if X_test.shape[1] % 30 == 0:
                    features = X_test.shape[1] // 30
                    X_test_3d = X_test.reshape(X_test.shape[0], 30, features)
            
            # 데이터셋 형태 로깅
            logger.info(f"GRU 훈련 데이터 형태: {X_train_3d.shape}")
            
            # 데이터가 충분한지 확인
            if X_train_3d.size > 0 and y_train.size > 0:
                # GRU 모델 훈련
                gru_metrics = gru_model.train(
                    X_train_3d, y_train,
                    X_val=X_val_3d if X_val_3d.size > 0 else None, 
                    y_val=y_val if y_val.size > 0 else None,
                    early_stopping_patience=5
                )
                
                logger.info(f"GRU 훈련 완료. 훈련 정확도: {gru_metrics.get('train_accuracy', 0):.4f}")
                
                # 테스트 데이터로 평가 (데이터가 충분한 경우)
                if X_test_3d is not None and X_test_3d.size > 0 and y_test is not None and y_test.size > 0 and len(X_test_3d) == len(y_test):
                    gru_test_metrics = gru_model.evaluate(X_test_3d, y_test)
                    logger.info(f"GRU 테스트 성능: 정확도={gru_test_metrics.get('test_accuracy', 0):.4f}")
                
                # 모델 저장
                gru_model.save()
                logger.info("GRU 모델 저장 완료")
                
                # 앙상블 모델 구성
                logger.info("앙상블 모델 구성 중...")
                ensemble = VotingEnsemble(
                    name="ML_Voting_Ensemble",
                    version="1.0.0",
                    voting='soft'
                )
                
                # 모델 추가 (가중치 업데이트: RF 0.6->0.55, GRU 0.4->0.45)
                try:
                    ensemble.add_model(rf_model, weight=0.55)
                    logger.info("RandomForest 모델을 앙상블에 추가했습니다.")
                except TypeError as e:
                    logger.error(f"RandomForest 모델 추가 실패: {str(e)}")
                    # 종속성 확인
                    logger.info(f"RF 모델 타입: {type(rf_model).__name__}")
                    logger.info(f"RF 모델은 ClassificationModel 상속 여부: {isinstance(rf_model, ClassificationModel)}")
                
                try:
                    ensemble.add_model(gru_model, weight=0.45)
                    logger.info("GRU 모델을 앙상블에 추가했습니다.")
                except TypeError as e:
                    logger.error(f"GRU 모델 추가 실패: {str(e)}")
                    # 종속성 확인
                    logger.info(f"GRU 모델 타입: {type(gru_model).__name__}")
                    logger.info(f"GRU 모델은 ClassificationModel 상속 여부: {isinstance(gru_model, ClassificationModel)}")
                
                # 앙상블에 모델이 추가되었는지 확인
                if len(ensemble.models) == 0:
                    logger.warning("앙상블에 모델이 추가되지 않았습니다. 앙상블 모델을 사용하지 않습니다.")
                else:
                    # 앙상블 모델 평가 (데이터가 충분한 경우)
                    if X_test_2d is not None and X_test_2d.size > 0 and y_test is not None and y_test.size > 0 and len(X_test_2d) == len(y_test):
                        ensemble_test_metrics = ensemble.evaluate(X_test_2d, y_test)
                        logger.info(f"앙상블 테스트 성능: 정확도={ensemble_test_metrics.get('accuracy', 0):.4f}")
                    
                    # 앙상블 모델 저장
                    ensemble.save()
                    logger.info("앙상블 모델 저장 완료")
            else:
                logger.warning("훈련 데이터가 부족하여 GRU 모델을 훈련할 수 없습니다")
            
        except Exception as e:
            logger.error(f"GRU 모델 훈련 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("GRU 모델 없이 Random Forest만 사용합니다.")
        
    except Exception as e:
        logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("모델 훈련을 건너뜁니다.")


@log_execution
def collect_all_required_data(ticker: str = None, include_hourly: bool = True, save_data: bool = True) -> Dict[str, pd.DataFrame]:
    """
    모델 학습과 백테스팅에 필요한 모든 데이터를 수집
    
    Args:
        ticker (str, optional): 티커 심볼. Defaults to settings.DEFAULT_MARKET.
        include_hourly (bool, optional): 시간별 데이터 포함 여부. Defaults to True.
        save_data (bool, optional): 데이터를 파일로 저장할지 여부. Defaults to True.
        
    Returns:
        Dict[str, pd.DataFrame]: 일별 및 시간별 데이터를 포함하는 딕셔너리
    """
    ticker = ticker or settings.DEFAULT_MARKET
    result = {}
    
    # 일별 데이터 수집 (최대 1000일)
    logger.info(f"일별 데이터 수집 중: {ticker}")
    daily_data = get_historical_data(
        ticker=ticker, 
        days=1000,
        indicators=True,
        verbose=True,
        source="upbit"
    )
    
    if daily_data is not None:
        result['daily'] = daily_data
        logger.info(f"{ticker}의 일별 데이터 {len(daily_data)}일 수집 완료")
        
        # 파일로 이미 저장되었으므로 추가 저장 불필요
    else:
        logger.warning(f"{ticker}의 일별 데이터 수집 실패")
    
    # 시간별 데이터 수집 (선택 사항)
    if include_hourly:
        logger.info(f"시간별 데이터 수집 중: {ticker}")
        try:
            collector = UpbitDataCollector()
            hourly_data = collector.get_hourly_ohlcv(ticker=ticker, count=2000)  # 약 83일
            
            if hourly_data is not None and not hourly_data.empty:
                result['hourly'] = hourly_data
                logger.info(f"{ticker}의 시간별 데이터 {len(hourly_data)}시간 수집 완료")
                
                # 시간별 데이터 저장
                if save_data:
                    try:
                        # 시간별 데이터 저장
                        timestamp = datetime.now().strftime('%Y%m%d')
                        filename = f"{ticker}_hour_{timestamp}.csv"
                        filepath = os.path.join(settings.OHLCV_DIR, filename)
                        
                        hourly_data.to_csv(filepath)
                        logger.info(f"시간별 데이터 저장 완료: {filepath}")
                        
                        # 기술적 지표 추가
                        if len(hourly_data) >= 100:
                            hourly_data_with_indicators = add_indicators(hourly_data)
                            
                            # 지표 데이터 저장
                            indicators_dir = os.path.join(settings.PROCESSED_DIR, ticker, "indicators")
                            os.makedirs(indicators_dir, exist_ok=True)
                            
                            indicator_filepath = os.path.join(
                                indicators_dir, 
                                f"hour_{timestamp}.csv"
                            )
                            hourly_data_with_indicators.to_csv(indicator_filepath)
                            logger.info(f"지표 포함 시간별 데이터 저장 완료: {indicator_filepath}")
                            
                            # 지표 포함 데이터로 업데이트
                            result['hourly'] = hourly_data_with_indicators
                    except Exception as e:
                        logger.error(f"시간별 데이터 저장 중 오류: {str(e)}")
            else:
                logger.warning(f"{ticker}의 시간별 데이터 수집 실패")
        except Exception as e:
            logger.error(f"시간별 데이터 수집 중 오류: {str(e)}")
            logger.debug(traceback.format_exc())
    
    return result 