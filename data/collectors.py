"""
Data collection module for Bitcoin Trading Bot

This module provides functionality to collect market data from Upbit exchange API.
"""

import time
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import concurrent.futures
import requests
import pyupbit
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
import os
from dateutil.relativedelta import relativedelta

from config import settings
from utils.logging import get_logger, log_execution
from utils.monitoring import track_api_call
from data.indicators import add_indicators
from data.storage import save_ohlcv_data
from models.ensemble import VotingEnsemble
from models.random_forest import RandomForestDirectionModel
from models.gru import GRUDirectionModel
from models.base import ClassificationModel

# Initialize logger
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
        Get OHLCV data
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            interval (str, optional): Time interval ('day', 'minute1', 'minute3', 'minute5', 'minute10', 
                                      'minute15', 'minute30', 'minute60', 'minute240', 'week', 'month'). 
                                      Defaults to "day".
            count (int, optional): Number of candles to retrieve (max 200). Defaults to 200.
            to (str, optional): End date in format 'YYYY-MM-DD HH:MM:SS'. Defaults to None.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing OHLCV data or None if error
        """
        try:
            ticker = ticker or settings.DEFAULT_MARKET
            track_api_call("upbit", f"ohlcv_{interval}")
            
            # If 'to' parameter is provided, use it
            if to:
                df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count, to=to)
            else:
                df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)
            
            if df is None or df.empty:
                logger.warning(f"No OHLCV data returned for {ticker} ({interval})")
                return None
            
            # Add ticker column for identification when storing multiple tickers
            df['ticker'] = ticker
            
            # Make sure all required columns are present
            expected_columns = ['open', 'high', 'low', 'close', 'volume', 'value']
            if not all(col in df.columns for col in expected_columns):
                logger.warning(f"Missing columns in OHLCV data for {ticker}: {set(expected_columns) - set(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {ticker} ({interval}): {str(e)}")
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
        Get daily OHLCV data for an extended period
        
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
            
            # Calculate number of days
            days_diff = (to_dt - since_dt).days
            
            # If more than 200 days (Upbit API limit), make multiple calls
            if days_diff > 200:
                all_data = []
                current_to_dt = to_dt
                
                while current_to_dt >= since_dt:
                    current_since_dt = max(current_to_dt - timedelta(days=199), since_dt)
                    current_since = current_since_dt.strftime('%Y-%m-%d')
                    current_to = current_to_dt.strftime('%Y-%m-%d')
                    
                    track_api_call("upbit", "ohlcv_day_extended")
                    
                    # Add a slight delay to avoid API rate limits
                    time.sleep(0.2)
                    
                    df = pyupbit.get_ohlcv(ticker=ticker, interval="day", to=current_to)
                    
                    if df is not None and not df.empty:
                        # Filter by date range
                        df = df[df.index >= pd.Timestamp(current_since)]
                        df = df[df.index <= pd.Timestamp(current_to)]
                        all_data.append(df)
                    
                    current_to_dt = current_since_dt - timedelta(days=1)
                
                if not all_data:
                    logger.warning(f"No data returned for {ticker} between {since} and {to}")
                    return None
                
                # Combine all data
                combined_df = pd.concat(all_data)
                combined_df = combined_df.sort_index()
                
                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                # Add ticker column
                combined_df['ticker'] = ticker
                
                return combined_df
            else:
                # Single API call is sufficient
                track_api_call("upbit", "ohlcv_day_single")
                df = pyupbit.get_ohlcv(ticker=ticker, interval="day", to=to)
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {ticker} between {since} and {to}")
                    return None
                
                # Filter by date range
                df = df[df.index >= pd.Timestamp(since)]
                df = df[df.index <= pd.Timestamp(to)]
                
                # Add ticker column
                df['ticker'] = ticker
                
                return df
        
        except Exception as e:
            logger.error(f"Error getting daily OHLCV data for {ticker}: {str(e)}")
            return None
    
    @log_execution
    def get_hourly_ohlcv(self, ticker: str = None, count: int = 2000) -> Optional[pd.DataFrame]:
        """
        Get hourly OHLCV data (uses minute60 interval)
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            count (int, optional): Number of hours to retrieve. Defaults to 2000.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing hourly OHLCV data or None if error
        """
        ticker = ticker or settings.DEFAULT_MARKET
        
        try:
            # Upbit API can only retrieve 200 candles at a time
            max_per_call = 200
            total_calls = (count + max_per_call - 1) // max_per_call  # Ceiling division
            
            all_data = []
            to_datetime = datetime.now()
            
            for _ in range(total_calls):
                to_date = to_datetime.strftime('%Y-%m-%d %H:%M:%S')
                
                track_api_call("upbit", "ohlcv_hour")
                
                # Add a slight delay to avoid API rate limits
                time.sleep(0.3)
                
                df = pyupbit.get_ohlcv(ticker=ticker, interval="minute60", to=to_date, count=min(count, max_per_call))
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    
                    # Update to_datetime for next batch
                    if len(df) > 0:
                        to_datetime = df.index[0].to_pydatetime() - timedelta(hours=1)
                        count -= len(df)
                    else:
                        break
                else:
                    break
                
                if count <= 0:
                    break
            
            if not all_data:
                logger.warning(f"No hourly OHLCV data returned for {ticker}")
                return None
            
            # Combine all data
            combined_df = pd.concat(all_data)
            combined_df = combined_df.sort_index()
            
            # Remove duplicates
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            
            # Add ticker column
            combined_df['ticker'] = ticker
            
            return combined_df
        
        except Exception as e:
            logger.error(f"Error getting hourly OHLCV data for {ticker}: {str(e)}")
            return None
    
    @log_execution
    def get_account_balance(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get account balance information
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of account balances or None if error
        """
        if not self.upbit:
            logger.error("Cannot get account balance: Upbit API not authenticated")
            return None
        
        try:
            track_api_call("upbit", "balance")
            balances = self.upbit.get_balances()
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return None
    
    @log_execution
    def get_ticker_info(self, ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a ticker
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing ticker information or None if error
        """
        ticker = ticker or settings.DEFAULT_MARKET
        
        try:
            url = f"https://api.upbit.com/v1/ticker?markets={ticker}"
            
            track_api_call("upbit", "ticker_info")
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                return data[0]
            else:
                logger.warning(f"No ticker information returned for {ticker}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting ticker information for {ticker}: {str(e)}")
            return None


# Create a global instance for convenience
upbit_collector = UpbitDataCollector()


@log_execution
def get_market_data(ticker: str = None, timeframe: str = "day", count: int = 200) -> Optional[pd.DataFrame]:
    """
    Convenience function to get market data from Upbit
    
    Args:
        ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        timeframe (str, optional): Time interval. Defaults to "day".
        count (int, optional): Number of candles to retrieve. Defaults to 200.
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing market data or None if error
    """
    ticker = ticker or settings.DEFAULT_MARKET
    
    if timeframe == "hour" or timeframe == "minute60":
        return upbit_collector.get_hourly_ohlcv(ticker=ticker, count=count)
    elif timeframe == "day":
        return upbit_collector.get_ohlcv(ticker=ticker, interval="day", count=min(count, 200))
    else:
        # Handle other timeframes
        return upbit_collector.get_ohlcv(ticker=ticker, interval=timeframe, count=min(count, 200))


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
    Get historical daily data for a given ticker
    
    Args:
        ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        days (int, optional): Number of days to fetch. Defaults to 100.
        indicators (bool, optional): Whether to add indicators. Defaults to True.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        source (str, optional): Data source. Defaults to "upbit".
        split (bool, optional): Whether to split data into train/validation/test sets. Defaults to False.
        train_days (int, optional): Number of days for training set. Defaults to 650.
        validation_days (int, optional): Number of days for validation set. Defaults to 150.
        test_days (int, optional): Number of days for test set. Defaults to 200.
        extend_with_synthetic (bool, optional): Whether to extend data with synthetic samples. Defaults to False.
        train_models (bool, optional): Whether to train models with the data. Defaults to False.
        
    Returns:
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]: Historical market data or data splits if split=True
    """
    try:
        ticker = ticker or settings.DEFAULT_MARKET
        logger.info(f"Getting historical data for {ticker} ({days} days)")
        
        # Check if we have recent data cached
        cache_path = f"data_storage/ohlcv/{ticker}_day_{datetime.now().strftime('%Y%m%d')}.csv"
        if os.path.exists(cache_path):
            logger.info(f"Using cached data from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) >= days:
                df = df.tail(days)
                if indicators:
                    df = add_indicators(df)
                
                # If splitting is requested, split and return the data
                if split:
                    data_splits = _split_data(df, train_days, validation_days, test_days)
                    
                    # Extend with synthetic data if requested
                    if extend_with_synthetic:
                        try:
                            from data.processors import extend_with_synthetic_data
                            data_splits['train'] = extend_with_synthetic_data(data_splits['train'])
                            logger.info(f"Extended training data with synthetic samples. New size: {len(data_splits['train'])}")
                        except (ImportError, AttributeError) as e:
                            logger.error(f"합성 데이터 확장 중 오류 발생: {e}")
                            logger.warning("data.processors 모듈이나 extend_with_synthetic_data 함수를 찾을 수 없습니다. 합성 데이터 확장을 건너뜁니다.")
                    # Train models if requested
                    if train_models:
                        _train_models_with_data(data_splits, ticker)
                        
                    return data_splits
                return df
        
        # Get data based on source
        if source.lower() == "upbit":
            total_days_needed = days
            if split:
                total_days_needed = max(days, train_days + validation_days + test_days)
            
            # Use get_ohlcv_from instead of get_ohlcv for better date range handling
            end_date = datetime.now()
            start_date = end_date - timedelta(days=total_days_needed)
            
            ohlcv = pyupbit.get_ohlcv_from(ticker=ticker, interval="day", fromDatetime=start_date, to=end_date)
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No OHLCV data returned for {ticker} (day)")
                logger.error(f"데이터를 가져오지 못했습니다: {ticker}")
                return None
                
            # Save the data for caching
            save_ohlcv_data(ohlcv, ticker, "day")
            logger.info(f"{ticker}의 일별 데이터 총 {len(ohlcv)}일 저장 완료")
            
            # Add indicators if requested
            if indicators:
                logger.info(f"일별 데이터 저장 완료: data_storage/ohlcv/{ticker}_day_{datetime.now().strftime('%Y%m%d')}.csv")
                ohlcv = add_indicators(ohlcv)
                from data.storage import save_indicator_data
                save_indicator_data(ohlcv, ticker, "day")
                logger.info(f"지표 포함 데이터 저장 완료: data_storage/processed/{ticker}/indicators/day_{datetime.now().strftime('%Y%m%d')}.csv")
            
            # If splitting is requested, split and return the data
            if split:
                data_splits = _split_data(ohlcv, train_days, validation_days, test_days)
                
                # Extend with synthetic data if requested
                if extend_with_synthetic:
                    try:
                        from data.processors import extend_with_synthetic_data
                        data_splits['train'] = extend_with_synthetic_data(data_splits['train'])
                        logger.info(f"Extended training data with synthetic samples. New size: {len(data_splits['train'])}")
                    except (ImportError, AttributeError) as e:
                        logger.error(f"합성 데이터 확장 중 오류 발생: {e}")
                        logger.warning("data.processors 모듈이나 extend_with_synthetic_data 함수를 찾을 수 없습니다. 합성 데이터 확장을 건너뜁니다.")
                
                # Train models if requested
                if train_models:
                    _train_models_with_data(data_splits, ticker)
                    
                return data_splits
            
            return ohlcv
            
        elif source.lower() == "binance":
            # Implement Binance API integration if needed
            logger.error("Binance API not yet implemented")
            return None
            
        else:
            logger.error(f"Unknown data source: {source}")
            return None
            
    except Exception as e:
        logger.error(f"Error in get_historical_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def _split_data(df: pd.DataFrame, train_days: int, validation_days: int, test_days: int) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    
    Args:
        df (pd.DataFrame): Data to split
        train_days (int): Number of days for training set
        validation_days (int): Number of days for validation set
        test_days (int): Number of days for test set
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train, validation, and test DataFrames
    """
    # Ensure we have enough data
    required_days = train_days + validation_days + test_days
    if len(df) < required_days:
        logger.warning(f"Not enough data for splitting. Required: {required_days}, Available: {len(df)}")
        # If not enough data, adjust the split ratios to available data
        total = len(df)
        train_days = int(total * (train_days / required_days))
        validation_days = int(total * (validation_days / required_days))
        test_days = total - train_days - validation_days
    
    # Sort by date to ensure proper splitting
    df = df.sort_index()
    
    # Split data
    test_start = len(df) - test_days
    validation_start = test_start - validation_days
    
    train = df.iloc[:validation_start].copy()
    validation = df.iloc[validation_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    
    logger.info(f"Data split: Train={len(train)}, Validation={len(validation)}, Test={len(test)}")
    
    return {
        'train': train,
        'validation': validation,
        'test': test
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
        from models.ensemble import VotingEnsemble
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
def collect_all_required_data(ticker: str = None) -> Dict[str, pd.DataFrame]:
    """
    Collect all required data for backtesting and model training
    
    Args:
        ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing dataframes for daily and hourly data
    """
    ticker = ticker or settings.DEFAULT_MARKET
    result = {}
    
    # Get daily data for as long as possible (up to 1000 days)
    daily_data = get_historical_data(ticker=ticker)
    if daily_data is not None:
        result['daily'] = daily_data
        logger.info(f"Collected {len(daily_data)} days of daily data for {ticker}")
    else:
        logger.warning(f"Failed to collect daily data for {ticker}")
    
    # Get hourly data (up to 2000 hours = ~83 days)
    hourly_data = upbit_collector.get_hourly_ohlcv(ticker=ticker)
    if hourly_data is not None:
        result['hourly'] = hourly_data
        logger.info(f"Collected {len(hourly_data)} hours of hourly data for {ticker}")
        
        # 시간별 데이터 저장
        from data.storage import save_ohlcv_data
        save_path = save_ohlcv_data(hourly_data, ticker, 'hour')
        logger.info(f"시간별 데이터가 {save_path}에 저장되었습니다.")
        
        # 지표가 있는 경우 지표 추가
        if len(hourly_data) >= 100:
            from data.indicators import calculate_all_indicators
            hourly_data_with_indicators = calculate_all_indicators(hourly_data, timeframe='hour')
            
            # 지표가 추가된 시간별 데이터 저장
            from data.storage import save_processed_data
            save_path = save_processed_data(hourly_data_with_indicators, ticker, 'indicators', 'hour')
            logger.info(f"지표가 추가된 시간별 데이터가 {save_path}에 저장되었습니다.")
    else:
        logger.warning(f"Failed to collect hourly data for {ticker}")
    
    return result 