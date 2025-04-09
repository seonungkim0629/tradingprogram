"""
Data Manager for Live Trading

This module provides functionality to manage data for live trading,
combining daily and hourly data for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime, timedelta
import os
import glob
import pickle
import traceback

from config import settings
from utils.logging import get_logger, log_execution
from data.collectors import UpbitDataCollector, get_historical_data
from data.indicators import add_indicators
from data.storage import (
    save_ohlcv_data, 
    stack_hourly_data, 
    load_dataframe_from_csv, 
    save_dataframe_to_csv,
    load_latest_combined_dataset,
    save_combined_dataset
)

# Initialize logger
logger = get_logger(__name__)

# 업비트 컬렉터 인스턴스화
upbit_collector = UpbitDataCollector()

class TradingDataManager:
    """Data manager for live trading that handles daily and hourly data"""
    
    def __init__(self, ticker: str = None):
        """
        Initialize the trading data manager
        
        Args:
            ticker (str, optional): Ticker symbol. Defaults to settings.DEFAULT_MARKET.
        """
        self.ticker = ticker or settings.DEFAULT_MARKET
        
        # 데이터 저장 구조 개선: 딕셔너리 사용
        self.data = {
            'day': None,
            'hour': None,
            'minute5': None,
            'minute1': None
        }
        
        # 마지막 업데이트 시간 기록을 위한 딕셔너리
        self.last_updates = {
            'day': None,
            'hour': None,
            'minute5': None,
            'minute1': None
        }
        
        # settings 모듈에서 설정 값 가져오기
        self.update_intervals = settings.DATA_UPDATE_INTERVALS
        self.max_candles = settings.DATA_MAX_CANDLES
        self.initial_counts = settings.DATA_INITIAL_COUNTS
        self.update_counts = settings.DATA_UPDATE_COUNTS
        
        # 타임프레임별 데이터 수집 함수 매핑
        self.data_collectors = {
            'day': lambda ticker, count: upbit_collector.get_ohlcv(ticker=ticker, interval="day", count=count),
            'hour': lambda ticker, count: upbit_collector.get_ohlcv(ticker=ticker, interval="minute60", count=count),
            'minute5': lambda ticker, count: upbit_collector.get_minute5_ohlcv(ticker=ticker, count=count),
            'minute1': lambda ticker, count: upbit_collector.get_minute1_ohlcv(ticker=ticker, count=count)
        }
        
        # settings 모듈에서 정의된 경로 사용
        self.stacked_data_dir = settings.STACKED_DIR
        
        # 초기 데이터 로드
        for timeframe in self.data.keys():
            self._load_data(timeframe)
        
        logger.info(f"Trading data manager initialized for {self.ticker}")
    
    @log_execution
    def _load_data(self, timeframe: str) -> None:
        """
        특정 타임프레임의 데이터를 로드하는 공통 함수
        
        Args:
            timeframe (str): 타임프레임 ('day', 'hour', 'minute5', 'minute1')
        """
        try:
            # 디스크에서 데이터 로드 시도
            pattern = f"{self.ticker}_{timeframe}_*.csv"
            files = glob.glob(os.path.join(settings.OHLCV_DIR, pattern))
            
            if files:
                # 날짜별 정렬 (최신순)
                files.sort(reverse=True)
                
                # 최신 파일 로드
                self.data[timeframe] = load_dataframe_from_csv(os.path.basename(files[0]), settings.OHLCV_DIR)
                logger.info(f"Loaded {timeframe} data from disk, {len(self.data[timeframe])} rows")
                
                # 지표 추가
                if len(self.data[timeframe]) >= 100:
                    self.data[timeframe] = add_indicators(self.data[timeframe])
            
            # 데이터가 없거나 불충분한 경우 API에서 가져오기
            if self.data[timeframe] is None or len(self.data[timeframe]) < 24:
                logger.info(f"Fetching {timeframe} data from API")
                
                # day 타임프레임은 특별 처리
                if timeframe == 'day':
                    self.data[timeframe] = get_historical_data(ticker=self.ticker, days=self.initial_counts[timeframe])
                else:
                    # 다른 타임프레임은 각 콜렉터 함수 사용
                    collector_func = self.data_collectors[timeframe]
                    self.data[timeframe] = collector_func(self.ticker, self.initial_counts[timeframe])
                
                if self.data[timeframe] is not None and len(self.data[timeframe]) > 0:
                    # 지표 추가
                    if len(self.data[timeframe]) >= 100:
                        self.data[timeframe] = add_indicators(self.data[timeframe])
                    
                    logger.info(f"Fetched {timeframe} data from API, {len(self.data[timeframe])} rows")
                else:
                    logger.warning(f"Failed to fetch {timeframe} data from API")
        
        except Exception as e:
            logger.error(f"Error loading {timeframe} data: {str(e)}")
            logger.debug(traceback.format_exc())
    
    @log_execution
    def update_data(self, force_daily: bool = False) -> dict:
        """
        모든 타임프레임의 데이터를 API에서 최신 데이터로 업데이트
        
        Args:
            force_daily (bool, optional): 일별 데이터 강제 업데이트 여부. Defaults to False.
            
        Returns:
            dict: 각 타임프레임 별 업데이트 성공 여부
        """
        now = datetime.now()
        
        # 각 타임프레임 업데이트 결과 추적
        updated = {timeframe: False for timeframe in self.data.keys()}
        
        # 각 타임프레임 별로 업데이트 필요 여부 확인 및 업데이트
        for timeframe in self.data.keys():
            # 특별 케이스: 일별 데이터 강제 업데이트
            if timeframe == 'day' and force_daily:
                needs_update = True
                    else:
                # 일반적인 업데이트 조건 확인
                needs_update = (
                    self.last_updates[timeframe] is None or
                    (now - self.last_updates[timeframe]).total_seconds() >= self.update_intervals[timeframe]
                )
            
            if needs_update:
                updated[timeframe] = self._update_timeframe_data(timeframe)
        
        # 모든 타임프레임이 업데이트된 경우 통합 데이터셋 생성
        if all(updated.values()):
            self._create_combined_dataset()
        
        return updated
    
    def _update_timeframe_data(self, timeframe: str) -> bool:
        """
        특정 타임프레임의 데이터를 업데이트
        
        Args:
            timeframe (str): 타임프레임 ('day', 'hour', 'minute5', 'minute1')
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            logger.info(f"Updating {timeframe} data")
            collector_func = self.data_collectors[timeframe]
            new_data = collector_func(self.ticker, self.update_counts[timeframe])
            
            if new_data is not None and not new_data.empty:
                # 기존 데이터와 병합
                if self.data[timeframe] is not None:
                    combined = pd.concat([self.data[timeframe], new_data])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    
                    # 데이터 크기 제한 (메모리 효율성)
                    self.data[timeframe] = combined.tail(self.max_candles[timeframe]).copy()
                else:
                    self.data[timeframe] = new_data
                
                # 지표 추가
                self.data[timeframe] = add_indicators(self.data[timeframe])
                
                # 시간별 데이터 이상은 쌓아두기
                if timeframe != 'day':
                    self._stack_timeframe_data(timeframe, new_data)
                
                # 마지막 업데이트 시간 기록
                self.last_updates[timeframe] = datetime.now()
                logger.info(f"{timeframe} data updated, now have {len(self.data[timeframe])} candles")
                return True
            else:
                logger.warning(f"No new {timeframe} data available")
                return False
                
        except Exception as e:
            logger.error(f"Error updating {timeframe} data: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def _stack_timeframe_data(self, timeframe: str, new_data: pd.DataFrame) -> None:
        """
        새로운 타임프레임 데이터를 기존 쌓인 데이터와 병합하여 저장
        
        Args:
            timeframe (str): 타임프레임 (hour, minute5, minute1)
            new_data (pd.DataFrame): 새로운 데이터
        """
        try:
            # 공통 데이터 저장 함수 사용
            stack_hourly_data(self.ticker, new_data, self.max_candles[timeframe] * 2)
            logger.debug(f"Stacked {timeframe} data for {self.ticker}")
        
        except Exception as e:
            logger.error(f"Error stacking {timeframe} data: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _create_combined_dataset(self) -> None:
        """모든 타임프레임의 데이터를 통합한 데이터셋 생성"""
        try:
            # 데이터 통합
            combined_data = {
                'day': self.data['day'],
                'hour': self.data['hour'],
                'minute5': self.data['minute5'],
                'minute1': self.data['minute1']
            }
            
            # storage 모듈의 함수를 사용하여 데이터셋 저장
            save_combined_dataset(combined_data, self.ticker)
            logger.info(f"Created combined dataset for {self.ticker}")
        
        except Exception as e:
            logger.error(f"Error creating combined dataset: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def get_data(self, timeframe: str = 'day') -> pd.DataFrame:
        """
        특정 타임프레임의 데이터 반환
        
        Args:
            timeframe (str): 타임프레임 ('day', 'hour', 'minute5', 'minute1')
            
        Returns:
            pd.DataFrame: 해당 타임프레임의 데이터
        """
        if timeframe in self.data:
            return self.data[timeframe]
        else:
            logger.warning(f"Unsupported timeframe: {timeframe}, returning daily data")
            return self.data['day']
    
    def clear_cache(self, timeframe: str = None) -> None:
        """
        메모리 데이터 캐시 정리
        
        Args:
            timeframe (str, optional): 정리할 타임프레임. None이면 모두 정리.
        """
        if timeframe is None:
            # 모든 타임프레임 정리
            for tf in self.data.keys():
                self.data[tf] = None
                logger.info(f"Cleared {tf} data cache")
        elif timeframe in self.data:
            # 특정 타임프레임만 정리
            self.data[timeframe] = None
            logger.info(f"Cleared {timeframe} data cache")
        else:
            logger.warning(f"Unknown timeframe: {timeframe}")
        
    def get_market_status(self) -> Dict[str, Any]:
        """
        현재 시장 상태 정보 반환
        
        Returns:
            Dict[str, Any]: 시장 상태 정보 (최신 가격, 거래량, 변동성 등)
        """
        status = {
            'last_update': datetime.now(),
            'ticker': self.ticker
        }
        
        # 일별 데이터에서 정보 추출
        if self.data['day'] is not None and len(self.data['day']) > 0:
            daily = self.data['day']
            status['daily'] = {
                'close': daily['close'].iloc[-1],
                'volume': daily['volume'].iloc[-1],
                'change_pct': daily['close'].pct_change().iloc[-1] * 100 if len(daily) > 1 else 0,
                'high': daily['high'].iloc[-1],
                'low': daily['low'].iloc[-1]
            }
            
            # 추가 지표 (가용한 경우)
            for indicator in ['rsi', 'ema_short', 'ema_long', 'macd']:
                if indicator in daily.columns:
                    status['daily'][indicator] = daily[indicator].iloc[-1]
        
        # 시간별 데이터에서 정보 추출
        if self.data['hour'] is not None and len(self.data['hour']) > 0:
            hourly = self.data['hour']
            status['hourly'] = {
                'close': hourly['close'].iloc[-1],
                'volume': hourly['volume'].iloc[-1],
                'change_pct': hourly['close'].pct_change().iloc[-1] * 100 if len(hourly) > 1 else 0
            }
            
            # 볼륨 변화 계산 (24시간)
            if len(hourly) >= 24:
                status['hourly']['volume_change_24h'] = (
                    hourly['volume'].iloc[-24:].sum() / 
                    hourly['volume'].iloc[-48:-24].sum() - 1
                ) * 100 if len(hourly) >= 48 else 0
        
        return status
    
    def get_current_data(self) -> Dict[str, pd.DataFrame]:
        """
        현재 가지고 있는 모든 타임프레임의 데이터를 반환
        
        Returns:
            Dict[str, pd.DataFrame]: 각 타임프레임별 데이터
        """
        # 모든 데이터를 딕셔너리 형태로 반환
        result = {}
        
        # 'day' -> 'daily', 'hour' -> 'hourly'로 변환하여 반환 (main.py와의 호환성)
        result['daily'] = self.data.get('day')
        result['hourly'] = self.data.get('hour')
        result['minute5'] = self.data.get('minute5')
        result['minute1'] = self.data.get('minute1')
        
        return result
    
    def get_latest_price(self) -> float:
        """
        가장 최근 가격(종가) 반환
        
        Returns:
            float: 최근 가격
        """
        # 가장 최근 데이터에서 가격 확인 (우선순위: 1분 > 5분 > 시간 > 일)
        for timeframe in ['minute1', 'minute5', 'hour', 'day']:
            if self.data[timeframe] is not None and len(self.data[timeframe]) > 0:
                return float(self.data[timeframe]['close'].iloc[-1])
        
        logger.warning("No price data available for any timeframe")
        return 0.0
    
    def load_initial_data(self) -> bool:
        """
        초기 데이터 로드 (모든 타임프레임)
        
        Returns:
            bool: 성공 여부
        """
        success = True
        
        # 모든 타임프레임에 대해 데이터 로드
        for timeframe in self.data.keys():
            try:
                self._load_data(timeframe)
                if self.data[timeframe] is None or len(self.data[timeframe]) < 10:
                    logger.warning(f"Failed to load sufficient {timeframe} data")
                    success = False
            except Exception as e:
                logger.error(f"Error loading initial {timeframe} data: {str(e)}")
                logger.debug(traceback.format_exc())
                success = False
        
        return success
    
    def load_combined_data(self) -> bool:
        """
        최신 통합 데이터셋 로드
        
        Returns:
            bool: 성공 여부
        """
        try:
            # storage 모듈의 함수를 사용해 최신 통합 데이터 로드
            combined_data = load_latest_combined_dataset(self.ticker)
            
            if combined_data is None:
                logger.warning(f"No combined dataset found for {self.ticker}")
                return False
                
            # 데이터 저장
            for timeframe, df in combined_data.items():
                if df is not None and not df.empty:
                    self.data[timeframe] = df
                    self.last_updates[timeframe] = datetime.now()
                    
            logger.info(f"Loaded combined dataset for {self.ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading combined dataset: {str(e)}")
            logger.debug(traceback.format_exc())
            return False