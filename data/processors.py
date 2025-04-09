"""
Data processing module for Bitcoin Trading Bot

This module provides functionality to process and prepare market data
for model training and prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
import traceback
from copy import deepcopy
from abc import ABC, abstractmethod
import logging
import os
import json
import ta

from config import settings
from utils.logging import get_logger, log_execution
from data.indicators import calculate_all_indicators, filter_indicators
from data.preprocessors import add_target_variable
from utils.data_utils import clean_dataframe

# Initialize logger
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/data.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 표준 피처 목록 정의 --- 
# data/indicators.py의 ALL_POSSIBLE_INDICATOR_COLUMNS 와 동기화 필요
STANDARD_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', # Base OHLCV
    # Basic Indicators
    'sma7', 'sma25', 'sma99', 'ema12', 'ema26', 'ema200', 'macd', 'macd_signal', 'macd_diff',
    'rsi14', 'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'atr', 'obv',
    'daily_return', 'volatility_14',
    # Advanced Indicators
    'cci', 'tsi', 'mfi', 'adi', 'roc', 'ichimoku_tenkan_sen',
    'ichimoku_kijun_sen', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 'ichimoku_chikou_span',
    'keltner_middle', 'keltner_upper', 'keltner_lower',
    'hma9', 'ao', 'williams_r', 'vwap',
    # Custom Indicators (based on implementation in indicators.py)
    'volume_sma7', 'volume_ema20', 'volume_surge', 'momentum_3d', 'momentum_7d', 'momentum_14d',
    'volatility_ratio', 'distance_from_ema200', 'rsi_divergence', 'efficiency_ratio', 'volume_price_ratio',
    # Pattern Recognition (based on implementation in indicators.py)
    'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star',
    'three_line_strike'
    # Date features can be added here if needed and handled consistently
]

@log_execution
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by handling missing values, outliers, etc.
    
    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Make a copy to avoid modifying original
        result = df.copy()
        
        # Check for missing values in essential columns
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_values = result[essential_columns].isna().sum()
        
        if missing_values.sum() > 0:
            logger.warning(f"Missing values in dataset: {missing_values}")
        
        # Forward-fill missing values in essential columns
        result[essential_columns] = result[essential_columns].ffill()
        
        # Check for negative or zero prices
        invalid_prices = (result[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if invalid_prices.any():
            invalid_count = invalid_prices.sum()
            logger.warning(f"Found {invalid_count} rows with invalid prices (<=0)")
            
            # Filter out invalid prices
            result = result[~invalid_prices]
        
        # Check for OHLC inconsistencies
        inconsistent = (
            (result['high'] < result['low']) | 
            (result['high'] < result['open']) | 
            (result['high'] < result['close']) |
            (result['low'] > result['open']) | 
            (result['low'] > result['close'])
        )
        
        if inconsistent.any():
            inconsistent_count = inconsistent.sum()
            logger.warning(f"Found {inconsistent_count} rows with OHLC inconsistencies")
            
            # Filter out inconsistent data
            result = result[~inconsistent]
        
        # Ensure the index is datetime and sorted
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, attempting to convert")
            try:
                result.index = pd.to_datetime(result.index)
            except Exception as e:
                logger.error(f"Failed to convert index to datetime: {str(e)}")
        
        # Sort by index
        result = result.sort_index()
        
        # Remove duplicate indices
        if result.index.duplicated().any():
            dup_count = result.index.duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate timestamps in index")
            result = result[~result.index.duplicated(keep='first')]
        
        # Handle extreme outliers in volume
        if 'volume' in result.columns:
            volume_mean = result['volume'].mean()
            volume_std = result['volume'].std()
            volume_outliers = result['volume'] > (volume_mean + 10 * volume_std)
            
            if volume_outliers.any():
                outlier_count = volume_outliers.sum()
                logger.warning(f"Found {outlier_count} extreme volume outliers")
                
                # Cap extreme volumes rather than removing the rows
                result.loc[volume_outliers, 'volume'] = volume_mean + 5 * volume_std
        
        logger.info(f"Data cleaning complete. Final dataset shape: {result.shape}")
        return result
    
    except Exception as e:
        logger.error(f"Error cleaning dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Return original if cleaning fails


@log_execution
def normalize_data(df: pd.DataFrame, feature_groups: Dict = None, scalers: Dict = None, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
    """
    데이터를 그룹별로 정규화합니다.
    
    Args:
        df: 정규화할 데이터프레임
        feature_groups: 피처 그룹 정보 {'price': [컬럼명들], 'volume': [컬럼명들], ...}
        scalers: 기존 스케일러 정보 (없으면 새로 생성)
        method: 정규화 방법 ('minmax' 또는 'standard')
        
    Returns:
        정규화된 데이터프레임과 스케일러 정보
    """
    if df.empty:
        logger.warning("빈 데이터프레임이 전달되어 정규화를 수행하지 않습니다.")
        return df, {}
    
    if scalers is None:
        scalers = {}
    
    # 기본 그룹 설정 (지정되지 않은 경우)
    if feature_groups is None:
        feature_groups = {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['volume'],
            'indicators': [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]
        }
    
    # 그룹별 정규화
    logger.info(f"데이터 정규화 수행 (방식: {method})")
    
    normalized_df = df.copy()  # 원본 보존
    
    # 각 그룹별 정규화 실행
    for group_name, columns in feature_groups.items():
        # 존재하는 컬럼만 필터링
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            logger.warning(f"그룹 {group_name}에 유효한 컬럼이 없습니다.")
            continue
        
        try:
            # 그룹별 공통 스케일러 사용 시도
            if group_name in ['price', 'volume']:
                group_data = normalized_df[valid_columns]
                try:
                    if np.isnan(group_data).any().any() or np.isinf(group_data).any().any():
                        logger.warning(f"{group_name} 그룹 데이터에 NaN 또는 Inf 값이 있습니다.")
                        # NaN 및 Inf 값 처리
                        group_data = group_data.replace([np.inf, -np.inf], np.nan)
                        group_data = group_data.fillna(group_data.mean())
                except:
                    logger.error(f"그룹 데이터 체크 중 오류: {group_name}")
                    try:
                        logger.error(f"데이터 형태: {group_data.shape}, 타입: {group_data.dtypes}")
                    except:
                        pass
                
                if group_name in scalers:
                    # 기존 스케일러 적용
                    min_val = scalers[group_name]['min']
                    max_val = scalers[group_name]['max']
                    
                    # 정규화 적용
                    for col in valid_columns:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                    
                    logger.info(f"{group_name} 그룹 ({len(valid_columns)}개 컬럼) 정규화 완료")
                else:
                    # 새 스케일러 생성
                    if method == 'minmax':
                        min_val = group_data.min().min()
                        max_val = group_data.max().max()
                        
                        if min_val == max_val:
                            logger.warning(f"{group_name} 그룹의 min_val과 max_val이 같습니다({min_val}). 정규화를 건너뜁니다.")
                            continue
                        
                        # 정규화 적용
                        for col in valid_columns:
                            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                        
                        # 스케일러 정보 저장
                        scalers[group_name] = {'min': min_val, 'max': max_val}
                        logger.info(f"{group_name} 그룹에 대한 새 스케일러가 생성되어 학습되었습니다.")
                    
                    logger.info(f"{group_name} 그룹 ({len(valid_columns)}개 컬럼) 정규화 완료")
            else:
                # 개별 컬럼 정규화 (indicators 그룹 등)
                logger.warning(f"각 컬럼에 대해 개별적으로 정규화를 시도합니다.")
                
                for col in valid_columns:
                    # 컬럼별 개별 정규화
                    if col not in normalized_df.columns:
                        continue  # 해당 컬럼이 없으면 건너뛰기
                    
                    # 스케일러 정보가 있는지 확인
                    scaler_key = f"{group_name}_{col}"
                    col_scalers = {col: scalers[scaler_key]} if scaler_key in scalers else None
                    
                    # 개별 컬럼 정규화 실행 및 스케일러 정보 업데이트
                    normalized_df, updated_scalers = _normalize_single_column(normalized_df, col, col_scalers)
                    
                    # 스케일러 정보 저장
                    if updated_scalers and col in updated_scalers:
                        scalers[scaler_key] = updated_scalers[col]
        
        except Exception as e:
            logger.error(f"{group_name} 그룹 정규화 중 오류: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 오류 발생 시 컬럼별로 개별 정규화 시도
            if group_name in ['price', 'volume']:
                logger.warning(f"{group_name} 그룹 정규화 중 오류 발생. 컬럼별 개별 정규화를 시도합니다.")
                for col in valid_columns:
                    normalized_df, updated_scalers = _normalize_single_column(normalized_df, col)
                    
                    # 스케일러 정보 저장
                    scaler_key = f"{group_name}_{col}"
                    if updated_scalers and col in updated_scalers:
                        scalers[scaler_key] = updated_scalers[col]
    
    # 정규화 결과 검증 (NaN 값 확인)
    nan_count = normalized_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"정규화 후 {nan_count}개의 NaN 값이 있습니다. 0으로 채웁니다.")
        normalized_df = normalized_df.fillna(0)
    
    return normalized_df, scalers


def _normalize_single_column(df: pd.DataFrame, col: str, scaler_dict: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    단일 컬럼을 정규화합니다.
    
    Args:
        df: 데이터프레임
        col: 정규화할 컬럼명
        scaler_dict: 기존 스케일러 정보 (없으면 새로 생성)
        
    Returns:
        정규화된 데이터프레임과 업데이트된 스케일러 정보
    """
    try:
        # 불리언 타입인지 확인
        if df[col].dtype == bool or 'bool' in str(df[col].dtype):
            # 불리언 데이터를 0과 1로 변환
            df[col] = df[col].astype(float)
            
        # 숫자형으로 변환 시도
        col_data_numeric = pd.to_numeric(df[col], errors='coerce')
        
        # NaN 혹은 Inf 값이 있는지 확인
        if col_data_numeric.isna().any() or np.isinf(col_data_numeric).any():
            logger.warning(f"컬럼 {col}에 NaN 또는 Inf 값이 있습니다. 평균값으로 대체합니다.")
            col_data_numeric = col_data_numeric.replace([np.inf, -np.inf], np.nan)
            # NaN 값을 평균으로 대체
            if col_data_numeric.isna().all():
                # 모든 값이 NaN인 경우 0으로 채움
                col_data_numeric = col_data_numeric.fillna(0)
            else:
                # 일부만 NaN인 경우 평균으로 채움
                col_data_numeric = col_data_numeric.fillna(col_data_numeric.mean())
        
        # 스케일러 정보가 없으면 새로 계산
        if scaler_dict is None or col not in scaler_dict:
            min_val = col_data_numeric.min()
            max_val = col_data_numeric.max()
            
            # min과 max가 같으면 정규화가 불가능하므로 모두 0 또는 0.5로 설정
            if min_val == max_val:
                logger.warning(f"컬럼 {col}의 min_val({min_val})과 max_val({max_val})이 유효하지 않습니다. 0으로 채웁니다.")
                df[col] = 0
                return df, {col: {'min': 0, 'max': 1}}  # 임의의 범위 설정
            
            if scaler_dict is None:
                scaler_dict = {}
            scaler_dict[col] = {'min': min_val, 'max': max_val}
        else:
            min_val = scaler_dict[col]['min']
            max_val = scaler_dict[col]['max']
            
            # 저장된 스케일러의 min과 max가 같으면 모두 0으로 설정
            if min_val == max_val:
                logger.warning(f"저장된 스케일러 {col}의 min_val({min_val})과 max_val({max_val})이 유효하지 않습니다. 0으로 채웁니다.")
                df[col] = 0
                return df, scaler_dict
        
        # 정규화 실행 (min_val과 max_val이 다른 경우에만)
        if min_val != max_val:
            df[col] = (col_data_numeric - min_val) / (max_val - min_val)
        else:
            df[col] = 0  # min_val과 max_val이 같으면 모두 0으로 설정
            
        return df, scaler_dict
    except Exception as e:
        logger.error(f"컬럼 {col} 개별 정규화 중 오류: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning(f"오류로 인해 컬럼 {col}의 모든 값을 0으로 설정했습니다.")
        
        # 오류 발생 시 해당 컬럼을 0으로 채우고 계속 진행
        df[col] = 0
        
        if scaler_dict is None:
            scaler_dict = {}
        scaler_dict[col] = {'min': 0, 'max': 1}  # 임의의 범위 설정
        
        return df, scaler_dict

@log_execution
def extend_with_synthetic_data(df: pd.DataFrame, target_days: int = 1000, 
                             balanced_direction: bool = True) -> pd.DataFrame:
    """
    실제 데이터를 기반으로 합성 데이터를 생성하여 데이터셋을 확장
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        target_days (int, optional): 목표 데이터 일수. Defaults to 1000.
        balanced_direction (bool, optional): 방향성(상승/하락) 균형 유지 여부. Defaults to True.
        
        df (pd.DataFrame): 원본 OHLCV 데이터
        target_days (int): 목표 데이터 일수
        balanced_direction (bool): 상승/하락 비율을 균형 있게 할지 여부
        
    Returns:
        pd.DataFrame: 합성 데이터가 추가된 확장 데이터프레임
    """
    if len(df) >= target_days:
        logger.info(f"데이터가 이미 {len(df)}일치 존재하여 확장이 필요 없음")
        return df
    
    logger.info(f"데이터를 {len(df)}일에서 {target_days}일로 합성 데이터를 이용해 확장")
    
    # 데이터프레임이 비어 있는 경우 처리
    if len(df) == 0:
        logger.warning("원본 데이터가 비어 있습니다. 모든 데이터를 합성으로 생성합니다.")
        
        # 샘플 데이터 생성을 위한 기본값 설정
        start_date = pd.Timestamp.now() - pd.Timedelta(days=target_days-1)
        result = pd.DataFrame(index=pd.date_range(start=start_date, periods=target_days))
        
        # 기본 비트코인 시세 범위 설정 (좀 더 현실적인 값으로)
        base_price = 30000.0  # 기본 시작 가격 (USD 기준)
        price_trend = np.linspace(-0.2, 0.2, target_days)  # 약한 추세 (-20%~+20%)
        noise = np.random.normal(0, 0.02, target_days)  # 일별 노이즈 (표준편차 2%)
        
        # 가격 시계열 생성 (추세 + 노이즈 + 랜덤워크)
        close_prices = []
        current_price = base_price
        
        # 방향성을 위한 상승/하락 패턴 설정 (약 50/50)
        if balanced_direction:
            # 상승/하락 교대 패턴 설정 (약간의 랜덤성 추가)
            direction_pattern = []
            
            # 1-3일 연속 상승/하락 패턴 생성
            current_direction = np.random.choice([1, -1])  # 1: 상승, -1: 하락
            days_remaining = target_days
            
            while days_remaining > 0:
                # 1-3일 연속으로 같은 방향 설정
                streak_length = min(np.random.randint(1, 4), days_remaining)
                direction_pattern.extend([current_direction] * streak_length)
                current_direction *= -1  # 방향 전환
                days_remaining -= streak_length
        else:
            # 자연적인 가격 변동에 따른 방향성 결정
            direction_pattern = np.random.choice([1, -1], size=target_days)
            
        # 가격 시계열 생성
        for i in range(target_days):
            # 기본 추세 + 노이즈 + 방향성 요소
            daily_return = price_trend[i] + noise[i] + direction_pattern[i] * 0.01
            current_price *= (1 + daily_return)
            close_prices.append(current_price)
        
        # 종가 시계열 설정
        result['close'] = close_prices
        
        # 시가/고가/저가 생성 (종가 기준)
        for i in range(target_days):
            # 당일 변동성 (종가의 1~3%)
            volatility = result['close'].iloc[i] * np.random.uniform(0.01, 0.03)
            
            # 시가는 전일 종가에서 약간 변동
            if i == 0:
                result.loc[result.index[i], 'open'] = result['close'].iloc[i] * np.random.uniform(0.98, 1.02)
            else:
                result.loc[result.index[i], 'open'] = result['close'].iloc[i-1] * np.random.uniform(0.99, 1.01)
            
            # 고가/저가 설정
            if direction_pattern[i] > 0:  # 상승일
                # 고가는 시가와 종가 중 높은 값보다 더 높게
                result.loc[result.index[i], 'high'] = max(result['open'].iloc[i], result['close'].iloc[i]) + volatility
                # 저가는 시가와 종가 중 낮은 값보다 더 낮게
                result.loc[result.index[i], 'low'] = min(result['open'].iloc[i], result['close'].iloc[i]) - (volatility * 0.5)
            else:  # 하락일
                # 고가는 시가와 종가 중 높은 값보다 더 높게 (하락폭보다 작게)
                result.loc[result.index[i], 'high'] = max(result['open'].iloc[i], result['close'].iloc[i]) + (volatility * 0.5)
                # 저가는 시가와 종가 중 낮은 값보다 더 낮게
                result.loc[result.index[i], 'low'] = min(result['open'].iloc[i], result['close'].iloc[i]) - volatility
        
        # 거래량 생성 (주가 변동에 따라 조정)
        base_volume = 100.0
        volumes = []
        
        for i in range(target_days):
            # 가격 변동률 계산 (절대값)
            if i == 0:
                price_change = 0.01
            else:
                price_change = abs(result['close'].iloc[i] / result['close'].iloc[i-1] - 1)
            
            # 가격 변동이 클수록 거래량 증가 (거래량 증폭 + 노이즈)
            volume_factor = 1.0 + (price_change * 10)  # 가격 변화에 따른 증폭
            volume_noise = np.random.uniform(0.7, 1.3)  # 30% 내외 노이즈
            volume = base_volume * volume_factor * volume_noise
            volumes.append(volume)
        
        result['volume'] = volumes
        
        # 거래대금 계산
        result['value'] = result['volume'] * result['close']
        
        # 티커 및 합성 데이터 표시
        result['ticker'] = 'KRW-BTC'
        result['synthetic'] = True
        
        # 방향성 칼럼 추가
        result['direction'] = (result['close'].shift(-1) > result['close']).astype(int)
        
        # 마지막 행의 방향성은 알 수 없으므로 이전 방향성과 동일하게 설정
        if len(result) > 0:
            result.loc[result.index[-1], 'direction'] = result['direction'].iloc[-2] if len(result) > 1 else np.random.randint(0, 2)
        
        logger.info(f"{target_days}일치의 기본 합성 데이터 생성 완료")
        direction_ratio = result['direction'].mean()
        logger.info(f"합성 데이터 상승 비율: {direction_ratio:.2f}, 하락 비율: {1-direction_ratio:.2f}")
        return result
    
    # 원본 데이터 보존
    result = df.copy()
    
    # 필요한 추가 일수 계산
    days_to_add = target_days - len(df)
    
    # 인덱스가 날짜 형식인지 확인하고 변환
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("데이터프레임 인덱스가 날짜 형식이 아닙니다. 날짜 인덱스로 변환합니다.")
        try:
            # 첫 번째 열을 인덱스로 가정
            result = result.reset_index()
            # 첫 번째 열을 날짜로 변환
            result['index'] = pd.to_datetime(result['index'])
            result = result.set_index('index')
        except Exception as e:
            logger.error(f"날짜 인덱스 변환 실패: {str(e)}")
            # 임의 날짜 인덱스 생성
            result = result.reset_index(drop=True)
            start_date = pd.Timestamp.now() - pd.Timedelta(days=len(result)-1)
            result.index = pd.date_range(start=start_date, periods=len(result))
    
    # 일별 변화율과 변동성 계산
    try:
        returns = result['close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std()
            mean_return = returns.mean()
        else:
            # 데이터가 부족하면 기본값 사용
            volatility = 0.02
            mean_return = 0.001
    except (KeyError, ValueError) as e:
        logger.warning(f"수익률 계산 중 오류: {str(e)}, 기본값 사용")
        volatility = 0.02
        mean_return = 0.001
    
    # 새 데이터의 날짜 범위 설정 (기존 데이터의 시작일 이전)
    try:
        last_date = result.index.min() - pd.Timedelta(days=1)
        # NaT 값 체크
        if pd.isna(last_date):
            logger.warning("인덱스에 NaT 값이 있습니다. 현재 시간 기준으로 날짜 설정")
            last_date = pd.Timestamp.now() - pd.Timedelta(days=1)
    except Exception as e:
        logger.warning(f"날짜 범위 설정 오류: {str(e)}, 현재 시간 기준으로 설정")
        last_date = pd.Timestamp.now() - pd.Timedelta(days=1)
    
    try:
        dates = pd.date_range(end=last_date, periods=days_to_add)
    except Exception as e:
        logger.error(f"날짜 범위 생성 오류: {str(e)}")
        start_date = last_date - pd.Timedelta(days=days_to_add-1)
        dates = pd.date_range(start=start_date, end=last_date)
        
    # 시작 가격 설정 (첫 번째 실제 데이터 가격 사용)
    try:
        first_close = result['close'].iloc[0]
    except (IndexError, KeyError) as e:
        logger.warning(f"가격 데이터 가져오기 오류: {str(e)}, 기본값 사용")
        first_close = 30000.0
    
    # 실제 데이터의 OHLC 비율 계산
    try:
        avg_open_close_ratio = (result['open'] / result['close']).mean()
        avg_high_close_ratio = (result['high'] / result['close']).mean()
        avg_low_close_ratio = (result['low'] / result['close']).mean()
        
        # NaN 값 체크
        if pd.isna(avg_open_close_ratio): avg_open_close_ratio = 1.0
        if pd.isna(avg_high_close_ratio): avg_high_close_ratio = 1.01
        if pd.isna(avg_low_close_ratio): avg_low_close_ratio = 0.99
    except Exception as e:
        logger.warning(f"OHLC 비율 계산 오류: {str(e)}, 기본값 사용")
        avg_open_close_ratio = 1.0
        avg_high_close_ratio = 1.01
        avg_low_close_ratio = 0.99
    
    # 거래량 통계 계산
    try:
        volume_mean = result['volume'].mean()
        volume_std = result['volume'].std()
        
        # NaN 값 체크
        if pd.isna(volume_mean): volume_mean = 100.0
        if pd.isna(volume_std): volume_std = 10.0
    except Exception as e:
        logger.warning(f"거래량 통계 계산 오류: {str(e)}, 기본값 사용")
        volume_mean = 100.0
        volume_std = 10.0
        
    # 방향성 균형을 위한 설정
    if balanced_direction:
        # 약 50/50의 상승/하락 비율로 방향 설정
        directions = np.random.choice([1, -1], size=days_to_add, p=[0.5, 0.5])
        
        # 1-3일 연속으로 같은 방향이 유지되도록 설정
        for i in range(1, days_to_add):
            if np.random.random() < 0.7:  # 70% 확률로 이전 방향 유지
                directions[i] = directions[i-1]
    else:
        # 자연적인 가격 변동에 따른 방향성 결정
        directions = [1 if np.random.normal(mean_return, volatility) > 0 else -1 for _ in range(days_to_add)]
    
    # 새 데이터 생성
    synthetic_data_list = []
    last_close = first_close
    
    for i in range(days_to_add):
        # 방향성에 따라 수익률 조정
        direction = directions[i]
        
        # 방향에 따른 수익률 조정 (방향성 강화)
        if direction > 0:  # 상승
            daily_return = np.random.normal(abs(mean_return) + 0.002, volatility)
        else:  # 하락
            daily_return = np.random.normal(-abs(mean_return) - 0.002, volatility)
        
        # 종가 계산
        close = last_close * (1 + daily_return)
        
        # 약간의 노이즈를 추가하여 실제 데이터와 비슷하게 만듦
        open_noise = np.random.normal(0, 0.005)
        high_noise = np.random.normal(0, 0.005)
        low_noise = np.random.normal(0, 0.005)
        
        # 방향성에 따라 OHLC 값을 더 현실적으로 설정
        if direction > 0:  # 상승일
            open_price = close / (1 + daily_return * np.random.uniform(0.5, 0.9))  # 시가는 종가보다 낮게
            high_price = close * (1 + np.random.uniform(0.001, 0.01))  # 고가는 종가보다 약간 높게
            low_price = min(open_price, close) * (1 - np.random.uniform(0.001, 0.01))  # 저가는 시가나 종가 중 낮은 값보다 더 낮게
        else:  # 하락일
            open_price = close / (1 + daily_return * np.random.uniform(0.5, 0.9))  # 시가는 종가보다 높게
            high_price = max(open_price, close) * (1 + np.random.uniform(0.001, 0.01))  # 고가는 시가나 종가 중 높은 값보다 더 높게
            low_price = close * (1 - np.random.uniform(0.001, 0.01))  # 저가는 종가보다 약간 낮게
        
        # 거래량 생성 (가격 변동 크기에 비례하도록)
        volume_factor = 1.0 + (abs(daily_return) * 10)  # 가격 변화에 따른 증폭
        volume = np.random.normal(volume_mean * volume_factor, volume_std)
        volume = max(0, volume)  # 음수 거래량 방지
        
        # 데이터 추가
        synthetic_data_list.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume,
            'value': volume * close,  # 대략적인 거래대금
            'ticker': result['ticker'].iloc[0] if 'ticker' in result.columns else 'KRW-BTC',
            'synthetic': True,  # 합성 데이터 표시
            'direction': 1 if direction > 0 else 0  # 방향성 표시 (1: 상승, 0: 하락)
        })
        
        # 다음 날을 위해 종가 업데이트
        last_close = close
    
    # 새 데이터를 데이터프레임으로 변환
    try:
        synthetic_df = pd.DataFrame(synthetic_data_list, index=dates)
        
        # 방향성 확인
        direction_ratio = synthetic_df['direction'].mean()
        logger.info(f"합성 데이터 방향성 비율 - 상승: {direction_ratio:.2f}, 하락: {1-direction_ratio:.2f}")
        
        # 원본 데이터와 합성 데이터 결합
        # 이미 방향성 컬럼이 있으면 유지, 없으면 추가
        if 'direction' not in result.columns and len(result) > 0:
            result['direction'] = (result['close'].shift(-1) > result['close']).astype(int)
            # 마지막 행 처리
            result.loc[result.index[-1], 'direction'] = result['direction'].iloc[-2] if len(result) > 1 else np.random.randint(0, 2)
        
        # 합성 데이터 컬럼 추가
        if 'synthetic' not in result.columns:
            result['synthetic'] = False
            
        extended_df = pd.concat([synthetic_df, result])
        extended_df = extended_df.sort_index()
        
        logger.info(f"{days_to_add}일치의 합성 데이터 생성 완료")
        
        # 전체 데이터 방향성 비율 확인
        if 'direction' in extended_df.columns:
            overall_direction_ratio = extended_df['direction'].mean()
            logger.info(f"전체 데이터 방향성 비율 - 상승: {overall_direction_ratio:.2f}, 하락: {1-overall_direction_ratio:.2f}")
        
        return extended_df
    except Exception as e:
        logger.error(f"합성 데이터 결합 오류: {str(e)}")
        # 오류 발생 시 원본 데이터 반환
        return result

def split_historical_data(df: pd.DataFrame, train_days: int = 650, 
                         validation_days: int = 150, test_days: int = 200) -> Dict[str, pd.DataFrame]:
    """
    과거 시장 데이터를 훈련, 검증, 테스트 세트로 분할
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        train_days (int, optional): 훈련 세트 일수. Defaults to 650.
        validation_days (int, optional): 검증 세트 일수. Defaults to 150.
        test_days (int, optional): 테스트 세트 일수. Defaults to 200.
        
    Returns:
        Dict[str, pd.DataFrame]: 'train', 'validation', 'test', 'all' 키를 가진 딕셔너리
    """
    from utils.data_utils import clean_dataframe
    
    # 데이터 정리
    df = clean_dataframe(df, handle_missing='interpolate')
    
    # 타임스탬프 형식 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
            logger.info("인덱스를 DatetimeIndex로 변환했습니다.")
        except Exception as e:
            logger.warning(f"인덱스 변환 실패: {str(e)}. 원본 인덱스를 유지합니다.")
    
    # 데이터 정렬
    df = df.sort_index()
    
    # 사용 가능한 총 데이터 일수
    total_available = len(df)
    total_days = train_days + validation_days + test_days
    
    # 충분한 데이터가 없는 경우 비율에 맞게 조정
    if total_available < total_days:
        logger.warning(f"지정된 분할에 충분한 데이터가 없습니다. 보유: {total_available}일, 필요: {total_days}일.")
        
        # 최소 테스트 세트 크기 보장 (적어도 1일)
        min_test_days = max(1, int(total_available * 0.1))
        
        # 남은 데이터를 train과 validation에 분배
        remaining_days = total_available - min_test_days
        ratio_train_to_val = train_days / (train_days + validation_days)
        
        train_size = max(1, int(remaining_days * ratio_train_to_val))
        val_size = max(1, remaining_days - train_size)
        test_size = min_test_days
        
        logger.info(f"조정된 분할: 훈련={train_size}, 검증={val_size}, 테스트={test_size}일")
    else:
        # 충분한 데이터가 있는 경우
        train_size = train_days
        val_size = validation_days
        test_size = test_days if test_days <= (total_available - train_size - val_size) else (total_available - train_size - val_size)
    
    # 최종 세트 크기 조정 (합이 총 데이터 일수보다 많을 수 없음)
    if train_size + val_size + test_size > total_available:
        excess = (train_size + val_size + test_size) - total_available
        # 비율에 맞게 excess 분배하여 감소
        train_reduction = int(excess * (train_size / (train_size + val_size + test_size)))
        val_reduction = int(excess * (val_size / (train_size + val_size + test_size)))
        test_reduction = excess - train_reduction - val_reduction
        
        train_size = max(1, train_size - train_reduction)
        val_size = max(1, val_size - val_reduction)
        test_size = max(1, test_size - test_reduction)
    
    # 최종 세트 크기 검증 및 조정
    total_split = train_size + val_size + test_size
    if total_split < total_available:
        # 남은 일수를 train에 추가
        train_size += (total_available - total_split)
    elif total_split > total_available:
        # 초과 일수 처리 (일어나지 않아야 하지만 안전 장치)
        excess = total_split - total_available
        if train_size > excess:
            train_size -= excess
        elif val_size > excess:
            val_size -= excess
        else:
            # 균등하게 감소
            train_size = max(1, int(train_size * total_available / total_split))
            val_size = max(1, int(val_size * total_available / total_split))
            test_size = total_available - train_size - val_size
    
    # 최종 분할 포인트 계산
    test_start = total_available - test_size
    val_start = test_start - val_size
    
    # 분할 경계 검증
    if val_start < 0:
        val_start = 0
    if test_start < val_start:
        test_start = val_start
    
    # 데이터 분할
    train_data = df.iloc[:val_start].copy() if val_start > 0 else pd.DataFrame(columns=df.columns)
    val_data = df.iloc[val_start:test_start].copy() if test_start > val_start else pd.DataFrame(columns=df.columns)
    test_data = df.iloc[test_start:].copy()
    
    # 빈 데이터프레임 검사 및 처리
    if train_data.empty and total_available > 0:
        logger.warning("훈련 데이터가 비어 있습니다. 최소 1개 샘플을 할당합니다.")
        train_data = df.iloc[0:1].copy()
    
    if val_data.empty and total_available > 1:
        logger.warning("검증 데이터가 비어 있습니다. 최소 1개 샘플을 할당합니다.")
        # 이미 훈련 데이터가 있다면 다른 샘플 사용
        if len(train_data) < total_available:
            val_data = df.iloc[len(train_data):len(train_data)+1].copy()
        else:
            val_data = df.iloc[0:1].copy()
    
    # 데이터 세트에 식별자 추가
    train_data['set'] = 'train'
    val_data['set'] = 'validation'
    test_data['set'] = 'test'
    
    # 데이터셋 크기 로깅
    logger.info(f"데이터 분할 완료: 훈련={len(train_data)}일, 검증={len(val_data)}일, 테스트={len(test_data)}일")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data,
        'all': df
    }

@log_execution
def denormalize_dataframe(df: pd.DataFrame, scalers: Dict[str, Any]) -> pd.DataFrame:
    """
    Denormalize DataFrame values using the provided scalers
    
    Args:
        df (pd.DataFrame): Normalized DataFrame
        scalers (Dict[str, Any]): Dictionary of scalers used for normalization
        
    Returns:
        pd.DataFrame: Denormalized DataFrame
    """
    try:
        # Make a copy to avoid modifying original
        result = df.copy()
        
        # Price columns to denormalize
        price_columns = ['open', 'high', 'low', 'close']
        price_columns = [col for col in price_columns if col in result.columns]
        
        # Volume columns to denormalize
        volume_columns = ['volume']
        volume_columns = [col for col in volume_columns if col in result.columns]
        
        # Indicator columns
        indicator_columns = [col for col in result.select_dtypes(include=np.number).columns 
                            if col not in price_columns + volume_columns]
        
        # Denormalize price columns
        if price_columns and 'price' in scalers:
            original_prices = scalers['price'].inverse_transform(result[price_columns].values)
            result[price_columns] = original_prices
        
        # Denormalize volume columns
        if volume_columns and 'volume' in scalers:
            try:
                # 차원 문제 해결: 1차원 열은 2차원으로 변환
                volume_data = result[volume_columns].values
                
                # reshape(-1, 1)로 2차원으로 변환하지 않은 경우 처리
                if len(volume_data.shape) == 1 or (len(volume_data.shape) == 2 and volume_data.shape[1] == 1):
                    # 1개 열일 때 2차원으로 명시적 변환
                    volume_data = volume_data.reshape(-1, 1)
                
                # 스케일러 적용 후 다시 데이터프레임에 삽입
                scaled_volume = scalers['volume'].inverse_transform(volume_data)
                result[volume_columns] = scaled_volume
                
                logger.debug(f"볼륨 데이터 정규화 성공: 형태 {volume_data.shape} -> {scaled_volume.shape}")
            except Exception as e:
                logger.warning(f"볼륨 데이터 정규화 중 오류 발생: {str(e)}")
                # 오류 발생 시 간단히 0-1 범위로 Min-Max 정규화 직접 수행
                for col in volume_columns:
                    if result[col].max() > result[col].min():
                        result[col] = (result[col] - result[col].min()) / (result[col].max() - result[col].min())
                    else:
                        result[col] = 0.5  # 모든 값이 같을 경우 0.5로 설정
        
        # Denormalize indicator columns
        if indicator_columns and 'indicators' in scalers:
            original_indicators = scalers['indicators'].inverse_transform(result[indicator_columns])
            result[indicator_columns] = original_indicators
        
        return result
    
    except Exception as e:
        logger.error(f"Error denormalizing dataframe: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Return original if denormalization fails


@log_execution
def create_features_targets(df: pd.DataFrame,
                           target_type: str = 'next_close',
                           target_column: str = 'close',
                           horizon: int = 1,
                           normalize: bool = True,
                           scaler_type: str = 'minmax',
                           drop_nan: bool = True) -> Dict[str, Any]:
    """
    피처와 타겟 생성
    ...
    """
    try:
        result = {}
        data = df.copy()

        # --- 1. 타겟 변수 추가 --- 
        data = add_target_variable(data, target_type=target_type, horizon=horizon)

        # --- 2. NaN 처리 --- 
        initial_len = len(data)
        if 'target' in data.columns:
            target_nan_mask = data['target'].isna()
            data = data.dropna(subset=['target'])
            logger.info(f"타겟 변수 생성 후 NaN 값 {target_nan_mask.sum()}개를 제거했습니다.")
        else:
            logger.warning("타겟 변수가 생성되지 않았습니다. NaN 제거를 건너뜁니다.")
            # Target이 없으면 이후 진행 불가
            return {"error": "Target variable could not be created."}

        # --- 3. 피처 추출 (표준 피처 목록 사용) --- 
        # 타겟을 제외하고 피처 추출 함수 적용
        features_to_extract = data.drop(columns=['target'], errors='ignore')
        features_extracted = extract_features(features_to_extract) # 수정된 extract_features 사용

        # NaN 처리 후에도 피처에 NaN이 남아있는지 확인 (extract_features에서 0으로 채워짐)
        # if features_extracted.isna().any().any():
        #     logger.warning("피처 추출 후에도 NaN 값이 남아있습니다. 0으로 채웁니다.")
        #     features_extracted = features_extracted.fillna(0)
        
        # --- 4. 타겟 변수 분리 --- 
        # features_extracted 와 동일한 인덱스를 가진 타겟 데이터 선택
        target = data.loc[features_extracted.index, 'target']

        # --- 5. 정규화 (표준 피처에 대해 수행) ---
        if normalize:
            logger.info(f"데이터 정규화 수행 (방식: {scaler_type})")
            # 이미 표준 컬럼만 있으므로 excludes 불필요
            normalized_df, scalers = normalize_data(features_extracted, method=scaler_type)
            result['normalized_df'] = normalized_df
            result['scalers'] = scalers
            result['X'] = normalized_df.values
        else:
            result['normalized_df'] = features_extracted
            result['scalers'] = {}
            result['X'] = features_extracted.values
        
        # 타겟 값 저장
        result['y'] = target.values # 기본적으로 1D 배열
        # 필요한 경우 reshape: .reshape(-1, 1)

        # --- 6. 결과 저장 --- 
        result['original_df'] = df
        result['feature_columns'] = STANDARD_FEATURES # 표준 피처 목록 사용
        result['target_type'] = target_type
        result['target_column'] = target_column
        result['horizon'] = horizon

        logger.info(f"피처 생성 완료: X 형태={result['X'].shape}, y 형태={result['y'].shape}")

        return result

    except Exception as e:
        logger.error(f"피처 및 타겟 생성 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@log_execution
def split_data(X: np.ndarray, y: np.ndarray, 
              train_ratio: float = 0.7, 
              val_ratio: float = 0.15,
              shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets
    
    Args:
        X (np.ndarray): Features array
        y (np.ndarray): Targets array
        train_ratio (float, optional): Ratio of training data. Defaults to 0.7.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.15.
        shuffle (bool, optional): Whether to shuffle data. Defaults to False.
        
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    try:
        if shuffle:
            # When shuffling, create a random permutation of indices
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        # Calculate split indices
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        # Split data
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Data split: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty arrays
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty


@log_execution
def prepare_data_for_training(
    data: pd.DataFrame, 
    sequence_length: int = 10, 
    n_future: int = 1, 
    target_column: str = 'close', 
    prediction_type: str = 'classification',
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sequence_model: bool = True,
    normalize: bool = True,
    price_difference: bool = True,
    return_type: str = 'dict',
    min_samples: int = 50,
    synthetic_data_ratio: float = 0.2,
    include_timestamp: bool = False,
    exclude_features: List[str] = None,
    custom_features: List[str] = None,
    feature_subset: str = 'all',
    scaling_method: str = 'minmax',
    existing_scalers: dict = None
) -> Dict[str, Any]:
    """
    모델 훈련을 위해 데이터를 준비하는 함수
    """
    logger.info("Beginning data preparation for model training")
    
    if data is None or data.empty:
        raise ValueError("입력 데이터가 비어있습니다. 모델 훈련을 위한 데이터 준비가 불가능합니다.")
        
    # 데이터 날짜 범위 기록
    start_date = data.index[0] if isinstance(data.index, pd.DatetimeIndex) else None
    end_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
    
    if start_date and end_date:
        logger.info(f"데이터 범위: {start_date} ~ {end_date}, 총 {len(data)}일")

    # 데이터 복사본 생성 (원본 보존)
    df = data.copy()
    
    # 타겟 변수 생성
    if prediction_type == 'classification':
        # n_future 기간 후의 종가와 현재 종가 비교하여 상승/하락 레이블 생성
        if price_difference:
            # 가격 차이에 기반한 방향 레이블 생성
            df['target'] = df[target_column].shift(-n_future) > df[target_column]
            df['target'] = df['target'].astype(int)
        else:
            # n_future 기간 동안의 변화 방향에 기반한 레이블 생성
            df['target'] = df[target_column].pct_change(periods=n_future).shift(-n_future) > 0
            df['target'] = df['target'].astype(int)
    elif prediction_type == 'regression':
        # n_future 기간 후의 종가를 타겟으로 설정
        df['target'] = df[target_column].shift(-n_future)
    else:
        raise ValueError(f"지원되지 않는 prediction_type: {prediction_type}. 'classification' 또는 'regression'을 사용하세요.")
    
    # 타겟 생성으로 인한 NaN 값 처리
    nan_count = df['target'].isna().sum()
    if nan_count > 0:
        logger.warning(f"타겟 변수에 {nan_count}개의 NaN 값이 있습니다. 대부분 데이터 끝 부분일 것입니다.")
        df = df.dropna(subset=['target'])
        logger.info(f"타겟 변수 생성 후 NaN 값 {nan_count}개를 제거했습니다.")
    
    if len(df) < min_samples:
        raise ValueError(f"유효한 데이터 샘플 수({len(df)})가 최소 요구사항({min_samples})보다 적습니다.")
    
    # 필요한 컬럼 선택
    if exclude_features is None:
        exclude_features = []
    
    # 사전 정의된 특성 그룹
    standard_features = [
        'open', 'high', 'low', 'close', 'volume',  # 기본 OHLCV
        'sma7', 'sma25', 'sma99', 'ema12', 'ema26', 'ema200',  # 이동평균
        'macd', 'macd_signal', 'macd_diff',  # MACD
        'rsi14', 'bb_upper', 'bb_middle', 'bb_lower',  # RSI, 볼린저
        'stoch_k', 'stoch_d', 'atr', 'obv', 'daily_return',  # 기타 기본 지표
        'volatility_14', 'cci', 'tsi', 'mfi', 'adi',  # 추가 기술적 지표
        'keltner_middle', 'keltner_upper', 'keltner_lower',  # 켈트너 채널
        'volume_sma7', 'volume_ema20', 'volume_surge',  # 거래량 지표
        'momentum_3d', 'momentum_7d', 'momentum_14d',  # 모멘텀 지표
        'distance_from_ema200',  # 이동평균 거리
        'rsi_divergence', 'efficiency_ratio', 'volume_price_ratio'  # 발산 및 효율성
    ]
    
    # 선택적 고급 특성
    advanced_features = [
        'volatility_ratio', 'trend_strength', 'market_momentum',
        'harmonic_pattern', 'volume_trend', 'price_volatility_relation'
    ]
    
    pattern_features = [
        'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing',
        'morning_star', 'evening_star', 'three_line_strike' 
    ]
    
    # 특성 서브셋 선택
    if feature_subset == 'minimal':
        selected_features = ['open', 'high', 'low', 'close', 'volume', 'rsi14', 'sma7', 'sma25', 'macd', 'daily_return']
    elif feature_subset == 'basic':
        selected_features = standard_features[:25]  # 기본 지표까지만 포함
    elif feature_subset == 'advanced':
        selected_features = standard_features + advanced_features[:3]  # 일부 고급 지표 포함
    elif feature_subset == 'patterns':
        selected_features = standard_features + pattern_features  # 패턴 인식 지표 포함
    elif feature_subset == 'all':
        selected_features = standard_features + advanced_features + pattern_features
    else:
        logger.warning(f"알 수 없는 feature_subset: {feature_subset}. 'all'로 설정합니다.")
        selected_features = standard_features + advanced_features + pattern_features
    
    # 커스텀 특성 추가
    if custom_features:
        selected_features.extend(custom_features)
    
    # 제외할 특성 처리
    selected_features = [feat for feat in selected_features if feat not in exclude_features]
    
    # 실제 데이터프레임에 있는 컬럼만 필터링
    available_features = [feat for feat in selected_features if feat in df.columns]
    
    # 필요한 컬럼이지만 데이터프레임에 없는 경우 NaN으로 추가
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        logger.warning(f"Added missing standard features (as NaN): {missing_features}")
        for feat in missing_features:
            df[feat] = np.nan
    
    # 타겟과 특성 분리
    X = df[selected_features]
    y = df['target']
    
    # NaN 값 있는지 확인하고 처리
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filled {nan_count} NaN values in extracted features (using 0).")
        X = X.fillna(0)
    
    logger.info(f"Extracted features. Final shape: {X.shape}. Columns: {X.columns.tolist()[:5]}...{X.columns.tolist()[-5:]}")
    
    # 데이터 정규화
    if normalize:
        logger.info(f"데이터 정규화 수행 (방식: {scaling_method})")
        
        # 피처 그룹 설정
        feature_groups = {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['volume'],
            'indicators': [col for col in X.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        }
        
        # 정규화 수행
        X_normalized, scalers = normalize_data(
            df=X,
            feature_groups=feature_groups,
            scalers=existing_scalers,
            method=scaling_method
        )
        
        X = X_normalized
    else:
        scalers = existing_scalers or {}
    
    # 시퀀스 모델을 위한 데이터 준비
    if sequence_model:
        try:
            # LSTM 모듈 임포트 시도
            try:
                from models.gru import create_sequence_data
                
                # 시퀀스 데이터 생성
                X_seq, y_seq = create_sequence_data(X.values, y.values, sequence_length)
                
                logger.info(f"시퀀스 데이터 생성 완료. X 형태: {X_seq.shape}, y 형태: {y_seq.shape}")
                
                # 훈련/검증/테스트 분할
                train_size = int(len(X_seq) * train_ratio)
                val_size = int(len(X_seq) * validation_ratio)
                
                X_train = X_seq[:train_size]
                y_train = y_seq[:train_size]
                
                X_val = X_seq[train_size:train_size + val_size]
                y_val = y_seq[train_size:train_size + val_size]
                
                X_test = X_seq[train_size + val_size:]
                y_test = y_seq[train_size + val_size:]
                
            except ImportError:
                logger.error("Error: 'models.gru' 모듈을 찾을 수 없습니다. 시퀀스 데이터를 생성할 수 없습니다.")
                logger.warning("대안으로 비시퀀스 데이터를 생성합니다.")
                
                # 비시퀀스 데이터로 대체
                train_size = int(len(X) * train_ratio)
                val_size = int(len(X) * validation_ratio)
                
                X_train = X.values[:train_size]
                y_train = y.values[:train_size]
                
                X_val = X.values[train_size:train_size + val_size]
                y_val = y.values[train_size:train_size + val_size]
                
                X_test = X.values[train_size + val_size:]
                y_test = y.values[train_size + val_size:]
                
        except Exception as e:
            logger.error(f"Error in data preparation pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    else:
        # 비시퀀스 모델을 위한 데이터 준비
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * validation_ratio)
        
        X_train = X.values[:train_size]
        y_train = y.values[:train_size]
        
        X_val = X.values[train_size:train_size + val_size]
        y_val = y.values[train_size:train_size + val_size]
        
        X_test = X.values[train_size + val_size:]
        y_test = y.values[train_size + val_size:]
    
    logger.info(f"피처 생성 완료: X 형태={X.shape}, y 형태={y.shape}")
    
    # 결과 반환
    if return_type == 'dict':
        result = {
            'X': X, 'y': y,
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'scalers': scalers,
            'sequence_length': sequence_length if sequence_model else 1,
            'prediction_type': prediction_type
        }
        return result
    elif return_type == 'arrays':
        return X_train, y_train, X_val, y_val, X_test, y_test, scalers

@log_execution
def prepare_latest_data_for_prediction(df: pd.DataFrame,
                                      window_size: int,
                                      scalers: Dict[str, Any]) -> np.ndarray:
    """
    Prepare the latest data for making predictions
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        window_size (int): Number of time steps for the sequence
        scalers (Dict[str, Any]): Dictionary of scalers for normalization
        
    Returns:
        np.ndarray: Prepared sequence for prediction
    """
    try:
        # Step 1: Clean dataset
        cleaned_df = clean_dataset(df)
        if cleaned_df.empty:
             logger.warning("Cleaned data is empty. Cannot prepare for prediction.")
             return np.array([])
             
        # Step 2: Calculate all indicators
        df_with_indicators = calculate_all_indicators(cleaned_df)
        if df_with_indicators.empty:
             logger.warning("Data after adding indicators is empty. Cannot prepare for prediction.")
             return np.array([])

        # Step 2.5: Extract standard features using the extract_features function
        features_df = extract_features(df_with_indicators)
        if features_df.empty:
            logger.warning("Feature extraction resulted in an empty DataFrame. Cannot prepare for prediction.")
            return np.array([])
        
        # Check if required scalers are provided
        required_scalers = ['price', 'volume', 'indicators']
        if not all(scaler_name in scalers for scaler_name in required_scalers):
             logger.error(f"Missing required scalers. Provided: {list(scalers.keys())}. Required: {required_scalers}")
             # Attempt to continue without scalers, logging a warning. Data will not be scaled.
             normalized_df = features_df.copy()
             logger.warning("Proceeding without full normalization due to missing scalers.")
        else:
            # Step 3: Normalize using provided scalers on the standard features
            # Use the standard feature set directly for normalization
            normalized_df = features_df.copy() # Start with the standard features

            # --- 정규화 로직 시작 ---
            # Identify column groups within STANDARD_FEATURES
            price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in STANDARD_FEATURES]
            volume_columns = [col for col in ['volume'] if col in STANDARD_FEATURES]
            # Indicator columns are all remaining standard numeric features
            indicator_columns = [
                col for col in STANDARD_FEATURES 
                if col not in price_columns and col not in volume_columns and pd.api.types.is_numeric_dtype(normalized_df[col])
            ]

            # Apply price scaler only to price columns present in the features_df
            active_price_cols = [col for col in price_columns if col in normalized_df.columns]
            if active_price_cols and 'price' in scalers:
                try:
                    price_data = normalized_df[active_price_cols].values
                    # Check for NaN/inf before scaling
                    if np.any(np.isnan(price_data)) or np.any(np.isinf(price_data)):
                        logger.warning("NaN/inf found in price data before scaling (post-extract_features). Filling with 0.")
                        price_data = np.nan_to_num(price_data, nan=0.0, posinf=0.0, neginf=0.0)
                    normalized_df[active_price_cols] = scalers['price'].transform(price_data)
                    logger.debug(f"가격 데이터 정규화 성공: 컬럼 {active_price_cols}")
                except Exception as e:
                    logger.warning(f"가격 정규화 중 오류: {str(e)}. 컬럼 {active_price_cols}은(는) 정규화되지 않을 수 있습니다.")
                    # Fallback: Fill with 0 if scaling fails
                    normalized_df[active_price_cols] = normalized_df[active_price_cols].fillna(0)


            # Apply volume scaler only to volume columns present in features_df
            active_volume_cols = [col for col in volume_columns if col in normalized_df.columns]
            if active_volume_cols and 'volume' in scalers:
                try:
                    volume_data = normalized_df[active_volume_cols].values
                    if len(volume_data.shape) == 1: volume_data = volume_data.reshape(-1, 1)
                    if np.any(np.isnan(volume_data)) or np.any(np.isinf(volume_data)):
                        logger.warning("NaN/inf found in volume data before scaling (post-extract_features). Filling with 0.")
                        volume_data = np.nan_to_num(volume_data, nan=0.0, posinf=0.0, neginf=0.0)
                    normalized_df[active_volume_cols] = scalers['volume'].transform(volume_data)
                    logger.debug("볼륨 데이터 정규화 성공")
                except Exception as e:
                    logger.warning(f"볼륨 데이터 정규화 중 오류 발생: {str(e)}. 컬럼 {active_volume_cols}은(는) 정규화되지 않을 수 있습니다.")
                    # Fallback: Fill with 0 if scaling fails
                    normalized_df[active_volume_cols] = normalized_df[active_volume_cols].fillna(0)

            # Apply indicator scaler only to indicator columns present in features_df
            active_indicator_cols = [col for col in indicator_columns if col in normalized_df.columns]
            if active_indicator_cols and 'indicators' in scalers:
                try:
                    indicator_data = normalized_df[active_indicator_cols].values
                    if np.any(np.isnan(indicator_data)) or np.any(np.isinf(indicator_data)):
                        logger.warning("NaN/inf found in indicator data before scaling (post-extract_features). Filling with 0.")
                        indicator_data = np.nan_to_num(indicator_data, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    # Ensure the scaler expects the correct number of features
                    if indicator_data.shape[1] != scalers['indicators'].n_features_in_:
                         logger.error(f"Indicator scaler feature count mismatch. Scaler expects {scalers['indicators'].n_features_in_}, data has {indicator_data.shape[1]}. Columns: {active_indicator_cols}")
                         # Attempt to align columns if possible (this is risky)
                         scaler_features = getattr(scalers['indicators'], 'feature_names_in_', None)
                         if scaler_features is not None:
                             logger.info(f"Attempting to reindex data to match scaler features: {list(scaler_features)}")
                             temp_df = pd.DataFrame(indicator_data, columns=active_indicator_cols, index=normalized_df.index)
                             temp_df = temp_df.reindex(columns=scaler_features, fill_value=0)
                             indicator_data = temp_df.values
                             active_indicator_cols = list(scaler_features) # Update columns being scaled
                             logger.info(f"Reindexed data shape: {indicator_data.shape}")
                         else:
                             raise ValueError(f"Cannot reconcile feature mismatch for indicator scaler without feature names.")

                    normalized_df[active_indicator_cols] = scalers['indicators'].transform(indicator_data)
                    logger.debug(f"지표 데이터 정규화 성공: 컬럼 수 {len(active_indicator_cols)}")
                except Exception as e:
                    logger.warning(f"지표 정규화 중 오류: {str(e)}. 컬럼 {active_indicator_cols}은(는) 정규화되지 않을 수 있습니다.")
                    # Fallback: Fill with 0 if scaling fails
                    normalized_df[active_indicator_cols] = normalized_df[active_indicator_cols].fillna(0)
            # --- 정규화 로직 끝 ---

        # 최종 NaN/inf 값 확인 및 처리 (정규화 또는 피처 추출 후에도 남을 수 있는 경우 대비)
        # Ensure we only check numeric columns that should exist based on STANDARD_FEATURES
        numeric_cols_in_df = [col for col in STANDARD_FEATURES if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col])]
        
        # Create a view for numeric columns to check/fill NaNs and Infs
        numeric_view = normalized_df[numeric_cols_in_df]

        # 데이터 타입 로깅 (디버깅용)
        logger.debug(f"숫자 컬럼 데이터 타입: {numeric_view.dtypes}")
        
        # 안전하게 NaN 및 inf 값 처리
        has_nulls = numeric_view.isnull().any().any()
        has_infs = False
        
        try:
            # 각 컬럼별로 개별 검사하여 숫자형 데이터만 inf 체크
            for col in numeric_cols_in_df:
                if pd.api.types.is_numeric_dtype(numeric_view[col]):
                    try:
                        col_vals = numeric_view[col].values
                        if np.any(np.isinf(col_vals)):
                            has_infs = True
                            logger.warning(f"무한값 발견: 컬럼 {col}")
                    except Exception as col_err:
                        logger.warning(f"컬럼 {col} inf 검사 중 오류: {str(col_err)}")
        except Exception as e:
            logger.warning(f"무한값 검사 중 오류 발생: {str(e)}")
        
        if has_nulls or has_infs:
            logger.warning("NaN/inf 값이 정규화 후에도 남아있습니다. 0으로 대체합니다.")
            # Fill NaNs and Infs directly in the original DataFrame for the identified numeric columns
            normalized_df[numeric_cols_in_df] = normalized_df[numeric_cols_in_df].fillna(0)
            
            # 안전하게 inf 값 대체
            for col in numeric_cols_in_df:
                if pd.api.types.is_numeric_dtype(normalized_df[col]):
                    try:
                        normalized_df[col] = normalized_df[col].replace([np.inf, -np.inf], 0)
                    except Exception as col_err:
                        logger.warning(f"컬럼 {col} inf 대체 중 오류: {str(col_err)}")
                        # 오류 발생 시 해당 컬럼을 0으로 채움
                        normalized_df[col] = 0

        # Step 4: Extract the latest window for prediction using the normalized standard features
        if len(normalized_df) < window_size:
            logger.warning(f"Not enough data for prediction. Need {window_size} samples, got {len(normalized_df)}")
            pad_size = window_size - len(normalized_df)

            # Identify numeric columns for padding based on STANDARD_FEATURES present in the df
            numeric_cols_to_pad = [col for col in STANDARD_FEATURES if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col])]

            # Padding logic (simplified - prioritize numeric index padding for consistency)
            logger.warning("Padding with zeros using numeric index.")
            pad_df = pd.DataFrame(np.zeros((pad_size, len(numeric_cols_to_pad))), columns=numeric_cols_to_pad)
            
            # Ensure consistent columns before concatenating
            current_data_numeric = normalized_df[numeric_cols_to_pad].reset_index(drop=True)
            
            # Concatenate padding and existing data
            combined_df = pd.concat([pad_df, current_data_numeric], ignore_index=True)

            # Restore original index type if possible, otherwise use RangeIndex
            try:
                 combined_df.index = pd.RangeIndex(start=0, stop=len(combined_df), step=1) # Use RangeIndex after padding
            except Exception as idx_err:
                 logger.warning(f"Could not set RangeIndex after padding: {idx_err}")
            
            normalized_df = combined_df # Update normalized_df with padded data

        # Ensure the final DataFrame before window extraction has the correct columns (STANDARD_FEATURES that are numeric)
        final_numeric_cols = [col for col in STANDARD_FEATURES if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col])]
        
        # Select only the numeric standard features for the final window
        numeric_data = normalized_df[final_numeric_cols]

        # Final check for NaN/inf after potential padding
        has_nulls = numeric_data.isnull().any().any()
        has_infs = False
        
        try:
            # 각 컬럼별로 개별 검사하여 숫자형 데이터만 inf 체크
            for col in final_numeric_cols:
                if pd.api.types.is_numeric_dtype(numeric_data[col]):
                    try:
                        col_vals = numeric_data[col].values
                        if np.any(np.isinf(col_vals)):
                            has_infs = True
                            logger.warning(f"패딩 후 무한값 발견: 컬럼 {col}")
                    except Exception as col_err:
                        logger.warning(f"패딩 후 컬럼 {col} inf 검사 중 오류: {str(col_err)}")
        except Exception as e:
            logger.warning(f"패딩 후 무한값 검사 중 오류 발생: {str(e)}")
        
        if has_nulls or has_infs:
            logger.warning("패딩 후에도 NaN/inf 값이 존재합니다. 0으로 채웁니다.")
            # NaN 채우기
            numeric_data = numeric_data.fillna(0)
            
            # 안전하게 inf 값 대체
            for col in final_numeric_cols:
                if pd.api.types.is_numeric_dtype(numeric_data[col]):
                    try:
                        numeric_data[col] = numeric_data[col].replace([np.inf, -np.inf], 0)
                    except Exception as col_err:
                        logger.warning(f"패딩 후 컬럼 {col} inf 대체 중 오류: {str(col_err)}")
                        # 오류 발생 시 해당 컬럼을 0으로 채움
                        numeric_data[col] = 0

        # Extract the latest window using only the numeric standard features
        if len(numeric_data) < window_size:
             logger.error(f"패딩 후에도 데이터가 부족합니다. ({len(numeric_data)}/{window_size})")
             return np.array([]) # 빈 배열 반환

        latest_window = numeric_data.values[-window_size:]

        # Reshape for model input (add batch dimension)
        # Shape should be (1, window_size, num_features) where num_features is len(final_numeric_cols)
        X = np.array([latest_window])

        logger.info(f"Prepared latest data for prediction, final shape: {X.shape}")
        return X

    except Exception as e:
        logger.error(f"Error preparing latest data for prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return np.array([])  # Return empty array if preparation fails

@log_execution
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a standard set of features from the DataFrame, ensuring consistency.

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV and indicators.

    Returns:
        pd.DataFrame: DataFrame with standard features selected, ordered, and NaN-filled.
    """
    try:
        result = df.copy()
        output_df = pd.DataFrame(index=result.index) # Create empty df with same index

        # Ensure all standard features exist, add missing as NaN
        added_missing = []
        for feature in STANDARD_FEATURES:
            if feature in result.columns:
                output_df[feature] = result[feature]
            else:
                # logger.warning(f"Standard feature '{feature}' not found in input DataFrame. Adding as NaN.")
                output_df[feature] = np.nan
                added_missing.append(feature)
        
        if added_missing:
            logger.warning(f"Added missing standard features (as NaN): {added_missing}")

        # Ensure the DataFrame contains exactly the STANDARD_FEATURES in the correct order
        output_df = output_df[STANDARD_FEATURES]

        # Fill NaN values using fillna(0)
        initial_nan_count = output_df.isna().sum().sum()
        if initial_nan_count > 0:
            output_df = output_df.fillna(0)
            final_nan_count = output_df.isna().sum().sum()
            if initial_nan_count - final_nan_count > 0:
                 logger.info(f"Filled {initial_nan_count - final_nan_count} NaN values in extracted features (using 0).")

        logger.info(f"Extracted features. Final shape: {output_df.shape}. Columns: {output_df.columns.tolist()[:5]}...{output_df.columns.tolist()[-5:]}") # Log first/last 5 cols
        return output_df

    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())

        # Fallback: Return basic OHLCV or empty df with standard columns
        logger.warning("Feature extraction failed. Returning fallback DataFrame.")
        fallback_df = pd.DataFrame(index=df.index)
        basic_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in STANDARD_FEATURES:
             fallback_df[col] = df[col].fillna(0) if col in df.columns and col in basic_cols else 0
        
        # Ensure standard columns and order even in fallback
        return fallback_df.reindex(columns=STANDARD_FEATURES, fill_value=0) 

class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.data_dir = 'data/raw'
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """데이터 로드"""
        try:
            # 데이터 파일 찾기
            data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not data_files:
                logger.error("데이터 파일이 없습니다.")
                return None
                
            # 가장 최근 파일 선택
            latest_file = max(data_files)
            data_path = os.path.join(self.data_dir, latest_file)
            
            # 데이터 로드
            data = pd.read_csv(data_path)
            
            # 시간 컬럼을 인덱스로 설정
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
                
            # 숫자형 컬럼만 선택
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_columns]
                
            logger.info(f"데이터 로드 완료: {data_path}")
            return data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return None
            
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        try:
            # 기본 가격 데이터 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error("필수 가격 데이터가 없습니다.")
                return data
            
            # 추천된 20개 핵심 특성을 위한 기술적 지표 계산
            # 추천된 20개 특성 목록:
            # 1. close, 2. high, 3. low, 4. volume (기본 OHLCV 중에서)
            # 5. sma7, 6. ema12, 7. ema26, 8. ema200 (이동평균)
            # 9. rsi14 (상대강도지수)
            # 10. macd_diff (MACD 오실레이터)
            # 11. stoch_k14 (스토캐스틱 %K)
            # 12. roc10 (10일 가격 변화율)
            # 13. atr14 (평균 실제 범위)
            # 14. bb_percent_b20 (볼린저 밴드 %B)
            # 15. obv (On Balance Volume)
            # 16. cmf20 (Chaikin Money Flow)
            # 17. adx14 (평균 방향성 지수)
            # 18. price_change_1 (1봉 가격 변화율)
            # 19. volume_change_1 (1봉 거래량 변화율)
            # 20. ema12_ema26_dist_norm (EMA 간 거리, 정규화)
            
            # 기본 데이터는 이미 있음 (close, high, low, volume)
            
            # 이동평균 지표
            data['sma7'] = ta.trend.sma_indicator(data['close'], window=7)
            data['ema12'] = ta.trend.ema_indicator(data['close'], window=12)
            data['ema26'] = ta.trend.ema_indicator(data['close'], window=26)
            data['ema200'] = ta.trend.ema_indicator(data['close'], window=200)
            
            # RSI 지표
            data['rsi14'] = ta.momentum.rsi(data['close'], window=14)
            
            # MACD 지표
            macd = ta.trend.MACD(data['close'])
            data['macd_diff'] = macd.macd_diff()
            
            # 스토캐스틱 지표
            stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14)
            data['stoch_k14'] = stoch.stoch()
            
            # 가격 변화율
            data['roc10'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100
            
            # ATR (Average True Range)
            data['atr14'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            
            # 볼린저 밴드 %B
            bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            data['bb_percent_b20'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # On Balance Volume
            data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
            
            # Chaikin Money Flow
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            volume = pd.to_numeric(data['volume'], errors='coerce')
            
            # Money Flow Multiplier 계산
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Money Flow Volume 계산
            mfv = mfm * volume
            
            # CMF 계산 (20일)
            data['cmf20'] = mfv.rolling(window=20).sum() / volume.rolling(window=20).sum()
            data['cmf20'] = data['cmf20'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Average Directional Index (ADX)
            adx = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14)
            data['adx14'] = adx.adx()
            
            # 1봉 가격 및 거래량 변화율
            data['price_change_1'] = data['close'].pct_change(periods=1)
            data['volume_change_1'] = data['volume'].pct_change(periods=1)
            
            # EMA 간 거리 정규화
            ema_dist = data['ema12'] - data['ema26']
            data['ema12_ema26_dist_norm'] = ema_dist / data['close']
            
            # NaN 값 처리
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            data.fillna(0, inplace=True)
            
            logger.info("기술적 지표 추가 완료")
            return data
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return data

def filter_indicators(df: pd.DataFrame, indicator_list: List[str] = None) -> pd.DataFrame:
    """
    필터링된 지표들만 포함하는 데이터프레임을 반환합니다.
    
    Args:
        df (pd.DataFrame): 모든 지표가 포함된 데이터프레임
        indicator_list (List[str], optional): 유지할 지표 리스트. None이면 기본 지표 사용.
        
    Returns:
        pd.DataFrame: 필터링된 지표들만 포함하는 데이터프레임
    """
    try:
        if indicator_list is None:
            # 기본 주요 지표 목록
            indicator_list = [
                'open', 'high', 'low', 'close', 'volume',  # 기본 OHLCV
                'sma7', 'sma25', 'ema12', 'ema26', 'ema200',  # 이동평균
                'macd', 'macd_signal', 'macd_diff',  # MACD
                'rsi14',  # RSI
                'bb_upper', 'bb_middle', 'bb_lower',  # 볼린저 밴드
                'stoch_k', 'stoch_d',  # 스토캐스틱
                'atr', 'obv',  # 기타 인기 지표
                'daily_return'  # 수익률
            ]
        
        # 실제 데이터프레임에 존재하는 컬럼만 필터링
        available_columns = [col for col in indicator_list if col in df.columns]
        
        if len(available_columns) < len(indicator_list):
            missing_columns = set(indicator_list) - set(available_columns)
            logger.warning(f"일부 요청된 지표가 데이터프레임에 없습니다: {missing_columns}")
        
        # 필터링된 데이터프레임 반환
        return df[available_columns].copy()
    
    except Exception as e:
        logger.error(f"지표 필터링 중 오류 발생: {str(e)}")
        # 기본 OHLCV 컬럼만 반환 (최소한의 안전망)
        basic_cols = ['open', 'high', 'low', 'close', 'volume']
        available_basic = [col for col in basic_cols if col in df.columns]
        return df[available_basic].copy()