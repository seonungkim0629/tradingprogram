"""
Data utilities for Bitcoin Trading Bot

This module provides utility functions for data preparation,
feature extraction, and other data-related operations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import requests
import json
from time import sleep
import os

from config import settings

# Initialize logger
logger = logging.getLogger(__name__)

def prepare_model_data(
    data: pd.DataFrame, 
    n_steps: int = 30, 
    target_column: str = 'close',
    target_shift: int = 1,
    features: Optional[List[str]] = None,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    머신러닝 모델용 데이터 준비. 시계열 데이터를 시퀀스 형태로 변환.
    
    Args:
        data (pd.DataFrame): 원본 시장 데이터프레임 (OHLCV 및 기술적 지표 포함)
        n_steps (int): 시퀀스 길이 (룩백 기간)
        target_column (str): 목표 변수 컬럼 (예: 'close')
        target_shift (int): 목표 예측 시점 (1이면 다음 날 예측)
        features (Optional[List[str]]): 사용할 특성 목록. None이면 가능한 모든 수치형 특성 사용
        scale (bool): 정규화 여부
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) 시퀀스 데이터와 목표 변수
    """
    if len(data) < n_steps + target_shift:
        logger.warning(f"Not enough data for sequence creation: {len(data)} rows, need {n_steps + target_shift}")
        return None, None
        
    # 사용할 특성 결정
    if features is None:
        # 수치형 컬럼만 선택
        features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and not pd.isna(data[col]).any()]
    
    # 데이터 확인
    for feature in features:
        if feature not in data.columns:
            logger.warning(f"Feature '{feature}' not in data columns. Available columns: {data.columns.tolist()}")
            features.remove(feature)
    
    if not features:
        logger.error("No valid features available for model preparation")
        return None, None
    
    # 데이터 준비
    df = data[features].copy()
    
    # 결측치 처리
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 데이터 정규화
    if scale:
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()
            if std > 0:
                df[column] = (df[column] - mean) / std
    
    # 방향성 계산 (분류 문제용)
    if target_shift > 0:
        # 다음 날 종가와 비교해 상승(1)/하락(0) 레이블 생성
        df['direction'] = (df[target_column].shift(-target_shift) > df[target_column]).astype(int)
    
    # 결측치 제거
    df = df.dropna()
    
    # 시퀀스 데이터 생성
    X, y = [], []
    
    for i in range(len(df) - n_steps - target_shift + 1):
        # 입력 시퀀스
        X.append(df.iloc[i:i+n_steps][features].values)
        
        # 목표 변수 (방향성)
        if target_shift > 0:
            y.append(df.iloc[i+n_steps-1+target_shift]['direction'])
    
    # 배열 변환
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared model data: X shape {X.shape}, y shape {y.shape}")
    return X, y

def extract_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    가격 기반 특성 추출
    
    Args:
        df (pd.DataFrame): OHLCV 데이터
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    df_features = df.copy()
    
    # 가격 변화율
    df_features['price_change'] = df_features['close'].pct_change()
    df_features['price_change_1d'] = df_features['close'].pct_change(1)
    df_features['price_change_3d'] = df_features['close'].pct_change(3)
    df_features['price_change_5d'] = df_features['close'].pct_change(5)
    
    # 고가-저가 범위
    df_features['high_low_range'] = (df_features['high'] - df_features['low']) / df_features['close']
    
    # 거래량 변화
    df_features['volume_change'] = df_features['volume'].pct_change()
    df_features['volume_change_1d'] = df_features['volume'].pct_change(1)
    df_features['volume_change_3d'] = df_features['volume'].pct_change(3)
    
    # 볼린저 밴드 위치
    if 'bb_upper' in df_features.columns and 'bb_lower' in df_features.columns:
        df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
    
    # 이동평균 대비 가격
    if 'sma20' in df_features.columns:
        df_features['price_to_sma20'] = df_features['close'] / df_features['sma20'] - 1
    if 'ema50' in df_features.columns:
        df_features['price_to_ema50'] = df_features['close'] / df_features['ema50'] - 1
    
    # 시가-종가 차이
    df_features['open_close_diff'] = (df_features['close'] - df_features['open']) / df_features['open']
    
    return df_features

def rolling_window_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    롤링 윈도우 기반 통계 특성 추출
    
    Args:
        df (pd.DataFrame): OHLCV 데이터
        windows (List[int]): 롤링 윈도우 크기
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    df_features = df.copy()
    
    # 각 윈도우 크기별 특성 추출
    for window in windows:
        # 가격 변동성
        df_features[f'volatility_{window}d'] = df_features['close'].pct_change().rolling(window).std()
        
        # 롤링 평균 대비 가격
        df_features[f'price_to_ma_{window}d'] = df_features['close'] / df_features['close'].rolling(window).mean() - 1
        
        # 거래량 이상치
        df_features[f'volume_to_ma_{window}d'] = df_features['volume'] / df_features['volume'].rolling(window).mean()
        
        # 롤링 최고가, 최저가 대비 현재가
        df_features[f'close_to_max_{window}d'] = df_features['close'] / df_features['high'].rolling(window).max()
        df_features[f'close_to_min_{window}d'] = df_features['close'] / df_features['low'].rolling(window).min()
    
    return df_features

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 관련 특성 추가
    
    Args:
        df (pd.DataFrame): 인덱스가 datetime인 데이터프레임
        
    Returns:
        pd.DataFrame: 날짜 특성이 추가된 데이터프레임
    """
    df_features = df.copy()
    
    # 인덱스가 datetime인지 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex. Converting index to datetime.")
        df_features.index = pd.to_datetime(df_features.index)
    
    # 요일 (0=월요일, 6=일요일)
    df_features['dayofweek'] = df_features.index.dayofweek
    
    # 월 (1-12)
    df_features['month'] = df_features.index.month
    
    # 분기 (1-4)
    df_features['quarter'] = df_features.index.quarter
    
    # 연도의 주 (1-53)
    df_features['weekofyear'] = df_features.index.isocalendar().week
    
    # 일년 중 날짜 (1-366)
    df_features['dayofyear'] = df_features.index.dayofyear
    
    # 사인 변환 (주기성 표현)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    
    return df_features

def ensure_feature_consistency(data_dict, feature_axis=2, common_strategy='intersection'):
    """
    여러 데이터셋 간의 특성 일관성을 유지합니다.
    
    Args:
        data_dict (dict): 데이터셋 딕셔너리 (예: {'train': X_train, 'val': X_val, 'test': X_test})
        feature_axis (int): 특성이 위치한 축 (기본값 2, 일반적으로 (samples, time_steps, features))
        common_strategy (str): 일관성 전략 ('intersection': 공통 특성만 유지, 'union': 모든 특성 유지하고 없는 값은 0으로 채움)
    
    Returns:
        dict: 특성이 조정된 데이터셋 딕셔너리
    """
    if len(data_dict) <= 1:
        return data_dict  # 하나 이하의 데이터셋만 있으면 변경 필요 없음
    
    # 각 데이터셋의 특성 수 확인
    feature_counts = {k: v.shape[feature_axis] for k, v in data_dict.items() if v is not None and len(v.shape) > feature_axis}
    
    # 특성 수가 모두 동일한지 확인
    if len(set(feature_counts.values())) == 1:
        logger.info(f"모든 데이터셋의 특성 수가 동일합니다: {list(feature_counts.values())[0]}")
        return data_dict  # 특성 수가 이미 동일하면 변경 필요 없음
    
    # 특성 수가 다른 경우, 조정 필요
    logger.warning(f"데이터셋 간 특성 수 불일치 발견: {feature_counts}")
    
    if common_strategy == 'intersection':
        # 모든 데이터셋의 공통 특성만 사용
        min_features = min(feature_counts.values())
        logger.info(f"공통 특성 수를 {min_features}로 조정합니다.")
        
        adjusted_data = {}
        for name, data in data_dict.items():
            if data is None or len(data.shape) <= feature_axis:
                adjusted_data[name] = data  # 변경 없음
                continue
                
            if data.shape[feature_axis] > min_features:
                # 필요한 특성만 선택
                if feature_axis == 2:  # (samples, time_steps, features)
                    adjusted_data[name] = data[:, :, :min_features]
                elif feature_axis == 1:  # (samples, features)
                    adjusted_data[name] = data[:, :min_features]
                else:
                    adjusted_data[name] = np.take(data, range(min_features), axis=feature_axis)
                    
                logger.info(f"데이터셋 '{name}'의 특성을 {data.shape[feature_axis]}에서 {min_features}로 조정했습니다.")
            else:
                adjusted_data[name] = data  # 이미 최소 특성 수보다 작거나 같음
        
        return adjusted_data
    
    elif common_strategy == 'union':
        # 모든 특성을 유지하고 필요한 경우 0으로 채움
        max_features = max(feature_counts.values())
        logger.info(f"모든 데이터셋의 특성 수를 {max_features}로 확장합니다.")
        
        adjusted_data = {}
        for name, data in data_dict.items():
            if data is None or len(data.shape) <= feature_axis:
                adjusted_data[name] = data  # 변경 없음
                continue
                
            if data.shape[feature_axis] < max_features:
                # 누락된 특성을 0으로 채움
                if feature_axis == 2:  # (samples, time_steps, features)
                    pad_width = ((0, 0), (0, 0), (0, max_features - data.shape[2]))
                elif feature_axis == 1:  # (samples, features)
                    pad_width = ((0, 0), (0, max_features - data.shape[1]))
                else:
                    pad_shape = list(data.shape)
                    pad_shape[feature_axis] = max_features - data.shape[feature_axis]
                    pad_width = [(0, 0)] * len(data.shape)
                    pad_width[feature_axis] = (0, max_features - data.shape[feature_axis])
                
                adjusted_data[name] = np.pad(data, pad_width, mode='constant', constant_values=0)
                logger.info(f"데이터셋 '{name}'의 특성을 {data.shape[feature_axis]}에서 {max_features}로 확장했습니다.")
            else:
                adjusted_data[name] = data  # 이미 최대 특성 수
        
        return adjusted_data
    
    else:
        logger.warning(f"알 수 없는 특성 조정 전략: {common_strategy}, 원본 데이터셋 반환")
        return data_dict 

def get_fear_greed_index() -> Dict[str, Any]:
    """
    Alternative.me의 Fear & Greed Index API에서 현재 공포탐욕지수를 가져옵니다.
    
    Returns:
        Dict[str, Any]: 공포탐욕지수 정보를 담은 사전
            {
                'value': (int) 공포탐욕지수 값 (0-100),
                'value_classification': (str) 지수 분류,
                'timestamp': (str) 타임스탬프,
                'time_until_update': (str) 다음 업데이트까지 남은 시간
            }
    """
    # API URL
    url = "https://api.alternative.me/fng/"
    
    try:
        # API 요청
        response = requests.get(url, timeout=10)
        
        # 응답 확인
        if response.status_code == 200:
            data = response.json()
            
            # 데이터 형식 확인
            if 'data' in data and len(data['data']) > 0:
                latest_data = data['data'][0]
                
                # 필요한 정보 추출
                result = {
                    'value': int(latest_data.get('value', 50)),  # 값이 없으면 중립값(50) 반환
                    'value_classification': latest_data.get('value_classification', 'Neutral'),
                    'timestamp': latest_data.get('timestamp', ''),
                    'time_until_update': latest_data.get('time_until_update', '')
                }
                
                logger.info(f"공포탐욕지수: {result['value']} ({result['value_classification']})")
                return result
            else:
                logger.warning("API 응답에 데이터가 없습니다. 기본값 반환.")
                return {
                    'value': 50,  # 중립값 반환
                    'value_classification': 'Neutral',
                    'timestamp': '',
                    'time_until_update': ''
                }
        else:
            logger.error(f"API 요청 실패. 상태 코드: {response.status_code}")
            return None
            
    except requests.RequestException as e:
        logger.error(f"API 요청 중 오류 발생: {str(e)}")
        # 네트워크 오류 시 재시도
        try:
            logger.info("API 요청 재시도 중...")
            sleep(2)  # 2초 대기 후 재시도
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    latest_data = data['data'][0]
                    result = {
                        'value': int(latest_data.get('value', 50)),
                        'value_classification': latest_data.get('value_classification', 'Neutral'),
                        'timestamp': latest_data.get('timestamp', ''),
                        'time_until_update': latest_data.get('time_until_update', '')
                    }
                    logger.info(f"공포탐욕지수 재시도 성공: {result['value']} ({result['value_classification']})")
                    return result
            logger.warning("API 재시도 실패. 기본값 반환.")
        except Exception as retry_error:
            logger.error(f"API 재시도 중 오류 발생: {str(retry_error)}")
        
        # 오류 시 기본값 반환
        return {
            'value': 50,  # 중립값 반환
            'value_classification': 'Neutral',
            'timestamp': '',
            'time_until_update': ''
        }
    except Exception as e:
        logger.error(f"알 수 없는 오류 발생: {str(e)}")
        # 오류 시 기본값 반환
        return {
            'value': 50,  # 중립값 반환
            'value_classification': 'Neutral',
            'timestamp': '',
            'time_until_update': ''
        }

def get_fear_greed_state() -> str:
    """
    공포탐욕지수 API에서 시장 상태를 가져옵니다.
    
    Returns:
        str: 시장 상태 ('extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed')
    """
    # 공포탐욕지수 가져오기
    fgi_data = get_fear_greed_index()
    
    if fgi_data is None:
        logger.warning("공포탐욕지수를 가져오지 못했습니다. 중립 상태를 반환합니다.")
        return "neutral"
    
    # 지수값을 이용해 시장 상태 결정
    fgi_value = fgi_data['value']
    
    if fgi_value < 25:
        return "extreme_fear"
    elif fgi_value < 45:
        return "fear"
    elif fgi_value < 55:
        return "neutral"
    elif fgi_value < 75:
        return "greed"
    else:
        return "extreme_greed"

def clean_dataframe(df: pd.DataFrame, handle_missing: str = 'interpolate') -> pd.DataFrame:
    """
    데이터프레임 정리: 결측치 처리, 문자열 열 제거, 중복 제거 등
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        handle_missing (str): 결측치 처리 방법 ('drop', 'interpolate', 'ffill', 'bfill', 'zero')
        
    Returns:
        pd.DataFrame: 정리된 데이터프레임
    """
    if df is None or df.empty:
        logger.warning("데이터프레임이 비어있습니다.")
        return pd.DataFrame()
    
    # 복사본으로 작업
    clean_df = df.copy()
    
    # 중복 인덱스 제거
    if clean_df.index.duplicated().any():
        clean_df = clean_df[~clean_df.index.duplicated(keep='last')]
        logger.info(f"{sum(df.index.duplicated())}개의 중복 인덱스가 제거되었습니다.")
    
    # 숫자형 열만 유지
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(clean_df.columns):
        logger.info(f"{len(clean_df.columns) - len(numeric_cols)}개의 비숫자형 열이 제거되었습니다.")
        clean_df = clean_df[numeric_cols]
    
    # 결측치 처리
    if clean_df.isna().any().any():
        na_count = clean_df.isna().sum().sum()
        
        if handle_missing == 'drop':
            clean_df = clean_df.dropna()
            logger.info(f"행 삭제로 {na_count}개의 결측치가 처리되었습니다.")
        elif handle_missing == 'interpolate':
            clean_df = clean_df.interpolate(method='linear')
            clean_df = clean_df.fillna(method='ffill').fillna(method='bfill')
            logger.info(f"보간으로 {na_count}개의 결측치가 처리되었습니다.")
        elif handle_missing == 'ffill':
            clean_df = clean_df.fillna(method='ffill')
            clean_df = clean_df.fillna(method='bfill')  # 시작 부분의 NaN 처리
            logger.info(f"순방향 채움으로 {na_count}개의 결측치가 처리되었습니다.")
        elif handle_missing == 'bfill':
            clean_df = clean_df.fillna(method='bfill')
            clean_df = clean_df.fillna(method='ffill')  # 끝부분의 NaN 처리
            logger.info(f"역방향 채움으로 {na_count}개의 결측치가 처리되었습니다.")
        elif handle_missing == 'zero':
            clean_df = clean_df.fillna(0)
            logger.info(f"0 대체로 {na_count}개의 결측치가 처리되었습니다.")
    
    # 인덱스가 시간순으로 정렬되었는지 확인
    if not clean_df.index.is_monotonic_increasing:
        clean_df = clean_df.sort_index()
        logger.info("인덱스가 시간순으로 정렬되었습니다.")
    
    return clean_df

def load_dataframe_with_cache(filepath: str, cache_dir: str = None, cache_key: str = None,
                             force_reload: bool = False, **read_kwargs) -> pd.DataFrame:
    """
    데이터프레임을 로드하고 캐싱합니다. 동일한 파일을 여러 번 읽을 때 성능 향상을 위한 함수입니다.
    
    Args:
        filepath (str): 로드할 파일 경로 (.csv, .parquet, .pkl 등)
        cache_dir (str, optional): 캐시 디렉토리. 기본값은 None (캐시 사용 안함).
        cache_key (str, optional): 캐시 키. 기본값은 None (파일명 사용).
        force_reload (bool, optional): 캐시를 무시하고 강제로 다시 로드할지 여부. 기본값은 False.
        **read_kwargs: pd.read_csv, pd.read_parquet 등에 전달할 추가 인자
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    # 캐시 키가 없으면 파일명에서 생성
    if cache_key is None:
        cache_key = os.path.basename(filepath)
    
    # 캐시 디렉토리 설정
    if cache_dir is None:
        cache_dir = os.path.join(settings.DATA_DIR, '.cache')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 파일 경로
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # 캐시가 있고 강제 리로드가 아니면 캐시에서 로드
    if not force_reload and os.path.exists(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            logger.debug(f"캐시에서 데이터 로드: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"캐시 로드 실패, 원본에서 다시 로드합니다: {str(e)}")
    
    # 파일 확장자에 따라 적절한 읽기 함수 선택
    _, ext = os.path.splitext(filepath.lower())
    
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, **read_kwargs)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath, **read_kwargs)
        elif ext in ['.pkl', '.pickle']:
            df = pd.read_pickle(filepath, **read_kwargs)
        elif ext == '.feather':
            df = pd.read_feather(filepath, **read_kwargs)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **read_kwargs)
        elif ext == '.json':
            df = pd.read_json(filepath, **read_kwargs)
        else:
            logger.error(f"지원되지 않는 파일 포맷: {ext}")
            return pd.DataFrame()
        
        # 캐시 저장
        if cache_dir is not None:
            df.to_pickle(cache_path)
            logger.debug(f"데이터 캐시 저장: {cache_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"파일 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame() 