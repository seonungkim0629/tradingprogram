"""
데이터 변환 유틸리티 모듈

이 모듈은 다양한 데이터 형식 간의 변환 함수를 제공합니다.
"""

from typing import Dict, Any, Union, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.constants import TimeFrame, DataColumn, SignalType
from models.signal import TradingSignal, standardize_signal, ModelOutput, standardize_model_output


def standardize_timeframe_key(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    데이터 딕셔너리의 타임프레임 키를 표준화
    
    Args:
        data (Dict[str, pd.DataFrame]): 원본 데이터 딕셔너리
        
    Returns:
        Dict[str, pd.DataFrame]: 표준화된 키를 가진 데이터 딕셔너리
    """
    if not data:
        return {}
    
    result = {}
    for key, df in data.items():
        std_key = TimeFrame.standardize(key)
        result[std_key] = df
    
    return result


def ensure_dict_data(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    default_timeframe: str = TimeFrame.DAY) -> Dict[str, pd.DataFrame]:
    """
    데이터를 항상 딕셔너리 형태로 변환
    
    Args:
        data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): 입력 데이터
        default_timeframe (str): DataFrame이 직접 주어진 경우 사용할 타임프레임 키
        
    Returns:
        Dict[str, pd.DataFrame]: 딕셔너리 형태의 데이터
    """
    if isinstance(data, dict):
        return standardize_timeframe_key(data)
    
    if isinstance(data, pd.DataFrame):
        std_key = TimeFrame.standardize(default_timeframe)
        return {std_key: data}
    
    return {}


def convert_signal_format(signal: Union[Dict[str, Any], TradingSignal, ModelOutput]) -> Dict[str, Any]:
    """
    신호를 딕셔너리 형식으로 표준화
    
    Args:
        signal (Union[Dict[str, Any], TradingSignal, ModelOutput]): 입력 신호
        
    Returns:
        Dict[str, Any]: 딕셔너리 형식의 신호
    """
    if isinstance(signal, dict):
        return signal
    
    if isinstance(signal, TradingSignal):
        return signal.to_dict()
    
    if isinstance(signal, ModelOutput):
        return signal.signal.to_dict()
    
    # 기본 신호 반환
    return {
        'signal': SignalType.HOLD,
        'confidence': 0.5,
        'reason': '신호 변환 실패',
        'metadata': {}
    }


def convert_model_output_format(output: Union[Dict[str, Any], TradingSignal, ModelOutput]) -> Dict[str, Any]:
    """
    모델 출력을 딕셔너리 형식으로 표준화
    
    Args:
        output (Union[Dict[str, Any], TradingSignal, ModelOutput]): 입력 모델 출력
        
    Returns:
        Dict[str, Any]: 딕셔너리 형식의 모델 출력
    """
    if isinstance(output, dict):
        return output
    
    if isinstance(output, ModelOutput):
        return output.to_dict()
    
    if isinstance(output, TradingSignal):
        # TradingSignal을 ModelOutput으로 변환 후 딕셔너리 반환
        model_output = ModelOutput.from_signal(output)
        return model_output.to_dict()
    
    # 기본 모델 출력 반환
    return {
        'signal': {
            'signal': SignalType.HOLD,
            'confidence': 0.5,
            'reason': '모델 출력 변환 실패',
            'metadata': {}
        },
        'confidence': 0.5,
        'metadata': {}
    }


def detect_timeframe(data: pd.DataFrame) -> str:
    """
    데이터의 타임프레임 자동 감지
    
    Args:
        data (pd.DataFrame): 입력 데이터
        
    Returns:
        str: 감지된 타임프레임 (표준 형식)
    """
    if data is None or data.empty or len(data) < 2:
        return TimeFrame.DAY  # 기본값
    
    # 인덱스가 datetime인지 확인
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            # 첫 번째 열이 datetime인지 확인
            if isinstance(data.iloc[0, 0], datetime):
                timestamps = data.iloc[:, 0]
            else:
                return TimeFrame.DAY  # datetime을 찾을 수 없음
        except:
            return TimeFrame.DAY  # 기본값
    else:
        timestamps = data.index
    
    # 연속된 두 타임스탬프 간의 평균 간격 계산
    time_diffs = []
    for i in range(1, min(len(timestamps), 10)):  # 최대 10개 샘플 사용
        diff = timestamps[i] - timestamps[i-1]
        time_diffs.append(diff.total_seconds())
    
    if not time_diffs:
        return TimeFrame.DAY  # 기본값
    
    avg_seconds = sum(time_diffs) / len(time_diffs)
    
    # 평균 간격에 따라 타임프레임 결정
    if avg_seconds < 120:  # 2분 미만
        return TimeFrame.MINUTE1
    elif avg_seconds < 600:  # 10분 미만
        return TimeFrame.MINUTE5
    elif avg_seconds < 1800:  # 30분 미만
        return TimeFrame.MINUTE15
    elif avg_seconds < 3600:  # 1시간 미만
        return TimeFrame.MINUTE30
    elif avg_seconds < 14400:  # 4시간 미만
        return TimeFrame.HOUR1
    elif avg_seconds < 86400:  # 1일 미만
        return TimeFrame.HOUR4
    elif avg_seconds < 604800:  # 1주일 미만
        return TimeFrame.DAY
    else:
        return TimeFrame.WEEK


def normalize_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
    """
    데이터프레임 형식 정규화
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        required_columns (List[str], optional): 필수 컬럼 목록
        
    Returns:
        pd.DataFrame: 정규화된 데이터프레임
    """
    if df is None or df.empty:
        return df
    
    # 복사본 생성
    result = df.copy()
    
    # 컬럼명 소문자로 변환
    result.columns = [col.lower() for col in result.columns]
    
    # 필수 컬럼이 지정되지 않았으면 OHLCV 사용
    if required_columns is None:
        required_columns = DataColumn.OHLCV_COLUMNS
    
    # 필수 컬럼 존재 확인 및 기본값 추가
    for col in required_columns:
        if col not in result.columns:
            if col in DataColumn.OHLCV_COLUMNS:
                # OHLCV 컬럼이면 다른 가용한 가격 컬럼에서 복사
                if col == DataColumn.OPEN and DataColumn.CLOSE in result.columns:
                    result[col] = result[DataColumn.CLOSE]
                elif col == DataColumn.HIGH and DataColumn.CLOSE in result.columns:
                    result[col] = result[DataColumn.CLOSE]
                elif col == DataColumn.LOW and DataColumn.CLOSE in result.columns:
                    result[col] = result[DataColumn.CLOSE]
                elif col == DataColumn.CLOSE and DataColumn.OPEN in result.columns:
                    result[col] = result[DataColumn.OPEN]
                elif col == DataColumn.VOLUME:
                    result[col] = 0  # 볼륨 기본값
            else:
                # 기타 컬럼은 NaN 추가
                result[col] = np.nan
    
    return result


def prepare_data_for_strategy(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    전략에 사용할 데이터 준비
    
    Args:
        data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): 원본 데이터
        
    Returns:
        Dict[str, pd.DataFrame]: 표준화된 데이터 딕셔너리
    """
    # 데이터를 딕셔너리 형태로 변환
    data_dict = ensure_dict_data(data)
    
    # 각 데이터프레임 정규화
    result = {}
    for timeframe, df in data_dict.items():
        # 타임프레임 키 표준화
        std_timeframe = TimeFrame.standardize(timeframe)
        
        # 데이터프레임 정규화
        result[std_timeframe] = normalize_dataframe(df)
    
    return result


def predict_and_format(model, data) -> ModelOutput:
    """
    모델 예측 및 출력 형식 표준화
    
    Args:
        model: 예측 모델
        data: 입력 데이터
        
    Returns:
        ModelOutput: 표준화된 모델 출력
    """
    try:
        # 모델 예측
        raw_output = model.predict(data)
        
        # 출력 형식 표준화
        model_output = standardize_model_output(raw_output)
        
        return model_output
    except Exception as e:
        # 예외 발생 시 기본 HOLD 신호 반환
        signal = TradingSignal(
            signal_type=SignalType.HOLD,
            reason=f"예측 오류: {str(e)}",
            confidence=0.0
        )
        return ModelOutput(signal=signal, metadata={"error": str(e)}) 