"""
데이터 전처리 모듈

시계열 데이터 전처리를 위한 다양한 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Union, Optional, Tuple, Any

from utils.logging import get_logger, log_execution

# 로거 초기화
logger = get_logger(__name__)

@log_execution
def add_target_variable(df: pd.DataFrame, 
                       target_type: str = 'next_close', 
                       horizon: int = 1) -> pd.DataFrame:
    """
    데이터프레임에 타겟 변수 추가
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        target_type (str, optional): 타겟 유형. 기본값은 'next_close'.
                      'next_close': 다음 기간의 종가
                      'next_return': 다음 기간의 수익률
                      'next_direction': 다음 기간의 방향 (상승=1, 하락=-1, 유지=0)
                      'binary_direction': 다음 기간의 방향 (상승=1, 하락=0)
        horizon (int, optional): 예측 기간. 기본값은 1.
        
    Returns:
        pd.DataFrame: 타겟 변수가 추가된 데이터프레임
    """
    try:
        result = df.copy()
        
        # 필수 컬럼 확인
        if 'close' not in result.columns:
            logger.warning("종가 컬럼이 데이터프레임에 없습니다. 타겟 변수를 추가할 수 없습니다.")
            return df
        
        # 타겟 유형에 따른 변수 추가
        if target_type == 'next_close':
            # 다음 기간의 종가
            result['target'] = result['close'].shift(-horizon)
            logger.info(f"{horizon}기간 후의 종가를 타겟으로 추가했습니다.")
            
        elif target_type == 'next_return':
            # 다음 기간의 수익률
            result['target'] = result['close'].pct_change(periods=-horizon) * 100
            logger.info(f"{horizon}기간 후의 수익률을 타겟으로 추가했습니다.")
            
        elif target_type == 'next_direction':
            # 다음 기간의 방향 (상승=1, 하락=-1, 유지=0)
            future_close = result['close'].shift(-horizon)
            current_close = result['close']
            
            # 방향 계산 - 명시적 비교로 변경
            direction = pd.Series(0, index=result.index)  # 기본값은 0 (유지)
            
            # 각 인덱스에 대해 개별적으로 방향 계산
            for idx in result.index:
                curr = current_close.loc[idx]
                if idx in future_close.index:
                    fut = future_close.loc[idx]
                    if not pd.isna(curr) and not pd.isna(fut):
                        if fut > curr:
                            direction.loc[idx] = 1  # 상승
                        elif fut < curr:
                            direction.loc[idx] = -1  # 하락
                        # else는 기본값 0 유지
            
            result['target'] = direction
            logger.info(f"{horizon}기간 후의 가격 방향을 타겟으로 추가했습니다.")
            
        elif target_type == 'binary_direction':
            # 다음 기간의 방향 (상승 또는 유지=1, 하락=0)
            future_close = result['close'].shift(-horizon)
            current_close = result['close']
            
            # 방향 계산 - 명시적 비교로 변경
            binary_direction = pd.Series(0, index=result.index)  # 기본값은 0 (하락)
            
            # 각 인덱스에 대해 개별적으로 방향 계산
            for idx in result.index:
                curr = current_close.loc[idx]
                if idx in future_close.index:
                    fut = future_close.loc[idx]
                    if not pd.isna(curr) and not pd.isna(fut):
                        if fut >= curr:  # 상승 또는 유지
                            binary_direction.loc[idx] = 1
            
            result['target'] = binary_direction
            logger.info(f"{horizon}기간 후의 이진 방향을 타겟으로 추가했습니다.")
            
        else:
            logger.warning(f"지원되지 않는 타겟 유형: {target_type}")
            return df
        
        # NaN 값 확인
        target_nan_count = result['target'].isna().sum()
        if target_nan_count > 0:
            logger.warning(f"타겟 변수에 {target_nan_count}개의 NaN 값이 있습니다. 대부분 데이터 끝 부분일 것입니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"타겟 변수 추가 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df 