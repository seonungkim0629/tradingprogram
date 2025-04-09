"""
시장 상황 모듈 (Market Context Module)

이 모듈은 시장 상황(변동성, 추세 등)에 따른 가중치 및 신뢰도 조정 함수를 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from ensemble.weights import normalize_weights


def calculate_market_volatility(prices: pd.Series, 
                             window: int = 20) -> float:
    """
    시장 변동성 계산 함수
    
    Args:
        prices (pd.Series): 가격 시계열
        window (int): 계산 윈도우 크기
        
    Returns:
        float: 변동성
    """
    if len(prices) < window:
        return 0.02  # 기본값
    
    # 수익률 계산
    returns = prices.pct_change().dropna()
    
    # 변동성 계산 (최근 window 기간)
    volatility = returns.tail(window).std()
    return volatility


def calculate_trend_strength(prices: pd.Series, 
                          ma_window: int = 20) -> float:
    """
    추세 강도 계산 함수
    
    Args:
        prices (pd.Series): 가격 시계열
        ma_window (int): 이동평균 기간
        
    Returns:
        float: 추세 강도
    """
    if len(prices) < ma_window:
        return 0.01  # 기본값
    
    # 이동평균 계산
    ma = prices.rolling(ma_window).mean()
    
    # 현재 가격과 이동평균 차이로 추세 강도 계산
    current_price = prices.iloc[-1]
    current_ma = ma.iloc[-1]
    
    if pd.isna(current_ma) or current_ma == 0:
        return 0.01  # 기본값
    
    # 추세 강도: 가격과 이동평균의 상대적 차이 (절대값)
    trend_strength = abs(current_price / current_ma - 1)
    return trend_strength


def extract_market_features(market_data: pd.DataFrame) -> Dict[str, float]:
    """
    시장 데이터에서 특징 추출 함수
    
    Args:
        market_data (pd.DataFrame): OHLCV 시장 데이터
        
    Returns:
        Dict[str, float]: 추출된 시장 특성
    """
    if not isinstance(market_data, pd.DataFrame) or len(market_data) < 20:
        return {
            'volatility': 0.02,
            'trend_strength': 0.01,
            'volume_ratio': 1.0,
            'avg_range': 0.02
        }
    
    try:
        # 필요한 컬럼 확인
        required_columns = ['close']
        if not all(col in market_data.columns for col in required_columns):
            return {
                'volatility': 0.02,
                'trend_strength': 0.01,
                'volume_ratio': 1.0,
                'avg_range': 0.02
            }
        
        # 종가 시리즈
        closes = market_data['close']
        
        # 1. 변동성 계산
        volatility = calculate_market_volatility(closes)
        
        # 2. 추세 강도 계산
        trend_strength = calculate_trend_strength(closes)
        
        # 추가 특성 계산
        features = {
            'volatility': volatility,
            'trend_strength': trend_strength,
        }
        
        # 3. 거래량 비율 (추가 특성)
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            if not volume.empty and not (volume.iloc[-20:] == 0).all():
                avg_volume = volume.iloc[-20:].mean()
                current_volume = volume.iloc[-1]
                volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                features['volume_ratio'] = min(volume_ratio, 5.0)  # 극단값 제한
        
        # 4. 평균 가격 범위 (추가 특성)
        if all(col in market_data.columns for col in ['high', 'low']):
            high = market_data['high']
            low = market_data['low']
            price_range = (high / low - 1).tail(20).mean()
            features['avg_range'] = float(price_range)
        
        return features
    
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return {
            'volatility': 0.02,
            'trend_strength': 0.01,
            'volume_ratio': 1.0,
            'avg_range': 0.02
        }


def adjust_weights_by_market_condition(weights: List[float],
                                    volatility: float = 0.02,
                                    trend_strength: float = 0.01,
                                    min_weight: float = 0.05) -> List[float]:
    """
    시장 상황에 따른 가중치 조정 함수 (리스트 버전)
    
    Args:
        weights (List[float]): 현재 모델 가중치
        volatility (float): 시장 변동성 (0-1)
        trend_strength (float): 추세 강도 (0-1)
        min_weight (float): 최소 가중치
        
    Returns:
        List[float]: 조정된 가중치
    """
    if not weights:
        return []
    
    # 시장 특성에 따른 가중치 조정
    adjusted_weights = weights.copy()
    num_models = len(weights)
    
    if num_models < 2:
        return weights
    
    # 간소화된 버전 - 가장 높은 가중치 모델에 약간 더 가중치를 부여
    max_idx = np.argmax(weights)
    
    # 변동성과 추세 강도를 결합하여 조정 강도 결정
    adjustment_strength = (volatility + trend_strength) / 2
    adjustment_factor = min(0.3, adjustment_strength)  # 최대 30%까지 조정
    
    # 가중치 조정
    for i in range(num_models):
        if i == max_idx:
            # 최대 가중치 모델 강화
            adjusted_weights[i] *= (1 + adjustment_factor)
        else:
            # 나머지 모델 약화
            adjusted_weights[i] *= (1 - adjustment_factor / (num_models - 1))
    
    # 정규화하여 합이 1이 되도록
    total = sum(adjusted_weights)
    if total <= 0:
        return weights
    
    normalized_weights = [max(min_weight, w / total) for w in adjusted_weights]
    
    # 최종 정규화
    total = sum(normalized_weights)
    return [w / total for w in normalized_weights]


def get_market_regime(market_features: Dict[str, float]) -> str:
    """
    시장 국면 분류 함수
    
    Args:
        market_features (Dict[str, float]): 시장 특성
        
    Returns:
        str: 시장 국면 ('high_volatility', 'strong_trend', 'sideways', 'normal')
    """
    volatility = market_features.get('volatility', 0.02)
    trend_strength = market_features.get('trend_strength', 0.01)
    
    # 시장 국면 분류
    if volatility > 0.04:
        return 'high_volatility'
    elif trend_strength > 0.06:
        return 'strong_trend'
    elif trend_strength < 0.01 and volatility < 0.02:
        return 'sideways'
    else:
        return 'normal'


def get_optimal_model_weights_for_regime(regime: str) -> Dict[str, Dict[str, float]]:
    """
    시장 국면별 최적 모델 가중치 반환 함수
    
    Args:
        regime (str): 시장 국면
        
    Returns:
        Dict[str, Dict[str, float]]: 모델 그룹 및 개별 모델 가중치
    """
    # 시장 국면별 최적 가중치 정의
    regime_weights = {
        'high_volatility': {
            'group_weights': {
                'ml': 0.55,
                'technical': 0.45
            },
            'model_weights': {
                'gru': 0.60,
                'rf': 0.40,
                'tech': 0.30,
                'rsi': 0.35,
                'macd': 0.35
            }
        },
        'strong_trend': {
            'group_weights': {
                'ml': 0.60,
                'technical': 0.40
            },
            'model_weights': {
                'gru': 0.70,
                'rf': 0.30,
                'tech': 0.25,
                'rsi': 0.25,
                'macd': 0.50
            }
        },
        'sideways': {
            'group_weights': {
                'ml': 0.45,
                'technical': 0.55
            },
            'model_weights': {
                'gru': 0.40,
                'rf': 0.60,
                'tech': 0.40,
                'rsi': 0.40,
                'macd': 0.20
            }
        },
        'normal': {
            'group_weights': {
                'ml': 0.55,
                'technical': 0.45
            },
            'model_weights': {
                'gru': 0.55,
                'rf': 0.45,
                'tech': 0.33,
                'rsi': 0.33,
                'macd': 0.34
            }
        }
    }
    
    return regime_weights.get(regime, regime_weights['normal']) 