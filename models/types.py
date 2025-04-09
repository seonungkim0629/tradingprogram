"""
타입 정의 모듈

비트코인 트레이딩 봇의 타입 정의를 포함합니다.
"""

from typing import Dict, Any, TypedDict, Optional, Union, List, Tuple


class TrainingMetrics(TypedDict, total=False):
    """
    모델 훈련 메트릭스
    
    훈련 과정에서 반환되는 메트릭을 정의한 타입입니다.
    """
    accuracy: float  # 정확도
    f1_score: float  # F1 점수
    precision: float  # 정밀도
    recall: float  # 재현율
    loss: float  # 손실 값
    training_time: float  # 훈련 소요 시간(초)
    feature_count: int  # 사용된 특성 수
    mae: float  # 평균 절대 오차
    mse: float  # 평균 제곱 오차
    rmse: float  # 평균 제곱근 오차
    r2: float  # 결정 계수
    error: str  # 오류 메시지
    is_trained: bool  # 훈련 성공 여부
    
    # 앙상블 모델 특화 필드
    direction_metrics: Dict[str, Dict[str, Any]]  # 방향 모델별 지표
    price_metrics: Dict[str, Dict[str, Any]]  # 가격 모델별 지표
    direction_models_trained: bool  # 모든 방향 모델 훈련 성공 여부
    price_models_trained: bool  # 모든 가격 모델 훈련 성공 여부 